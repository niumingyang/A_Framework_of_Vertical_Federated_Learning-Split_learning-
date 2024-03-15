import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as data
import logging

from utils.training import AverageMeter
from utils.metric import compute_auc_acc, precision_recall


class BottomModelPlus(nn.Module):
    def __init__(self, bottom_model, emb_size, out_size):
        super(BottomModelPlus, self).__init__()
        self.bottom_model = bottom_model
        self.fc_final = nn.Linear(emb_size, out_size)
        self.bn_final = nn.BatchNorm1d(emb_size)
        self.relu = nn.ReLU()
        self.fc_final.apply(self.weights_init_ones)

    def weights_init_ones(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.ones_(m.weight)

    def forward(self, x):
        x = self.bottom_model(x)
        x = self.bn_final(x)
        x = self.relu(x)
        x = self.fc_final(x)
        return x


class ModelCompletion():
    def __init__(self, resume, dataset, config):
        self.device = config['training']['device']
        self.n_labeled = int(config['privacy']['mcNlabeled'])
        self.batch_size = int(config['privacy']['mcBatch'])
        self.lr = float(config['privacy']['mcLr'])
        self.epochs = int(config['privacy']['mcEpochs'])
        self.val_iteration = int(config['privacy']['mcIter'])
        self.alpha = float(config['privacy']['mcalpha'])
        self.lambda_u = float(config['privacy']['mcLambdau'])
        self.T = float(config['privacy']['mcT'])
        self.ema_decay = float(config['privacy']['mcEmadecay'])
        self.class_num = dataset.class_num

        # create inference model
        vfl_model = torch.load(resume)
        bottom_model = vfl_model.host_bottom_model
        emb_size = int(config['dataset']['embeddingDim']) // 2
        top_fc = config['dataset']['topFcDim'].split(',')
        self.mc_model = BottomModelPlus(bottom_model, emb_size, self.class_num)
        self.mc_model.to(self.device)

        # create ema model
        vfl_model = torch.load(resume)
        bottom_model = vfl_model.host_bottom_model
        self.ema_model = BottomModelPlus(bottom_model, emb_size, self.class_num)
        for param in self.ema_model.parameters():
            param.detach_()
        self.ema_model.to(self.device)

        # prepare dataset
        labeled_set = dataset.labeled_set
        unlabeled_set = dataset.unlabeled_set
        train_set = dataset.train_set
        val_set = dataset.val_set
        self.labeled_loader = data.DataLoader(
            labeled_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.unlabeled_loader = data.DataLoader(
            unlabeled_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        dataset_bs = self.batch_size * 10
        self.trainset_loader = data.DataLoader(
            train_set, batch_size=dataset_bs, shuffle=True, drop_last=True)
        self.valset_loader = data.DataLoader(
            val_set, batch_size=dataset_bs, shuffle=False)

        # prepare training
        self.train_criterion = SemiLoss(self.lambda_u)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.mc_model.parameters(), lr=self.lr)
        self.ema_optimizer = WeightEMA(
            self.mc_model, self.ema_model, lr=self.lr, alpha=self.ema_decay)

    def ssl_training(self):
        best_auc = 0
        for epoch in range(self.epochs):
            logging.info('')
            logging.info('Epoch: [%d | %d] LR: %f' % (
                epoch + 1, self.epochs, self.optimizer.state_dict()['param_groups'][0]['lr']))
            total_loss, train_loss_x, train_loss_u = self.train_per_epoch(epoch)
            logging.info("---Label inference on complete training dataset:")
            trainset_loss, trainsec_acc, trainset_auc = self.validate(self.ema_model, self.trainset_loader)
            logging.info("---Label inference on complete testing dataset:")
            valset_loss, valsec_acc, valset_auc = self.validate(self.ema_model, self.valset_loader)
            best_auc = max(best_auc, trainset_auc)
        logging.info('Best AUC: {}'.format(best_auc))
            

    def train_per_epoch(self, epoch):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        ws = AverageMeter()

        labeled_train_iter = iter(self.labeled_loader)
        unlabeled_train_iter = iter(self.unlabeled_loader)

        self.mc_model.train()

        for batch_idx in range(self.val_iteration):
            try:
                inputs_x, targets_x = next(labeled_train_iter)
            except StopIteration:
                labeled_train_iter = iter(self.labeled_loader)
                inputs_x, targets_x = next(labeled_train_iter)
            try:
                inputs_u, = next(unlabeled_train_iter)
            except StopIteration:
                unlabeled_train_iter = iter(self.unlabeled_loader)
                inputs_u, = next(unlabeled_train_iter)

            # Transform label to one-hot
            targets_x = targets_x.view(-1, 1).type(torch.long)
            targets_x = torch.zeros(self.batch_size, self.class_num).scatter_(1, targets_x, 1)

            inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
            inputs_u = inputs_u.to(self.device)

            with torch.no_grad():
                # compute guessed labels of unlabel samples
                targets_x.view(-1, 1).type(torch.long)
                outputs_u = self.mc_model(inputs_u)
                p = torch.softmax(outputs_u, dim=1)
                pt = p ** (1 / self.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            # mixup
            all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
            all_targets = torch.cat([targets_x, targets_u], dim=0)

            l = np.random.beta(self.alpha, self.alpha)

            l = max(l, 1 - l)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabeled samples between batches to get correct batch norm calculation
            mixed_input = list(torch.split(mixed_input, self.batch_size))
            mixed_input = interleave(mixed_input, self.batch_size)

            logits = [self.mc_model(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(self.mc_model(input))

            # put interleaved samples back
            logits = interleave(logits, self.batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            Lx, Lu, w = self.train_criterion(logits_x, mixed_target[:self.batch_size], logits_u, mixed_target[self.batch_size:], epoch + batch_idx / self.val_iteration)

            loss = Lx + w * Lu

            # record loss
            losses.update(loss.item(), inputs_x.size(0))
            losses_x.update(Lx.item(), inputs_x.size(0))
            losses_u.update(Lu.item(), inputs_x.size(0))
            ws.update(w, inputs_x.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema_optimizer.step()

            if batch_idx % 250 == 0:
                logging.info("batch_idx: {} loss: {}".format(batch_idx, losses.avg))
        return losses.avg, losses_x.avg, losses_u.avg

    def validate(self, model, dataloader):
        losses = AverageMeter()
        acc = AverageMeter()
        auc = AverageMeter()
        precision = AverageMeter()
        recall = AverageMeter()

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, _, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # compute output
                inputs = inputs.type(torch.float)
                outputs = model(inputs)
                targets = targets.type(torch.long)
                loss = self.criterion(outputs, targets)

                # measure accuracy and record loss
                preds = F.softmax(outputs, dim=1)
                preds = preds[:, 1]
                batch_auc, batch_acc = compute_auc_acc(targets, preds)
                if self.class_num == 2:
                    prec, rec = precision_recall(outputs, targets)
                    precision.update(prec, inputs.size(0))
                    recall.update(rec, inputs.size(0))

                losses.update(loss.item(), inputs.size(0))
                acc.update(batch_acc.item(), inputs.size(0))
                auc.update(batch_auc.item(), inputs.size(0))

        logging.info("Dataset Overall Statistics:")
        if self.class_num == 2:
            if (precision.avg + recall.avg) != 0:
                f1 = (precision.avg * recall.avg) / (precision.avg + recall.avg)
            else:
                f1 = 0
            logging.info('  precision: {} recall: {} F1: {}'.format(precision.avg, recall.avg, f1))
        logging.info('  accuracy: {}, auc: {}'.format(acc.avg, auc.avg))
        return losses.avg, acc.avg, auc.avg


def linear_rampup(current, rampup_length=10):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __init__(self, lambda_u):
        self.lambda_u = lambda_u

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x,
                         dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, self.lambda_u * linear_rampup(epoch)


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.type(torch.float)
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param = param.type(torch.float)
            param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]
