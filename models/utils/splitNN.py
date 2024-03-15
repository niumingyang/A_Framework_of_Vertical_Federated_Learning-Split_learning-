import torch
import torch.nn as nn
import logging

from utils.training import AverageMeter, keep_predict_loss
from utils.metric import compute_auc, compute_auc_acc, compute_prec_rec, compute_f1
from privacy import active_learn, norm_attack, emb_attack, distance_correlation

# guest is label party, host is non-label party


class splitNN(nn.Module):
    def __init__(self, train_config, privacy_config):
        super(splitNN, self).__init__()
        self.host_bottom_model = None
        self.guest_bottom_model = None
        self.guest_top_model = None

        self.train_loader = None
        self.test_loader = None

        self.loss_func_top_model = None
        self.loss_func_bottom_model = keep_predict_loss

        self.host_bottom_optimizer = None
        self.guest_bottom_optimizer = None
        self.guest_top_optimizer = None

        self.print_threshold = int(train_config['printInterval'])
        self.epochs = int(train_config['epoch'])
        self.device = train_config['device']
        self.batch_size = int(train_config['batchSize'])
        self.lr = float(train_config['learningRate'])
        self.mal_opt = int(train_config['useMalOptimizer'])

        self.norm_attack = int(privacy_config['normAttack'])
        self.emb_attack = int(privacy_config['embAttack'])

        self.dcor_loss = int(privacy_config['dcorLoss'])
        self.dcor_coef = float(privacy_config['dcorCoef'])
        self.loss_func_dcor = distance_correlation.DisCorLoss()

    def forward(self, x_host, x_guest):
        host_out = self.host_bottom_model(x_host)
        guest_out = self.guest_bottom_model(x_guest)
        return self.guest_top_model(host_out, guest_out)

    def train_per_epoch(self, epoch):
        self.train()
        logging.info("")
        logging.info("Epoch [{} | {}]:   BatchSize = {}\thostLR = {:.7f}  guestLR = {:.7f}  topLR = {:.7f}"
                     .format(epoch, self.epochs, self.batch_size,
                             self.host_bottom_optimizer.state_dict()[
                                 'param_groups'][0]['lr'],
                             self.guest_bottom_optimizer.state_dict()[
                                 'param_groups'][0]['lr'],
                             self.guest_top_optimizer.state_dict()[
                                 'param_groups'][0]['lr']))
        size = len(self.train_loader.dataset)
        model_auc = AverageMeter()
        model_acc = AverageMeter()
        norm_auc = AverageMeter()
        emb_auc = AverageMeter()
        progress_percent = 0
        for batch_idx, (x_host, x_guest, y) in enumerate(self.train_loader):
            # prepare data
            x_host = x_host.float().to(self.device)
            x_guest = x_guest.float().to(self.device)
            y = y.float().to(self.device)

            # bottom model forward
            host_embedding = self.host_bottom_model(x_host)
            if self.emb_attack == 1:
                emb_batch_auc = emb_attack.emb_attack(host_embedding, y)
                emb_auc.update(emb_batch_auc)

            guest_embedding = self.guest_bottom_model(x_guest)

            # top model forward
            guest_input = guest_embedding
            host_input = torch.tensor([], requires_grad=True)
            host_input.data = host_embedding.data
            top_output = self.guest_top_model(host_input, guest_input)

            # update guest top and bottom model
            top_loss = self.loss_func_top_model(top_output, y)
            if self.dcor_loss == 1:
                top_loss = top_loss + self.dcor_coef * \
                    self.loss_func_dcor(host_input, y)
            self.guest_top_optimizer.zero_grad()
            self.guest_bottom_optimizer.zero_grad()
            top_loss.backward()
            self.guest_top_optimizer.step()
            self.guest_bottom_optimizer.step()

            # update host bottom model
            host_grad = host_input.grad
            if self.norm_attack == 1:
                norm_batch_auc = norm_attack.norm_attack(host_grad, y)
                norm_auc.update(norm_batch_auc)

            host_loss = self.loss_func_bottom_model(host_embedding, host_grad)
            self.host_bottom_optimizer.zero_grad()
            host_loss.backward()
            self.host_bottom_optimizer.step()

            # print to screen
            current = (batch_idx + 1) * self.batch_size
            current = min(current, size)
            percent = current / size * 100
            batch_auc, batch_acc = compute_auc_acc(y, top_output)
            model_auc.update(batch_auc)
            model_acc.update(batch_acc)
            # less print
            if self.print_threshold != None:
                if percent - progress_percent < self.print_threshold:
                    continue
                else:
                    progress_percent += self.print_threshold
            logging.info('Epoch [{:d}]: [{:d}/{:d}] ({:.2f}%)\tLoss: {:.4f}\tAUC: {:.4f}({:.4f})'.format(
                epoch, current, size, percent, top_loss, model_auc.val, model_auc.avg))
            if self.norm_attack == 1:
                logging.info('\tnorm attack AUC: {:.4f}({:.4f})'.format(
                    norm_auc.val, norm_auc.avg))
            if self.emb_attack == 1:
                logging.info('\temb attack AUC: {:.4f}({:.4f})'.format(
                    emb_auc.val, emb_auc.avg))

    def test_model(self, istrain):
        self.eval()
        logging.info("")
        logging.info("Evaluate model on {} set:".format(
            'train' if istrain else 'test'))
        data_loader = self.train_loader if istrain else self.test_loader
        test_loss = AverageMeter()
        precision = AverageMeter()
        recall = AverageMeter()
        size = len(data_loader.dataset)
        target_list = []
        output_list = []
        print_threshold = 20
        for batch_idx, (x_host, x_guest, y) in enumerate(data_loader):
            # prepare data
            x_host = x_host.float().to(self.device)
            x_guest = x_guest.float().to(self.device)
            y = y.float().to(self.device)

            # compute and evaluate
            top_out = self.forward(x_host, x_guest)
            test_loss.update(self.loss_func_top_model(top_out, y).data.item())
            target_list.append(y.detach().cpu())
            output_list.append(top_out.detach().cpu())
            prec, rec = compute_prec_rec(y, top_out)
            precision.update(prec)
            recall.update(rec)

            # print
            current = (batch_idx + 1) * self.batch_size
            percent = current / size * 100
            if percent >= print_threshold:
                logging.info('\tprocessing data {:d}%'.format(print_threshold))
                print_threshold += 20
        target_list = torch.concat(target_list)
        output_list = torch.concat(output_list)
        test_auc, test_acc = compute_auc_acc(target_list, output_list)
        logging.info('Loss: {:.5f}\tAUC: {:.4f}\tAccuracy: {:.4f}'.format(
            test_loss.avg, test_auc, test_acc))
        logging.debug('precision: {:.4f}\trecall: {:.4f}\tf1: {:.4f}'.format(
            precision.avg, recall.avg, compute_f1(precision.avg, recall.avg)))

    def train_model(self):
        for epoch in range(self.epochs):
            self.train_per_epoch(epoch+1)
            self.test_model(istrain=True)
            self.test_model(istrain=False)

    def save_model(self, path):
        torch.save(self, path)

    def get_optimizer(self, part, optimizer='adagrad'):
        def choose_opt(model, lr, opt_name):
            if opt_name == 'adagrad':
                opt = torch.optim.Adagrad(
                    params=model.parameters(), lr=lr, initial_accumulator_value=0.1)
            elif opt_name == "adam":
                opt = torch.optim.Adam(params=model.parameters(), lr=lr)
            elif opt_name == "sgd":
                opt = torch.optim.SGD(params=model.parameters(), lr=lr)
            elif opt_name == "adadelta":
                opt = torch.optim.Adadelta(params=model.parameters(), lr=lr)
            else:
                raise ValueError(
                    "Do not support optimizer: {}.".format(opt_name))
            return opt

        if part == 'host':
            if self.mal_opt:
                return active_learn.MaliciousSGD(params=self.host_bottom_model.parameters(), lr=self.lr)
            else:
                return choose_opt(self.host_bottom_model, self.lr, optimizer)
        elif part == 'guest':
            return choose_opt(self.guest_bottom_model, self.lr, optimizer)
        elif part == 'top':
            return choose_opt(self.guest_top_model, self.lr, optimizer)
        else:
            raise ValueError("Do not support part: {}.".format(part))

    def to(self, device):
        self.host_bottom_model = self.host_bottom_model.to(device)
        self.guest_bottom_model = self.guest_bottom_model.to(device)
        self.guest_top_model = self.guest_top_model.to(device)

    def train(self):
        self.host_bottom_model.train()
        self.guest_bottom_model.train()
        self.guest_top_model.train()

    def eval(self):
        self.host_bottom_model.eval()
        self.guest_bottom_model.eval()
        self.guest_top_model.eval()

    def print_log(self):
        logging.info('host_bottom_model:\n' + str(self.host_bottom_model))
        logging.info('guest_bottom_model:\n' + str(self.guest_bottom_model))
        logging.info('guest_top_model:\n' + str(self.guest_top_model))
