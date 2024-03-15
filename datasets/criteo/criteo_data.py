import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import logging

TRAIN_FILE = './datasets/criteo/criteo_train.csv'
VAL_FILE = './datasets/criteo/criteo_test.csv'
FEAT_FILE = './datasets/criteo/fea_num.npy'


class Criteo(data.Dataset):
    def __init__(self, guest_cols, host_cols, istrain=True):
        super(Criteo, self).__init__()
        self.istrain = istrain

        if istrain:
            input_file = TRAIN_FILE
        else:
            input_file = VAL_FILE

        df = pd.read_csv(input_file)

        self.host_data = df[host_cols].values
        self.guest_data = df[guest_cols].values
        self.labels = df['label'].values

        if istrain:
            self.host_feat_num = []
            self.guest_feat_num = []
            fea_num = np.load(FEAT_FILE, allow_pickle=True)[0]
            for fea in host_cols:
                if fea in fea_num:
                    self.host_feat_num.append(fea_num[fea])
            for fea in guest_cols:
                if fea in fea_num:
                    self.guest_feat_num.append(fea_num[fea])

        self.print_log()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.host_data[index], self.guest_data[index], self.labels[index]

    def print_log(self):
        logging.info(('train' if self.istrain else 'test') +
                     ' set: ' + str(len(self.labels)))

    def get_feat(self):
        if self.istrain:
            return self.host_feat_num, self.guest_feat_num
        else:
            return None


def split_for_ssl(labels, n_labeled_per_class, num_classes):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs


class CriteoForSSL():
    def __init__(self, cfg):
        self.class_num = 2
        n_labeled = int(cfg['privacy']['mcNlabeled'])
        guest_cols = cfg['dataset']['guestCols'].split(',')
        host_cols = cfg['dataset']['hostCols'].split(',')
        self.train_set = Criteo(guest_cols, host_cols, istrain=True)
        self.val_set = Criteo(guest_cols, host_cols, istrain=False)

        df = pd.read_csv(TRAIN_FILE)
        all_data = df[host_cols].values
        labels = df['label'].values
        labeled_idx, unlabeled_idx = split_for_ssl(
            labels, int(n_labeled / self.class_num), self.class_num)
        self.labeled_set = data.TensorDataset(torch.tensor(
            all_data[labeled_idx], dtype=torch.float32), torch.tensor(labels[labeled_idx], dtype=torch.float32))
        self.unlabeled_set = data.TensorDataset(
            torch.tensor(all_data[unlabeled_idx], dtype=torch.float32))
        logging.info('#Labeled: {}    #Unlabeled: {}'.format(
            len(labeled_idx), len(unlabeled_idx)))
