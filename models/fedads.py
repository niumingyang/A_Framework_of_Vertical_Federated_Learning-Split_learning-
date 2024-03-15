import torch
import torch.nn as nn

from models.utils import splitNN, wide_deep
from datasets.fedads.fedads_data import *


class FedadsModel(splitNN.splitNN):
    def __init__(self, cfg):
        super(FedadsModel, self).__init__(
            train_config=cfg['training'], privacy_config=cfg['privacy'])
        sec = cfg['dataset']
        guest_cols = sec['guestCols'].split(',')
        host_cols = sec['hostCols'].split(',')
        guest_dense_dim = int(sec['guestDenseNum'])
        guest_sparse_dim = int(sec['guestSparseNum'])
        host_dense_dim = int(sec['hostDenseNum'])
        host_sparse_dim = int(sec['hostSparseNum'])
        embedding_size = int(sec['embeddingSize'])
        bottom_fc = sec['bottomFcDim'].split(',')
        bottom_fc = [int(x) for x in bottom_fc]
        top_fc = sec['topFcDim'].split(',')
        top_fc = [int(x) for x in top_fc]
        activation_dim = int(sec['embeddingDim'])

        train_set = Fedads(guest_cols, host_cols, istrain=True)
        test_set = Fedads(guest_cols, host_cols, istrain=False)
        host_fea_num, guest_fea_num = train_set.get_feat()

        self.host_bottom_model = wide_deep.HostBottomModel(fc_dims=bottom_fc + [activation_dim // 2],
                                                           dense_num=host_dense_dim,
                                                           sparse_num=host_sparse_dim,
                                                           embedding_size=embedding_size,
                                                           fea_num=host_fea_num)
        self.guest_bottom_model = wide_deep.GuestBottomModel(fc_dims=bottom_fc + [activation_dim // 2],
                                                             dense_num=guest_dense_dim,
                                                             sparse_num=guest_sparse_dim,
                                                             embedding_size=embedding_size,
                                                             fea_num=guest_fea_num)
        self.guest_top_model = wide_deep.GuestTopModel(
            fc_dims=[activation_dim] + top_fc)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.loss_func_top_model = nn.BCEWithLogitsLoss()

        self.to(self.device)

        self.host_bottom_optimizer = self.get_optimizer('host')
        self.guest_bottom_optimizer = self.get_optimizer('guest')
        self.guest_top_optimizer = self.get_optimizer('top')

        self.print_log()
