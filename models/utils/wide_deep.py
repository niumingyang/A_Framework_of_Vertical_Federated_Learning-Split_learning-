import torch
import torch.nn as nn

class Wide(nn.Module):
    def __init__(self, input_dim):
        super(Wide, self).__init__()
        self.fc = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.fc(x)


class Deep(nn.Module):
    def __init__(self, hidden_units, dropout=0.):
        super(Deep, self).__init__()
        self.dnn = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(
            zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i, fc in enumerate(self.dnn):
            x = fc(x)
            if i < len(self.dnn)-1:  
                x = self.relu(x)
        return self.dropout(x)


class Embed(nn.Module):
    def __init__(self, fea_num, embedding_size):
        super(Embed, self).__init__()
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat, embedding_dim=embedding_size)
            for i, feat in enumerate(fea_num)
        })

    def forward(self, x):
        sparse_embeds = [self.embed_layers['embed_' +
                                           str(i)](x[:, i]) for i in range(x.shape[1])]
        return torch.cat(sparse_embeds, axis=-1)


class HostBottomModel(nn.Module):
    def __init__(self, fc_dims, dense_num, sparse_num, embedding_size, fea_num):
        super(HostBottomModel, self).__init__()
        self.dense_num = dense_num
        self.sparse_num = sparse_num
        self.embed_part = Embed(fea_num, embedding_size)
        self.deep_part = Deep(
            hidden_units=[sparse_num * embedding_size + dense_num] + fc_dims)

    def forward(self, x):
        dense_input, sparse_input = x[:,
                                      :self.dense_num], x[:, self.dense_num:]
        sparse_input = sparse_input.long()
        sparse_embeds = self.embed_part(sparse_input)
        deep_input = torch.cat([sparse_embeds, dense_input], axis=-1)
        deep_out = self.deep_part(deep_input)

        return deep_out


class GuestBottomModel(nn.Module):
    def __init__(self, fc_dims, dense_num, sparse_num, embedding_size, fea_num):
        super(GuestBottomModel, self).__init__()
        self.dense_num = dense_num
        self.sparse_num = sparse_num
        self.wide_embed = Embed(fea_num, embedding_size)
        self.deep_embed = Embed(fea_num, embedding_size)
        self.wide_sparse_part = Wide(input_dim=sparse_num * embedding_size)
        self.wide_dense_part = Wide(input_dim=dense_num)
        self.deep_part = Deep(
            hidden_units=[sparse_num * embedding_size + dense_num] + fc_dims)

    def forward(self, x):
        dense_input, sparse_input = x[:,
                                      :self.dense_num], x[:, self.dense_num:]
        sparse_input = sparse_input.long()
        wide_embeds = self.wide_embed(sparse_input)
        deep_embeds = self.deep_embed(sparse_input)
        deep_input = torch.cat([deep_embeds, dense_input], axis=-1)

        wide_out = self.wide_sparse_part(wide_embeds) + self.wide_dense_part(dense_input)
        deep_out = self.deep_part(deep_input)

        return torch.concat([deep_out, wide_out], dim=1)


class GuestTopModel(nn.Module):
    def __init__(self, fc_dims):
        super(GuestTopModel, self).__init__()
        self.deep_part = Deep(fc_dims)
        self.relu = nn.ReLU()
        
    def forward(self, host_input, guest_input):
        wide_out = guest_input[:, -1]
        deep_input = torch.concat([host_input, guest_input[:, :-1]], dim=1)
        deep_input = self.relu(deep_input)
        deep_out = self.deep_part(deep_input)
        deep_out = deep_out.squeeze()
        return wide_out + deep_out