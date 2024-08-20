from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GATv2Conv, global_mean_pool
from torch.nn import Linear
import torch
import torch.nn as nn
import torch.nn.functional as F


class STConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, heads):
        super(STConv, self).__init__()
        self.num_layers = num_layers
        self.conv1 = torch.nn.ModuleList()
        self.conv1.append(GATConv(in_channels, hidden_channels, heads))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 1):
            self.conv1.append(GATConv(heads * hidden_channels, hidden_channels, heads))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        # self.conv1.append(GATConv(heads * hidden_channels, out_channels, 1))

        self.GCN_conv = torch.nn.ModuleList()
        self.bns_gcn = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.GCN_conv.append(GCNConv(heads * hidden_channels, heads * hidden_channels))
            self.bns_gcn.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        self.GCN_conv.append(GCNConv(heads * hidden_channels, out_channels))
        self.dropout = dropout
        self.lin = Linear(hidden_channels, out_channels)
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        for conv in self.conv1:
            conv.reset_parameters()
        for conv in self.GCN_conv:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for bn in self.bns_gcn:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr, add_index, batch):
        # z = x
        for i, conv in enumerate(self.conv1):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < self.num_layers - 3:
                x = self.bns[i](x)
            x = F.leaky_relu(x)  # 对交互边进行卷积
        # x = self.conv1_out(x, edge_index, edge_attr=edge_attr)

        for i, conv in enumerate(self.GCN_conv):
            x = conv(x, add_index)  # 注意力层附加的边权值是乘以一个矩阵与计算的注意力系数相加然后在进行特征更新
            if i < self.num_layers - 3:
                x = self.bns_gcn[i](x)
            x = F.leaky_relu(x)
        # z = self.conv3(z, add_index)
        # z = F.leaky_relu(z)
        # z = self.bn1(z)
        # x = z + x
        x = F.leaky_relu(x)
        x = global_mean_pool(x, batch)
        # x = TopKPooling(x, batch)

        return self.sigmoid(x)

