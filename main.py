import argparse
import pickle
import torch
from sklearn.metrics import roc_auc_score
from attacks import flag
from model import STConv
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class MyGraphDataset(Dataset):
    def __init__(self, graph_data_list):
        super(MyGraphDataset, self).__init__()
        self.graphs = graph_data_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def get(self, idx):
        return self.graphs[idx]

    def len(self):
        return len(self.graphs)
# pretrain:D:/dataset/rwf/train_150_400_9.pickle+D:/dataset/rwf/test_150_400_9.pickle
with open("D:/dataset/hockey/hockey_train_50_800.pickle", "rb") as file:
    datasets = pickle.load(file)
with open("D:/dataset/hockey/hockey_test_50_200.pickle", "rb") as f:
    test_datasets = pickle.load(f)
parser = argparse.ArgumentParser(description='GNN with Pytorch Geometrics')
parser.add_argument('-m', type=int, default=3)
parser.add_argument('--step-size', type=float, default=8e-3)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--lr', type=int, default=0.005)

args = parser.parse_args()
# 创建数据加载器1
dataloader = DataLoader(datasets, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_datasets, batch_size=8, shuffle=False)
print("已加载数据")
criterion = torch.nn.BCELoss().cuda()
# 遍历数据加载器中的批次数据
# model = GCN(in_channels=4, hidden_channels=2, out_channels=1,num_layers=3,dropout=0.05).cuda()
model = GATV1(in_channels=4, hidden_channels=2, out_channels=1, num_layers=3, dropout=0.2, heads=8).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
model.reset_parameters()
# model.load_state_dict(torch.load('D:/pre_model/gnn_rwf111.pt'))
label_list = pred_list = []
max_ac = max_auc = 0
for epoch in range(args.epochs):
    # print("第{}次训练开始：".format(epoch))
    model.train()
    model = model.cuda()
    i = 0
    # optimizer.zero_grad()
    for data in dataloader:
        data = data.cuda()
        i += 1
        features = data.x
        edge_index = data.edge_index    # interation edge
        edge_attr = data.edge_attr      # weight of interation edge
        self_link = data.self_link_index
        temp_link = data.temporal_index
        add_index = torch.cat([self_link, temp_link], dim=1)    # self-link edge
        # add_index = int(add_index.item())
        y = data.y.to(torch.float32)
        y = torch.unsqueeze(y, dim=1)
        # 定义一个函数，函数输入是perturb
        # forward = lambda perturb: model(features + perturb, edge_index, edge_attr, add_index, data.batch).to(torch.float32)
        # model_forward = (model, forward)
        # perturb_shape = data.x.shape
        # loss, output = flag(model_forward, perturb_shape, y, args, optimizer, args.device, criterion)
        optimizer.zero_grad()
        output = model(features, edge_index, edge_attr, add_index, data.batch)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    #   test
    model.eval()
    acc = 0
    # model = model.cpu()
    for data in test_dataloader:
        data = data.cuda()
        features = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        self_link = data.self_link_index
        temp_link = data.temporal_index
        add_index = torch.cat([self_link, temp_link], dim=1)

        y = data.y.to(torch.float32)
        label_list = label_list + y.cpu().numpy().tolist()
        # print(y, label_list)
        y_pred = model(features, edge_index, edge_attr, add_index, data.batch)
        pred_list = pred_list + y_pred.cpu().detach().numpy().tolist()
        y = torch.unsqueeze(y, dim=1)
        y_pred_round = y_pred.round()
        acc_score = y_pred_round.eq(y).sum()
        acc += acc_score
    ac = acc / len(test_datasets)
    auc = roc_auc_score(label_list, pred_list)
    ac = ac.cpu().numpy()
    # print(f"AUC: {auc}")
    # print("第{}轮测试：".format(epoch),ac)
    print(ac)
    if ac > max_ac:
        max_ac = ac
        # torch.save(model.state_dict(), 'G:/gcn/online/pretrained_models/gnn_hockey.pt')
        # torch.save(model.state_dict(), 'D:/pre_model/gnn_rwf111.pt')
        # torch.save(model, 'pre_model_no_flag_avdv.pth')
    if auc > max_auc:
        max_auc = auc
print(f"max accuracy is：{max_ac} ")
print(f"max auc is：{max_auc} ")
