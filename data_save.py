
from data_process import *
import os
import pickle
class MyGraphDataset(Dataset):
    def __init__(self, graph_data_list):
        super(MyGraphDataset, self).__init__()
        self.graphs = graph_data_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

def process_data():
    # folder_path = "G:/gcn/GNNcv/input/test"  # 文件夹路径
    folder_path = "G:/gcn/GNNcv/input/RWF2000/RWF_train"
    file_names = os.listdir(folder_path)
    graph_data_list = []
    file_paths = [os.path.join(folder_path, file_name).replace("\\", "/") for file_name in file_names]
    frame = 150
    i = 0
    for path in file_paths:
        data = data_process(path, frame)
        # print(data)
        graph_data_list += data
        i += 1
        print("已处理的视频数：", i)
    return graph_data_list

def save_data(graph_data_list):
    # 构建多图数据集
    dataset = MyGraphDataset(graph_data_list)

    # 保存dataset到文件
    # with open("D:/dataset/test.pickle", "wb") as file:
    with open("D:/dataset/rwf/rwf_800_8.pickle", "wb") as file:
        pickle.dump(dataset, file)
    print("已保存数据集文件")

# 调用数据处理和保存函数
list1 = process_data()
print("data has finished")
save_data(list1)
print("Number of Graph is {}".format(len(list1)))


