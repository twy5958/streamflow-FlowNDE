import os
import csv
import numpy as np
from fastdtw import fastdtw
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from einops import repeat
import pandas as pd
import controldiffeq
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据目录相对于项目根目录的路径
data_dir = os.path.join(project_root, 'data')
files = {
    'PEMS03': ['PEMS03/pems03.npz', 'PEMS03/distance.csv'],
    'PEMS04': ['PEMS04/PEMS04.npz', 'PEMS04/PEMS04.csv'],
    'PEMS07': ['PEMS07/pems07.npy', 'PEMS07/distance.csv'],
    'PEMS08': ['PEMS08/pems08.npy', 'PEMS08/distance.csv'],
    'PEMSBAY': ['PEMSBAY/pems_bay.npy', 'PEMSBAY/distance.csv'],
    'PeMSD7M': ['PeMSD7M/PeMSD7M.npy', 'PeMSD7M/distance.csv'],
    'PeMSD7L': ['PeMSD7L/PeMSD7L.npy', 'PeMSD7L/distance.csv'],
    'syn': ['syn/syndata.npy', 'syn/distance.csv'],
    'changjiang':['changjiang/changjiang_com.npz','changjiang/distance.csv','changjiang/mask.npz']
}

def read_data(args):
    """read data, generate spatial adjacency matrix and semantic adjacency matrix by dtw

    Args:
        sigma1: float, default=0.1, sigma for the semantic matrix
        sigma2: float, default=10, sigma for the spatial matrix
        thres1: float, default=0.6, the threshold for the semantic matrix
        thres2: float, default=0.5, the threshold for the spatial matrix

    Returns:
        data: tensor, T * N * 1
        sp_matrix: array, spatial adjacency matrix
    """
    filename = args.filename
    file = files[filename]
    data = np.load(os.path.join(data_dir,file[0]))['arr_0']
    mask= np.load(os.path.join(data_dir,file[2]))['arr_0']
    #data = np.load(filepath + file[0])['data']
    # PEMS04 == shape: (16992, 307, 3)    feature: flow,occupy,speed
    # PEMSD7M == shape: (12672, 228, 1)
    # PEMSD7L == shape: (12672, 1026, 1)
    num_node = data.shape[1]
    #diff_data = np.diff(data, axis=0)
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]
    # use directed graph
    adj_dir = os.path.join(data_dir, filename)
    os.makedirs(adj_dir, exist_ok=True)  # 确保目录存在
    adj_file = os.path.join(adj_dir, f'{filename}_adj.npy')
    if not os.path.exists(adj_file):
        df = pd.read_csv(os.path.join(data_dir, 'changjiang/distance.csv'))
        laneindexMap = {
            "沙市": 6, "枝城": 1, "监利": 10, "湘潭": 2, "桃江": 7, "桃源": 4, 
            "石门": 8, "弥陀寺": 0, "沙道观": 11, "新江口": 3, "管家铺": 5, "城陵矶": 9
        }
        laneNum = len(laneindexMap)
        dist_matrix = np.zeros((laneNum, laneNum))
        for index, row in df.iterrows():
            from_lane = row['from']
            to_lane = row['to']
            if from_lane in laneindexMap and to_lane in laneindexMap:
                from_index = laneindexMap[from_lane]
                to_index = laneindexMap[to_lane]
                dist_matrix[from_index][to_index] = 1
        np.save(adj_file, dist_matrix)
    
    adj = np.load(adj_file)
    return torch.from_numpy(data).to(torch.float32), torch.from_numpy(mask).to(torch.float32), mean_value, std_value, adj


def get_normalized_adj_tensor(A):
    A2 = torch.mm(A, A)
    A3 = torch.mm(A, A2)
    A = A + A2 + A3
    row_sum = torch.sum(A, axis=1) + 1e-9
    col_sum = torch.sum(A, axis=0) + 1e-9
    row_sum_sqrt_inv = 1 / torch.sqrt(row_sum)
    col_sum_sqrt_inv = 1 / torch.sqrt(col_sum)
    A_wave = torch.einsum('i, ij, j->ij', row_sum_sqrt_inv, A, col_sum_sqrt_inv)
    return A_wave

def get_normalized_adj(A):
    A2 = np.dot(A, A)
    A3 = np.dot(A, A2)
    A = A + A2 + A3
    row_sum = np.sum(A, axis=1) + 1e-9
    col_sum = np.sum(A, axis=0) + 1e-9
    row_sum_sqrt_inv = 1 / np.sqrt(row_sum)
    col_sum_sqrt_inv = 1 / np.sqrt(col_sum)
    A_wave = np.einsum('i, ij, j->ij', row_sum_sqrt_inv, A, col_sum_sqrt_inv)
    return A_wave
    # A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    # return A_reg
    # return torch.from_numpy(A_reg.astype(np.float32))


def generate_dataset(data, mask,args):
    """
    Args:
        data: input dataset, shape like T * N
        batch_size: int 
        train_ratio: float, the ratio of the dataset for training
        his_length: the input length of time series for prediction
        pred_length: the target length of time series of prediction

    Returns:
        train_dataloader: torch tensor, shape like batch * N * his_length * features
        test_dataloader: torch tensor, shape like batch * N * pred_length * features
    """
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    his_length = args.his_length
    pred_length = args.pred_length
    total_length = data.shape[0]

    train_data = data[: int(total_length * train_ratio)]
    valid_data = data[int(total_length * train_ratio): int(total_length * (train_ratio + valid_ratio))]
    test_data = data[int(total_length * (train_ratio + valid_ratio)): ]
    train_mask = mask[: int(total_length * train_ratio)]
    valid_mask = mask[int(total_length * train_ratio): int(total_length * (train_ratio + valid_ratio))]
    test_mask = mask[int(total_length * (train_ratio + valid_ratio)): ]
    def add_window(data,mask):
        max_index = data.shape[0] 
        index = 0 
        X, Y,m = [], [],[]
        index = 0
        while index + his_length + pred_length < max_index:
            X.append(data[index: index + his_length])
            Y.append(data[index + his_length: index + pred_length + his_length][:, :, 0])
            m.append(mask[index + his_length: index + pred_length + his_length][:, :, 0])
            index += 1
        X = torch.stack(X, dim=0).permute(0, 2, 1, 3)
        aug_time = repeat(torch.linspace(0, 11, 12), 'h -> a b h c', a=X.shape[0], b=X.shape[1], c=1)
        X = torch.cat([X, aug_time], dim=3)#(S,N,T,F->3152,12,12,3)
        Y = torch.stack(Y, dim=0).permute(0, 2, 1)#(S,N,T->3152,12,12)
        m = torch.stack(m, dim=0).permute(0, 2, 1)
        return X, Y,m

    x_train, y_train,mask_train = add_window(train_data,train_mask)#(3152,12,12)
    x_valid, y_valid,mask_valid = add_window(valid_data,valid_mask)#(1035,12,12)
    x_test, y_test,mask_test = add_window(test_data,test_mask)#(1035,12,12)

    times = torch.linspace(0, 11, 12)
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_train)#(t,(s,n,t,f))->
    valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_valid)
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_test)

    def get_loader(X, Y,mask, batch_size, shuffle):
        data = torch.utils.data.TensorDataset(*X, Y,mask)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    train_loader = get_loader(train_coeffs, y_train,mask_train, batch_size=batch_size, shuffle=True)
    valid_loader = get_loader(valid_coeffs, y_valid,mask_valid, batch_size=batch_size, shuffle=False)
    test_loader = get_loader(test_coeffs, y_test, mask_test,batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
