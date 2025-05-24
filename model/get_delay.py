import numpy as np 
import scipy
from scipy.interpolate import interp1d, UnivariateSpline
import os
import torch 
from tqdm import tqdm 
import matplotlib.pyplot as plt
from scipy.signal import correlate
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, 'data')
def correlate(a, b, lag):
    corrs = []
    for k in range(len(a) - lag, len(a) + 1):
        corrs.append(np.dot(a[:k], b[-k:])/k)
    for k in range(1, lag + 1):
        corrs.append(np.dot(a[k:], b[:-k])/(len(a)-k))
    return np.array(corrs)

def cal_all_delay(data, filename):
    delay_dir = os.path.join(data_dir, filename)
    delay_file = os.path.join(delay_dir, 'delay.npy')
    
    # 确保目录存在
    os.makedirs(delay_dir, exist_ok=True)
    if os.path.exists(delay_file):
        return np.load(delay_file)
    data = data[:, :, 0]
    num_node = data.shape[1]
    all_delays = np.zeros((3, num_node, num_node))
    t = np.arange(12*7, 12*14)
    bias = 0
    for k in tqdm(range(3)):
        bias += 1
        interp_y = []
        for i in range(num_node):
            x = data[12*7+bias: 12*14+bias, i]
            f = interp1d(t, x, kind='cubic')
            sample = np.linspace(12*7, 12*14-1, num=(7*12-1)*5)
            y = f(sample)
            interp_y.append(y)
        for i in range(num_node):
            for j in range(num_node):
                y1 = interp_y[i]
                y2 = interp_y[j]
                cor = correlate(y1, y2, 50)
                delay = np.argmax(cor) - (len(cor) - 1)/2
                all_delays[k, i, j] = delay 

    all_delay = np.zeros((num_node, num_node))

    for i in range(num_node):
        for j in range(num_node):
            vals = sorted(all_delays[:, i, j], key=lambda x: abs(x))
            val = vals[1]
            # -1e-3 for zero delay
            if abs(val) < 1e-7:
                val = -1e-3
            all_delay[i, j] = val 
    all_delay = all_delay / 10
    np.save(delay_file, all_delay)
    print('calculate all delays over')
    return all_delay

def cal_all_delay_day(data, filename):
    delay_dir = os.path.join(data_dir, filename)
    delay_file = os.path.join(delay_dir, 'delay_day.npy')
    os.makedirs(delay_dir, exist_ok=True)
    
    if os.path.exists(delay_file):
        return np.load(delay_file)
    data = data[:, :, 0]
    num_node = data.shape[1]
    all_delays = np.zeros((3, num_node, num_node))
    t = np.arange(30*2, 30*7)
    bias = 0
    for k in tqdm(range(3)):
        bias += 30*12
        interp_y = []
        for i in range(num_node):
            x = data[30*2+bias: 30*7+bias, i]
            f = interp1d(t, x, kind='cubic')
            sample = np.linspace(30*2, 30*7-1, num=(30*6-1)*10)
            y = f(sample)
            interp_y.append(y)
        for i in range(num_node):
            for j in range(num_node):
                y1 = interp_y[i]
                y2 = interp_y[j]
                cor = correlate(y1, y2, 100)
                delay = np.argmax(cor) - (len(cor) - 1)/2
                all_delays[k, i, j] = delay 

    all_delay = np.zeros((num_node, num_node))

    for i in range(num_node):
        for j in range(num_node):
            vals = sorted(all_delays[:, i, j], key=lambda x: abs(x))
            val = vals[1]
            # -1e-3 for zero delay
            if abs(val) < 1e-7:
                val = -1e-3
            all_delay[i, j] = val 
    all_delay = all_delay / 10
    np.save(delay_file, all_delay)
    print('calculate all delays over')
    return all_delay


def check_delay(adj, all_delay, back, threshold):
    # all_delay = np.load('data/delay.npy')
    all_delay = torch.from_numpy(all_delay).to(torch.float32).to(adj.device)

    edge_exist = adj > threshold
    positive_edge = all_delay < 0 
    negative_edge = all_delay > 0 
    positive = positive_edge * edge_exist
    negative = negative_edge * edge_exist
    
    correct_adj = torch.zeros_like(adj)
    correct_adj[positive] = adj[positive]
    correct_adj[negative.T] = adj.T[negative.T]

    correct_delay = torch.zeros_like(adj)
    correct_delay[positive] = all_delay[positive]
    correct_delay[negative.T] = -all_delay.T[negative.T]

    # set max delay
    correct_delay[correct_delay < -back] = -back 

    return correct_adj, correct_delay


def plot_delay(data, adj):
    num_node = adj.shape[0]
    data = data[:, :, 0]
    all_delays = np.zeros((5, num_node, num_node))
    t = np.arange(12*7, 12*21)
    bias = 0
    for k in tqdm(range(5)):
        bias += 12 * 24
        interp_y = []
        for i in range(num_node):
            x = data[12*7+bias: 12*21+bias, i]
            f = interp1d(t, x, kind='cubic')
            sample = np.linspace(12*7, 12*21-1, num=(14*12-1)*50)
            y = f(sample)
            interp_y.append(y)
        for i in range(num_node):
            for j in range(num_node):
                if adj[i, j]:
                    y1 = interp_y[i]
                    y2 = interp_y[j]
                    cor = correlate(y1, y2, 200)
                    delay = np.argmax(cor) - (len(cor) - 1)/2
                    all_delays[k, i, j] = delay 
    all_delay = np.zeros((num_node, num_node))

    for i in range(num_node):
        for j in range(num_node):
            if adj[i, j]:
                vals = sorted(all_delays[:, i, j], key=lambda x: abs(x))
                val = vals[2]
                # -1e-3 for zero delay
                if abs(val) < 1e-7:
                    val = -1e-3
                all_delay[i, j] = val 
    all_delay = np.abs(all_delay / 50)
# plt.hist(ysc)
    # fig=plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_yscale("log", base=4)
    # ax.hist(all_delay[adj != 0] * 5, bins=50, range=(0, 3.9*5), log=True)
    # ax.set_title('PEMS04')
    # ax.set_xlabel('delay')
    # ax.set_ylabel('count')
    # plt.savefig('delay04.png')

    vals = all_delay[adj != 0] * 5 
    # vals = np.concatenate([vals])
    # plt.figure(figsize=(18, 12))
    plt.hist(vals, bins=50, range=(0, 3.9*5), log=True)
    # plt.title('PEMS04', fontsize=20)
    plt.xlabel('delay', fontsize=18)
    plt.ylabel('count', fontsize=18)
    plt.savefig('delay04.png', dpi=120, bbox_inches='tight')
    print('calculate all delays over')
    return all_delay
