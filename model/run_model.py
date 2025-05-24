import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from loguru import logger
import dgl 
from scipy import sparse
from datetime import datetime
from collections import defaultdict
# from torch.utils.tensorboard import SummaryWriter
import json
from lib.args import args
from model import Flownde
from lib.dataset import generate_dataset, read_data, get_normalized_adj
from lib.eval import masked_mae_np, masked_mape_np, masked_rmse_np
from get_delay import cal_all_delay,cal_all_delay_day
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.__version__)
print(torch.version.cuda)
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)

def train(loader, model, A, all_delay, optimizer, criterion, device):
    batch_loss = 0
    batch_regression_loss = 0
    for idx, batch in enumerate(tqdm(loader)):
        model=model.to(device)
        model.train()
        optimizer.zero_grad()

        batch = tuple(b.to(device) for b in batch)
        *inputs, targets,mask = batch
        outputs, regression_loss = model(A, all_delay, inputs) 
        loss = criterion(outputs, targets) 
        loss += regression_loss
        loss.backward()
        optimizer.step()

        batch_loss += loss.detach().cpu().item() 
        batch_regression_loss += regression_loss.detach().cpu().item()

    # print(f'##on train data## total loss: {batch_loss/(idx+1)}, pred loss: {(batch_loss - batch_regression_loss)/(idx+1)}, regression loss: {batch_regression_loss/(idx+1)}')
    return batch_loss/(idx + 1), (batch_loss - batch_regression_loss)/(idx+1), batch_regression_loss/(idx+1)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
@torch.no_grad()
def eval(loader, model, A, all_delay, std, mean, device):
    batch_rmse_loss = 0  
    batch_mae_loss = 0
    batch_mape_loss = 0
    for idx, batch in enumerate(tqdm(loader)):
        model.eval()

        batch = tuple(b.to(device) for b in batch)
        *inputs, targets,mask = batch
        output = model(A, all_delay, inputs).reshape(*targets.shape)
        out_unnorm = output.detach().cpu().numpy()*std + mean
        target_unnorm = targets.detach().cpu().numpy()*std + mean
        mask = mask.detach().cpu().numpy()
        mae_loss = masked_mae_np(target_unnorm, out_unnorm,mask,0)
        rmse_loss = masked_rmse_np(target_unnorm, out_unnorm,mask,0)
        mape_loss = masked_mape_np(target_unnorm, out_unnorm,mask,0)
        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1)
def eval_test(loader, model, A, all_delay, std, mean, device):
    batch_rmse_loss = 0  
    batch_mae_loss = 0
    batch_mape_loss = 0
    predictions, running_targets,masks = list(), list(),list()
    for idx, batch in enumerate(tqdm(loader)):
        model.eval()
        batch = tuple(b.to(device) for b in batch)
        *inputs, targets,mask = batch
        output = model(A, all_delay, inputs).reshape(*targets.shape)
        out_unnorm = output.detach().cpu().numpy()*std + mean
        target_unnorm = targets.detach().cpu().numpy()*std + mean
        mask = mask.detach().cpu().numpy()
        running_targets.append(target_unnorm)
        predictions.append(out_unnorm)
        masks.append(mask)
    running_targets, predictions,masks = np.concatenate(running_targets, axis=0),\
                               np.concatenate(predictions, axis=0),\
                                np.concatenate(masks,axis=0)
    n_samples = running_targets.shape[0]
    running_targets=running_targets[:,-3:-2,0:3]
    predictions=predictions[:,-3:-2,0:3]
    masks=masks[:,-3:-2,0:3]
    scores = defaultdict(dict)
    for horizon in range(3):
        y_true = running_targets[:, 0,horizon:horizon+1]
        y_pred = predictions[:, 0,horizon:horizon+1]
        mask = masks[:,0, horizon:horizon+1]
        scores['MAE'][f'horizon-{horizon + 1}'] = masked_mae_np(y_true, y_pred,mask,0)
        scores['RMSE'][f'horizon-{horizon + 1}'] = masked_rmse_np(y_true, y_pred,mask,0)
        scores['MAPE'][f'horizon-{horizon+1}'] = masked_mape_np(y_true,y_pred,mask,0) 
    y_true = running_targets
    y_pred = predictions
    scores['rmse'] = masked_mae_np(y_true, y_pred,mask,0)
    scores['mae'] =  masked_rmse_np(y_true, y_pred,mask,0)
    scores['mape']= masked_mape_np(y_true,y_pred,mask,0)
    return scores

def main(args):
    # random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

    if args.log:
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename=os.path.join(file_dir, f"logs/log_{time}_{args.filename}.log")
        logger.add(log_filename)
    options = vars(args)
    if args.log:
        logger.info(options)
    else:
        print(options)

    data,mask, mean, std, A = read_data(args)
    train_loader, valid_loader, test_loader = generate_dataset(data,mask,args)
    num_node = data.shape[1]
    all_delay = cal_all_delay_day(data, args.filename)

    # A = get_normalized_adj(A)
    # # A, delay = cal_delay(A, data, args.back)
    # A_tensor = torch.from_numpy(A).to(torch.float32).to(device)
    # A_tensor, delay = check_delay(A_tensor, data, back=args.back)
    # A_numpy = A_tensor.cpu().numpy()
    
    # # delay = torch.from_numpy(delay).to(torch.float32).to(device)

    # graph = dgl.from_scipy(sparse.coo_matrix(A_numpy)).to(device)
    # print('num of nodes:', graph.num_nodes(), 'num of edges:', graph.num_edges())
    # graph.edata['delay'] = delay[delay < 0]
    # # graph.edata['delay'] = torch.zeros(graph.num_edges()).to(device)
    # graph.edata['w'] = A_tensor[A_tensor > 0]

    net = Flownde(adj=A, num_node=num_node, in_dim=data.shape[2]+1, hidden_dim=args.hidden_dim, out_dim=12, 
                step_size=args.step_size, back=args.back, thres=args.thres, extra=[mean, std])
    net = net.to(device)

    lr = args.lr
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    best_valid_rmse = 3000 
    scheduler = StepLR(optimizer, step_size=30, gamma=0.7)
    patience = 10  # Number of epochs to wait for improvement
    early_stopping_counter = 0
    for epoch in range(1, args.epochs+1):
        print("=====Epoch {}=====".format(epoch))
        print('Training...')
        total_loss, pred_loss, regression_loss = train(train_loader, net, A, all_delay, optimizer, criterion, device)
        print('Evaluating...')
        train_rmse, train_mae, train_mape = eval(train_loader, net, A, all_delay, std, mean, device)
        valid_rmse, valid_mae, valid_mape = eval(valid_loader, net, A, all_delay, std, mean, device)
        test_rmse, test_mae, test_mape = eval(test_loader, net, A, all_delay, std, mean, device)
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            print('New best results!')
            torch.save(net.state_dict(), f'net_params_{args.filename}_{args.device}.pkl')
        else:
            early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f'Early stopping triggered after {epoch} epochs.')
            break
        if args.log:
            logger.info(f'=====Epoch {epoch}=====\n' + 
                        f'\n##on train data## total loss: {total_loss}, pred loss: {pred_loss}, regression loss: {regression_loss} \n' + 
                        f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
                        f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n' + 
                        f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}\n')
        else:
            print(f'\n##on train data## total loss: {total_loss}, pred loss: {pred_loss}, regression loss: {regression_loss} \n' + 
                    f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
                    f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n' 
                    f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}\n'
                )
        
        scheduler.step()
    model_load_path = os.path.join(file_dir, f'model/net_params_{args.filename}_{args.device}.pkl')
    net.load_state_dict(torch.load(model_load_path))
    # net.load_state_dict(torch.load('net_params_PEMS04_7.pkl', map_location=torch.device('cpu')))
    test_rmse, test_mae, test_mape = eval(test_loader, net, A, all_delay, std, mean, device)
    print(f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}')

def test(args):
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    data, mask,mean, std, A = read_data(args)
    train_loader, valid_loader, test_loader = generate_dataset(data,mask, args)
    num_node = data.shape[1]
    all_delay = cal_all_delay(data, args.filename)
    net = Flownde(adj=A, num_node=num_node, in_dim=data.shape[2]+1, hidden_dim=args.hidden_dim, out_dim=12, 
                step_size=args.step_size, back=args.back, thres=args.thres, extra=[mean, std])
    net = net.to(device)
    net.load_state_dict(torch.load(os.path.join(file_dir,f'model/net_params_{args.filename}_{args.device}.pkl')))
    # net.load_state_dict(torch.load('net_params_PEMS04_7.pkl', map_location=torch.device('cpu')))
    test_rmse, test_mae, test_mape = eval(test_loader, net, A, all_delay, std, mean, device)
    scores=eval_test(test_loader, net, A, all_delay, std, mean, device)
    print(f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}')
    print('test results:')
    print(json.dumps(scores, cls=MyEncoder, indent=4))
    test_scores_path = os.path.join(file_dir, 'logs/test-scores.json')
    config_path = os.path.join(file_dir, 'logs/config.json')
    with open(test_scores_path, 'w+') as f:
       json.dump(scores, f, cls=MyEncoder, indent=4)
    with open(config_path,'w') as cf:
       json.dump(config,cf,cls=MyEncoder,indent=4)
if __name__ == '__main__':
    args.station='test'
    args.filename='changjiang'
    if(args.station=='train'):
        main(args)
    else:
        test(args)
