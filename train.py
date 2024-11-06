import os
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm

import util
import random
import torch
import argparse
import numpy as np
from agcrn import AGCRN
from engine import trainer
from data_provider.dataloader import get_dataloader

torch.set_num_threads(3)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='device name')
parser.add_argument('--dataset', type=str, default='SDWPF', help='dataset name')
parser.add_argument('--in_dim', type=int, default=13, help='input dimension')
parser.add_argument('--out_dim', type=int, default=1, help='output dimension')
parser.add_argument('--rnn_units', type=int, default=64, help='hidden dimension')
parser.add_argument('--num_layers', type=int, default=2, help='number of layer')
parser.add_argument('--cheb_k', type=int, default=3, help='cheb order')
parser.add_argument('--history', type=int, default=36, help='history sequence length')
parser.add_argument('--horizon', type=int, default=12, help='sequence length')
parser.add_argument('--method', type=str, default='contrastive_learning', help='two choices: pure, contrastive_learning')
parser.add_argument('--im_t', type=float, default=0.01, help='input masking threshold')
parser.add_argument('--tempe', type=float, default=0.05, help='temperature parameter')
parser.add_argument('--lam1', type=float, default=0.1, help='loss lambda')
parser.add_argument('--lam2', type=float, default=0.05, help='loss lambda')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=300, help='epochs')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()
print(args)

if args.dataset == 'SDWPF':
    dtw_graph_path = './data/SDWPF/dtw_graph_top5.npy'
    geo_graph_path = './data/SDWPF/geo_graph_no_weight.npy'
    num_nodes = 134
    embed_dim = 10

save = 'save'
result_save = 'result/' + args.dataset
if not os.path.exists(save):
    os.makedirs(save)

if not os.path.exists(result_save):
    os.makedirs(result_save)

save += '/'


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


set_seed(args.seed)
device = torch.device(args.device)

dtw_graph = np.load(dtw_graph_path)
if geo_graph_path is not None:
    geo_graph = np.load(geo_graph_path)
else:
    geo_graph = None
train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args.dataset, args.history, args.horizon, args.batch_size,
                                                                           val_ratio=0.2, test_ratio=0.2, normalizer='std', single=False)

model = AGCRN(args, num_nodes, embed_dim, dtw_graph, geo_graph)
engine = trainer(device, model, scaler, args.method, args.lam1, args.lam2, args.lrate)


nparam = sum([p.nelement() for p in model.parameters()])
print('Total parameters:', nparam)

print('Start training...')
his_loss =[]
train_time = []
val_time = []
min_loss = float('inf')
train_val_result = pd.DataFrame(
        columns=['Epoch', 'Train Loss', 'Train SupLoss', 'Train UnsupNodeLoss', 'Train UnsupTcLoss', 'Train RMSE',
                 'Train MAPE', 'Train Time', 'Valid Loss', 'Valid RMSE', 'Valid MAPE', 'Valid Time'])
for i in range(1, args.epochs + 1):
    print(f"Epoch {i} of {args.epochs}")
    train_loss = []
    train_rmse = []
    train_sloss = []
    train_node_uloss = []
    train_tc_loss = []
    t1 = time.time()
    for iter, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        trainx = x.to(device)
        trainy = y.to(device)
        metrics = engine.train(trainx, trainy)

        train_loss.append(metrics[0])
        train_rmse.append(metrics[1])
        train_sloss.append(metrics[2])
        train_node_uloss.append(metrics[3])
        train_tc_loss.append(metrics[4])

    t2 = time.time()
    train_time.append(t2-t1)

    valid_loss = []
    valid_rmse = []
    s1 = time.time()
    for iter, (x, y) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        testx = x.to(device)
        testy = y.to(device)
        metrics = engine.eval(testx, testy)

        valid_loss.append(metrics[0])
        valid_rmse.append(metrics[1])

    s2 = time.time()
    val_time.append(s2-s1)

    mtrain_loss = np.mean(train_loss)
    mtrain_rmse = np.mean(train_rmse)
    mtrain_sloss = np.mean(train_sloss)
    mtrain_node_uloss = np.mean(train_node_uloss)
    mtrain_tc_loss = np.mean(train_tc_loss)

    mvalid_loss = np.mean(valid_loss)
    mvalid_rmse = np.mean(valid_rmse)
    his_loss.append(mvalid_loss)

    train_val_result = train_val_result.append(
        {'Epoch': i, 'Train Loss': round(mtrain_loss, 4), 'Train SupLoss': round(mtrain_sloss, 4),
         'Train UnsupNodeLoss': round(mtrain_node_uloss, 4), 'Train UnsupTcLoss': round(mtrain_tc_loss, 4),
         'Train RMSE': round(mtrain_rmse, 4), 'Train Time': round((t2 - t1), 4), 'Valid Loss': round(mvalid_loss, 4),
         'Valid RMSE': round(mvalid_rmse, 4), 'Valid Time': round((s2 - s1), 4)}, ignore_index=True)

    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train SupLoss: {:.4f}, Train UnsupNodeLoss: {:.4f}, Train UnsupTcLoss: ' \
          '{:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f},' \
          'Train Time: {:.4f}/epoch, Valid Time: {:.4f}/epoch'
    print(log.format(i, mtrain_loss, mtrain_sloss, mtrain_node_uloss, mtrain_tc_loss, mtrain_rmse, mvalid_loss, mvalid_rmse, (t2 - t1), (s2 - s1)))

    if min_loss > mvalid_loss:
        torch.save(engine.model.state_dict(), save + 'epoch_' + str(i) + '_' + str(round(mvalid_loss, 2)) + '.pth')
        min_loss = mvalid_loss

current_time = datetime.now().strftime("%Y.%m.%d_%H-%M")
train_val_result.to_csv(f'{result_save}/{args.dataset}_{current_time}_train_val_result.csv', index=False)

bestid = np.argmin(his_loss)
engine.model.load_state_dict(torch.load(save + 'epoch_' + str(bestid + 1) + '_' + str(round(his_loss[bestid], 2)) + '.pth'))
log = 'Best Valid MAE: {:.4f}'
print(log.format(round(his_loss[bestid], 4)))

valid_loss = []
valid_rmse = []
for iter, (x, y) in enumerate(val_dataloader):
    testx = x.to(device)
    testy = y.to(device)
    metrics = engine.eval(testx, testy)

    valid_loss.append(metrics[0])
    valid_rmse.append(metrics[1])
mvalid_loss = np.mean(valid_loss)
mvalid_rmse = np.mean(valid_rmse)
log = 'Recheck Valid MAE: {:.4f}, Valid RMSE: {:.4f}'
print(log.format(np.mean(mvalid_loss), np.mean(mvalid_rmse)))

outputs, realy = [], []
test_result = pd.DataFrame(columns=['Horizon', 'MAE', 'RMSE'])
for iter, (x, y) in enumerate(test_dataloader):
    testx = x.to(device)
    testy = y.to(device)
    engine.model.eval()
    with torch.no_grad():
        output = engine.model(testx)
        output = output.squeeze(-1)
        output = output.transpose(1, 2)
        outputs.append(output)
    testy = testy.transpose(1, 2)
    realy.append(testy)
realy = torch.cat(realy, dim=0)
preds = torch.cat(outputs, dim=0)

test_loss = []
test_rmse = []
res = []
for k in range(args.horizon):
    pred = scaler.inverse_transform(preds[:, :, k])
    real = realy[:, :, k]
    metrics = util.metric(pred, real)
    log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(k + 1, metrics[0], metrics[1]))
    test_result = test_result.append({'Horizon': k + 1, 'MAE': round(metrics[0], 4), 'RMSE': round(metrics[1], 4)}, ignore_index=True)
    test_loss.append(metrics[0])
    test_rmse.append(metrics[1])
    if k in [2, 5, 11]:
        res += [metrics[0], metrics[1]]

test_result.to_csv(f'{result_save}/{args.dataset}_{current_time}_test_result.csv', index=False)

mtest_loss = np.mean(test_loss)
mtest_rmse = np.mean(test_rmse)

log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}'
print(log.format(mtest_loss, mtest_rmse))
res += [mtest_loss, mtest_rmse]
res = [round(r, 4) for r in res]
print(res)

print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
print("Average Inference Time: {:.4f} secs/epoch".format(np.mean(val_time)))
