import os
import random
import argparse
import pandas as pd
import time

import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd

import dgl
from models.gat_net import GATNet_ss
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold


def add_dimension_glasso(var, dim=0):
    return var.pow(2).sum(dim=dim).add(1e-8).pow(1/2.).sum()


def parser_loader():
    parser = argparse.ArgumentParser(description='Self-Supervised GAT')
    parser.add_argument('--dataset', type=str, default='no2')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--timestep', type=int, default=26)
    parser.add_argument('--heads', nargs='+', type=float, default=[13, 4, 1])
    parser.add_argument('--dropout', nargs='+', type=float, default=[0.75, 0.15, 0.75])
    parser.add_argument('--layers', nargs='+', type=int, default=[3, 1, 1])
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[43, 88, 18, 8, 1, 16])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=6e-7)
    parser.add_argument('--reduced-dimension', type=int, default=1)
    parser.add_argument('--loss-weight', type=float, default=6e3)
    parser.add_argument('--regularization-weight', nargs='+', type=int, default=[0.3, 0.2])
    parser.add_argument('--grid-search', type=bool, default=False)
    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def spilt_timestep(input_timestep, features):
    features_timestep = torch.zeros((features.shape[0]-input_timestep, input_timestep, 999, args['embedding_dim'][0]), device='cuda')
    for i in range(features.shape[0]-input_timestep):
        features_timestep[i] = features[i:i+input_timestep, :, :]
    return features_timestep


def create_1d_position_embedding(n_pos_vec, dim):
    assert dim % 2 == 0, 'wrong dimension'
    position_embedding = torch.zeros(n_pos_vec.numel(), dim, dtype=torch.float)
    omega = torch.arange(dim/2, dtype=torch.float)
    omega /= dim / 2.
    omega = 1. / (10000 ** omega)
    out = n_pos_vec[:, None] @ omega[None, :]
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    position_embedding[:, 0::2] = emb_sin
    position_embedding[:, 1::2] = emb_cos
    return position_embedding


def create_2d_position_embedding(height, width, dim):
    assert dim % 4 == 0, 'wrong dimension!'
    position_embedding = torch.zeros(height*width, dim)
    coord = torch.stack(torch.meshgrid(torch.arange(height, dtype=torch.float), torch.arange(width, dtype=torch.float)))
    height_embedding = create_1d_position_embedding(torch.flatten(coord[0]), dim // 2)
    width_embedding = create_1d_position_embedding(torch.flatten(coord[1]), dim // 2)
    position_embedding[:, :dim//2] = height_embedding
    position_embedding[:, dim // 2:] = width_embedding
    return position_embedding


def run(args, seed):
    torch.set_default_tensor_type(torch.FloatTensor)
    setup_seed(seed)

    adj_od = np.load('./dataset/adj.npy')
    adj_od = csr_matrix(adj_od)

    adj_sem = np.load('./dataset/adj_static.npy')
    adj_sem = csr_matrix(adj_sem)

    features = np.load(f"./dataset/{args['dataset']}/no2_features.npy")
    features = torch.from_numpy(features)

    labels = np.load(f"./dataset/{args['dataset']}/no2_label.npy")
    labels = torch.from_numpy(labels)

    ss_labels = np.load(f"./dataset/{args['dataset']}/no2_ss_label.npy")
    ss_labels = torch.from_numpy(ss_labels)

    position_embedding = create_2d_position_embedding(height=27, width=37, dim=4)
    position_embedding = position_embedding.unsqueeze(0)
    position_embedding = position_embedding.repeat(features.shape[0], 1, 1)

    features = torch.cat((features, position_embedding), dim=2)

    grid_label = pd.read_csv('./dataset/grid_label.csv', index_col=0)
    idx_label = grid_label['grid_id'].values
    permutation = np.random.permutation(idx_label.shape[0])
    spilt_line1 = int(idx_label.shape[0] * 0.2)
    spilt_line2 = int(idx_label.shape[0] * 0.3)
    indices_test = permutation[spilt_line1:spilt_line2]
    idx_test = idx_label[indices_test]
    idx_test = torch.tensor(idx_test)
    idx_test = idx_test.reshape(-1)
    idx_test = idx_test.to(torch.long)

    indices_label = permutation[spilt_line2:]
    idx_label = idx_label[indices_label]
    k = 3
    mae = np.zeros(k)
    rmse = np.zeros(k)
    r2 = np.zeros(k)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    j = 0
    for indices_train, indices_val in kf.split(idx_label):
        idx_train = idx_label[indices_train]
        idx_val = idx_label[indices_val]

        idx_train = torch.tensor(idx_train)
        idx_val = torch.tensor(idx_val)

        idx_train = idx_train.reshape(-1)
        idx_val = idx_val.reshape(-1)

        idx_train = idx_train.to(torch.long)
        idx_val = idx_val.to(torch.long)

        node_num = features.shape[1]

        g_od = dgl.DGLGraph()
        g_od.add_nodes(node_num)
        adj_od = adj_od.tocoo()
        g_od.add_edges(adj_od.row, adj_od.col)
        g_od = g_od.to('cuda')

        g_sem = dgl.DGLGraph()
        g_sem.add_nodes(node_num)
        adj_sem = adj_sem.tocoo()
        g_sem.add_edges(adj_sem.row, adj_sem.col)
        g_sem = g_sem.to('cuda')

        loss_func = nn.L1Loss()
        loss_func_ss = nn.L1Loss()

        categories = [24, 7]

        net_gcn = GATNet_ss(args['embedding_dim'], args['reduced_dimension'], args['heads'], args['dropout'],
                            args['layers'], args['timestep'], categories)

        g_od.add_edges(list(range(node_num)), list(range(node_num)))
        g_sem.add_edges(list(range(node_num)), list(range(node_num)))

        features_timestep = spilt_timestep(input_timestep=args['timestep'], features=features)
        ss_labels_timestep = ss_labels[args['timestep']:, :]
        labels_timestep = labels[args['timestep']:, :]

        permutation = torch.randperm(features_timestep.shape[0])
        features_timestep = features_timestep[permutation]
        ss_labels_timestep = ss_labels_timestep[permutation]
        ss_labels_timestep = ss_labels_timestep.unsqueeze(2)
        labels_timestep = labels_timestep[permutation]
        labels_timestep = labels_timestep.unsqueeze(2)

        split_line = int(features_timestep.shape[0] * 0.2)
        test_indices = permutation[0: split_line]
        test_features = features_timestep[test_indices].cuda()
        test_labels = labels_timestep[test_indices].cuda()

        net_gcn = net_gcn.cuda()
        optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        best_mae = float('inf')
        best_rmse = float('inf')
        best_r2 = 0
        best_r2_epoch = 0
        patience = 15
        counter = 0

        for epoch in range(500):
            for i in range(0, features_timestep.shape[0] - args['batch_size'], args['batch_size']):
                net_gcn.train()
                optimizer.zero_grad()

                indices = permutation[i:i + args['batch_size']]
                batch_features = features_timestep[indices].cuda()
                batch_labels = labels_timestep[indices].cuda()
                batch_ss_labels = ss_labels_timestep[indices].cuda()

                batch_features_num = torch.cat((batch_features[:, :, :, :37], batch_features[:, :, :, 39:]), dim=3)
                batch_features_cat = batch_features[:, :, :, 37:39]
                # dataset is PM2.5
                # batch_features_num = torch.cat((batch_features[:, :, :, :38], batch_features[:, :, :, 40:]), dim=3)
                # batch_features_cat = batch_features[:, :, :, 38:40]

                batch_features_num.requires_grad_()
                output, output_ss, batch_features_cat_e, weight = net_gcn(g_od, g_sem, batch_features_num.float(),
                                                                          batch_features_cat.int(), 0, 0)

                batch_labels_train = batch_labels[:, idx_train, :]
                output = output[:, idx_train, :]
                loss_target = loss_func(output.float(), batch_labels_train.float())
                loss_ss = loss_func_ss(output_ss.float(), batch_ss_labels.float())
                train_loss = loss_target + loss_ss * args['loss_weight']

                grad_params_num = autograd.grad(train_loss, batch_features_num, create_graph=True, allow_unused=True)
                grad_params_cat = autograd.grad(train_loss, batch_features_cat_e, create_graph=True, allow_unused=True)

                reg_num = add_dimension_glasso(grad_params_num[0], dim=0)
                reg_cat = add_dimension_glasso(grad_params_cat[0], dim=0)

                reg_num = reg_num.item()
                reg_cat = reg_cat.item()
          
                train_loss = ((1-args['regularization_weight'][0]-args['regularization_weight'][1]) * train_loss +
                              args['regularization_weight'][0] * reg_num + args['regularization_weight'][1] * reg_cat)
                train_loss.backward()
                optimizer.step()

            with torch.no_grad():
                net_gcn.eval()

                test_features_num = torch.cat((test_features[:, :, :, :37], test_features[:, :, :, 39:]), dim=3)
                test_features_cat = test_features[:, :, :, 37:39]
                # dataset is PM2.5
                # batch_features_num = torch.cat((batch_features[:, :, :, :38], batch_features[:, :, :, 40:]), dim=3)
                # batch_features_cat = batch_features[:, :, :, 38:40]

                output, _, features_cat_e, weight = net_gcn(g_od, g_sem, test_features_num.float(), test_features_cat.int(), 0, 0)

                val_mae = loss_func(output[:, idx_val, :].float(), test_labels[:, idx_val, :].float())
                test_mae = loss_func(output[:, idx_test, :].float(), test_labels[:, idx_test, :].float())
                
                val_pre = output[:, idx_val, :].detach().cpu().numpy()
                val_true = test_labels[:, idx_val, :].detach().cpu().numpy()
                val_pre = val_pre.squeeze()
                val_true = val_true.squeeze()

                test_pre = output[:, idx_test, :].detach().cpu().numpy()
                test_true = test_labels[:, idx_test, :].detach().cpu().numpy()
                test_pre = test_pre.squeeze()
                test_true = test_true.squeeze()

                val_true_mean = np.mean(val_true)
                val_total_sum_of_squares = np.sum((val_true - val_true_mean) ** 2)
                val_residual_sum_of_squares = np.sum((val_true - val_pre) ** 2)
                val_r2 = 1 - (val_residual_sum_of_squares / val_total_sum_of_squares)
                val_rmse = np.sqrt(np.mean((val_pre - val_true) ** 2))
                val_mae = val_mae.detach().cpu().numpy()

                test_true_mean = np.mean(test_true)
                test_total_sum_of_squares = np.sum((test_true - test_true_mean) ** 2)
                test_residual_sum_of_squares = np.sum((test_true - test_pre) ** 2)
                test_r2 = 1 - (test_residual_sum_of_squares / test_total_sum_of_squares)
                test_rmse = np.sqrt(np.mean((test_pre - test_true) ** 2))
                test_mae = test_mae.detach().cpu().numpy()

                if val_mae < best_mae:
                    best_mae = val_mae
                    counter = 0
                    best_mae_epoch = epoch
                else:
                    counter += 1
                if counter >= patience:
                    print("Early stopping!", 'epoch', epoch)
                    break

                if val_r2 > best_r2:
                    best_r2 = val_r2
                    best_r2_epoch = epoch

                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_rmse_epoch = epoch

            print('epoch:', epoch)
            print('train_loss', loss_target)
            print('val_mae', val_mae, 'val_r2', val_r2, 'val_mse', val_rmse)
            print('bset_mae', best_mae, 'best_r2', best_r2, 'best_rmse', best_rmse)
            print('test_mae', test_mae, 'test_r2', test_r2, 'test_rmse', test_rmse)
            print('bset_mae_epoch', best_mae_epoch, 'best_r2_epoch', best_r2_epoch,
                  'best_rmse_epoch', best_rmse_epoch)
        mae[j] = best_mae
        rmse[j] = best_rmse
        r2[j] = best_r2
        print('j:', j)
        print('bset_mae', best_mae,  'best_r2', best_r2, 'best_rmse', best_rmse)
        j = j+1
    print('finish')
    return 0


if __name__ == "__main__":
    start_time = time.time()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)
    run(args, seed=42)
    end_time = time.time()
    execution_time = end_time - start_time
    print("time: {} 秒".format(execution_time))
