import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
from sklearn.cluster import KMeans




def _parse_model_output(model_out):
    if not isinstance(model_out, tuple):
        return model_out, None, None

    if len(model_out) >= 3:
        return model_out[0], model_out[1], model_out[2]

    if len(model_out) == 2:
        return model_out[0], model_out[1], None

    return model_out[0], None, None


def loss_func(
    y_pred,
    y_true,
    pi_t=None,
    p_prior=None,
    prev_pi=None,
    eta=1.0,
    lambda_balance=0.0,
    lambda_smooth=0.0,
):
    recon_loss = F.mse_loss(y_pred, y_true, reduction='mean')

    balance_loss = torch.zeros((), device=y_pred.device)
    if pi_t is not None and p_prior is not None:
        mean_pi = pi_t.mean(dim=0)
        balance_loss = F.kl_div(torch.log(mean_pi + 1e-8), p_prior.detach(), reduction='batchmean')

    smooth_loss = torch.zeros((), device=y_pred.device)
    if pi_t is not None and prev_pi is not None:
        valid_bs = min(pi_t.shape[0], prev_pi.shape[0])
        cur_pi = pi_t[:valid_bs]
        last_pi = prev_pi[:valid_bs]

        x_t = y_true[:valid_bs]
        x_hat_t = y_pred[:valid_bs]
        g_t = torch.exp(-eta * F.mse_loss(x_t, x_hat_t, reduction='none').mean(-1)).detach()

        pi_delta = cur_pi - last_pi
        smooth_term = torch.sum(pi_delta * pi_delta, dim=-1)
        smooth_loss = torch.mean(g_t * smooth_term)

    total_loss = recon_loss + lambda_balance * balance_loss + lambda_smooth * smooth_loss

    return total_loss, recon_loss, balance_loss, smooth_loss



def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()


    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 15

    warmup_epoch = int(config.get('warmup_epoch', 10))
    default_cluster_num = int(config.get('kmeans_clusters', config.get('moe_num', 4)))
    lambda_balance = float(config.get('lambda_balance', 0.1))
    lambda_smooth = float(config.get('lambda_smooth', 0.1))
    eta = float(config.get('eta', 1.0))

    h_sys_list = []
    p_prior = None
    route_dim = None

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()
        prev_pi = None

        for x, labels, attack_labels, edge_index in dataloader:
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            model_out = model(x, edge_index)
            out, h_sys, pi_t = _parse_model_output(model_out)
            out = out.float().to(device)

            if h_sys is not None and i_epoch < warmup_epoch:
                h_sys_list.append(h_sys.detach().cpu().numpy())

            if pi_t is not None and route_dim is None:
                route_dim = int(pi_t.shape[-1])

            active_balance = lambda_balance if (p_prior is not None and pi_t is not None) else 0.0

            loss, recon_loss, balance_loss, smooth_loss = loss_func(
                out,
                labels,
                pi_t=pi_t,
                p_prior=p_prior,
                prev_pi=prev_pi,
                eta=eta,
                lambda_balance=active_balance,
                lambda_smooth=lambda_smooth,
            )
            
            loss.backward()
            optimizer.step()

            if pi_t is not None:
                prev_pi = pi_t.detach()

            
            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1

        if i_epoch == warmup_epoch - 1 and p_prior is None and len(h_sys_list) > 0:
            all_h_sys = np.concatenate(h_sys_list, axis=0)
            cluster_num = route_dim if route_dim is not None else default_cluster_num
            kmeans = KMeans(n_clusters=cluster_num, random_state=seed, n_init=10)
            cluster_labels = kmeans.fit_predict(all_h_sys)

            counts = np.bincount(cluster_labels, minlength=cluster_num).astype(np.float64)
            prior_np = counts / max(np.sum(counts), 1.0)

            p_prior = torch.tensor(prior_np, dtype=torch.float32, device=device)
            p_prior = p_prior / p_prior.sum().clamp_min(1e-12)
            h_sys_list = []


        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        acu_loss/len(dataloader), acu_loss), flush=True
            )

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_result = test(model, val_dataloader)

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1


            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss



    return train_loss_list
