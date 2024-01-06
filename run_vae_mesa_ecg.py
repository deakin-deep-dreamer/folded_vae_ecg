
import os
import sys
import random
import time
from datetime import datetime
import logging
import argparse
import traceback
import matplotlib.pyplot as plt

import torch
from torch import nn

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, \
    accuracy_score, classification_report, confusion_matrix

import datasource_mesa, models, vae_model
import fn_loss

logger = logging.getLogger(__name__)


def log(msg):
    logger.debug(msg)


def config_logger(log_file):
    r"""Config logger."""
    global logger
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(format)
    logger.addHandler(ch)
    # logging.basicConfig(format=format, level=logging.DEBUG, datefmt="%H:%M:%S")
    # logger = logging.getLogger(__name__)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setFormatter(format)
    fh.setLevel(logging.DEBUG)
    # add the handlers to the logger
    logger.addHandler(fh)


def fix_randomness():
    RAND_SEED = 2021
    random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)
    np.random.seed(RAND_SEED)


def calculate_class_weights(labels, n_classes=2):
    freq = np.zeros(n_classes)
    for label in labels:
        freq[label] += 1
    min_freq = freq[np.argmin(freq)]
    weights = min_freq / freq
    return freq, weights


def viz_epoch_batch(epoch, x_batch, x_hat_batch, log_filename):
    folder = os.path.join("logs", "recon_vae", log_filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    x_batch = x_batch.detach().cpu().numpy()
    x_hat_batch = x_hat_batch.detach().cpu().numpy()
    for i in range(2):
        orig = x_batch[i, 0, :]
        recon = x_hat_batch[i, 0, :]
        _, ax = plt.subplots()
        ax.plot(range(len(orig)), orig)
        # plt.savefig(
        #     f"{folder}/epoch{epoch}_item{i}_orig.png",
        #     format='png', dpi=300, bbox_inches='tight')
        ax.plot(range(len(recon)), recon)
        plt.savefig(
            f"{folder}/epoch{epoch}_item{i}.png",
            format='png', dpi=300, bbox_inches='tight')
    plt.close()


def score(
    labels, preds
):
    r"""Calculate scores."""
    _preds = preds
    _labels = labels
    # _preds = _preds.argmax(axis=1)
    score_prec = precision_score(_labels, _preds, average='macro')
    score_recall = recall_score(_labels, _preds, average='macro')
    score_f1 = f1_score(_labels, _preds, average='macro')
    score_acc = accuracy_score(_labels, _preds)
    report_dict = classification_report(_labels, _preds, output_dict=True)
    cm = confusion_matrix(_labels, _preds)
    return score_prec, score_recall, score_f1, score_acc, report_dict, cm


def training(
        net, ds_train, ds_val, ds_test=None, model_file=None, class_w=True,
        device=None, batch_sz=32, early_stop_patience=30, early_stop_delta=0.0001,
        weight_decay=0., init_lr=0.001, min_lr=1e-6, lr_scheduler_patience=15,
        max_epoch=200, n_epoch_pre_train=100, dann_loss=True, eta_regulariser=1., 
        loss_type='min', log=print, log_filename=None
):
    class_weights = None
    if class_w:
        labels_ = None
        labels_ = [
            ds_train.memory_ds.seg_labels[i] for i in ds_train.indexes
        ]
        class_weights = torch.from_numpy(
            calculate_class_weights(
                labels_, n_classes=net.output_dim,
            )[-1]
        ).type(torch.FloatTensor)
        class_weights = class_weights.to(device)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset=ds_train, batch_size=batch_sz, shuffle=True, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=ds_val, batch_size=batch_sz, shuffle=False, drop_last=True)
    
    net.to(device)
    optimizer = torch.optim.Adam(
            net.parameters(), lr=init_lr, weight_decay=weight_decay)
    # crit_mmd = fn_loss.MMD
    crit_mmd = fn_loss.MMD_loss()
    crit_classif = nn.CrossEntropyLoss(weight=class_weights)
    eta_regulariser = float(eta_regulariser)
    log(
        f"[model_training:fit] model:{net.__class__.__name__}, "
        f"dann_loss:{dann_loss}, eta_regulariser:{eta_regulariser}, "
        f"train-db:{len(ds_train)}, "
        f"val-db:{len(ds_val)}, max-epoch:{max_epoch}, "
        f"n_epoch_pre_train:{n_epoch_pre_train}, device:{device}, "
        f"model_file:{model_file}, "
        f"early_stop_pt/delta:{early_stop_patience}/{early_stop_delta}, "
        f"lr_schd_pt:{lr_scheduler_patience}, batch-sz:{batch_sz}, "
        f"init_lr:{init_lr}, min_lr:{min_lr}, "        
    )
    for epoch in range(max_epoch):
        since = time.time()
        train_loss = 0.
        recon_loss = 0.
        enc_d_loss = 0.
        net.train()
        data_loader_train.dataset.on_epoch_end()
        for i_batch, (inputs, labels) in enumerate(data_loader_train):
            # inputs = inputs.to(device)
            labels = labels.to(device)
            
            x_src, x_ref = inputs[:, :1], inputs[:, 1:2]
            x_src = x_src.to(device)
            x_ref = x_ref.to(device)

            optimizer.zero_grad()
            latent, x_hat, out_classif = net(x_src)
            latent_ref, _, _ = net(x_ref)

            loss_classif = crit_classif(out_classif, labels)      

            # reconstruction loss
            loss_recon = net.gaussian_likelihood(x_hat, x_src)
            # elbo
            elbo = (net.encoder.kl - loss_recon).mean()
            # loss_recon = ((x_src - x_hat)**2).sum() + net.encoder.kl   

            # encoding distribution similarity loss
            loss_enc_d = eta_regulariser*crit_mmd(latent, latent_ref)
            
            # total loss
            loss = elbo + loss_classif + loss_enc_d
            
            recon_loss += elbo.detach().cpu().numpy()
            train_loss += loss.detach().cpu().numpy()
            enc_d_loss += loss_enc_d.detach().cpu().numpy()

            loss.backward()
            optimizer.step()

            if i_batch == 0 and epoch % 10 == 0:
                viz_epoch_batch(epoch, x_src, x_hat, log_filename)


        train_loss = train_loss / len(data_loader_train)
        recon_loss = recon_loss / len(data_loader_train)
        enc_d_loss = enc_d_loss / len(data_loader_train)

        time_elapsed = time.time() - since

        # validate model
        val_loss = 0.
        net.eval()
        with torch.no_grad():
            for inputs, labels in data_loader_val:
                # inputs.to(device)
                labels = labels.to(device)
                x_src = inputs[:, :1]
                x_src = x_src.to(device)
                _, _, out_classif = net(x_src)
                loss = crit_classif(out_classif, labels)
                val_loss += loss.detach().item()
            val_loss = val_loss / len(data_loader_val)

        # test model
        f1_test = 0.
        if ds_test is not None and epoch % 10==0:
            t_preds, t_labels = predict(
                net, ds_test, device=device)
            _prec, _recl, f1_test, _acc, _report_dict, _cm = score(
                t_labels, t_preds)

        log(
            f"Epoch:{epoch}, train_loss:{train_loss:.5f}, val_loss:{val_loss:.5f}, "
            f"recon_loss:{recon_loss:.5f}, enc_d_loss:{enc_d_loss:.5f}, test_F1:{f1_test:.02f}, "
            f"lr:{optimizer.param_groups[0]['lr']}, "
            f"time:{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    pass
    net.load_state_dict(torch.load(model_file))  # return best model
    log('Training is done.')
    return 0., 0.


def predict(net, dataset, device="cpu"):
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, drop_last=True)
    out_preds, out_labels = [], []
    net.to(device)
    net.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            # inputs.to(device)
            labels = labels.to(device)
            x_src = inputs[:, :1]
            x_src = x_src.to(device)
            _, _, output = net(x_src)
            out_preds.extend(output.detach().cpu().numpy().argmax(axis=1))
            out_labels.extend(labels.detach().cpu().numpy())
    return out_preds, out_labels


def train_single_fold(
    data_path=None, base_path=None, log_path=None, model_path=None, n_classes=None,
    tm_sim_start=None, device=None, max_epoch=None, early_stop_delta=None, 
    early_stop_patience=None, lr_scheduler_patience=None, init_lr=None, w_decay=None, 
    batch_sz=None, seg_sec=None, n_skip_seg=None, hz=None, class_w=None, dann_loss=True, 
    eta_regulariser=1., loss_type='dann', n_epoch_pre_train=0, n_subjects=50, 
    sig_modality='eeg', log_filename=None, train_set=None, test_set=None, ecg_modality='raw',
):
    if n_classes == 2:
        class_map = {0:0, 1:1, 2:1, 3:1, 5:1}
    elif n_classes == 3:
        class_map = {0:0, 1:1, 2:1, 3:1, 5:2}
    elif n_classes == 4:
        class_map = {0:0, 1:1, 2:1, 3:2, 5:3}    
    elif n_classes == 5:
        class_map = {0:0, 1:1, 2:2, 3:3, 5:4}    
    else:
        raise Exception(f"Invalid n_class: {n_classes}.")
    dataset = datasource_mesa.MesaDb(
        base_data_dir="data/mesa/polysomnography", data_subdir=train_set, 
        log=log, hz=hz, class_map=class_map, rr_seg_dim=128,
        n_subjects=n_subjects, sig_modality=sig_modality)   

    for i_fold in range(1):
        kf = KFold(n_splits=5, shuffle=True, random_state=2021)
        train_rec_idx, test_rec_idx = next(
            kf.split(
                np.zeros((len(dataset.record_names), 1)),
                dataset.record_names
            )
        )
        test_index = []
        for i_test_rec in test_rec_idx:
            test_index.extend(dataset.record_wise_segments[dataset.record_names[i_test_rec]])
        train_idx = []
        for i_train_rec in train_rec_idx:
            train_idx.extend(dataset.record_wise_segments[dataset.record_names[i_train_rec]])

        log(
            f"**Train({len(train_rec_idx)}): {[dataset.record_names[i] for i in train_rec_idx]}, "
            f"Test({len(test_rec_idx)}): {[dataset.record_names[i] for i in test_rec_idx]}, ")
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
        for i_fold in range(1):
            # train/validation split
            train_index, val_index = next(
                skf.split(
                    np.zeros((len(train_idx), 1)),
                    [dataset.seg_labels[i] for i in train_idx])
            )
            train_index = [train_idx[i] for i in train_index]
            val_index = [train_idx[i] for i in val_index]

            train_dataset = datasource_mesa.PartialDataset(
                dataset, seg_index=train_index, shuffle=True, log=log)
            val_dataset = datasource_mesa.PartialDataset(
                dataset, seg_index=val_index, test=True, log=log)
            test_dataset = datasource_mesa.PartialDataset(
                dataset, seg_index=test_index, test=True, log=log)

            # net = models.VariationalAutoEncoder_MLP512(
            #     input_dim=100, latent_dim=128, log=log)
            hidden_dim = 3840
            net = vae_model.VariationalAutoEncoderCNN(
                input_dim=hz*30, hidden_dim=hidden_dim, latent_dim=hidden_dim, decoder_in_chan=64, 
                kernel_sz=21, is_cuda=device!='cpu', log=log)
            if i_fold == 0:
                log(net)

            r"Training."
            model_file = (
                f"{model_path}/{log_filename}_fold{i_fold}.pt"
            )

            training(
                net, train_dataset, val_dataset, ds_test=test_dataset, model_file=model_file, device=device, 
                batch_sz=batch_sz, early_stop_patience=early_stop_patience, 
                early_stop_delta=early_stop_delta, lr_scheduler_patience=lr_scheduler_patience,
                max_epoch=max_epoch, dann_loss=dann_loss, eta_regulariser=eta_regulariser, 
                loss_type=loss_type, n_epoch_pre_train=n_epoch_pre_train, log=log, log_filename=log_filename)
            
            t_preds, t_labels = predict(
                net, test_dataset, device=device)
            r"Persist test preds and labels"
            pred_path = f"{log_path}/sleep_preds/{tm_sim_start}"
            if not os.path.exists(pred_path):
                os.makedirs(pred_path)
            pred_file = f"{pred_path}/fold{i_fold}_.preds.csv"
            df = pd.DataFrame(
                {"preds": t_preds, "labels": t_labels})
            df.to_csv(pred_file, index=True)
            _prec, _recl, _f1, _acc, _report_dict, _cm = score(
                t_labels, t_preds)
            log(
                f"[Fold{i_fold}] Prec:{_prec:.02f}, Recal:{_recl:.02f}, "
                f"F1:{_f1:.02f}, acc:{_acc:.02f}, cm:\n{_cm}")

def load_config():
    parser = argparse.ArgumentParser(description="SleepECGNet")
    parser.add_argument("--i_cuda", default=0, help="CUDA")
    parser.add_argument("--class_w", default=True, help="Weighted class")
    parser.add_argument("--n_classes", default=2, help="No. of sleep stages")
    parser.add_argument("--hz", default=128, help="Hz")
    parser.add_argument("--seg_sec", default=30, help="Segment len in sec")
    parser.add_argument("--max_epoch", default=400, help="Max no. of epoch")
    parser.add_argument("--early_stop_patience", default=30,
                        help="Early stop patience")
    parser.add_argument("--early_stop_delta", default=0.0001,
                        help="Early stop delta")
    parser.add_argument("--lr_scheduler_patience",
                        default=15, help="LR scheduler patience")
    parser.add_argument("--init_lr", default=0.001, help="Initial LR")
    parser.add_argument("--w_decay", default=0, help="LR weight decay")
    parser.add_argument("--base_path", default=None, help="Sim base path")
    parser.add_argument("--data_path", default=None, help="Data dir")
    parser.add_argument("--batch_sz", default=32, help="Batch size")
    parser.add_argument("--loss_type", default='normal', 
                        choices=['normal', 'dann', 'triplet', 'mean', 'mix', 'mix_freeze', 'feat_only'], help="DANN loss?")
    parser.add_argument("--n_epoch_pre_train", default=0, help="No. of pre-training epoch")
    parser.add_argument("--eta_regulariser", default=1, help="eta_regulariser dann loss")
    parser.add_argument("--n_subjects", default=-1, help="No. of subjects")
    parser.add_argument("--sig_modality", default='ekg', help="Data channel")
    parser.add_argument("--train_set", default='set1x30', help="Data channel")
    parser.add_argument("--test_set", default='set2', help="Data channel")
    parser.add_argument("--ecg_modality", default='rr', choices=['raw', 'rr'], help="Data modality")
    

    args = parser.parse_args()

    args.tm_sim_start = f"{datetime.now():%Y%m%d%H%M%S}"
    if args.base_path is None:
        args.base_path = os.getcwd()
    args.log_path = f"{args.base_path}/logs"
    args.model_path = f"{args.base_path}/models"
    if args.data_path is None:
        args.data_path = f"{args.base_path}/data/mesa/polysomnography/"

    # Convert commonly used parameters to integer, if required.
    if isinstance(args.i_cuda, str):
        args.i_cuda = int(args.i_cuda)
    if isinstance(args.n_classes, str):
        args.n_classes = int(args.n_classes)
    if isinstance(args.hz, str):
        args.hz = int(args.hz)
    if isinstance(args.batch_sz, str):
        args.batch_sz = int(args.batch_sz)
    if isinstance(args.max_epoch, str):
        args.max_epoch = int(args.max_epoch)
    if isinstance(args.lr_scheduler_patience, str):
        args.lr_scheduler_patience = int(args.lr_scheduler_patience)
    if isinstance(args.eta_regulariser, str):
        args.eta_regulariser = float(args.eta_regulariser)
    if isinstance(args.n_epoch_pre_train, str):
        args.n_epoch_pre_train = int(args.n_epoch_pre_train)
    if isinstance(args.n_subjects, str):
        args.n_subjects = int(args.n_subjects)
    
    # GPU device?
    if args.i_cuda > 0:
        args.device = torch.device(
            f"cuda:{args.i_cuda}" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "cpu":
            args.device = torch.device(
                f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
            )
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    return args


def main():
    fix_randomness()
    args = load_config()
    # create unique attribute log filename
    args.log_filename = f"vae_mesa_ecg{args.ecg_modality}_clz{args.n_classes}_{args.sig_modality}_{args.loss_type}_eta{args.eta_regulariser}_preTr{args.n_epoch_pre_train}_{args.tm_sim_start}"
    config_logger(f"{args.log_path}/{args.log_filename}.log")
    param_dict = vars(args)
    log(param_dict)
    # Exclude non-existing arguments
    #
    param_dict.pop("i_cuda", None)
    try:
        train_single_fold(**param_dict)

    except Exception as e:
        log(f"Exception in kfold, {str(e)}, caused by - \n{traceback.format_exc()}")
        logger.exception(e)


if __name__ == '__main__':
    main()
