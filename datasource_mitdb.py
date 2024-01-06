import os
import sys
import random
import traceback
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import scipy
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
# from scipy import signal


import wfdb

import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset, TensorDataset

from ecgdetectors import Detectors
# https://github.com/berndporr/py-ecg-detectors
import heartpy.filtering as hf

minmax_scaler = MinMaxScaler()


def get_rr_signal(y, hz, target_n_samp, rr_min=0.3, rr_max=1.0, log=print):
    detectors = Detectors(hz)
    r_peaks = detectors.pan_tompkins_detector(y)
    rr_signal = np.diff(r_peaks) / hz  # normalise for 0-1 range
    rr_signal = rr_signal[(rr_signal >= rr_min) & (rr_signal <= rr_max)]  # ignore < 0.5
    rr_signal = scipy.signal.resample(rr_signal, target_n_samp) if rr_signal.shape[0]>5 else None
    return rr_signal


def read_random_file_segments(
        data_path, n_subject=-1, fs_target=64, seg_len=None,
        filter_records=None):
    segments = []
    count_recording = 0
    for f in os.listdir(data_path):        
        if not f.endswith(".hea"):
            continue

        if filter_records and f[:-4] not in filter_records:
            continue

        print(f"Reading {f}")
        count_recording += 1
        if count_recording > n_subject:
            break
        
        rec_name = f[:-4]
        signals, info = wfdb.rdsamp(f"{data_path}/{rec_name}")

        # grab first channel
        ecg_sig = signals[:, 0]  # flatten vector
        
        # read annotation
        annot = wfdb.rdann(f"{data_path}/{rec_name}", extension='atr')

        ecg_sig = preprocess_ecg(ecg_sig, hz=annot.fs)

        # resample ECG signal -> target_hz * n_same / src_hz
        recording = scipy.signal.resample(ecg_sig, fs_target*len(ecg_sig)//annot.fs)
        
        while len(recording) >= seg_len:
            seg = recording[:seg_len]

            seg = np.expand_dims(seg, axis=1)
            seg = minmax_scaler.fit_transform(seg).T
            
            segments.append(seg)
            recording = recording[seg_len:]
        

    tensor_x = torch.Tensor(np.array(segments))
    tensor_y = torch.Tensor(np.zeros((len(segments))))
    return segments, TensorDataset(tensor_x, tensor_y)


def preprocess_ecg(recording, hz=None, baseline_wander=False):
    # differentiated signal
    # 
    # w = len(recording)    
    # recording = np.diff(recording, n=1)
    # if w != len(recording):
    #     recording = scipy.signal.resample(recording, w)

    # remove baseline wander
    # 
    if baseline_wander:
        recording = hf.remove_baseline_wander(recording, sample_rate=hz, )

    # minmax scaling
    # 
    # recording = np.expand_dims(recording, axis=1)
    # recording = minmax_scaler.fit_transform(recording).flatten()
    return recording




def main():
    segments, dataset = read_random_file_segments(
        data_path="data/physionet.org/files/mitdb/1.0.0", n_subject=1, fs_target=64, seg_len=64*3,
        filter_records=['103'])
    print(f"dataset:{len(dataset)}, segments:{len(segments)}")
    for i in range(3):
        seg, lbl = dataset[i]
        seg = seg.flatten()
        plt.plot(range(len(seg)),seg)
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except:
        # traceback.print_exc(file=sys.stdout)
        print(traceback.format_exc())