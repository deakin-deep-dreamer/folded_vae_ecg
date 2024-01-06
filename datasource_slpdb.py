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


def read_repeated_segments(data_path, n_segments=1000, fs_target=64, seg_len=None):
    ecg_segment = None
    segments = []
    # count_recording = 0
    for f in os.listdir(data_path):        
        if not f.endswith(".hea"):
            continue
        rec_name = f[:-4]
        signals, info = wfdb.rdsamp(f"{data_path}/{rec_name}")

        # ECG signal exists - if yes, where?
        i_ecg = -1
        for i_sig, name_sig in enumerate(info['sig_name']):
            if name_sig.lower().find("ecg") > -1:
                i_ecg = i_sig
                break
        if i_ecg == -1:
            print(f"ERROR no ECG signal in record '{f}'")
            continue
        
        # count_recording += 1
        # if count_recording > n_subject:
        #     break
        count_segment = 0
        
        ecg_sig = signals[:, i_ecg]  # flatten vector
        
        # read annotation
        annot = wfdb.rdann(f"{data_path}/{rec_name}", extension='st')

        ecg_sig = preprocess_ecg(ecg_sig, hz=annot.fs)

        # resample ECG signal -> target_hz * n_same / src_hz
        recording = scipy.signal.resample(ecg_sig, fs_target*len(ecg_sig)//annot.fs)
        
        
        if ecg_segment is None:  #len(recording) >= seg_len:            
            seg = recording[:seg_len]

            seg = np.expand_dims(seg, axis=1)
            seg = minmax_scaler.fit_transform(seg).T
            ecg_segment = seg

        while count_segment < n_segments:  
            segments.append(ecg_segment)
            count_segment += 1

    tensor_x = torch.Tensor(np.array(segments))
    tensor_y = torch.Tensor(np.zeros((len(segments))))
    return segments, TensorDataset(tensor_x, tensor_y)


def read_random_file_segments(data_path, n_subject=-1, fs_target=64, seg_len=None):
    segments = []
    count_recording = 0
    for f in os.listdir(data_path):        
        if not f.endswith(".hea"):
            continue
        rec_name = f[:-4]
        signals, info = wfdb.rdsamp(f"{data_path}/{rec_name}")

        # ECG signal exists - if yes, where?
        i_ecg = -1
        for i_sig, name_sig in enumerate(info['sig_name']):
            if name_sig.lower().find("ecg") > -1:
                i_ecg = i_sig
                break
        if i_ecg == -1:
            print(f"ERROR no ECG signal in record '{f}'")
            continue
        
        count_recording += 1
        if n_subject > -1 and count_recording > n_subject:
            break
        
        ecg_sig = signals[:, i_ecg]  # flatten vector
        
        # read annotation
        annot = wfdb.rdann(f"{data_path}/{rec_name}", extension='st')

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


def preprocess_ecg(recording, hz=None):
    # differentiated signal
    # 
    # w = len(recording)    
    # recording = np.diff(recording, n=1)
    # if w != len(recording):
    #     recording = scipy.signal.resample(recording, w)

    # remove baseline wander
    # 
    recording = hf.remove_baseline_wander(recording, sample_rate=hz, )

    # minmax scaling
    # 
    # recording = np.expand_dims(recording, axis=1)
    # recording = minmax_scaler.fit_transform(recording).flatten()
    return recording


class PhysionetSlpdb():
    r"""
    Cap sleep dataset resamples at 64Hz as CSV files.
    """
    def __init__(
            self, data_directory, hz=64, seg_sec=30, class_map=None, 
            log=print, rr_seg_dim=100, rr_min=0.2, rr_max=2,  filter_records=[], 
            n_subjects=-1, is_rr_sig=False
    ):
        self.data_directory = data_directory
        self.hz = hz
        self.seg_sec = seg_sec
        self.seg_sz = self.seg_sec * self.hz
        self.class_map = class_map
        self.n_classes = len(set(class_map.values()))
        self.log = log
        self.rr_seg_dim = rr_seg_dim
        self.rr_min = rr_min
        self.rr_max = rr_max
        self.filter_records = filter_records
        self.is_rr_sig = is_rr_sig
        self.n_subjects = n_subjects
        self.record_names = []
        self.record_wise_segments = {}
        self.segments = []
        self.seg_labels = []
        self.log(
            f"data:{data_directory}, hz:{hz}, seg_sec:{seg_sec}, class_map:{class_map}, "
            f"n_classes:{self.n_classes}")
        self._initialise()
        self.indexes = [i for i in range(len(self.segments))]
        np.random.shuffle(self.indexes)

    def _initialise(self):
        if self.n_subjects > 0:
            rec_names = []
            # randomly choose n recordings
            for f in os.listdir(self.data_directory):
                rec_name = f[:-4]
                rec_names.append(rec_name)
            random_rec_names = set()
            while len(set(random_rec_names)) < self.n_subjects:
                random_rec_names.add(
                    rec_names[random.randint(0, len(rec_names))])
            self.filter_records = list(random_rec_names)
            self.log(f"Filter {len(self.filter_records)} records from {len(rec_names)}")

        count_file = 0
        for f in os.listdir(self.data_directory):
            if not f.endswith(".hea"):
                continue
            rec_name = f[:-4]
            if len(self.filter_records) > 0 and not rec_name in self.filter_records:
                continue            
            
            self.log(f"Loading {rec_name}...")
            self.record_names.append(rec_name)
            if self.record_wise_segments.get(rec_name) is None:
                    self.record_wise_segments[rec_name] = []
                    
            signals, info = wfdb.rdsamp(f"{self.data_directory}/{rec_name}")
            
            # ECG signal exists - if yes, where?
            i_ecg = -1
            for i_sig, name_sig in enumerate(info['sig_name']):
                if name_sig.lower().find("ecg") > -1:
                    i_ecg = i_sig
                    break
            if i_ecg == -1:
                self.log(f"ERROR no ECG signal in record '{f}'")
                continue
            ecg_sig = signals[:, i_ecg]  # flatten vector

            # read annotation
            annot = wfdb.rdann(f"{self.data_directory}/{rec_name}", extension='st')
            # resample ECG signal -> target_hz * n_same / src_hz            
            
            # preprocessing ECG
            ecg_sig = preprocess_ecg(ecg_sig, hz=annot.fs)
            
            ecg_sig = scipy.signal.resample(ecg_sig, self.hz*len(ecg_sig)//annot.fs)
            n_samples = len(ecg_sig)
            
            # segmentation
            # 
            seg_count = 0
            annot_label_dist, clz_label_dist = {}, {}
            for i_seg in range(len(annot.aux_note)):
                # aux_note is a list where first char is [W, 1, 2, 3, 4, R] 
                sleep_lbl = annot.aux_note[i_seg][:1]
                # count annot label distribution
                if annot_label_dist.get(sleep_lbl) is None:
                    annot_label_dist[sleep_lbl] = 0
                annot_label_dist[sleep_lbl] += 1

                # Map to output clz label
                label = self.class_map.get(sleep_lbl)
                if label is None:
                    self.log(f"No label for annot '{sleep_lbl}' in {f}")
                    continue
                if clz_label_dist.get(label) is None:
                    clz_label_dist[label] = 0
                clz_label_dist[label] += 1

                start = i_seg * self.seg_sz
                if start + self.seg_sz > n_samples:
                    self.log(f"Remaining samples:{len(ecg_sig)-start}, annots:{len(annot.aux_note)-i_seg}")
                    break
                seg = ecg_sig[start : start + self.seg_sz]
                
                # Normalisation: standard score
                # 
                # seg = zscore(seg)
                # Normalisation minmax
                seg = np.expand_dims(seg, axis=1)
                seg = minmax_scaler.fit_transform(seg)
                seg = seg.flatten()

                # # Generate RR signal, ignore segment with poor signal quality
                # rr_sig = get_rr_signal(
                #     seg, self.hz, target_n_samp=self.rr_seg_dim, rr_min=self.rr_min, rr_max=self.rr_max)
                # if rr_sig is None:
                #     self.log(f"RR with less than {self.rr_min} R-peaks")
                #     continue

                # if self.is_rr_sig:
                #     # seg_z = np.expand_dims(rr_sig, axis=1)  
                #     seg_z = rr_sig 
                # else:
                #     # seg_z = np.expand_dims(seg, axis=1)
                #     seg_z = seg

                seg_z = seg
                if self.is_rr_sig:
                    raise Exception("RR signal not supported now.")

                self.segments.append(seg_z)
                self.record_wise_segments[rec_name].append(len(self.segments)-1)
                self.seg_labels.append(label)
                # Update start
                start += self.seg_sz
                seg_count += 1
            self.log(
                f"... n_seg:{seg_count}, remain_samp:{len(ecg_sig)-(len(annot.aux_note)*self.seg_sz)}, "
                f"ignored_seg:{len(annot.aux_note)-seg_count}, annot_dist:{annot_label_dist}, "
                f"clz_lbl_dist:{clz_label_dist}")
            count_file += 1
        # sample distribution
        # 
        self.indexes = range(len(self.segments))
        _dist = np.unique(
            [self.seg_labels[i] for i in self.indexes], return_counts=True)
        self.log(f"Total files:{count_file}, n_seg:{len(self.segments)}, distribution:{_dist}")


class PartialDataset(Dataset):
    r"""Generate dataset from a parent dataset and indexes."""

    def __init__(
        self, dataset=None, seg_index=None, test=False, shuffle=False, as_np=False, log=print
    ):
        r"""Instantiate dataset from parent dataset and indexes."""
        self.memory_ds = dataset
        self.indexes = seg_index[:]
        self.test = test
        self.shuffle = shuffle
        self.as_np = as_np
        self.log = log
        self.label_idx_dict = {}
        self._initialise()

    def _initialise(self):
        # label segregation
        # 
        for i_clz in range(self.memory_ds.n_classes):
            self.label_idx_dict[i_clz] = []
        for idx in self.indexes:
            self.label_idx_dict[self.memory_ds.seg_labels[idx]].append(idx)
        dist_str = [f"{i_clz}:{len(self.label_idx_dict[i_clz])}" for i_clz in range(self.memory_ds.n_classes)]
        self.log(f"label distribution: {dist_str}")

    def on_epoch_end(self):
        r"""End of epoch."""
        if self.shuffle and not self.test:
            np.random.shuffle(self.indexes)

    def __len__(self):
        r"""Dataset length."""
        return len(self.indexes)
    
    def find_another_sample(self, q_idx, target_label):
        target_indexes = self.label_idx_dict[target_label]        
        rand_idx = random.randint(0, len(target_indexes)-1)
        while target_indexes[rand_idx] == q_idx:
            rand_idx = random.randint(0, len(target_indexes)-1)
        return target_indexes[rand_idx]

    def __getitem__(self, idx):
        r"""Find and return item."""
        ID = self.indexes[idx]
        # trainX = np.array(self.memory_ds.segments[ID])
        trainX = self.memory_ds.segments[ID]
        trainY = self.memory_ds.seg_labels[ID]

        idx_secondary_x = self.find_another_sample(ID, trainY)
        # self.log(f"[idx:{idx}] ID:{ID}, 2nd-ID:{idx_secondary_x}")
        trainX = np.stack([trainX, self.memory_ds.segments[idx_secondary_x]], axis=1).T

        # if self.as_np:
        #     return trainX, trainY

        X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        # print(f"X_tensor before: {X_tensor.size()}")
        r"numpy array shape: (n_samp, n_chan), reshape it to (n_chan, n-samp)."
        # Segment in (1, n_samp) form, still need below line?
        # X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        Y_tensor = trainY
        if torch.any(torch.isnan(X_tensor)):
            X_tensor = torch.nan_to_num(X_tensor)
        return X_tensor, Y_tensor
    


def test_ecg_datasource():
    hz = 64
    datasource = PhysionetSlpdb(
        data_directory="data/physionet.org/files/slpdb/1.0.0/",
        class_map={
            "W": 0,
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 1,
            "R": 1,
        },
        hz=hz,
        # n_subjects=1,
        rr_seg_dim=30*10,
        is_rr_sig=False,
        filter_records=['slp01a']
    )
    p_ds = PartialDataset(
        dataset=datasource, 
        seg_index=datasource.record_wise_segments[datasource.record_names[0]], 
        as_np=True)
    sleep_seg_count = 0
    for i in range(2):
        seg, lbl = p_ds[i]
        print(f"partial-ds, seg:{seg.shape}, label:{lbl}")
        # plt.ylim((0, 1.2))
        for ch in range(seg.shape[0]):
            seg = seg[ch, :1000]
            plt.plot(range(seg.shape[-1]), seg, linewidth=4.0, color='orange')
            plt.axis('off')
            # plt.plot(range(seg.shape[-1]), seg[ch, :])
            # plt.title(f"ch:{ch}")
            plt.savefig(
                f"logs/misc/ecg_seg_recon_ch{ch}_{i}.png", format='png', dpi=300,
                bbox_inches='tight', transparent=True, pad_inches=0)
            plt.show()
            break

        if sleep_seg_count > 5:
            break
        if lbl == 1:
            sleep_seg_count += 1


def main():
    test_ecg_datasource()


if __name__ == "__main__":
    try:
        main()
    except:
        # traceback.print_exc(file=sys.stdout)
        print(traceback.format_exc())