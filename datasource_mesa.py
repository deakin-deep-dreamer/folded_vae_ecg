import os
import sys
import random
import traceback
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy.stats import zscore
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import scipy

import mne
import xml.etree.ElementTree as ET

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


def load_edf_channel(edf_file, fs_target=100, ch_name=None, log=print):
    try:
        raw = mne.io.read_raw_edf(edf_file, preload=False)
        log(f"channels: {raw.info.get('ch_names')}")
        ch_idx = -1        
        for cname in raw.info.get('ch_names'):
            ch_idx += 1
            if cname.upper().find(ch_name.upper()) > -1:
                break
        # else:
        #     raise Exception(f"No channel by name: {ch_name}")
        hz = mne.pick_info(raw.info, [ch_idx], verbose=False)['sfreq']
        hz = int(hz)
        raw.pick_channels([cname])
        recording = raw.get_data().flatten()
        log(f"channels: {raw.info.get('ch_names')}, search:{ch_name}, src_hz:{hz}")

        return scipy.signal.resample(recording, fs_target*len(recording) // int(hz))
    except:
        log(f"Error reading {edf_file}, caused by - {traceback.format_exc()}")
        return


def read_sleep_annot_xml(annot_xml):
    tree = ET.parse(annot_xml)
    root = tree.getroot()

    sleep_annot = []
    for se in root.findall('ScoredEvents'): 
        for evt in se:
            evt_meta = {}
            for item in evt:
                if item.tag == "EventType" and item.text != "Stages|Stages":
                    break
                if item.tag == "EventType":
                    continue
                val = item.text
                if item.tag == "EventConcept":
                    # <EventConcept>Wake|0</EventConcept>
                    # <EventConcept>Stage 1 sleep|1</EventConcept>
                    evt_meta['stage'] = int(val.split("|")[-1])
                if item.tag == "Start":
                    evt_meta["start"] = int(float(val))
                if item.tag == "Duration":
                    evt_meta["duration"] = int(float(val))
            else:
                # print(evt_meta)
                sleep_annot.append(evt_meta)
    return sleep_annot


def preprocess_ecg(recording, hz, diff=False, baseline_wader=False, fir_filter=None):
    # differentiated signal
    # 
    if diff:
        w = len(recording)    
        recording = np.diff(recording, n=1)
        if w != len(recording):
            recording = scipy.signal.resample(recording, w)

    # bandpass filter
    if fir_filter:
        filter = scipy.signal.firwin(400, [fir_filter[0], fir_filter[1]], pass_zero=False, fs=hz)
        recording = scipy.signal.convolve(recording, filter, mode='same')

    # remove baseline wander
    # 
    if baseline_wader:
        recording = hf.remove_baseline_wander(recording, sample_rate=hz, )

    # minmax scaling
    # 
    # recording = np.expand_dims(recording, axis=1)
    # recording = minmax_scaler.fit_transform(recording).flatten()
    return recording


def query_channel_names(
        data_path, n_subject=1, fs_target=64, seg_len=None, ch_name='ekg'
):
    count_recording = 0
    for f in os.listdir(data_path):
        if not f.endswith(".edf"):
            continue
        count_recording += 1
        if count_recording > n_subject:
            break
        recording = load_edf_channel(
                f"{data_path}/{f}", ch_name=ch_name, fs_target=fs_target)


def read_random_file_segments(
        data_path, n_subject=1, fs_target=64, seg_len=None, ch_name='ekg'
):
    segments = []
    count_recording = 0
    for f in os.listdir(data_path):
        if not f.endswith(".edf"):
            continue
        count_recording += 1
        if count_recording > n_subject:
            break
        recording = load_edf_channel(
                f"{data_path}/{f}", ch_name=ch_name, fs_target=fs_target)
        
        # recording = preprocess_ecg(recording, hz=fs_target, fir_filter=(0.2, 10.0))
        recording = preprocess_ecg(recording, hz=fs_target, baseline_wader=True)

        while len(recording) >= seg_len:
            seg = recording[:seg_len]
            
            seg = np.expand_dims(seg, axis=1)
            seg = minmax_scaler.fit_transform(seg).T

            segments.append(seg)
            recording = recording[seg_len:]

    tensor_x = torch.Tensor(np.array(segments))
    tensor_y = torch.Tensor(np.zeros((len(segments))))  # dummy label
    return segments, TensorDataset(tensor_x, tensor_y)


class MesaDb():
    def __init__(
            self, base_data_dir, data_subdir="edfs", annot_subdir="annotations-events-nsrr", 
            hz=128, seg_sec=30, class_map=None, log=print, rr_seg_dim=256, rr_min=0.2, 
            rr_max=2, sig_modality="ekg", filter_records=[], n_subjects=-1, is_rr_sig=False) -> None:
        self.base_data_dir = base_data_dir
        self.data_dir = os.path.join(base_data_dir, data_subdir)
        self.annot_dir = os.path.join(base_data_dir, annot_subdir)
        self.hz = hz
        self.seg_sec = seg_sec
        self.seg_dim = seg_sec * hz
        self.class_map = class_map
        self.n_classes = len(set(class_map.values()))
        self.log = log
        self.sig_modality = sig_modality
        self.rr_seg_dim = rr_seg_dim
        self.rr_min = rr_min
        self.rr_max = rr_max
        self.filter_records = filter_records
        self.n_subjects = n_subjects
        self.is_rr_sig = is_rr_sig
        self.record_names = []
        self.record_wise_segments = {}
        self.segments = []
        self.seg_labels = []
        self.log(
            f"Data base-dir:{base_data_dir}, data:{data_subdir}, annot:{annot_subdir}, "
            f"hz:{hz}, seg_sec:{seg_sec}, class_map:{class_map}, "
            f"n_classes:{self.n_classes}")
        self.scaler = MinMaxScaler()
        self._initialise()
        self.indexes = [i for i in range(len(self.segments))]
        np.random.shuffle(self.indexes)

    def _initialise(self):
        if self.n_subjects > 0:
            rec_names = []
            # randomly choose n recordings
            for f in os.listdir(self.data_dir):
                rec_name = f[:-4]
                rec_names.append(rec_name)
            random_rec_names = set()
            while len(set(random_rec_names)) < self.n_subjects:
                random_rec_names.add(
                    rec_names[random.randint(0, len(rec_names))])
            self.filter_records = list(random_rec_names)
            self.log(f"Filter {len(self.filter_records)} records from {len(rec_names)}")

        count_file = 0
        for f in os.listdir(self.data_dir):
            rec_name = f[:-4]
            if len(self.filter_records) > 0 and not rec_name in self.filter_records:
                continue
            data_file = f"{self.data_dir}/{f}"
            annot_file = f"{self.annot_dir}/{rec_name}-nsrr.xml"
  
            self.log(f"Loading {rec_name}...")
            sleep_annot = read_sleep_annot_xml(annot_file)
            
            recording = load_edf_channel(
                data_file, ch_name=self.sig_modality, fs_target=self.hz, log=self.log)
            
            if recording is None:
                continue

            self.record_names.append(rec_name)
            if self.record_wise_segments.get(rec_name) is None:
                self.record_wise_segments[rec_name] = []
                      
            # pre-processing
            recording = preprocess_ecg(recording, self.hz, fir_filter=(0.2, 10.0))

            self.log(f"[{f[:-4]}] {len(sleep_annot)} events")

            # segmentation
            # 
            seg_count = 0
            evt_count = 0
            annot_label_dist, clz_label_dist = {}, {}
            for dict_annot in sleep_annot:
                sleep_stage, start_sec, duration_sec = dict_annot['stage'], \
                    dict_annot['start'], dict_annot['duration']
                evt_count += 1
                label = self.class_map.get(sleep_stage)
                if annot_label_dist.get(sleep_stage) is None:
                    annot_label_dist[sleep_stage] = 0
                for i_epoch in range(duration_sec//self.seg_sec):
                    # Map to output clz label
                    annot_label_dist[sleep_stage] += 1                    
                    label = self.class_map.get(sleep_stage)
                    if label is None:
                        self.log(f"No label for annot '{sleep_stage}' in {f}")
                        continue                    

                    seg_start = (start_sec*self.hz) + (i_epoch*self.seg_dim)
                    seg = recording[seg_start:seg_start+self.seg_dim]
                                        
                    # seg_z = zscore(seg)

                    # seg_z = seg

                    seg = np.expand_dims(seg, axis=1)
                    seg_z = self.scaler.fit_transform(seg)
                    seg_z = seg_z.flatten()

                    # Generate RR signal, and ignore segment with poor signal quality
                    if self.sig_modality in ['ecg', 'ekg']:
                        rr_sig = get_rr_signal(
                            seg_z, self.hz, target_n_samp=self.rr_seg_dim, rr_min=self.rr_min, rr_max=self.rr_max)
                        if rr_sig is None:
                            self.log(f"[{f}] Bad segment (less than 5 R-peaks), {dict_annot}")
                            continue                    

                        if self.is_rr_sig:
                            seg_z = rr_sig
                    
                    # Valid segment, include them.
                    # 
                    self.segments.append(seg_z)
                    self.seg_labels.append(label)
                    self.record_wise_segments[rec_name].append(len(self.segments)-1)
                    
                    if clz_label_dist.get(label) is None:
                        clz_label_dist[label] = 0
                    clz_label_dist[label] += 1

                    seg_count += 1
                pass

            self.log(
                f"\tn_seg:{seg_count}, n_evt:{evt_count}, annot_dist:{annot_label_dist}, "
                f"clz_lbl_dist:{clz_label_dist}, remain:{len(recording)-(seg_start+self.seg_dim)}")
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
    
    def find_positive_negative(self, q_idx, target_label):
        idx_positive_x = self.find_another_sample(q_idx, target_label)
        neg_lbl = (target_label+1) % self.memory_ds.n_classes
        idx_negative_x = self.find_another_sample(q_idx, neg_lbl)
        return idx_positive_x, idx_negative_x, neg_lbl

    def __getitem__(self, idx):
        r"""Find and return item."""
        ID = self.indexes[idx]
        # trainX = np.array(self.memory_ds.segments[ID])
        trainX = self.memory_ds.segments[ID]
        trainY = self.memory_ds.seg_labels[ID]

        # idx_secondary_x = self.find_another_sample(ID, trainY)
        # # self.log(f"[idx:{idx}] ID:{ID}, 2nd-ID:{idx_secondary_x}")
        # trainX = np.stack([trainX, self.memory_ds.segments[idx_secondary_x]], axis=1)

        # triplet samples (anchor, positive, negative)
        # 
        # seg = np.linspace(-np.pi, np.pi, self.memory_ds.seg_dim)
        # seg = np.sin(seg)
        # seg = np.expand_dims(seg, axis=1)
        # seg = self.memory_ds.scaler.fit_transform(seg)
        # seg = seg.flatten()
        # trainX = np.stack([seg, seg, seg], axis=1).T

        if self.memory_ds.n_classes > 2:
            idx_pos, idx_neg, neg_lbl = self.find_positive_negative(ID, trainY)
            trainX = np.stack([
                trainX, 
                self.memory_ds.segments[idx_pos],
                self.memory_ds.segments[idx_neg]
                ], axis=1).T
            # add neg to trainY
            trainY = np.array([trainY, neg_lbl])
        else:
            idx_pos = self.find_another_sample(ID, trainY)
            trainX = np.stack([
                trainX, 
                self.memory_ds.segments[idx_pos],
                ], axis=1).T
        
        # Make sure data shape (n_chan, n_samp)
        assert trainX.shape[0] < trainX.shape[-1]

        X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        # print(f"X_tensor before: {X_tensor.size()}")
        # r"numpy array shape: (n_samp, n_chan), reshape it to (n_chan, n-samp)."
        # Segment in (1, n_samp) form, still need below line?
        # X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        Y_tensor = trainY
        if torch.any(torch.isnan(X_tensor)):
            X_tensor = torch.nan_to_num(X_tensor)
        return X_tensor, Y_tensor
    

class MesaEcgRRDb():
    def __init__(
            self, base_data_dir, data_subdir="edfs", annot_subdir="annotations-events-nsrr", 
            hz=128, seg_sec=30, class_map=None, log=print, rr_seg_dim=100, rr_min=0.2, 
            rr_max=2, sig_modality="ekg", filter_records=[], n_subjects=-1) -> None:
        self.base_data_dir = base_data_dir
        self.data_dir = os.path.join(base_data_dir, data_subdir)
        self.annot_dir = os.path.join(base_data_dir, annot_subdir)
        # self.data_dir = base_data_dir
        # self.annot_dir = base_data_dir
        self.hz = hz
        self.seg_sec = seg_sec
        self.seg_dim = seg_sec * hz
        self.class_map = class_map
        self.n_classes = len(set(class_map.values()))
        self.log = log
        self.sig_modality = sig_modality
        self.rr_seg_dim = rr_seg_dim
        self.rr_min = rr_min
        self.rr_max = rr_max
        self.filter_records = filter_records
        self.n_subjects = n_subjects
        self.record_names = []
        self.record_wise_segments = {}
        self.segments = []
        self.seg_labels = []
        self.log(
            f"Data base-dir:{base_data_dir}, "
            f"hz:{hz}, seg_sec:{seg_sec}, class_map:{class_map}, "
            f"n_classes:{self.n_classes}")
        self._initialise()
        self.indexes = [i for i in range(len(self.segments))]
        np.random.shuffle(self.indexes)

    def _initialise(self):
        if self.n_subjects > 0:
            rec_names = []
            # randomly choose n recordings
            for f in os.listdir(self.data_dir):
                rec_name = f[:-4]
                rec_names.append(rec_name)
            random_rec_names = set()
            while len(set(random_rec_names)) < self.n_subjects:
                random_rec_names.add(
                    rec_names[random.randint(0, len(rec_names))])
            self.filter_records = list(random_rec_names)
            self.log(f"Filter {len(self.filter_records)} records from {len(rec_names)}")

        count_file = 0
        exclude_file = 0
        for f in os.listdir(self.data_dir):
            if not f.endswith(".csv"):
                continue
            rec_name = f[:-4]
            if len(self.filter_records) > 0 and not rec_name in self.filter_records:
                continue
            data_file = f"{self.data_dir}/{f}"
            # annot_file = f"{self.annot_dir}/{rec_name}-nsrr.xml"

            try:
                seg_count = 0
                evt_count = 0
                clz_label_dist = {}
                _segments, _stages = [], []
                df = pd.read_csv(os.path.join(self.data_dir, f))
                epoch_list = np.unique(df['epoch'])
                epoch_list.sort()
                for ep in epoch_list:
                    evt_count += 1
                    rr_signal = df[df['epoch']==ep]['RPoint']
                    rr_signal = np.diff(rr_signal)
                    if rr_signal is None or len(rr_signal) == 0:
                        # skip segment
                        continue
                    # scale RR from src Hz to target Hz
                    # 
                    rr_signal = rr_signal / 256  # normalise for 0-1 range
                    rr_signal = rr_signal[(rr_signal >= self.rr_min) & (rr_signal <= self.rr_max)]
                    rr_signal = scipy.signal.resample(rr_signal, self.rr_seg_dim)
                    sleep_stage = np.unique(df[df['epoch']==ep]['stage'])[0]
                    label = self.class_map.get(sleep_stage)
                    if label is None:
                        self.log(f"No label for annot '{sleep_stage}' in {f}")
                        continue
                    if clz_label_dist.get(label) is None:
                        clz_label_dist[label] = 0
                    clz_label_dist[label] += 1
                    # add signal and label
                    # 
                    _segments.append(rr_signal)
                    _stages.append(label)
                    seg_count += 1
            except:
                # self.log(f"Error in {f}, {traceback.format_exc()}")
                exclude_file += 1
                continue
            
            self.record_names.append(rec_name)
            if self.record_wise_segments.get(rec_name) is None:
                self.record_wise_segments[rec_name] = []
            # Add segment/label to global list
            # 
            for i_seg in range(len(_segments)):
                self.segments.append(_segments[i_seg])
                self.seg_labels.append(_stages[i_seg])
                self.record_wise_segments[rec_name].append(len(self.segments)-1)
            
            self.log(
                f"\tn_seg:{seg_count}, n_evt:{evt_count}, clz_lbl_dist:{clz_label_dist}")
            count_file += 1
        # sample distribution
        # 
        self.indexes = range(len(self.segments))
        _dist = np.unique(
            [self.seg_labels[i] for i in self.indexes], return_counts=True)
        self.log(f"Total files:{count_file}, n_seg:{len(self.segments)}, distribution:{_dist}")


def main():
    datasource = MesaDb(
        base_data_dir="data/mesa/polysomnography", hz=64,
        # class_map={0:0, 1:1, 2:1, 3:1, 5:1}, 
        class_map={0:0, 1:1, 2:1, 3:1, 5:2},
        n_subjects=1, is_rr_sig=True)
    # datasource = MesaEcgRRDb(
    #     base_data_dir="data/mesa/polysomnography/annotations-rpoints",
    #     class_map={
    #         0:0, 1:1, 2:1, 3:1, 5:1
    #     }, n_subjects=1)
    p_ds = PartialDataset(
        dataset=datasource, 
        seg_index=datasource.indexes[:100], 
        # as_np=as_np
        )
    for i in range(2):
        seg, lbl = p_ds[i]
        print(f"partial-ds, seg:{seg.shape}, label:{lbl.shape}")               
        for ch in range(seg.shape[0]):
            plt.plot(range(seg.shape[-1]), seg[ch, :])
            plt.ylim((0, 2)) 
            plt.title(f"ch:{ch}")
            plt.show()

if __name__ == "__main__":
    try:
        main()
    except:
        # traceback.print_exc(file=sys.stdout)
        print(traceback.format_exc())