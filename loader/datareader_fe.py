import numpy as np
import torchaudio
import torch

def get_firstchannel_read(path, fs=16000):
    wave_data, sr = torchaudio.load(path)
    if sr != fs:
        wave_data = torchaudio.functional.resample(wave_data, sr, fs)
    if len(wave_data.shape) > 1:
        wave_data = wave_data[0,...]
    wave_data = wave_data.cpu().numpy()
    return wave_data

def parse_scp(scp, path_list):
    with open(scp) as fid: 
        for line in fid:
            tmp = line.strip().split()
            if len(tmp) > 1:
                path_list.append({"inputs": tmp[0], "duration": tmp[1]})
            else:
                path_list.append({"inputs": tmp[0]})

class DataReader(object):
    def __init__(self, filename, sample_rate): 
        self.file_list = []
        self.sample_rate = sample_rate
        parse_scp(filename, self.file_list)

    def extract_feature(self, path):
        path = path["inputs"]
        name = path.split("/")[-1].split(".")[0]
        data = get_firstchannel_read(path, fs=self.sample_rate).astype(np.float32)
        max_norm = np.max(np.abs(data))
        if max_norm == 0:
            max_norm = 1      
        data = data / max_norm
        inputs = np.reshape(data, [1, data.shape[0]])
        inputs = torch.from_numpy(inputs)

        egs = {
            "mix": inputs,
            "max_norm": max_norm,
            "name": name
        }
        return egs

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])

    def get_utt2spk(self, path):
        lines = open(path, "r").readlines()
        for line in lines:
            line = line.strip().split()
            utt_path, spk_id = line[0], line[1]
            self.utt2spk[utt_path] = spk_id
    
    def get_spk2utt(self, path):
        lines = open(path, "r").readlines()
        for line in lines:
            line = line.strip().split()
            utt_path, spk_id = line[0], line[1]
            self.spk2aux[spk_id] = utt_path
