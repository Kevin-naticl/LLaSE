import os
import sys

from pathlib import Path
from typing import Optional

import torch
import torch as th
import torch.nn as nn
import soundfile as sf
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import yaml
from tqdm import tqdm
import argparse
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from collections import OrderedDict
import numpy as np
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# for WavLM
from nnet.WavLM import WavLM, WavLMConfig

# for encodec
from vq.codec_encoder import CodecEncoder_Transformer
from vq.codec_decoder_vocos import CodecDecoderVocos
from vq.module import SemanticEncoder

# Simple Datareader
from loader.datareader_fe import DataReader

# llama
from nnet.llama import LLM_AR as model

class Encodec():
    '''
    load Xcodec2 
    '''
    def __init__(self,device="cpu") -> None:
        self.device=device
        ckpt = './ckpt/codec_ckpt/epoch=4-step=1400000.ckpt'
        ckpt = torch.load(ckpt, map_location='cpu')
        state_dict = ckpt['state_dict']
        filtered_state_dict_codec = OrderedDict()
        filtered_state_dict_semantic_encoder = OrderedDict()
        filtered_state_dict_gen = OrderedDict()
        filtered_state_dict_fc_post_a = OrderedDict()
        filtered_state_dict_fc_prior = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('CodecEnc.'):
                new_key = key[len('CodecEnc.'):]
                filtered_state_dict_codec[new_key] = value
            elif key.startswith('generator.'):
                new_key = key[len('generator.'):]
                filtered_state_dict_gen[new_key] = value
            elif key.startswith('fc_post_a.'):
                new_key = key[len('fc_post_a.'):]
                filtered_state_dict_fc_post_a[new_key] = value
            elif key.startswith('SemanticEncoder_module.'):
                new_key = key[len('SemanticEncoder_module.'):]
                filtered_state_dict_semantic_encoder[new_key] = value
            elif key.startswith('fc_prior.'):
                new_key = key[len('fc_prior.'):]
                filtered_state_dict_fc_prior[new_key] = value
        
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            "./ckpt/codec_ckpt/hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b",
            output_hidden_states=True)
        self.semantic_model=self.semantic_model.eval().to(self.device)
        
        self.SemanticEncoder_module = SemanticEncoder(1024,1024,1024)
        self.SemanticEncoder_module.load_state_dict(filtered_state_dict_semantic_encoder)
        self.SemanticEncoder_module = self.SemanticEncoder_module.eval().to(self.device)

        self.encoder = CodecEncoder_Transformer()
        self.encoder.load_state_dict(filtered_state_dict_codec)
        self.encoder = self.encoder.eval().to(self.device)

        self.decoder = CodecDecoderVocos()
        self.decoder.load_state_dict(filtered_state_dict_gen)
        self.decoder = self.decoder.eval().to(self.device)

        self.fc_post_a = nn.Linear( 2048, 1024 )
        self.fc_post_a.load_state_dict(filtered_state_dict_fc_post_a)
        self.fc_post_a = self.fc_post_a.eval().to(self.device)

        self.fc_prior = nn.Linear( 2048, 2048 )
        self.fc_prior.load_state_dict(filtered_state_dict_fc_prior)
        self.fc_prior = self.fc_prior.eval().to(self.device)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "./ckpt/codec_ckpt/hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b")
        
    
    def get_feat(self, wav_batch, pad=None):

        if len(wav_batch.shape) != 2:
            return self.feature_extractor(F.pad(wav_batch, pad), sampling_rate=16000, return_tensors="pt") .data['input_features']
        
        padded_wavs = torch.stack([F.pad(wav, pad) for wav in wav_batch])
        batch_feats = []

        for wav in padded_wavs:
            feat = self.feature_extractor(
                wav,
                sampling_rate=16000,
                return_tensors="pt"
            ).data['input_features']

            batch_feats.append(feat)
        feat_batch = torch.concat(batch_feats, dim=0).to(self.device)
        return feat_batch 

    def get_embedding(self, wav_cpu):
        wav_cpu = wav_cpu.cpu()
        feat = self.get_feat(wav_cpu,pad=(160,160))
        feat = feat.to(self.device)

        if(len(wav_cpu.shape)==1):
            wav = wav_cpu.unsqueeze(0).to(self.device)
        else:
            wav = wav_cpu.to(self.device)

        wav = torch.nn.functional.pad(wav, (0, (200 - (wav.shape[1] % 200))))
        with torch.no_grad():
            vq_emb = self.encoder(wav.unsqueeze(1))
            vq_emb = vq_emb.transpose(1, 2) 

            if vq_emb.shape[2]!=feat.shape[1]:
                feat = self.get_feat(wav_cpu)
                feat = feat.to(self.device)

            semantic_target = self.semantic_model(feat[:,  :,:])
            semantic_target = semantic_target.hidden_states[16]
            semantic_target = semantic_target.transpose(1, 2)
            semantic_target = self.SemanticEncoder_module(semantic_target)

            vq_emb = torch.cat([semantic_target, vq_emb], dim=1)
            # vq_emb =  self.fc_prior(vq_emb.transpose(1, 2)).transpose(1, 2)

        return vq_emb
    
    def emb2token(self, emb):
        emb.to(self.device)
        emb =  self.fc_prior(emb.transpose(1, 2)).transpose(1, 2)
        _, vq_code, _ = self.decoder(emb, vq=True)
        return vq_code

    def token2wav(self, vq_code):
        vq_code.to(self.device)
        vq_post_emb = self.decoder.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
        vq_post_emb = vq_post_emb.transpose(1, 2)
        vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1,2)).transpose(1,2)
        recon = self.decoder(vq_post_emb.transpose(1, 2), vq=False)[0].squeeze()
        # if write the wav, add .squeeze().detach().cpu().numpy()
        # if need gradient use the config right now
        return recon

class WavLM_feat(object):
    '''
    reload pretrained wavlm and extract audio feature
    '''
    
    def __init__(self, device):
        self.wavlm = self._reload_wavLM_large(device=device)
        self.wavlm.eval()

    def __call__(self, wav):
        T = wav.shape[-1]
        wav = wav.reshape(-1, T)
        with torch.no_grad():
            feat = self.wavlm.extract_features(wav, output_layer=6, ret_layer_results=False)[0]
            # B x T x 768(1024) -> B*T x 768(1024)
            B, T, D = feat.shape
            feat = torch.reshape(feat, (-1, D))

            return feat 

    def _reload_wavLM_large(self, path="./ckpt/WavLM-Large.pt", device: Optional[torch.device] = None):
        cpt = torch.load(path, map_location="cpu")
        cfg = WavLMConfig(cpt['cfg'])
        wavLM = WavLM(cfg)
        wavLM.load_state_dict(cpt['model'])
        wavLM.eval()
        if device != None:
            wavLM = wavLM.to(device)
        for p in wavLM.parameters():
            p.requires_grad = False
        print('successful to reload wavLM', path)
        return wavLM 

def load_obj(obj, device):
    '''
    Offload tensor object in obj to cuda device
    '''
    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj
    
    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)

def run(args):
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])    
    dist.init_process_group(args.backend, rank=WORLD_RANK, world_size=WORLD_SIZE)    
    torch.cuda.set_device(LOCAL_RANK)
    
    device = torch.device('cuda', LOCAL_RANK) 
    print(f"[{os.getpid()}] using device: {device}", torch.cuda.current_device(), "local rank", LOCAL_RANK)

    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    data_reader = DataReader(**conf["datareader"])

    # Encodec and WavLM
    codec = Encodec(device)
    wavlm_feat = WavLM_feat(device)

    nnet = model(**conf["nnet_conf"])
    cpt_fname = Path(conf["test"]["checkpoint"])
    cpt = th.load(cpt_fname, map_location="cpu")

    nnet = nnet.to(device)
    nnet = DistributedDataParallel(nnet, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True) 
    nnet.load_state_dict(cpt["model_state_dict"])
    nnet.eval()

    if not os.path.exists(conf["save"]["wav_dir"]):
        os.makedirs(conf["save"]["wav_dir"])
    if not os.path.exists(conf["save"]["feat_dir"]):
        os.makedirs(conf["save"]["feat_dir"])

    # inference is by chunk
    chunk_seconds = conf["test"]["chunk_seconds"]
    overlap_seconds = conf["test"]["overlap_seconds"]

    # Feature Extraction
    if_chunk = conf["test"]["if_chunk"]

    with th.no_grad():
        for egs in tqdm(data_reader, desc="Feature Extraction"):
            egs = load_obj(egs, device)
            audio = egs["mix"].contiguous()
            
            if if_chunk:
                total_samples = audio.shape[-1]
                feat_list = []
                
                chunk_size=16000 * chunk_seconds
                overlap_size = 16000 * overlap_seconds
                
                for start in range(0, total_samples, chunk_size):
                    left = max(0, start - overlap_size)
                    right = min(start + chunk_size + overlap_size, total_samples)
                    
                    left_overlap = (start - left) 
                    right_overlap = (right - (start + chunk_size)) 
                    
                    chunk = audio[:, left:right]
                    
                    # too short to process
                    if total_samples - start < 400:
                        break
                    
                    feat_chunk = wavlm_feat(chunk)  # (1, seq_len, feat_dim)
                    if len(feat_chunk.shape)!=2:
                        continue
                    zeros_row = torch.zeros((1, 1024)).to(device)
                    feat_chunk = torch.concat((feat_chunk, zeros_row), dim = 0)
                    
                    if right_overlap <= 0:
                        feat_chunk = feat_chunk[left_overlap//320:, :]
                    else:
                        feat_chunk = feat_chunk[left_overlap//320: -right_overlap//320, :]
                    
                    feat_chunk = feat_chunk.detach().squeeze(0).cpu().numpy() 
                    feat_list.append(feat_chunk)
                    
                    del chunk, feat_chunk, zeros_row
                    th.cuda.empty_cache()
                    
                full_feat = np.concatenate(feat_list, axis=0)
                
                del audio, feat_list
                
            else:
                full_feat = wavlm_feat(audio)
                zeros_row = torch.zeros((1, 1024)).to(device)
                full_feat = torch.concat((full_feat, zeros_row), dim = 0).detach().squeeze(0).cpu().numpy()      
               
            np.save(os.path.join(conf["save"]["feat_dir"], egs["name"]), full_feat)
            
            del full_feat
            th.cuda.empty_cache()
            
    with th.no_grad():
        for egs in tqdm(data_reader, desc="Audio Generation"):
            feat_path = os.path.join(conf["save"]["feat_dir"], egs["name"] + ".npy")
            full_feat = np.load(feat_path)
            total_frames = full_feat.shape[0]
            
            if if_chunk:
            
                recon_list = []
                chunk_step = chunk_seconds * 50 
                overlap_step = overlap_seconds * 50 
                
                for start in range(0, total_frames, chunk_step):
                    
                    left = max(0, start - overlap_step)
                    right = min(start + chunk_step + overlap_step, total_frames)
                    
                    left_overlap = (start - left) 
                    right_overlap = (right - (start + chunk_step)) 
                    
                    feat_chunk = th.from_numpy(full_feat[left:right, :]).unsqueeze(0)
                    feat_chunk = feat_chunk.to(device)
                    
                    est = nnet(feat_chunk)
                    max_indices = th.argmax(est, dim=1)
                    
                    recon_chunk = codec.token2wav(max_indices.unsqueeze(0))
                    
                    if right_overlap <= 0:
                        recon_chunk = recon_chunk[left_overlap//50 * 16000 :]
                    else:
                        recon_chunk = recon_chunk[left_overlap//50 * 16000 : - right_overlap//50 * 16000]
                    
                    recon_chunk = recon_chunk.squeeze().detach().cpu().numpy()
                    recon_list.append(recon_chunk)
                    
                    del feat_chunk, est, max_indices, recon_chunk
                    th.cuda.empty_cache()
                
                full_recon = np.concatenate(recon_list)
                del recon_list
                
            else:
                est = nnet(th.from_numpy(full_feat).unsqueeze(0))
                max_indices = th.argmax(est, dim=1)   
                full_recon = codec.token2wav(max_indices.unsqueeze(0)).squeeze().detach().cpu().numpy()
                
            sf.write(
                os.path.join(conf["save"]["wav_dir"], egs["name"] + ".wav"),
                full_recon,
                16000
            )
            
            del full_feat,full_recon
            th.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Command to test separation model in Pytorch",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-conf",
                        type=str,
                        required=True,
                        help="Yaml configuration file for training")
    parser.add_argument("--backend",
                        type=str,
                        default="nccl",
                        choices=["nccl", "gloo"])                          
    args = parser.parse_args()
    
    os.environ["NCCL_DEBUG"] = "INFO"
    run(args)