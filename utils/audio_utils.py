# %%
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import os
import random
import torch
import torch.nn as nn
from IPython.display import display, Audio

SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = int(SAMPLE_RATE * 0.04)
HOP_LEN = int(SAMPLE_RATE * 0.02)

mel_spectrogram = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LEN,
    n_mels=N_MELS
)

# %%
def load_audio(aud_filename):
    wav, sr = torchaudio.load(aud_filename, normalize=True)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        wav = resampler(wav)

    return wav

def get_duration(aud_filename):
    audio_info = torchaudio.info(aud_filename)
    return (audio_info.num_frames/audio_info.sample_rate)

def round_up(audio):
    rem = SAMPLE_RATE - audio.shape[1]%SAMPLE_RATE
    baggage = torch.zeros([1,rem], dtype=audio.dtype)
    audio = torch.cat((audio,baggage), 1)
    while audio.shape[1] < SAMPLE_RATE*5:
        audio = audio.repeat(1,2)
    return audio

def same_dur_as(audio0, audio1):
    while audio0.shape[1] > audio1.shape[1]:
        audio1 = audio1.repeat(1,2)
        
    return audio1[:,0:audio0.shape[1]]

def add_two_noise(aud_filename, snr_list):
    
    snr = torch.tensor(snr_list)
    wav0 = round_up(load_audio(aud_filename[0]))
    wav1 = round_up(load_audio(aud_filename[1]))
    
    if get_duration(aud_filename[0])>get_duration(aud_filename[1]):
        wav1 = same_dur_as(wav0, wav1)
    else:
        wav0 = same_dur_as(wav1, wav0)
    
    noisy = F.add_noise(wav0, wav1, snr)
    noisy = nn.functional.normalize(noisy)
    return noisy

def get_log_melSpectrogram(audio):
    mel_feats = mel_spectrogram(audio)
    log_mel_feats = T.AmplitudeToDB()(mel_feats)
    return log_mel_feats

def get_random_audioFeatures(audio):
    
    t_audio = (int)(audio.shape[1]/SAMPLE_RATE)
    #rand_sec = random.choice([i for i in range(0,t_audio-5)])
    rand_sec = 0
    start_sample = rand_sec*SAMPLE_RATE
    end_sample = (rand_sec+5)*SAMPLE_RATE
    log_mel_features = get_log_melSpectrogram(audio[:,start_sample:end_sample])

    return log_mel_features

# %%
'''sample_ = '../../LibriVox_Kaggle/achtgesichterambiwasse/achtgesichterambiwasse_0022.wav'
#sample_1 = '../../LibriVox_Kaggle/achtgesichterambiwasse/achtgesichterambiwasse_0001.wav'
noise = '../audioData/NIGENS/NIGENS/footsteps/FootstepsWood+6017_16_1.wav'
snrlist = [5]

noisy = add_two_noise([sample_, noise], snrlist)
display(Audio(noisy, rate=SAMPLE_RATE))
get_random_audioFeatures(noisy).shape''' 


