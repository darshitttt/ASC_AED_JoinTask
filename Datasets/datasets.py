from torch.utils.data import Dataset
import os
import torch
import torchaudio
import pandas as pd
import numpy as np

# Synthetic Scrapper Dataset with events and scenes
class scraperDataset(Dataset):

    '''datadir = '../../audioData/sythenticSoundscenes/test/'
    csv = 'scrapper_dataset.csv'''

    def __init__(self, dataset_csv, data_dir, only_scene=False):

        self.dataset_csv = dataset_csv
        self.data_directory = data_dir
        self.only_scene = only_scene
        self.dataframe = pd.read_csv(dataset_csv)

        self.audio_files = self.dataframe['audio_fileNames']
        self.label_files = self.dataframe['label_fileNames']
        self.scene_labels = self.dataframe['acoustic_scene_label']
        self.events_label_list = self.dataframe['events_label_list']

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load Audio file
        audio_file = os.path.join(self.data_directory, self.audio_files[idx])
        audio_data, sr = torchaudio.load(audio_file)

        if self.only_scene:
            sample = {'audio':audio_data, 'scene_label':self.scene_labels[idx]}
        else:
            sample = {'audio':audio_data, 'scene_label':self.scene_labels[idx], 'event_list':self.events_label_list[idx]}
        
        return sample


# Real-world TUTUrban18 Dataset with Scene Labels
class TUT18_Dataset(Dataset):

    '''datadir = '../../audioData/TUTUrban2018/developmentDataset/TUT-urban-acoustic-scenes-2018-development/'
    csv = 'TUT18_train.csv'''

    def __init__(self, dataset_csv, data_dir):

        self.dataset_csv = dataset_csv
        self.data_directory = data_dir
        self.dataframe = pd.read_csv(dataset_csv)

        self.audio_files = self.dataframe['files']
        self.scene_labels = self.dataframe['labels']

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file = os.path.join(self.data_directory, self.audio_files[idx])
        audio_data, sr = torchaudio.load(audio_file)

        sample = {'audio':audio_data, 'scene_label':self.scene_labels[idx]}
        return sample
