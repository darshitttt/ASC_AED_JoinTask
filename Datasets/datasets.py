from torch.utils.data import Dataset
import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import sys
import audio_utils

TUT_AUD_DIR = '/work/dpandya/giggityGit/audioData/TUTUrban2018/developmentDataset/TUT-urban-acoustic-scenes-2018-development/'
SCAPPER_AUD_DIR = '/work/dpandya/giggityGit/audioData/sythenticSoundscenes/'
TUT_TRAIN_CSV = '/work/dpandya/giggityGit/ASC_AED_JoinTask/Datasets/TUT18_train.csv'
TUT_TEST_CSV = '/work/dpandya/giggityGit/ASC_AED_JoinTask/Datasets/TUT18_test.csv'
SCAPPER_TRAIN_CSV = '/work/dpandya/giggityGit/ASC_AED_JoinTask/Datasets/scrapper_train_dataset.csv'
SCAPPER_TEST_CSV = '/work/dpandya/giggityGit/ASC_AED_JoinTask/Datasets/scrapper_test_dataset.csv'


# Synthetic Scrapper Dataset with events and scenes
class scraperDataset(Dataset):

    '''datadir = '../../audioData/sythenticSoundscenes/test/'
    csv = 'scrapper_dataset.csv'''

    def __init__(self, dataset_csv, data_dir, only_scene=False, transforms=None):

        self.dataset_csv = dataset_csv
        self.data_directory = data_dir
        self.only_scene = only_scene
        self.dataframe = pd.read_csv(dataset_csv)
        self.transforms = transforms

        self.audio_files = self.dataframe['audio_fileNames']
        self.label_files = self.dataframe['label_fileNames']
        self.scene_labels = self.dataframe['acoustic_scene_label']
        self.events_label_list = self.dataframe['events_label_list']
        self.all_scenes = self.scene_labels.unique()
        self.all_events = self.events_label_list.unique()

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load Audio file
        audio_file = os.path.join(self.data_directory, self.audio_files[idx])
        #audio_data, sr = torchaudio.load(audio_file)
        audio = audio_utils.load_audio_from_file(audio_file)
        scene_label = label_to_one_hot(self.scene_labels[idx], (list)(self.all_scenes))
        if self.transforms:
            audio = self.transforms(audio)

        if self.only_scene:
            sample = {'data':audio, 'scene_label':scene_label}
        else:
            sample = {'data':audio, 'scene_label':scene_label}
        
        return sample


# Real-world TUTUrban18 Dataset with Scene Labels
class TUT18_Dataset(Dataset):

    '''datadir = '../../audioData/TUTUrban2018/developmentDataset/TUT-urban-acoustic-scenes-2018-development/'
    csv = 'TUT18_train.csv'''

    def __init__(self, dataset_csv, data_dir, transforms=None):

        self.dataset_csv = dataset_csv
        self.data_directory = data_dir
        self.dataframe = pd.read_csv(dataset_csv)
        self.transforms = transforms
        self.audio_files = self.dataframe['files']
        self.scene_labels = self.dataframe['labels']
        self.all_scenes = self.scene_labels.unique()

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file = os.path.join(self.data_directory, self.audio_files[idx])
        #audio_data, sr = torchaudio.load(audio_file)
        audio = audio_utils.load_audio_from_file(audio_file)
        scene_label = label_to_one_hot(self.scene_labels[idx], self.all_scenes)
        
        if self.transforms:
            audio = self.transforms(audio)
        
        sample = {'data':audio, 'scene_label':scene_label}
        
        return sample
    


def label_to_one_hot(label, label_array):
    """
    Convert string labels to one-hot encoded labels based on the provided array of labels.

    Args:
    - labels (list of str): List of string labels to convert.
    - label_array (numpy array): Array containing all possible labels.

    Returns:
    - one_hot_encoded (numpy array): One-hot encoded labels corresponding to the input labels.
    """
    label_dict = {label: i for i, label in enumerate(label_array)}
    one_hot_encoded = np.zeros(len(label_array), dtype=int)
    one_hot_encoded[label_dict[label]] = 1
    return torch.tensor(one_hot_encoded)
