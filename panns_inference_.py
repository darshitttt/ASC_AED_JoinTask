# %%
import panns_inference
import librosa
from panns_inference import AudioTagging, SoundEventDetection, labels
import numpy as np
import pandas as pd
import os

# %%
model_path00 = 'models/Cnn14_mAP=0.431.pth'
model_path01 = 'models/Cnn14_DecisionLevelMax.pth'

def get_event_list(clipwise_output):

    # With the slicing, we are reversing the sorted index to get the descending order of event probs.
    sorted_indexes = np.argsort(clipwise_output)[::-1]
    event_list = []

    # Creating a list of events with more than 0.05 prob
    for i in range(0, len(sorted_indexes)):
        event_prob = clipwise_output[sorted_indexes[i]]
        if event_prob > 0.05:
            event_list.append(np.array(labels)[sorted_indexes[i]])
    
    return event_list
    

def get_panns_inference(audio_file_name, model_path):

    (audio, _) = librosa.core.load(audio_file_name, sr=32000, mono=True)
    audio = audio[None, :]
    at = AudioTagging(checkpoint_path=model_path, device='cuda')
    (clipwise_output, embedding) = at.inference(audio)

    event_list = get_event_list(clipwise_output[0])
    return event_list

def get_panns_event_labels(files_list, audio_dir):
    event_labels_list = []
    
    for file in files_list:
        audio_fname = os.path.join(audio_dir, file)
        audio_event_list = get_panns_inference(audio_fname, model_path00)
        event_labels_list.append(audio_event_list)

    return event_labels_list



