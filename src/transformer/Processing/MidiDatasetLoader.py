import os
import torch
from torch.utils.data import Dataset

class MidiDatasetLoader(Dataset):
    def __init__(self, video_data_chunks_save_path, music_data_chunks_save_path):
        self.video_data_chunks_save_path = video_data_chunks_save_path
        self.music_data_chunks_save_path = music_data_chunks_save_path

    def __len__(self):
        return len(os.listdir(self.video_data_chunks_save_path))

    def __getitem__(self, idx):
        video_load_path = f"{self.video_data_chunks_save_path}/{idx}.pt"
        music_load_path = f"{self.music_data_chunks_save_path}/{idx}.pt"
        video_data = torch.load(video_load_path, weights_only=True)
        music_data = torch.load(music_load_path, weights_only=True)
    
        return music_data['melody'], music_data['chords'], music_data['chords_context_inputs'], video_data['video']