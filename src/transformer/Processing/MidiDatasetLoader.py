import os
import torch
from torch.utils.data import Dataset

class MidiDatasetLoader(Dataset):
    def __init__(self, video_data_chunks_save_path, music_data_chunks_save_path, is_using_video = True):
        self.is_using_video = is_using_video
        self.music_data_chunks_save_path = music_data_chunks_save_path
        
        if is_using_video:
            self.video_data_chunks_save_path = video_data_chunks_save_path

    def __len__(self):
        return len(os.listdir(self.music_data_chunks_save_path))

    def __getitem__(self, idx):
        
        music_load_path = f"{self.music_data_chunks_save_path}/{idx}.pt"
        music_data = torch.load(music_load_path, weights_only=True)
        
        if self.is_using_video:
            video_load_path = f"{self.video_data_chunks_save_path}/{idx}.pt"
            video_data = torch.load(video_load_path, weights_only=True)
            
            return music_data['melody'], music_data['chords'], video_data['video']
            
        return music_data['melody'], music_data['chords']