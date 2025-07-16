import os
import torch
from torch.utils.data import Dataset
from Processing.const import CONSTANTS

class MidiDatasetLoader(Dataset):
    # load_data_type = "melody" | "chords"
    def __init__(self, load_data_type = "chords"):
        self.chords_data_chunks_save_path = CONSTANTS.CHORDS_DATA_CHUNKS_SAVE_PATH
        self.melody_data_chunks_save_path = CONSTANTS.MELODY_DATA_CHUNKS_SAVE_PATH
        self.melody_video_data_chunks_save_path = CONSTANTS.MELODY_VIDEO_CHUNKS_SAVE_PATH
        self.chords_video_data_chunks_save_path = CONSTANTS.CHORDS_VIDEO_CHUNKS_SAVE_PATH
        
        self.load_path = load_data_type == "chords" and self.chords_data_chunks_save_path or self.melody_data_chunks_save_path
        self.video_load_path = load_data_type == "chords" and self.chords_video_data_chunks_save_path or self.melody_video_data_chunks_save_path
        
        self.load_data_type = load_data_type

    def __len__(self):
        return len(os.listdir(self.load_path))

    def __getitem__(self, idx):
        music_load_path = f"{self.load_path}/{idx}.pt"
        music_data = torch.load(music_load_path, weights_only=True)
        
        video_load_path = f"{self.video_load_path}/{idx}.pt"
        video_data = torch.load(video_load_path, weights_only=True)
        
        pitches_tokens = music_data['chords_pitches'] if self.load_data_type == "chords" else music_data['melody_pitches']
        duration_tokens = music_data['chords_duration'] if self.load_data_type == "chords" else music_data['melody_duration']
        
        return pitches_tokens, duration_tokens, video_data['video']
