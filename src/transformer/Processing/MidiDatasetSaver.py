import os
import re
import sys
import torch
import numpy as np
from torch.utils.data import Dataset

np.set_printoptions(threshold=sys.maxsize, linewidth=100000)

class MidiDatasetSaver(Dataset):
    def __init__(self, songsEncoder, songsMapper, videoProcessor, song_lengths, CONSTANTS):
        self.CONSTANTS = CONSTANTS
        self.songsEncoder = songsEncoder
        self.songsMapper = songsMapper
        self.videoProcessor = videoProcessor    

        batches_count_per_song = [x - CONSTANTS.DEFAULT_SEQUENCE_LENGTH for x in song_lengths]
        self.chunk_size = _find_greatest_common_divisor(batches_count_per_song, upper_bound=50, lower_bound=30)
        self.classes_size = self._initialize_classes_size()

        self.folders = self._get_numeric_subfolders(CONSTANTS.MAPPED_MIDI_DATA_FOLDER_PATH)
        self.current_total_chunks = 0

    def _initialize_classes_size(self):
        return {
            "melody": self.songsMapper.get_mappings_size(self.CONSTANTS.MELODY_MAPPINGS_PATH),
            "chords": self.songsMapper.get_mappings_size(self.CONSTANTS.CHORDS_MAPPINGS_PATH),
            "chords_context_inputs": self.songsMapper.get_mappings_size(self.CONSTANTS.CHORDS_CONTEXT_MAPPINGS_PATH),
        }

    def _get_numeric_subfolders(self, base_folder):
        subfolders = [
            os.path.join(path, folder) for path, folders, _ in os.walk(base_folder)
            for folder in folders if _has_only_numbers(folder)
        ]
        return subfolders

    def save_training_data(self):
        for folder in self.folders:
            data = self._extract_single_training_file(folder)
            self._save_single_type_training_sequence(data)

    def _extract_single_training_file(self, folder_path):
        melody, chords, chords_context, song_length = self.songsEncoder.load_single_encoded_txt_song(folder_path)
        
        padded_melody = self.songsEncoder.padd_encoded_song_with_rest(melody, to_length=song_length)
        padded_chords = self.songsEncoder.padd_encoded_song_with_rest(chords, to_length=song_length)
        padded_chords_context = self.songsEncoder.padd_encoded_song_with_rest(chords_context, to_length=song_length, is_chords_context=True)
        
        video = self._process_video(folder_path, song_length)
        
        return {
            "melody": self.songsMapper.convert_songs_to_int(padded_melody, self.CONSTANTS.MELODY_MAPPINGS_PATH),
            "chords": self.songsMapper.convert_songs_to_int(padded_chords, self.CONSTANTS.CHORDS_MAPPINGS_PATH),
            "chords_context": self.songsMapper.convert_songs_to_int(padded_chords_context, self.CONSTANTS.CHORDS_CONTEXT_MAPPINGS_PATH),
            "video": video
        }

    def _process_video(self, folder_path, song_length):
        video_path = f"{folder_path.replace(self.CONSTANTS.MAPPED_MIDI_DATA_FOLDER_PATH, self.CONSTANTS.VIDEO_DATA_FOLDER_PATH)}/original-180.mp4"
        video, _ = self.videoProcessor.load_video_frames(video_path)
        return self.videoProcessor.process_video(video, target_video_length_in_frames=song_length)

    def _save_single_type_training_sequence(self, data):
        video = data["video"].to(self.CONSTANTS.DEVICE)
        frames_count = video.shape[0]
        frames_per_data_item = frames_count // len(data["melody"])
        num_sequences = len(data["melody"]) - self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH
        num_chunks = num_sequences // self.chunk_size

        for chunk_index in range(num_chunks):
            self._save_chunk(chunk_index, data, frames_per_data_item)

    def _save_chunk(self, chunk_index, data, frames_per_data_item):
        print(chunk_index, '- chunk')
        
        music_data_save_path = f"{self.CONSTANTS.MUSIC_DATA_CHUNKS_SAVE_PATH}/{self.current_total_chunks}.pt"
        video_save_path = f"{self.CONSTANTS.VIDEO_CHUNKS_SAVE_PATH}/{self.current_total_chunks}.pt"

        sequence_data = {
            "melody_inputs": [],
            "chords_inputs": [],
            "chords_context_inputs": [],
            "video_inputs": []
        }
        
        self._prepare_chunk_data(chunk_index, data, frames_per_data_item, sequence_data)
        
        encoded_music_data = self._one_hot_encode_data(sequence_data)
        
        torch.save(encoded_music_data, music_data_save_path)
        torch.save({"video": torch.stack(sequence_data["video_inputs"])}, video_save_path)
        
        self.current_total_chunks += 1
        
    def _prepare_chunk_data(self, chunk_index, data, frames_per_data_item, sequence_data):
        for i in range(self.chunk_size):
            idx = i + (chunk_index * self.chunk_size)
            sequence_data["melody_inputs"].append(data["melody"][idx:idx + self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH])
            sequence_data["chords_inputs"].append(data["chords"][idx:idx + self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH])
            sequence_data["chords_context_inputs"].append(data["chords_context"][idx + self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH])
            
            frames = data["video"][(idx * frames_per_data_item): (idx + self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH) * frames_per_data_item]
            sequence_data["video_inputs"].append(frames[[0, int(self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH / 2)]])

    def _one_hot_encode_data(self, sequence_data):
        return {
            'melody': torch.tensor(torch.nn.functional.one_hot(torch.tensor(sequence_data["melody_inputs"]), num_classes=self.classes_size['melody']), dtype=torch.float32),
            'chords': torch.tensor(torch.nn.functional.one_hot(torch.tensor(sequence_data["chords_inputs"]), num_classes=self.classes_size['chords']), dtype=torch.float32),
            'chords_context_inputs': torch.tensor(torch.nn.functional.one_hot(torch.tensor(sequence_data["chords_context_inputs"]), num_classes=self.classes_size['chords_context_inputs']), dtype=torch.float32)
        }

def _has_only_numbers(input_str):
    return bool(re.match(r'^\d+$', input_str))

def _find_greatest_common_divisor(numbers_list, upper_bound=50, lower_bound=30):
    min_remainder = float('inf')
    chosen_gcd = upper_bound

    for potential_gcd in range(upper_bound, lower_bound, -1):
        remainders = [length % (100 * potential_gcd) for length in numbers_list]
        if max(remainders) < min_remainder:
            min_remainder = max(remainders)
            chosen_gcd = potential_gcd

    return chosen_gcd
