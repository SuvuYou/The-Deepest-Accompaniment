import os
import re
import sys
import torch
import numpy as np
from Processing.SongsEncoder import SongsEncoder
from Processing.SongsMapper import SongsMapper

np.set_printoptions(threshold=sys.maxsize, linewidth=100000)

class MidiDatasetSaver():
    def __init__(self, videoProcessor, chord_sequence_lengths, melody_sequence_lengths, CONSTANTS):
        self.CONSTANTS = CONSTANTS
        self.videoProcessor = videoProcessor    

        batches_count_per_chord_sequence = [x - CONSTANTS.DEFAULT_SEQUENCE_LENGTH for x in chord_sequence_lengths]
        batches_count_per_melody_sequence = [x - CONSTANTS.DEFAULT_SEQUENCE_LENGTH for x in melody_sequence_lengths]
        
        self.chord_sequence_chunk_size = _find_greatest_common_divisor(batches_count_per_chord_sequence, upper_bound=50, lower_bound=30)
        self.melody_sequence_chunk_size = _find_greatest_common_divisor(batches_count_per_melody_sequence, upper_bound=50, lower_bound=30)
        
        self.classes_size = {
            "melody_pitch": SongsMapper.get_mappings_size(self.CONSTANTS.MELODY_PITCH_MAPPINGS_PATH),
            "melody_duration": SongsMapper.get_mappings_size(self.CONSTANTS.MELODY_DURATION_MAPPINGS_PATH),
            "chords_pitch": SongsMapper.get_mappings_size(self.CONSTANTS.CHORDS_PITCH_MAPPINGS_PATH),
            "chords_duration": SongsMapper.get_mappings_size(self.CONSTANTS.CHORDS_DURATION_MAPPINGS_PATH),
        }
        
        self.mappings = {
            "melody_pitch_mapping": SongsMapper.load_mappings(self.CONSTANTS.MELODY_PITCH_MAPPINGS_PATH),
            "melody_duration_mapping": SongsMapper.load_mappings(self.CONSTANTS.MELODY_DURATION_MAPPINGS_PATH),
            "chord_pitch_mapping": SongsMapper.load_mappings(self.CONSTANTS.CHORDS_PITCH_MAPPINGS_PATH),
            "chord_duration_mapping": SongsMapper.load_mappings(self.CONSTANTS.CHORDS_DURATION_MAPPINGS_PATH),
        }

        self.folders = self._get_numeric_subfolders(CONSTANTS.MAPPED_MIDI_DATA_FOLDER_PATH)
        self.current_total_melody_chunks = 0
        self.current_total_chords_chunks = 0

    def _get_numeric_subfolders(self, base_folder):
        subfolders = [
            os.path.join(path, folder) for path, folders, _ in os.walk(base_folder)
            for folder in folders if _has_only_numbers(folder)
        ]
        
        return subfolders

    def save_training_data(self):
        for folder in self.folders:
            data = self._extract_single_training_file(folder)
            self._save_chords_training_sequence(data)
            self._save_melody_training_sequence(data)

    def _extract_single_training_file(self, folder_path):
        melody_pitch_tokens, melody_duration_tokens, chords_pitch_tokens, chords_duration_tokens = SongsEncoder.load_single_encoded_txt_song(folder_path)
        
        melody_length = max(len(melody_pitch_tokens.split()), len(melody_duration_tokens.split()))
        chords_length = max(len(chords_pitch_tokens.split()), len(chords_duration_tokens.split()))

        chords_training_video, melody_training_video = self._process_video(folder_path, chords_length, melody_length)
        
        return {
            "melody_length": melody_length,
            "chords_length": chords_length,
            "melody_pitch_tokens": SongsMapper.convert_song_to_int(melody_pitch_tokens, self.mappings["melody_pitch_mapping"]["mappings"]),
            "melody_duration_tokens": SongsMapper.convert_song_to_int(melody_duration_tokens, self.mappings["melody_duration_mapping"]["mappings"]),
            "chords_pitch_tokens": SongsMapper.convert_song_to_int(chords_pitch_tokens, self.mappings["chord_pitch_mapping"]["mappings"]),
            "chords_duration_tokens": SongsMapper.convert_song_to_int(chords_duration_tokens, self.mappings["chord_duration_mapping"]["mappings"]),
            "melody_training_video": melody_training_video,
            "chords_training_video": chords_training_video
        }

    def _process_video(self, folder_path, chords_length, melody_length):
        video_path = f"{folder_path.replace(self.CONSTANTS.MAPPED_MIDI_DATA_FOLDER_PATH, self.CONSTANTS.VIDEO_DATA_FOLDER_PATH)}/original-180.mp4"
        video, _ = self.videoProcessor.load_video_frames(video_path)
        
        chords_training_video = self.videoProcessor.process_video(video, target_video_length_in_frames=chords_length)
        melody_training_video = self.videoProcessor.process_video(video, target_video_length_in_frames=melody_length)
        
        return chords_training_video, melody_training_video

    def _save_chords_training_sequence(self, data):
        chords_training_video = data["chords_training_video"].to(self.CONSTANTS.DEVICE)
        
        frames_count = chords_training_video.shape[0]
        frames_per_data_item = frames_count // data["chords_length"]
        num_sequences = data["chords_length"] - self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH
        num_chunks = num_sequences // self.chord_sequence_chunk_size

        for chunk_index in range(num_chunks):
            self._save_chords_data_to_chunk(chunk_index, data, frames_per_data_item)
            
    def _save_melody_training_sequence(self, data):
        melody_training_video = data["melody_training_video"].to(self.CONSTANTS.DEVICE)
        
        frames_count = melody_training_video.shape[0]
        frames_per_data_item = frames_count // data["melody_length"]
        num_sequences = data["melody_length"] - self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH
        num_chunks = num_sequences // self.melody_sequence_chunk_size

        for chunk_index in range(num_chunks):
            self._save_melody_data_to_chunk(chunk_index, data, frames_per_data_item)

    def _save_chords_data_to_chunk(self, chunk_index, data, frames_per_data_item):
        print(chunk_index, '- chunk')
        
        chords_data_save_path = f"{self.CONSTANTS.CHORDS_DATA_CHUNKS_SAVE_PATH}/{self.current_total_chords_chunks}.pt"
        chords_video_data_save_path = f"{self.CONSTANTS.CHORDS_VIDEO_CHUNKS_SAVE_PATH}/{self.current_total_chords_chunks}.pt"

        sequence_data = {
            "chords_pitches": [],
            "chords_duration": [],
            "video": []
        }
        
        self._prepare_chords_chunk_data(chunk_index, data, frames_per_data_item, sequence_data)
        
        encoded_music_data = self._one_hot_encode_chords_data(sequence_data)
        
        torch.save(encoded_music_data, chords_data_save_path)
        torch.save({"video": torch.stack(sequence_data["video"])}, chords_video_data_save_path)
        
        self.current_total_chords_chunks += 1
        
    def _save_melody_data_to_chunk(self, chunk_index, data, frames_per_data_item):
        print(chunk_index, '- chunk')
        
        melody_save_path = f"{self.CONSTANTS.MELODY_DATA_CHUNKS_SAVE_PATH}/{self.current_total_melody_chunks}.pt"
        melody_video_save_path = f"{self.CONSTANTS.MELODY_VIDEO_CHUNKS_SAVE_PATH}/{self.current_total_melody_chunks}.pt"

        sequence_data = {
            "melody_pitches": [],
            "melody_duration": [],
            "video": []
        }
        
        self._prepare_melody_chunk_data(chunk_index, data, frames_per_data_item, sequence_data)
        
        encoded_music_data = self._one_hot_encode_melody_data(sequence_data)
        
        torch.save(encoded_music_data, melody_save_path)
        torch.save({"video": torch.stack(sequence_data["video"])}, melody_video_save_path)
        
        self.current_total_melody_chunks += 1
        
    def _prepare_chords_chunk_data(self, chunk_index, data, frames_per_data_item, sequence_data):
        for i in range(self.chord_sequence_chunk_size):
            idx = i + (chunk_index * self.chord_sequence_chunk_size)
            sequence_data["chords_pitches"].append(data["chords_pitch_tokens"][idx:idx + self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH])
            sequence_data["chords_duration"].append(data["chords_duration_tokens"][idx:idx + self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH])
            
            frames = data["chords_training_video"][(idx * frames_per_data_item): (idx + self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH) * frames_per_data_item]
            sequence_data["video"].append(frames[[0, int(self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH / 2)]])
            
    def _prepare_melody_chunk_data(self, chunk_index, data, frames_per_data_item, sequence_data):
        for i in range(self.melody_sequence_chunk_size):
            idx = i + (chunk_index * self.melody_sequence_chunk_size)
            sequence_data["melody_pitches"].append(data["melody_pitch_tokens"][idx:idx + self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH])
            sequence_data["melody_duration"].append(data["melody_duration_tokens"][idx:idx + self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH])
            
            frames = data["melody_training_video"][(idx * frames_per_data_item): (idx + self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH) * frames_per_data_item]
            sequence_data["video"].append(frames[[0, int(self.CONSTANTS.DEFAULT_SEQUENCE_LENGTH / 2)]])

    def _one_hot_encode_chords_data(self, sequence_data):
        return {
            'chords_pitches': torch.nn.functional.one_hot(torch.tensor(sequence_data["chords_pitches"]), num_classes=self.classes_size['chords_pitch']), 
            'chords_duration': torch.nn.functional.one_hot(torch.tensor(sequence_data["chords_duration"]), num_classes=self.classes_size['chords_duration']), 
        }

        
    def _one_hot_encode_melody_data(self, sequence_data):
        return {
            'melody_pitches': torch.nn.functional.one_hot(torch.tensor(sequence_data["melody_pitches"]), num_classes=self.classes_size['melody_pitch']),
            'melody_duration': torch.nn.functional.one_hot(torch.tensor(sequence_data["melody_duration"]), num_classes=self.classes_size['melody_duration']),
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
