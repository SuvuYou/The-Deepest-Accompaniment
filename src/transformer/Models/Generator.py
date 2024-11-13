import json
import numpy as np
import torch
import music21

class Generator:
    def __init__(self, chords_generation_model, save_file_name, CONSTANTS):
        self.save_file_name = save_file_name
        self.CONSTANTS = CONSTANTS
        self.step_duration = CONSTANTS.ACCEPTABLE_DURATIONS[0]
        
        self.chords_generation_model = chords_generation_model
        
        with open(self.CONSTANTS.CHORDS_MAPPINGS_PATH, "r") as fp:
            self._chords_mappings = json.load(fp)['mappings']
        
        self.chords_mappings_inv = {v: k for k, v in self._chords_mappings.items()}  # Reverse mapping for easy decoding

    def generate(self, chords_seed, video, num_steps, max_sequence_length, temperature):
        with torch.no_grad():
            chords_seed = chords_seed.split()
            chords = chords_seed
        
            # Convert seed chords to indices
            chords_seed = [self._chords_mappings[symbol] for symbol in chords_seed]

            for idx in range(num_steps):
                # Use only the last max_sequence_length items for generation
                chords_seed = chords_seed[-max_sequence_length:]
                seed_length = len(chords_seed)

                # Prepare video input corresponding to the current seed length
                video_seed = video[idx: idx + seed_length]
                video_seed = video_seed[[0, len(video_seed) // 2]]  # Use two frames per prediction
                
                num_frames, height, width, channels = video_seed.shape
                video_seed = video_seed.reshape(channels, num_frames, height, width)

                # Prepare one-hot encoded chord seed as input
                onehot_chords_seed = torch.nn.functional.one_hot(torch.tensor(chords_seed, dtype=torch.int64), num_classes=len(self._chords_mappings))
                
                # Transformer expects input in the shape [seq_len, batch_size, feature_dim], so we permute and add batch dimension
                chords_output = self.chords_generation_model(
                    onehot_chords_seed.unsqueeze(0).float(),  # [batch_size, seq_len, feature_dim]
                    video_seed.unsqueeze(0).float()  # [batch_size, channels, num_frames, height, width] for video
                )
                
                # Apply temperature sampling
                chords_probabilities = chords_output.squeeze(0)[-1]  # Get probabilities of last step

                chords_output_int = self._sample_with_temperature(chords_probabilities, temperature)
                
                # Decode the predicted chord to its symbol and append
                chords_output_symbol = self.chords_mappings_inv[chords_output_int]
                chords_seed.append(chords_output_int)
                chords.append(chords_output_symbol)
                
            return chords

    def _sample_with_temperature(self, probabilities, temperature):
        probabilities = torch.nn.functional.softmax(probabilities / temperature, dim=0)
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities.numpy())
        
        return index

    def _save_chord_progression(self, chords, step_duration, file_name):
        stream = music21.stream.Stream()
        chords = [x for x in ' '.join(chords).lstrip('_ ').split()]

        current_symbol = chords[0]
        current_symbol_step_counter = 1
        
        chords_length = len(chords[1:])

        for i, symbol in enumerate(chords[1:]):     
            if symbol == "_" and i != chords_length - 1:
                current_symbol_step_counter += 1
            else:
                quarter_length_duration = step_duration * current_symbol_step_counter
                if current_symbol == "r":
                    event = music21.note.Rest(quarterLength=quarter_length_duration)
                else:
                    current_symbol = current_symbol.replace('(', '').replace(')', '').split('-')
                    event = music21.chord.Chord(notes=current_symbol, quarterLength=quarter_length_duration)
                    
                stream.append(event)
                
                current_symbol = symbol
                current_symbol_step_counter = 1
                    
        stream.write('midi', file_name)
        
    def save_to_file(self, chords):
        self._save_chord_progression(chords, self.step_duration, file_name=f'generated/{self.save_file_name}_chords.mid')
