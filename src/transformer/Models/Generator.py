import json
import numpy as np
import torch
import music21

class Generator:
    def __init__(self, chords_generation_model, melody_generation_model, save_file_name, CONSTANTS):
        self.save_file_name = save_file_name
        self.CONSTANTS = CONSTANTS
        self.step_duration = CONSTANTS.ACCEPTABLE_DURATIONS[0]
        
        self.chords_generation_model = chords_generation_model
        self.melody_generation_model = melody_generation_model
        
        with open(self.CONSTANTS.CHORDS_MAPPINGS_PATH, "r") as fp:
            self._chords_mappings = json.load(fp)['mappings']
        
        self.chords_mappings_inv = {v: k for k, v in self._chords_mappings.items()} 
        
        with open(self.CONSTANTS.MELODY_MAPPINGS_PATH, "r") as fp:
            self._melody_mappings = json.load(fp)['mappings']
        
        self.melody_mappings_inv = {v: k for k, v in self._melody_mappings.items()}
        
        with open(self.CONSTANTS.CHORDS_CONTEXT_MAPPINGS_PATH, "r") as fp:
            self._chords_context_mappings = json.load(fp)['mappings']
        
        self._chords_context_mappings_inv = {v: k for k, v in self._chords_context_mappings.items()}

    def generate(self, chords_seed, melody_seed, chords_context_seed, video, num_steps, max_sequence_length, temperature):
        with torch.no_grad():
            chords_seed = chords_seed.split()
            melody_seed = melody_seed.split()
            chords_context_seed = chords_context_seed.split()
            
            chords = chords_seed
            melody = melody_seed
            chords_context = chords_context_seed
            
            # Convert seed chords and melody to indices
            chords_seed = [self._chords_mappings[symbol] for symbol in chords_seed]
            melody_seed = [self._melody_mappings[symbol] for symbol in melody_seed]
            chords_context_seed = [self._chords_context_mappings[symbol] for symbol in chords_context_seed]

            for idx in range(num_steps):
                # Use only the last max_sequence_length items for generation
                chords_seed = chords_seed[-max_sequence_length:]
                chords_context_seed = chords_context_seed[-max_sequence_length:]
                melody_seed = melody_seed[-max_sequence_length:]
                seed_length = len(chords_seed)

                # Prepare video input corresponding to the current seed length
                video_seed = video[idx: idx + seed_length]
                video_seed = video_seed[[0, len(video_seed) // 2]]  # Use two frames per prediction
                
                num_frames, height, width, channels = video_seed.shape
                video_seed = video_seed.reshape(channels, num_frames, height, width)

                # Prepare one-hot encoded chord and melody seeds as input
                onehot_chords_seed = torch.nn.functional.one_hot(torch.tensor(chords_seed, dtype=torch.int64), num_classes=len(self._chords_mappings))
                onehot_chords_context_seed = torch.nn.functional.one_hot(torch.tensor(chords_context_seed, dtype=torch.int64), num_classes=len(self._chords_context_mappings))
                onehot_melody_seed = torch.nn.functional.one_hot(torch.tensor(melody_seed, dtype=torch.int64), num_classes=len(self._melody_mappings))
                
                # Transformer expects input in the shape [seq_len, batch_size, feature_dim], so we permute and add batch dimension
                chords_output = self.chords_generation_model(
                    onehot_chords_seed.unsqueeze(0).float(),  # [batch_size, seq_len, feature_dim]
                    video_seed.unsqueeze(0).float()  # [batch_size, channels, num_frames, height, width] for video
                )

                melody_output = self.melody_generation_model(
                    onehot_melody_seed.unsqueeze(0).float(),  # [batch_size, seq_len, feature_dim]
                    onehot_chords_context_seed.unsqueeze(0).float(),  # [batch_size, seq_len, feature_dim] for chord context
                    video_seed.unsqueeze(0).float()  # [batch_size, channels, num_frames, height, width] for video
                )
                
                # Apply temperature sampling for both chords and melody
                chords_probabilities = chords_output.squeeze(0)[-1]  # Get probabilities of last step
                melody_probabilities = melody_output.squeeze(0)[-1]  # Get probabilities of last step
                
                # Sample from the distributions
                chords_output_int = self._sample_with_temperature(chords_probabilities, temperature)
                melody_output_int = self._sample_with_temperature(melody_probabilities, temperature)

                # Decode the predicted chords and melody to their symbols and append
                chords_output_symbol = self.chords_mappings_inv[chords_output_int]
                melody_output_symbol = self.melody_mappings_inv[melody_output_int]
                
                if chords_output_symbol == "_" or chords_output_symbol == "r":
                    chords_context_seed.append(chords_context_seed[-1])
                    chords_context.append(chords_context[-1])
                else:
                    chords_context.append(chords_output_symbol)
                    chords_context_seed.append(self._chords_context_mappings[chords_output_symbol])
                    
                chords_seed.append(chords_output_int)
                melody_seed.append(melody_output_int)
                
                chords.append(chords_output_symbol)
                melody.append(melody_output_symbol)
                
            return chords, melody

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
        
    def _save_melody(self, melody, step_duration, file_name):
        stream = music21.stream.Stream()
        melody = [x for x in ' '.join(melody).lstrip('_ ').split()]

        current_symbol = melody[0]
        current_symbol_step_counter = 1
        
        melody_length = len(melody[1:])
        
        for i, symbol in enumerate(melody[1:]):     
            if symbol == "_" and i != melody_length - 1:
                current_symbol_step_counter += 1
            else:
                quarter_length_duration = step_duration * current_symbol_step_counter
                
                if current_symbol == "r":
                    event = music21.note.Rest(quarterLength=quarter_length_duration)
                else:
                    event = music21.note.Note(int(current_symbol), quarterLength=quarter_length_duration)
                    
                stream.append(event)
                
                current_symbol = symbol
                current_symbol_step_counter = 1
        
        stream.write('midi', file_name)
         
    def save_to_file(self, chords, melody):
        self._save_chord_progression(chords, self.step_duration, file_name=f'generated/{self.save_file_name}_chords.mid')
        self._save_melody(melody, self.step_duration, file_name=f'generated/{self.save_file_name}_melody.mid')
