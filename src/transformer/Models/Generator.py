import json
import numpy as np
import torch
import music21

class Generator:
    def __init__(self, melody_generation_model, chords_generation_model, save_file_name, CONSTANTS):
        self.save_file_name = save_file_name
        self.CONSTANTS = CONSTANTS
        self.step_duration = CONSTANTS.ACCEPTABLE_DURATIONS[0]
            
        self.melody_generation_model = melody_generation_model
        self.chords_generation_model = chords_generation_model
        
        with open(self.CONSTANTS.MELODY_MAPPINGS_PATH, "r") as fp:
            self._melody_mappings = json.load(fp)['mappings']
            
        with open(self.CONSTANTS.CHORDS_MAPPINGS_PATH, "r") as fp:
            self._chords_mappings = json.load(fp)['mappings']    
            
        with open(self.CONSTANTS.CHORDS_CONTEXT_MAPPINGS_PATH, "r") as fp:
            self._chords_context_mappings = json.load(fp)['mappings'] 

    def generate(self, melody_seed, chords_seed, chords_context_seed, video, num_steps, max_sequence_length, temperature):
        with torch.no_grad():
            melody_seed = melody_seed.split()
            chords_seed = chords_seed.split()
            chords_context_seed = chords_context_seed.split()
            
            melody = melody_seed
            chords = chords_seed
        
            melody_seed = [self._melody_mappings[symbol] for symbol in melody_seed]
            chords_seed = [self._chords_mappings[symbol] for symbol in chords_seed]
            chords_context_seed = [self._chords_context_mappings[symbol] for symbol in chords_context_seed]

            for idx in range(num_steps):
                # Slice seeds  
                melody_seed = melody_seed[-max_sequence_length:]
                seed_length = len(melody_seed)
                chords_seed = chords_seed[-max_sequence_length:]
                chords_context_seed = chords_context_seed[-max_sequence_length:]
                video_seed = video[idx : idx + seed_length]
                video_seed = video_seed[[0, len(video_seed) / 2]]
                
                # Generate next chord symbol
                onehot_chords_seed = torch.nn.functional.one_hot(torch.tensor(chords_seed, dtype=torch.int64), num_classes=len(self._chords_mappings))
                chords_output = self.chords_generation_model(onehot_chords_seed.unsqueeze(0).float(), video_seed.unsqueeze(0).float())
                chords_probabilities = chords_output[0]
                chords_output_int = self._sample_with_temperature(chords_probabilities, temperature)      
                chords_output_symbol = [k for k, v in self._chords_mappings.items() if v == chords_output_int][0]
                chords_seed.append(chords_output_int)
                chords.append(chords_output_symbol)
                
                # Get chord context from generated chord for melody model
                if chords_output_symbol == '_':
                    chords_context_output_int = chords_context_seed[seed_length - 1]
                else:
                    chords_context_output_int = self._chords_context_mappings[chords_output_symbol]
                
                selected_chord_context = torch.nn.functional.one_hot(torch.tensor([chords_context_output_int], dtype=torch.int64), num_classes=len(self._chords_context_mappings))
                chords_context_seed.append(chords_context_output_int)
                
                # Generate next melody symbol
                onehot_melody_seed = torch.nn.functional.one_hot(torch.tensor(melody_seed, dtype=torch.int64), num_classes=len(self._melody_mappings))
                melody_output = self.melody_generation_model(onehot_melody_seed.unsqueeze(0).float(), selected_chord_context.float(), video_seed.unsqueeze(0).float())
                melody_probabilities = melody_output[0]
                melody_output_int = self._sample_with_temperature(melody_probabilities, temperature)
                melody_output_symbol = [k for k, v in self._melody_mappings.items() if v == melody_output_int][0]
                melody_seed.append(melody_output_int)
                melody.append(melody_output_symbol)
                
            return chords, melody

    def _sample_with_temperature(self, probabilites, temperature):
        predictions = np.log(probabilites.numpy()) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites))
        index = np.random.choice(choices, p=probabilites)

        return index
    
    
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
    
    def _save_chord_progression(self, chords, step_duration, file_name):
        stream = music21.stream.Stream()    
        chords = [x for x in ' '.join(chords).lstrip('_ ').split()]

        current_symbol = chords[0]
        current_symbol_step_counter = 1
        
        melody_length = len(chords[1:])

        for i, symbol in enumerate(chords[1:]):     
            if symbol == "_" and i != melody_length - 1:
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
        
    def save_to_file(self, chords, melody):
        self._save_chord_progression(chords, self.step_duration, file_name=f'generated/{self.save_file_name}_chords.mid')
        self._save_melody(melody, self.step_duration, file_name=f'generated/{self.save_file_name}_melody.mid')        