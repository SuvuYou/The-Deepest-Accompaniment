import numpy as np
import torch
from Processing.SongsMapper import SongsMapper

class ClassWeightManager:
    def __init__(self):
        self.mappings = {
            "melody_pitch_mapping": SongsMapper.load_mappings(self.CONSTANTS.MELODY_PITCH_MAPPINGS_PATH),
            "melody_duration_mapping": SongsMapper.load_mappings(self.CONSTANTS.MELODY_DURATION_MAPPINGS_PATH),
            "chord_pitch_mapping": SongsMapper.load_mappings(self.CONSTANTS.CHORDS_PITCH_MAPPINGS_PATH),
            "chord_duration_mapping": SongsMapper.load_mappings(self.CONSTANTS.CHORDS_DURATION_MAPPINGS_PATH),
        }
        
        self._init_class_weights()

    def _init_class_weights(self):
        self.chords_class_weights = self._compute_class_weights(self.chords_mappings)
        self.melody_class_weights = self._compute_class_weights(self.melody_mappings)
        
        self._print_class_weights(self.chords_class_weights, self.chords_mappings["mappings"], "Initial chords_class_weights weights")
        self._print_class_weights(self.melody_class_weights, self.melody_mappings["mappings"], "Initial melody_class_weights weights")
    
    def _compute_class_weights(self, mappings):
        symbol_counts = np.array(list(mappings['counter']['mapped_symbols'].values()))
        class_weights = torch.tensor(sum(symbol_counts) / (len(symbol_counts) * symbol_counts), dtype=torch.float32)
        return class_weights

    def _print_class_weights(self, weights, mappings, message):
        print(message)
        for idx, weight in enumerate(weights):
            symbol = list(mappings.values())[idx]
            print(f"Symbol {symbol}: {weight}")

    def update_melody_class_weights(self, updates):
        updated_weights = self.melody_class_weights.clone()
        for symbol, percentage in updates.items():
            mapped_value = self.melody_mappings['mappings'][symbol]
            updated_weights[mapped_value] *= (0.01 * percentage)
        self._print_class_weights(updated_weights, self.melody_mappings["mappings"], "Updated melody class weights")
        self.melody_class_weights = updated_weights
        
    def update_chords_class_weights(self, updates):
        updated_weights = self.chords_class_weights.clone()
        for symbol, percentage in updates.items():
            mapped_value = self.chords_mappings['mappings'][symbol]
            updated_weights[mapped_value] *= (0.01 * percentage)
        self._print_class_weights(updated_weights, self.chords_mappings["mappings"], "Updated chords class weights")
        self.chords_class_weights = updated_weights
