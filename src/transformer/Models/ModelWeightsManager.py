import os
import torch
from Processing.const import CONSTANTS

class ModelWeightsManager:
    def __init__(self, chords_model, melody_model, starting_weights_idx=-1):
        self.save_weight_idx = starting_weights_idx
        
        self.chords_model = chords_model
        self.melody_model = melody_model
        
        self._load_model_weights_if_needed()

    def _load_model_weights_if_needed(self):
        if self.save_weight_idx != -1:
            chords_weights_path = CONSTANTS.DEFAULT_CHORDS_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(self.save_weight_idx)
            melody_weights_path = CONSTANTS.DEFAULT_MELODY_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(self.save_weight_idx)
            
            if self.chords_model:
                self.chords_model.load_state_dict(torch.load(chords_weights_path))
            
            if self.melody_model:
                self.melody_model.load_state_dict(torch.load(melody_weights_path))

    def save_chords_model_weights(self):
        self.save_weight_idx += 1
        weights_folder = CONSTANTS.DEFAULT_MODEL_WEIGHTS_FOLDER_NAME(idx=self.save_weight_idx)
        os.makedirs(weights_folder, exist_ok=True)
        
        chords_weights_path = CONSTANTS.DEFAULT_CHORDS_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(idx=self.save_weight_idx)
        torch.save(self.chords_model.state_dict(), chords_weights_path)
        
    def save_melody_model_weights(self):
        self.save_weight_idx += 1
        weights_folder = CONSTANTS.DEFAULT_MODEL_WEIGHTS_FOLDER_NAME(idx=self.save_weight_idx)
        os.makedirs(weights_folder, exist_ok=True)
         
        melody_weights_path = CONSTANTS.DEFAULT_MELODY_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(idx=self.save_weight_idx)
        torch.save(self.melody_model.state_dict(), melody_weights_path)
