from Processing.const import CONSTANTS
from Processing.SongsMapper import SongsMapper

class ModelSettings:
    def __init__(self): 
        self.classes_size = {
            "chords_pitch": SongsMapper.get_mappings_size(CONSTANTS.CHORDS_PITCH_MAPPINGS_PATH),
            "chords_duration": SongsMapper.get_mappings_size(CONSTANTS.CHORDS_DURATION_MAPPINGS_PATH),
            "melody_pitch": SongsMapper.get_mappings_size(CONSTANTS.MELODY_PITCH_MAPPINGS_PATH),
            "melody_duration": SongsMapper.get_mappings_size(CONSTANTS.MELODY_DURATION_MAPPINGS_PATH),
        }
    
    def get_model_settings(self):
        MODEL_SETTINGS = {
            "chords_transformer":
            {
                "chord_pitch_dim": self.classes_size["chords_pitch"], 
                "chord_duration_dim": self.classes_size["chords_duration"], 
                
                "d_model": 1024, 
                "dim_feedforward": 2048,
                "nhead": 16, 
                # "num_encoder_layers": 8, 
                "num_decoder_layers": 16, 
                
                "video_out_dim": 512, 
                "seq_len": CONSTANTS.DEFAULT_SEQUENCE_LENGTH - 1
            },
            "melody_transformer":
            {
                "melody_pitch_dim": self.classes_size["melody_pitch"], 
                "melody_duration_dim": self.classes_size["melody_duration"], 
                
                "d_model": 512, 
                "dim_feedforward": 2048,
                "nhead": 16, 
                "num_encoder_layers": 8, 
                "num_decoder_layers": 8, 
                
                "video_out_dim": 512, 
                "seq_len": CONSTANTS.DEFAULT_SEQUENCE_LENGTH
            },
            "LR": 1e-5,
            "num_epochs": 10
        }
        
        return MODEL_SETTINGS

