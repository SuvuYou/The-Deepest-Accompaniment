class ModelSettings:
    def __init__(self, melody_mappings_size, chords_mappings_size, chords_context_mappings_size): 
        self.classes_size = {
            "melody": melody_mappings_size,
            "chords": chords_mappings_size,
            "chords_context": chords_context_mappings_size
        }
    
    def get_model_settings(self):
        MODEL_SETTINGS = {
            "chords_transformer":
            {
                "chord_dim": self.classes_size["chords"], 
                "video_out_dim": 256, 
                "num_layers": 16, 
                "num_decoder_layers": 32, 
                "nhead": 16, 
                "d_model": 256, 
                "dim_feedforward": 2048,
            },
            "LR": 0.0001,
            "num_epochs": 15
        }
        
        return MODEL_SETTINGS

