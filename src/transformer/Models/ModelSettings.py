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
                "video_out_dim": 512, 
                "num_encoder_layers": 8, 
                "num_decoder_layers": 8, 
                "nhead": 16, 
                "d_model": 256, 
                "dim_feedforward": 2048,
            },
            "melody_transformer":
            {
                "melody_dim": self.classes_size["melody"], 
                "chord_context_dim": self.classes_size["chords_context"], 
                "video_out_dim": 512, 
                "num_encoder_layers": 8, 
                "num_decoder_layers": 8, 
                "nhead": 16, 
                "d_model": 256, 
                "dim_feedforward": 2048,
            },
            "LR": 0.0001,
            "num_epochs": 15
        }
        
        return MODEL_SETTINGS

