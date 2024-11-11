class ModelSettings:
    def __init__(self, melody_mappings_size, chords_mappings_size, chords_context_mappings_size): 
        self.classes_size = {
            "melody": melody_mappings_size,
            "chords": chords_mappings_size,
            "chords_context": chords_context_mappings_size
        }
    
    def get_model_settings(self):
        MODEL_SETTINGS = {
            "melody": {
                "melody_input_size": self.classes_size["melody"],
                "chords_context_input_size": self.classes_size["chords_context"],
                "hidden_size": 1024,
                "output_size": self.classes_size["melody"],
                "cnn_feature_size": 128,
                "chords_feature_size": 32,
                "num_layers": 3
            },
            "chords": {
                "input_size": self.classes_size["chords"],
                "hidden_size": 1024,
                "output_size": self.classes_size["chords"],
                "cnn_feature_size": 128,
                "num_layers": 3
            },
            "LR": 0.00001,
            "num_epochs": 15
        }
        
        return MODEL_SETTINGS

