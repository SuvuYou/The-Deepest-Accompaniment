import torch 
from Models.Trainer import Trainer
from Models.ModelSettings import ModelSettings
from Models.ModelModules import ChordGeneratorTransformer, MelodyGeneratorTransformer
from Processing.MidiDatasetLoader import MidiDatasetLoader
from Processing.const import CONSTANTS

print("DEVICE -", CONSTANTS.DEVICE)

modelSettings = ModelSettings()
MODEL_SETTINGS = modelSettings.get_model_settings()

chords_model = ChordGeneratorTransformer(MODEL_SETTINGS["chords_transformer"]).to(CONSTANTS.DEVICE)
melody_model = MelodyGeneratorTransformer(MODEL_SETTINGS["melody_transformer"]).to(CONSTANTS.DEVICE)

trainer = Trainer(chords_model, melody_model, MODEL_SETTINGS, starting_weights_idx = -1)

# melody_updates = {
#     '_': 200,
#     'r': 120,
# }

# chords_updates = {
#     '_': 550,
#     'r': 120,
# }

# trainer.update_melody_class_weights(melody_updates)
# trainer.update_class_weights(chords_updates)

chords_dataset = MidiDatasetLoader(load_data_type="chords")
melody_dataset = MidiDatasetLoader(load_data_type="melody")

chords_data_loader = torch.utils.data.DataLoader(chords_dataset, batch_size=6, shuffle=True) 
melody_data_loader = torch.utils.data.DataLoader(melody_dataset, batch_size=6, shuffle=True)   
        
trainer.train_chords(chords_data_loader)
trainer.train_melody(melody_data_loader)

# trainer._plot_metrics_from_file()