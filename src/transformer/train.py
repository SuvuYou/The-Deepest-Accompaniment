import torch 
from Models.Trainer import Trainer
from Models.ModelSettings import ModelSettings
from Models.ModelModules import ChordGeneratorTransformer, MelodyGeneratorTransformer
from Processing.SongsMapper import SongsMapper
from Processing.MidiDatasetLoader import MidiDatasetLoader
from Processing.const import CONSTANTS

print("DEVICE -", CONSTANTS.DEVICE)
        
melody_mappings_size = SongsMapper.get_mappings_size(CONSTANTS.MELODY_MAPPINGS_PATH)
chords_mappings_size = SongsMapper.get_mappings_size(CONSTANTS.CHORDS_MAPPINGS_PATH)

modelSettings = ModelSettings(melody_mappings_size, chords_mappings_size)
MODEL_SETTINGS = modelSettings.get_model_settings()

chords_model = ChordGeneratorTransformer(**MODEL_SETTINGS["chords_transformer"]).to(CONSTANTS.DEVICE)
# melody_model = MelodyGeneratorTransformer(**MODEL_SETTINGS["melody_transformer"]).to(CONSTANTS.DEVICE)

trainer = Trainer(chords_model, None, MODEL_SETTINGS, CONSTANTS, starting_weights_idx = -1)

melody_updates = {
    '_': 200,
    'r': 120,
}

chords_updates = {
    '_': 550,
    'r': 120,
}

trainer.update_melody_class_weights(melody_updates)
trainer.update_class_weights(chords_updates)

dataset = MidiDatasetLoader(video_data_chunks_save_path = CONSTANTS.VIDEO_CHUNKS_SAVE_PATH, music_data_chunks_save_path = CONSTANTS.MUSIC_DATA_CHUNKS_SAVE_PATH)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)  
        
trainer.train_chords(data_loader)
trainer.train_melody(data_loader)

trainer._plot_metrics_from_file()