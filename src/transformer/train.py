import torch 
from Models.Trainer import Trainer
from Models.ModelSettings import ModelSettings
from Models.ModelModules import MelodyLSTM, ChordsLSTM
from Processing.SongsMapper import SongsMapper
from Processing.MidiDatasetLoader import MidiDatasetLoader
from Processing.const import CONSTANTS

print("DEVICE -", CONSTANTS.DEVICE)
        
melody_mappings_size = SongsMapper.get_mappings_size(CONSTANTS.MELODY_MAPPINGS_PATH)
chords_mappings_size = SongsMapper.get_mappings_size(CONSTANTS.CHORDS_MAPPINGS_PATH)
chords_context_mappings_size = SongsMapper.get_mappings_size(CONSTANTS.CHORDS_CONTEXT_MAPPINGS_PATH)

modelSettings = ModelSettings(melody_mappings_size, chords_mappings_size, chords_context_mappings_size)
MODEL_SETTINGS = modelSettings.get_model_settings()

melody_generation_model = MelodyLSTM(**MODEL_SETTINGS['melody']).to(CONSTANTS.DEVICE)
chords_generation_model = ChordsLSTM(**MODEL_SETTINGS['chords']).to(CONSTANTS.DEVICE)

trainer = Trainer(melody_generation_model, chords_generation_model, MODEL_SETTINGS, CONSTANTS, starting_weights_idx = -1)

melody_updates = {
    '_': 185,
    'r': 120,
    '60': 102.5,
    '62': 105,
    '64': 37,
    '69': 66.7,
    '65': 64,
    '67': 84.5,
    '71': 48.35
}

chords_updates = {
    '_': 200,
    'r': 140,
    '(C-E-A)': 80
}

trainer.update_melody_class_weights(melody_updates)
trainer.update_chords_class_weights(chords_updates)

dataset = MidiDatasetLoader(video_data_chunks_save_path = CONSTANTS.VIDEO_CHUNKS_SAVE_PATH, music_data_chunks_save_path = CONSTANTS.MUSIC_DATA_CHUNKS_SAVE_PATH)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)  
        
trainer.train(data_loader)