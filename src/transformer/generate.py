import torch
from Processing.VideoProcessor import VideoProcessor
from Processing.SongsMapper import SongsMapper
from Models.Generator import Generator
from Models.ModelModules import ChordGeneratorTransformer
from Models.ModelSettings import ModelSettings
from Processing.const import CONSTANTS

def init_seed(type):
    videoProcessor = VideoProcessor(CONSTANTS.DEVICE)
    
    notes_to_generate_count = 200
    seeds = {
        "fast":{
            "seed_melody": "64 71 69 64 64 71 69 64 64 71 69 64",
            "seed_chords": "(C-E-A) _ _ _ _ _ _ _ _ _ _ _",
            "seed_chords_context": "(C-E-A) _ _ _ _ _ _ _ _ _ _ _".replace("_", "(C-E-A)")
        },
        "fast1":{
            "seed_melody": "60 64 69 60 60 64 69 60 _ _ _ 67 71 64 67 71 71 64 67 71 71 64 67 71",
            "seed_chords": "(C-E-G) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (C-E-G) _ _ _ _ _ _ _",
            "seed_chords_context": "(C-E-G) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (C-E-G) _ _ _ _ _ _ _".replace("_", "(C-E-G)")
        },
        "slow1":{
            "seed_melody": "69 _ _ _ _ _ _ _ 69 _ 64 _ 71 _ 60 _ 60 _ _ _ _ _",
            "seed_chords": "(D-F-A) _ _ _ _ _ _ _ (C-E-A) _ _ _ _ _ _ _ (C-E-A) _ _ _ _ _ _",
            "seed_chords_context": "(D-F-A) _ _ _ _ _ _ _ (C-E-A) _ _ _ _ _ _ _ (C-E-A) _ _ _ _ _ _".replace("_", "(C-E-A)")
        },
        "slow":{
            "seed_melody": "r _ _ _ _ _ _ _ _ _ _ _",
            "seed_chords": "(D-F-B) _ _ _ _ _ _ _ _ _ _ _",
            "seed_chords_context": "(D-F-B) _ _ _ _ _ _ _ _ _ _ _".replace("_", "(D-F-B)")
        }
    }
    
    seed_melody = seeds[type]["seed_melody"]
    seed_chords = seeds[type]["seed_chords"]
    seed_chords_context = seeds[type]["seed_chords_context"]
    
    seed_video, fps = videoProcessor.load_video_frames(folder_path=f"generated/original-180-{type}.mp4")

    video_frames = videoProcessor.process_video(video=seed_video, target_video_length_in_frames=notes_to_generate_count + len(seed_melody.split()))
    
    return seed_melody, seed_chords, seed_chords_context, video_frames, notes_to_generate_count

def init_generator(weights_path, save_file_name, MODEL_SETTINGS, CONSTANTS):
    chords_model_weights_path = CONSTANTS.DEFAULT_CHORDS_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(weights_path)
    
    # Initialize Transformer model here for chords generation
    chords_generation_model = ChordGeneratorTransformer(**MODEL_SETTINGS['chords_transformer'])
    chords_generation_model.load_state_dict(torch.load(chords_model_weights_path))

    return Generator(chords_generation_model, save_file_name, CONSTANTS)

def generate_music(generator, seed_chords, video_frames, notes_to_generate_count):
    chords = generator.generate(
        chords_seed=seed_chords,
        video=video_frames,
        num_steps=notes_to_generate_count, 
        max_sequence_length=CONSTANTS.DEFAULT_SEQUENCE_LENGTH, 
        temperature=0.9
    )
    
    print(chords)
    generator.save_to_file(chords)
    
# Main function to initialize and start generation
if __name__ == "__main__":
    melody_mappings_size = SongsMapper.get_mappings_size(CONSTANTS.MELODY_MAPPINGS_PATH)
    chords_mappings_size = SongsMapper.get_mappings_size(CONSTANTS.CHORDS_MAPPINGS_PATH)
    chords_context_mappings_size = SongsMapper.get_mappings_size(CONSTANTS.CHORDS_CONTEXT_MAPPINGS_PATH)

    modelSettings = ModelSettings(melody_mappings_size, chords_mappings_size, chords_context_mappings_size)
    MODEL_SETTINGS = modelSettings.get_model_settings()
    generator = init_generator(weights_path="0", save_file_name="generated", MODEL_SETTINGS=MODEL_SETTINGS, CONSTANTS=CONSTANTS)
    
    seed_melody, seed_chords, seed_chords_context, video_frames, notes_to_generate_count = init_seed(type="fast1")
    
    generate_music(generator, seed_chords, video_frames, notes_to_generate_count)