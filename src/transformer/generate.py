import torch
from Processing.VideoProcessor import VideoProcessor
from Processing.SongsMapper import SongsMapper
from Models.Generator import Generator
from Models.ModelModules import ChordGeneratorTransformer, MelodyGeneratorTransformer
from Models.ModelSettings import ModelSettings
from Processing.const import CONSTANTS
from Models.ModelWeightsManager import ModelWeightsManager

def init_seed(type):
    videoProcessor = VideoProcessor(CONSTANTS.DEVICE)
    
    notes_to_generate_count = 200
    
    seeds = {
        "fast":{
            "seed_melody": "64 71 69 64 64 71 69 64 64 71 69 64 64 71 69 64 64 71 69 64 64 71 69 64",
            "seed_chords": "(C-E-A) _ _ _ _ _ _ _ _ _ _ _ (C-E-A) _ _ _ _ _ _ _ _ _ _ _",
            "seed_chords_context": "(C-E-A) _ _ _ _ _ _ _ _ _ _ _ (C-E-A) _ _ _ _ _ _ _ _ _ _ _".replace("_", "(C-E-A)")
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
        },
        "fast_new": 
        {
            "melody_pitch_seed": "64 71 69 64 64 71 69 64 64 71 69 64 64 71 69 64 64 71 69 64 64 71 69 64",
            "melody_duration_seed": "0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25",
            "chords_pitch_seed": "(C-E-A) (C-E-A)",
            "chords_duration_seed": "3.0 3.0",
        }
    }
    
    chords_pitch_seed = seeds[type]["chords_pitch_seed"]
    chords_duration_seed = seeds[type]["chords_duration_seed"]
    melody_pitch_seed = seeds[type]["melody_pitch_seed"]
    melody_duration_seed = seeds[type]["melody_duration_seed"]
    
    seed_video, fps = videoProcessor.load_video_frames(folder_path=f"generated/original-180-{type}.mp4")

    video_frames = videoProcessor.process_video(video=seed_video, target_video_length_in_frames=notes_to_generate_count + len(melody_pitch_seed.split()))
    
    return chords_pitch_seed, chords_duration_seed, melody_pitch_seed, melody_duration_seed, video_frames, notes_to_generate_count

def init_generator(weights_path, save_file_name):
    modelSettings = ModelSettings()
    MODEL_SETTINGS = modelSettings.get_model_settings()
    
    # Initialize Transformer model here for chords generation
    chords_generation_model = ChordGeneratorTransformer(MODEL_SETTINGS['chords_transformer'])
    melody_generation_model = MelodyGeneratorTransformer(MODEL_SETTINGS['melody_transformer'])
    
    ModelWeightsManager(chords_generation_model, melody_generation_model, starting_weights_idx = 5)

    return Generator(chords_generation_model, melody_generation_model, save_file_name)

def generate_music(generator, chords_pitch_seed, chords_duration_seed, melody_pitch_seed, melody_duration_seed, video_frames, MAX_GENERATION_STEPS):
    generated_chords, generated_melody = generator.generate(
        chords_pitch_seed=chords_pitch_seed, 
        chords_duration_seed=chords_duration_seed, 
        melody_pitch_seed=melody_pitch_seed, 
        melody_duration_seed=melody_duration_seed, 
        video=video_frames, 
        num_steps=MAX_GENERATION_STEPS, 
        temperature=0.3
    )
    
    print(generated_chords)
    print(generated_melody)
    
    generator.save_to_file(generated_chords, generated_melody)

if __name__ == "__main__":
    generator = init_generator(weights_path="2", save_file_name="generated")
    
    chords_pitch_seed, chords_duration_seed, melody_pitch_seed, melody_duration_seed, video_frames, notes_to_generate_count = init_seed(type="fast_new")
    
    generate_music(generator, chords_pitch_seed, chords_duration_seed, melody_pitch_seed, melody_duration_seed, video_frames, notes_to_generate_count)