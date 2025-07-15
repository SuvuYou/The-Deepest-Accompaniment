from Processing.const import CONSTANTS
from Processing.MidiDatasetSaver import MidiDatasetSaver
from Processing.SongsEncoder import SongsEncoder
from Processing.SongsMapper import SongsMapper
from Processing.VideoProcessor import VideoProcessor
    
if __name__ == "__main__":
    songsEncoder = SongsEncoder(CONSTANTS.ACCEPTABLE_DURATIONS, CONSTANTS.PROCESSED_DATASET_PATH, CONSTANTS.MAPPED_MIDI_DATA_FOLDER_PATH, CONSTANTS.SORTED_CHORDS_LIST)
    videoProcessor = VideoProcessor(CONSTANTS.DEVICE, CONSTANTS.VIDEO_CHUNK_SIZE)

    songsEncoder.encode_songs_and_save_to_files(type='melody')
    songsEncoder.encode_songs_and_save_to_files(type='chords')

    melody_pitch_songs, melody_duration_songs, melody_lengths = songsEncoder.load_encoded_txt_songs(type='melody')
    chords_pitch_songs, chords_duration_songs, chords_lengths = songsEncoder.load_encoded_txt_songs(type='chords')
    
    SongsMapper.create_int_mapping(melody_pitch_songs, CONSTANTS.MELODY_PITCH_MAPPINGS_PATH)
    SongsMapper.create_int_mapping(melody_duration_songs, CONSTANTS.MELODY_DURATION_MAPPINGS_PATH)
    
    SongsMapper.create_int_mapping(chords_pitch_songs, CONSTANTS.CHORDS_PITCH_MAPPINGS_PATH)
    SongsMapper.create_int_mapping(chords_duration_songs, CONSTANTS.CHORDS_DURATION_MAPPINGS_PATH)
    
    melody_pitch_mapping = SongsMapper.load_mappings(CONSTANTS.MELODY_PITCH_MAPPINGS_PATH)
    melody_duration_mapping = SongsMapper.load_mappings(CONSTANTS.MELODY_DURATION_MAPPINGS_PATH)
    
    chord_pitch_mapping = SongsMapper.load_mappings(CONSTANTS.CHORDS_PITCH_MAPPINGS_PATH)
    chord_duration_mapping = SongsMapper.load_mappings(CONSTANTS.CHORDS_DURATION_MAPPINGS_PATH)
    
    SongsMapper.plot_mappings_data(melody_pitch_mapping, title="Melody Pitch Distribution")
    SongsMapper.plot_mappings_data(melody_duration_mapping, title="Melody Duration Distribution")
    SongsMapper.plot_mappings_data(chord_pitch_mapping, title="Chord Pitch Distribution")
    SongsMapper.plot_mappings_data(chord_duration_mapping, title="Chord Duration Distribution")

    dataSaver = MidiDatasetSaver(videoProcessor, song_lengths = melody_lengths + chords_lengths, CONSTANTS = CONSTANTS)

    dataSaver.save_training_data()
    