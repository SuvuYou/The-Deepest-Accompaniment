from Processing.const import CONSTANTS
from Processing.MidiDatasetSaver import MidiDatasetSaver
from Processing.SongsEncoder import SongsEncoder
from Processing.SongsMapper import SongsMapper
from Processing.VideoProcessor import VideoProcessor
    
if __name__ == "__main__":
    songsEncoder = SongsEncoder(CONSTANTS.ACCEPTABLE_DURATIONS, CONSTANTS.PROCESSED_DATASET_PATH, CONSTANTS.MAPPED_MIDI_DATA_FOLDER_PATH, CONSTANTS.SORTED_CHORDS_LIST)
    songsMapper = SongsMapper(CONSTANTS.MAPPINGS_PATH)
    videoProcessor = VideoProcessor(CONSTANTS.DEVICE, CONSTANTS.VIDEO_CHUNK_SIZE)

    songsEncoder.encode_songs_and_save_to_files(type='melody')
    songsEncoder.encode_songs_and_save_to_files(type='chords')

    melody_songs, melody_lengths = songsEncoder.load_encoded_txt_songs(type='melody')
    chords_songs, chords_lengths = songsEncoder.load_encoded_txt_songs(type='chords')
    chords_context_songs, chords_context_lengths = songsEncoder.load_encoded_txt_songs(type='chords-context')

    songsMapper.create_int_mapping(melody_songs, CONSTANTS.MELODY_MAPPINGS_PATH)
    songsMapper.create_int_mapping(chords_songs, CONSTANTS.CHORDS_MAPPINGS_PATH)
    songsMapper.create_int_mapping(chords_context_songs, CONSTANTS.CHORDS_CONTEXT_MAPPINGS_PATH)

    dataSaver = MidiDatasetSaver(songsEncoder, songsMapper, videoProcessor, song_lengths = melody_lengths + chords_lengths, CONSTANTS = CONSTANTS)

    dataSaver.save_training_data()