import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_DATASET_PATH = "data"

PROCESSED_DATASET_PATH = f"{RAW_DATASET_PATH}/3_processed_synced_midi"
MAPPED_MIDI_DATA_FOLDER_PATH = f"{RAW_DATASET_PATH}/4_mapped_synced_midi"
VIDEO_DATA_FOLDER_PATH = f"{RAW_DATASET_PATH}/2_synced_midi_and_video"

MAPPINGS_PATH = f"{RAW_DATASET_PATH}/5_mappings"
MELODY_MAPPINGS_PATH = f"{MAPPINGS_PATH}/melody_mappings.json"
CHORDS_MAPPINGS_PATH = f"{MAPPINGS_PATH}/chords_mappings.json"
CHORDS_CONTEXT_MAPPINGS_PATH = f"{MAPPINGS_PATH}/chords_context_mappings.json"

MUSIC_DATA_CHUNKS_SAVE_PATH = f"{RAW_DATASET_PATH}/6_stored_chunked_data/tensors"
VIDEO_CHUNKS_SAVE_PATH = f"{RAW_DATASET_PATH}/6_stored_chunked_data/tensors_video"

DEFAULT_SEQUENCE_LENGTH = 24
DEFAULT_MODEL_WEIGHTS_FOLDER_NAME = lambda idx: f"weights/{idx}"
DEFAULT_MELODY_MODEL_WEIGHTS_FILE_NAME = lambda idx: f"weights/{idx}/melody_model_weights.pth"
DEFAULT_CHORDS_MODEL_WEIGHTS_FILE_NAME = lambda idx: f"weights/{idx}/chords_model_weights.pth"

ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4]

SORTED_CHORDS_LIST = ["C", "D", "E", "F", "G", "A", "B"]

VIDEO_CHUNK_SIZE = 50

class DotDict(dict):
    """A dictionary that supports dot notation."""
    def __getattr__(self, name):
        return self[name]
    
    def __setattr__(self, name, value):
        self[name] = value
    
    def __delattr__(self, name):
        del self[name]

CONSTANTS = DotDict({
    "DEVICE": DEVICE,
    "RAW_DATASET_PATH": RAW_DATASET_PATH,
    "PROCESSED_DATASET_PATH": PROCESSED_DATASET_PATH,
    "MAPPED_MIDI_DATA_FOLDER_PATH": MAPPED_MIDI_DATA_FOLDER_PATH,
    "VIDEO_DATA_FOLDER_PATH": VIDEO_DATA_FOLDER_PATH,
    "MAPPINGS_PATH": MAPPINGS_PATH,
    "MELODY_MAPPINGS_PATH": MELODY_MAPPINGS_PATH,
    "CHORDS_MAPPINGS_PATH": CHORDS_MAPPINGS_PATH,
    "CHORDS_CONTEXT_MAPPINGS_PATH": CHORDS_CONTEXT_MAPPINGS_PATH,
    "MUSIC_DATA_CHUNKS_SAVE_PATH": MUSIC_DATA_CHUNKS_SAVE_PATH,
    "VIDEO_CHUNKS_SAVE_PATH": VIDEO_CHUNKS_SAVE_PATH,
    "DEFAULT_SEQUENCE_LENGTH": DEFAULT_SEQUENCE_LENGTH,
    "DEFAULT_MODEL_WEIGHTS_FOLDER_NAME": DEFAULT_MODEL_WEIGHTS_FOLDER_NAME,
    "DEFAULT_MELODY_MODEL_WEIGHTS_FILE_NAME": DEFAULT_MELODY_MODEL_WEIGHTS_FILE_NAME,
    "DEFAULT_CHORDS_MODEL_WEIGHTS_FILE_NAME": DEFAULT_CHORDS_MODEL_WEIGHTS_FILE_NAME,
    "ACCEPTABLE_DURATIONS": ACCEPTABLE_DURATIONS,
    "SORTED_CHORDS_LIST": SORTED_CHORDS_LIST,
    "VIDEO_CHUNK_SIZE": VIDEO_CHUNK_SIZE
})