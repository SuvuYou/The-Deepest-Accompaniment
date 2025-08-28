import os
import music21
import numpy as np

class SongsEncoder:
    def __init__(self, acceptable_durations, processed_dataset_path, mapped_midi_data_folder_path, sorted_chords_list):
        self.load_path = processed_dataset_path
        self.save_path = mapped_midi_data_folder_path
        self.acceptable_durations = acceptable_durations
        self.sorted_chords_list = sorted_chords_list
        self.black_keys_pitches = [1, 3, 6, 8, 10]

    def load_encoded_txt_songs(self, type):
        """
        type: "melody" | "chords"
        Returns:
            pitch_songs: list of strings (each pitch sequence space-separated)
            duration_songs: list of strings (each duration sequence space-separated)
            song_lengths: list of integers (lengths of pitch token sequences)
        """
        pitch_songs = []
        duration_songs = []

        for path, _, files in os.walk(self.save_path):    
            pitch_file = f"{type}-pitch.txt"
            duration_file = f"{type}-duration.txt"

            if pitch_file in files and duration_file in files:
                with open(os.path.join(path, pitch_file), "r") as f:
                    pitch = f.read()

                with open(os.path.join(path, duration_file), "r") as f:
                    duration = f.read()

                pitch_songs.append(pitch)
                duration_songs.append(duration)

        song_lengths = [len(song.split()) for song in pitch_songs]

        return pitch_songs, duration_songs, song_lengths
  
    @staticmethod
    def load_single_encoded_txt_song(load_path):
        melody_pitch_tokens, melody_duration_tokens = SongsEncoder._load_single_encoded_txt_song(load_path, type='melody')
        chords_pitch_tokens, chords_duration_tokens = SongsEncoder._load_single_encoded_txt_song(load_path, type='chords')

        return melody_pitch_tokens, melody_duration_tokens, chords_pitch_tokens, chords_duration_tokens
    
    @staticmethod            
    def _load_single_encoded_txt_song(load_path, type):
        """
        type: "melody" | "chords"
        """
                        
        with open(f"{os.path.join(load_path, type)}-pitch.txt", "r") as f:
            pitch_tokens = f.read()
        
        with open(f"{os.path.join(load_path, type)}-duration.txt", "r") as f:
            duration_tokens = f.read()

        return pitch_tokens, duration_tokens
    
    def encode_songs_and_save_to_files(self, type):
        """
        type: "melody" | "chords"
        """
        songs = self._load_songs_from_midi_files(type)
        
        for song, paths in songs:
            _, _, save_path = paths
            
            print(paths)
            
            if type == 'chords':
                encoded_pitch_tokens, encoded_duration_tokens = self._encode_chords_to_txt(song)
                self._save_to_file(data=encoded_pitch_tokens, folder_path=save_path, save_path=f"{save_path}/{type}-pitch.txt")
                self._save_to_file(data=encoded_duration_tokens, folder_path=save_path, save_path=f"{save_path}/{type}-duration.txt")
            else:
                encoded_pitch_tokens, encoded_duration_tokens = self._encode_melody_to_txt(song)  
                self._save_to_file(data=encoded_pitch_tokens, folder_path=save_path, save_path=f"{save_path}/{type}-pitch.txt")    
                self._save_to_file(data=encoded_duration_tokens, folder_path=save_path, save_path=f"{save_path}/{type}-duration.txt") 

    def _load_songs_from_midi_files(self, type):
        """
        type: "melody" | "chords"
        """
        songs = []
        paths = []

        for path, _, files in os.walk(self.load_path):
            if len(files) > 0:
                song_load_path = path
                song_transposed_path = path
                song_save_path = path.replace(self.load_path, self.save_path)
                
                for file in files:
                    if file[-3:] == "mid":  
                        if ((type == "melody" and "midi-melody" in file) or 
                            (type == "chords" and "midi-chords" in file)):
                            print(song_load_path, file)
                            song = music21.converter.parse(os.path.join(path, file))
                            songs.append(song)
                            paths.append((song_load_path, song_transposed_path, song_save_path))

        return zip(songs, paths)
    
    def _encode_melody_to_txt(self, song):
        pitch_tokens = []
        duration_tokens = []
        
        for event in song.flatten().notesAndRests:
            event_duration = self._closest_acceptable_durations(event.duration.quarterLength)
            
            if isinstance(event, music21.note.Note):
                symbol = self._select_pitch(event.pitch.midi)
            elif isinstance(event, music21.chord.Chord):
                symbol = self._select_pitch(event.pitches[0].midi)
            elif isinstance(event, music21.note.Rest):
                symbol = "r"
                
                if event.duration.quarterLength < self.acceptable_durations[0]:
                    continue
            else:
                continue
                        
            pitch_tokens.append(symbol)
            duration_tokens.append(event_duration)
        
        return " ".join(map(str, pitch_tokens)), " ".join(map(str, duration_tokens))

    def _encode_chords_to_txt(self, song):
        pitch_tokens = []
        duration_tokens = []

        for event in song.flatten().notesAndRests:
            event_duration = self._closest_acceptable_durations(event.duration.quarterLength)
                    
            if isinstance(event, music21.chord.Chord):
                chord = "-".join(sorted(event.pitchNames, key=self._custom_sort_key))
                symbol = f"({chord})"

            elif isinstance(event, music21.note.Rest):
                symbol = "r"
                
                if event.duration.quarterLength < self.acceptable_durations[0]:
                    continue
            else:
                continue
            
            pitch_tokens.append(symbol)
            duration_tokens.append(event_duration)

        return " ".join(map(str, pitch_tokens)), " ".join(map(str, duration_tokens))
    
    def _save_to_file(self, data, folder_path, save_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        with open(save_path, "w") as fp:
            fp.write(data)   
            
    def _custom_sort_key(self, chord):
        return self.sorted_chords_list.index(chord)

    def _closest_acceptable_durations(self, duration):
        lst = np.asarray(self.acceptable_durations)
        idx = (np.abs(lst - duration)).argmin()
        
        return lst[idx]

    def _select_pitch(self, pitch):
        remainder = pitch % 12
        
        if remainder in self.black_keys_pitches:
            remainder = remainder - 1
        
        return remainder + 60
               