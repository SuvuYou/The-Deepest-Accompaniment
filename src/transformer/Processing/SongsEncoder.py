import os
import music21
import numpy as np

class SongsEncoder:
    def __init__(self, acceptable_durations, processed_dataset_path, mapped_midi_data_folder_path, sorted_chords_list):
        self.load_path = processed_dataset_path
        self.save_path = mapped_midi_data_folder_path
        self.acceptable_durations = acceptable_durations
        self.sorted_chords_list = sorted_chords_list
        self.time_step = self.acceptable_durations[0]
        self.black_keys_pitches = [1, 3, 6, 8, 10]

    def load_encoded_txt_songs(self, type):
        """
        type: "melody" | "chords" | "chords-context"
        """
        load_path = self.save_path
        songs = []

        for path, _, files in os.walk(load_path):    
            for file in files:
                if file[-3:] == "txt":  
                    if ((type == "melody" and "melody" in file) or 
                        (type == "chords" and "chords" in file and "chords-context" not in file) or
                        (type == "chords-context" and "chords-context" in file)):
                        
                        with open(os.path.join(path, file), "r") as f:
                            song = f.read()

                        songs.append(song)

        song_lengths = [len(song.split()) for song in songs]
        
        return songs, song_lengths
  
    def load_single_encoded_txt_song(self, load_path):
        melody = self._load_single_encoded_txt_song(load_path, type='melody');
        chords = self._load_single_encoded_txt_song(load_path, type='chords');
        chords_context = self._load_single_encoded_txt_song(load_path, type='chords-context');
        
        song_length = max(len(melody.split()), len(chords.split()))

        return melody, chords, chords_context, song_length
             
    def _load_single_encoded_txt_song(self, load_path, type):
        """
        type: "melody" | "chords" | "chords-context"
        """
                        
        with open(f"{os.path.join(load_path, type)}.txt", "r") as f:
            song = f.read()

        return song
    
    def encode_songs_and_save_to_files(self, type):
        """
        type: "melody" | "chords"
        """
        songs = self._load_songs_from_midi_files(type)
        
        for song, paths in songs:
            _, _, save_path = paths
            
            print(paths)
            
            if type == 'chords':
                encoded_song = self._encode_chords_to_txt(song)
                encoded_song_context = self._encode_chords_context_to_txt(song)
                self._save_to_file(data=encoded_song, folder_path=save_path, save_path=f"{save_path}/{type}.txt")
                self._save_to_file(data=encoded_song_context, folder_path=save_path, save_path=f"{save_path}/{type}-context.txt")
            else:
                encoded_song = self._encode_melody_to_txt(song)  
                self._save_to_file(data=encoded_song, folder_path=save_path, save_path=f"{save_path}/{type}.txt")    
                     
    def padd_encoded_song_with_rest(self, song, to_length, is_chords_context = False):
        encoded_song = song.split()
        
        if len(encoded_song) >= to_length:
            return song

        symbols = [symbol for symbol in encoded_song if symbol != '_']
        last_symbol = symbols[len(symbols) - 1]
        
        if last_symbol != 'r':
            encoded_song.append('r')
            
        if is_chords_context:
            encoded_song.extend(["r"] * (to_length - len(encoded_song)))
        else:
            encoded_song.extend(["_"] * (to_length - len(encoded_song)))   
        
        encoded_song = " ".join(map(str, encoded_song))
        
        return encoded_song

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
        encoded_song = []

        for event in song.flatten().notesAndRests:
            event_duration = self._closest_acceptable_durations(event.duration.quarterLength)
            
            if isinstance(event, music21.note.Note):
                symbol = self._select_pitch(event.pitch.midi)
                # symbol = event.pitch.midi
                
            if isinstance(event, music21.chord.Chord):
                symbol = self._select_pitch(event.pitches[0].midi)
                # symbol = event.pitches[0].midi

            elif isinstance(event, music21.note.Rest):
                symbol = "r"
                
                if event.duration.quarterLength < self.acceptable_durations[0]:
                    continue
                        
            steps = int(event_duration / self.time_step)

            encoded_song.append(symbol)

            encoded_song.extend(["_"] * (steps - 1))

        encoded_song = " ".join(map(str, encoded_song))
        
        return encoded_song

    def _encode_chords_to_txt(self, song):
        encoded_song = []

        for event in song.flatten().notesAndRests:
            event_duration = self._closest_acceptable_durations(event.duration.quarterLength)
                    
            if isinstance(event, music21.chord.Chord):
                chord = "-".join(sorted(event.pitchNames, key=self._custom_sort_key))
                symbol = f"({chord})"

            elif isinstance(event, music21.note.Rest):
                symbol = "r"
                
                if event.duration.quarterLength < self.acceptable_durations[0]:
                    continue    
                
            steps = int(event_duration / self.time_step)

            encoded_song.append(symbol)

            encoded_song.extend(["_"] * (steps - 1))

        encoded_song = " ".join(map(str, encoded_song))

        return encoded_song

    def _encode_chords_context_to_txt(self, song):
        encoded_song = []

        for event in song.flatten().notesAndRests:
            event_duration = self._closest_acceptable_durations(event.duration.quarterLength)
                    
            if isinstance(event, music21.chord.Chord):
                chord = "-".join(sorted(event.pitchNames, key=self._custom_sort_key))
                symbol = f"({chord})"

            elif isinstance(event, music21.note.Rest):
                symbol = "r"
                
                if event.duration.quarterLength < self.acceptable_durations[0]:
                    continue    
                        
            steps = int(event_duration / self.time_step)
            
            encoded_song.extend([symbol] * steps)

        encoded_song = " ".join(map(str, encoded_song))

        return encoded_song
    
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
               



            
