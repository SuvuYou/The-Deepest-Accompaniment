import json
import numpy as np
import torch
import music21
from Processing.const import CONSTANTS
from Processing.SongsMapper import SongsMapper

class Generator:
    def __init__(self, chords_generation_model, melody_generation_model, save_file_name):
        self.save_file_name = save_file_name
        self.step_duration = CONSTANTS.ACCEPTABLE_DURATIONS[0]
        
        self.chords_generation_model = chords_generation_model
        self.melody_generation_model = melody_generation_model
        
        self.mappings = {
            "chord_pitch_mapping": SongsMapper.load_mappings(CONSTANTS.CHORDS_PITCH_MAPPINGS_PATH)['mappings'],
            "chord_duration_mapping": SongsMapper.load_mappings(CONSTANTS.CHORDS_DURATION_MAPPINGS_PATH)['mappings'],
            "melody_pitch_mapping": SongsMapper.load_mappings(CONSTANTS.MELODY_PITCH_MAPPINGS_PATH)['mappings'],
            "melody_duration_mapping": SongsMapper.load_mappings(CONSTANTS.MELODY_DURATION_MAPPINGS_PATH)['mappings'],
        }
        
        self.mappings_inverted = {
            "chord_pitch_mapping": {v: k for k, v in self.mappings['chord_pitch_mapping'].items()},
            "chord_duration_mapping": {v: k for k, v in self.mappings['chord_duration_mapping'].items()},
            "melody_pitch_mapping": {v: k for k, v in self.mappings['melody_pitch_mapping'].items()},
            "melody_duration_mapping": {v: k for k, v in self.mappings['melody_duration_mapping'].items()},
        }
        
    def generate(self, chords_pitch_seed, chords_duration_seed, melody_pitch_seed, melody_duration_seed, video, num_steps, temperature):
        with torch.no_grad():
            chords_pitch_seq = [self.mappings["chord_pitch_mapping"][symbol] for symbol in chords_pitch_seed.split()]
            chords_dur_seq   = [self.mappings["chord_duration_mapping"][symbol] for symbol in chords_duration_seed.split()]
            melody_pitch_seq = [self.mappings["melody_pitch_mapping"][symbol] for symbol in melody_pitch_seed.split()]
            melody_dur_seq   = [self.mappings["melody_duration_mapping"][symbol] for symbol in melody_duration_seed.split()]

            generated_chords = list(zip(chords_pitch_seed.split(), chords_duration_seed.split()))
            generated_melody = list(zip(melody_pitch_seed.split(), melody_duration_seed.split()))

            for idx in range(num_steps):
                # truncate to max seq length
                chords_pitch_seq = chords_pitch_seq[-(CONSTANTS.DEFAULT_SEQUENCE_LENGTH - 1):]
                chords_dur_seq   = chords_dur_seq[-(CONSTANTS.DEFAULT_SEQUENCE_LENGTH - 1):]
                melody_pitch_seq = melody_pitch_seq[-(CONSTANTS.DEFAULT_SEQUENCE_LENGTH - 1):]
                melody_dur_seq   = melody_dur_seq[-(CONSTANTS.DEFAULT_SEQUENCE_LENGTH - 1):]

                # get video context
                seed_len = len(chords_pitch_seq)
                video_seed = video[idx: idx + seed_len]
                video_seed = video_seed[[0, len(video_seed) // 2]]
                num_frames, h, w, c = video_seed.shape
                video_seed = video_seed.reshape(c, num_frames, h, w)

                # forward pass
                chords_out_pitch, chords_out_dur = self.chords_generation_model(
                    torch.tensor(chords_pitch_seq).unsqueeze(0),
                    torch.tensor(chords_dur_seq).unsqueeze(0),
                    # torch.tensor(video_seed).unsqueeze(0),
                )

                # sample with temperature
                next_chord_pitch = self._sample_with_temperature(chords_out_pitch[-1].squeeze(0), temperature)
                next_chord_dur   = self._sample_with_temperature(chords_out_dur[-1].squeeze(0), temperature)

                # decode
                cp, cd = self.mappings_inverted['chord_pitch_mapping'][next_chord_pitch], self.mappings_inverted['chord_duration_mapping'][next_chord_dur]

                # append
                chords_pitch_seq.append(next_chord_pitch)
                chords_dur_seq.append(next_chord_dur)

                generated_chords.append((cp, cd))

            return generated_chords, generated_melody

    def _sample_with_temperature(self, logits, temperature):
        scaled_logits = logits / temperature
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)

        return torch.multinomial(probs, num_samples=1).item()

    def save_to_file(self, chords, melody):
        self._save_chords(chords, f'generated/{self.save_file_name}_chords.mid')
        self._save_melody(melody, f'generated/{self.save_file_name}_melody.mid')

    def _save_chords(self, chords, file_name):
        stream = music21.stream.Stream()
        for pitch, dur in chords:
            ql = self._duration_to_quarter_length(dur)
            if pitch == "r":
                stream.append(music21.note.Rest(quarterLength=ql))
            else:
                notes = pitch.replace('(', '').replace(')', '').split('-')
                stream.append(music21.chord.Chord(notes, quarterLength=ql))
        stream.write('midi', file_name)

    def _save_melody(self, melody, file_name):
        stream = music21.stream.Stream()
        for pitch, dur in melody:
            ql = self._duration_to_quarter_length(dur)
            if pitch == "r":
                stream.append(music21.note.Rest(quarterLength=ql))
            else:
                stream.append(music21.note.Note(int(pitch), quarterLength=ql))
        stream.write('midi', file_name)

    def _duration_to_quarter_length(self, duration_symbol: str) -> float:
        try:
            return float(duration_symbol)
        except ValueError:
            return 1.0
