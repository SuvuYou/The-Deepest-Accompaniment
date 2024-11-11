from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup
import music21

class MusicScraperAndChordGenerator:
    def __init__(self, url, should_snap_to_white_key = False):
        self.url = url
        self.should_snap_to_white_key = should_snap_to_white_key
        self.driver = None
        self.page_source = None
        self.data_list = []
        self.black_keys_pitches = [1, 3, 6, 8, 10]
        self.white_keys_pitches = [0, 2, 4, 5, 7, 9, 11]
        self.valid_symbols = {'r', '_', 'C', 'D', 'E', 'F', 'G', 'A', 'B', 'Em', 'Bm', 'Am', 'Cm', 'Dm', 'F#m', 'G#m', 'C#m'} 

    def scrape_webpage(self):
        self.driver = webdriver.Chrome()
        self.driver.get(self.url)
        self._scroll_page()
        self.page_source = self.driver.page_source
        self.driver.quit()
        self._parse_data()
        
    def _validate_melody(self, melody):
        return [symbol if symbol in self.valid_symbols else 'r' for symbol in melody]

    def _scroll_page(self):
        for _ in range(1):
            self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
            time.sleep(1)

    def _parse_data(self):
        soup = BeautifulSoup(self.page_source, 'html.parser')
        spans = soup.find_all('span', {'id': lambda x: x and x.startswith('ac')})
        
        for span in spans:
            p_tag = span.find('p')
            if p_tag:
                self.data_list.append(p_tag.get_text(strip=True) or "_")

    def _snap_to_white_key(self, pitch):
        if not self.should_snap_to_white_key:
            return pitch
            
        pitch_class = pitch.pitchClass
        
        if pitch_class in self.white_keys_pitches:
            return pitch

        closest_white_key = min(self.white_keys_pitches, key=lambda x: abs(x - pitch_class))
        adjusted_pitch = pitch.transpose(closest_white_key - pitch_class)
        
        return adjusted_pitch

    def _get_chord_pitches(self, chord_symbol, interval=0):
        root_note = chord_symbol[0]  
        quality = chord_symbol[1:]  

        root_pitch = music21.pitch.Pitch(root_note)
        root_pitch = root_pitch.transpose(interval)  
        root_pitch = self._snap_to_white_key(root_pitch)  
        
        if quality == 'm':
            pitches = [root_pitch, self._snap_to_white_key(root_pitch.transpose('m3')), self._snap_to_white_key(root_pitch.transpose('P5'))]
        elif quality == 'dim':
            pitches = [root_pitch, self._snap_to_white_key(root_pitch.transpose('m3')), self._snap_to_white_key(root_pitch.transpose('d5'))]
        elif quality == 'aug':
            pitches = [root_pitch, self._snap_to_white_key(root_pitch.transpose('M3')), self._snap_to_white_key(root_pitch.transpose('A5'))]
        else:
            pitches = [root_pitch, self._snap_to_white_key(root_pitch.transpose('M3')), self._snap_to_white_key(root_pitch.transpose('P5'))]
        
        return pitches

    def detect_key_and_transpose_interval(self, melody):
        stream = music21.stream.Stream()
        
        for note in melody:
            if note == "r" or note == "_":
                stream.append(music21.note.Rest())
            else: 
                chord_pitches = self._get_chord_pitches(note)
                chord = music21.chord.Chord(chord_pitches)
                stream.append(chord)
        
        key = stream.analyze('key')
        print(f"Detected Key: {key}")
        
        if key.mode == 'minor':
            interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch('A'))
        else:
            interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch('C'))
        
        return interval

    def save_chord_progression(self, melody, step_duration, file_name):
        transposition_interval = self.detect_key_and_transpose_interval(melody)
        
        stream = music21.stream.Stream()
        melody = [x for x in ' '.join(melody).lstrip('_ ').split()]
        
        current_symbol = melody[0]
        current_symbol_step_counter = 1
        melody_length = len(melody[1:])

        for i, symbol in enumerate(melody[1:]):
            if symbol == "_" and i != melody_length - 1:
                current_symbol_step_counter += 1
            else:
                quarter_length_duration = step_duration * current_symbol_step_counter
                if current_symbol == "r":
                    event = music21.note.Rest(quarterLength=quarter_length_duration)
                else:
                    chord_pitches = self._get_chord_pitches(current_symbol, transposition_interval)
                    event = music21.chord.Chord(chord_pitches, quarterLength=quarter_length_duration)
                stream.append(event)
                
                current_symbol = symbol
                current_symbol_step_counter = 1

        stream.write('midi', file_name)

    def run(self, step_duration=1, file_name="chords.mid"):
        self.scrape_webpage()
        
        self.data_list = self._validate_melody(self.data_list)
        
        print("Scraped Data:", self.data_list)

        self.save_chord_progression(self.data_list, step_duration, file_name)


url = 'https://chordu.com/chords-tabs-celeste-piano-collections-11-reach-for-the-summit-lena-raine-trevor-alan-gomes--id_tZEW4X7RGMs?vers=sim'
music_processor = MusicScraperAndChordGenerator(url, should_snap_to_white_key = True)
music_processor.run()
