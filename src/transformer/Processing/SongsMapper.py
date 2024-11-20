import json
from collections import Counter
import matplotlib.pyplot as plt

class SongsMapper:
    def __init__(self, mapping_path): 
        self.mapping_path = mapping_path
        
    def create_int_mapping(self, songs, mapping_path):
        mappings = {}
        
        songs = [song.split() for song in songs]
        songs_flat = sum(songs, [])
        
        unique_symbols = list(set(songs_flat))
        symbols_counts = Counter(songs_flat)
        mapped_symbols_counts = {}
    
        for i, symbol in enumerate(unique_symbols):
            mappings[symbol] = i
            mapped_symbols_counts[i] = symbols_counts[symbol]
            
        data = {
            'mappings': mappings,
            'counter': {
                "symbols": symbols_counts,
                "mapped_symbols": mapped_symbols_counts
                }
            }    

        with open(mapping_path, "w") as fp:
            json.dump(data, fp, indent=4)
            
    def convert_songs_to_int(self, songs, mapping_path):
        int_songs = []

        with open(mapping_path, "r") as fp:
            data = json.load(fp)

        songs = songs.split()

        for symbol in songs:
            int_songs.append(data["mappings"][symbol])

        return int_songs

    @staticmethod
    def load_mappings(path):
        with open(path, "r") as fp:
            mappings = json.load(fp)
            
        return mappings
            
    @staticmethod
    def get_mappings_size(mapping_path):
        with open(mapping_path, "r") as fp:
            data = json.load(fp)

        return len(data["mappings"].items())
    
    @staticmethod
    def plot_mappings_data(mapping_data, title="Symbol Distribution"):
        symbols = list(mapping_data["counter"]["symbols"].keys())
        counts = list(mapping_data["counter"]["symbols"].values())

        plt.figure(figsize=(12, 6))
        plt.bar(symbols, counts, color='skyblue', alpha=0.8)
        plt.title(title)
        plt.xlabel("Symbols")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
