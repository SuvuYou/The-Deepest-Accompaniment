import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from Processing.const import CONSTANTS

class MetricsManager:
    def __init__(self):
        self.chords_metrics_log = {
            "accuracy": [],
            "loss": [],
            "class_correct": [],
            "class_total": [],
        }
        
        self.melody_metrics_log = {
            "accuracy": [],
            "loss": [],
            "class_correct": [],
            "class_total": []
        }
    
    def log_melody_metrics(self, accuracy, loss, class_correct, class_total):
        self.log_metrics(self.melody_metrics_log, accuracy, loss, class_correct, class_total)
        
    def log_chords_metrics(self, accuracy, loss, class_correct, class_total):
        self.log_metrics(self.chords_metrics_log, accuracy, loss, class_correct, class_total)

    def log_metrics(self, metrics, accuracy, loss, class_correct, class_total):
        metrics["accuracy"].append(accuracy)
        metrics["loss"].append(loss)
        metrics["class_correct"].append(class_correct)
        metrics["class_total"].append(class_total)
        
    def save_metrics(self, metrics, is_chords):
        def convert_to_serializable(obj):
            # Recursively convert numpy arrays and tensors to lists
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()  # Convert PyTorch tensor to list
            elif isinstance(obj, list):
                return [convert_to_serializable(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            else:
                return obj  # Leave other types untouched

        serializable_metrics_log = convert_to_serializable(metrics)
        
        with open(CONSTANTS.CHORDS_METRICS_SAVE_FILE_PATH if is_chords else CONSTANTS.MELODY_METRICS_SAVE_FILE_PATH, "w") as f:
            json.dump(serializable_metrics_log, f)

    def plot_metrics_from_file(self):
        with open(CONSTANTS.MELODY_METRICS_SAVE_FILE_PATH, "r") as f:
            melody_metrics = json.load(f)
            
        with open(CONSTANTS.CHORDS_METRICS_SAVE_FILE_PATH, "r") as f:
            chords_metrics = json.load(f)
        
        epochs = range(1, len(melody_metrics["melody_accuracy"]) + 1)
        
        # Accuracy plots
        plt.figure()
        plt.plot(epochs, melody_metrics["melody_accuracy"], label="Melody Accuracy")
        plt.plot(epochs, chords_metrics["chords_accuracy"], label="Chords Accuracy")
        plt.title("Model Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig("metrics/accuracy_plot.png")

        # Loss plots
        plt.figure()
        plt.plot(epochs, melody_metrics["melody_loss"], label="Melody Loss")
        plt.plot(epochs, chords_metrics["chords_loss"], label="Chords Loss")
        plt.title("Model Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("metrics/loss_plot.png")

        self.plot_class_correctness(melody_metrics, chords_metrics)

        print("Plots saved as files.")

    def plot_class_correctness(self, melody_metrics, chords_metrics):
        melody_labels = {v: k for k, v in self.melody_mappings['mappings'].items()}
        chords_labels = {v: k for k, v in self.chords_mappings['mappings'].items()}
        
        melody_class_correct = melody_metrics["melody_class_correct"]
        melody_class_total = melody_metrics["melody_class_total"]
        chords_class_correct = chords_metrics["chords_class_correct"]
        chords_class_total = chords_metrics["chords_class_total"]

        for symbol_idx, symbol in melody_labels.items():
            correct_counts = [epoch[symbol_idx] for epoch in melody_class_correct]
            total_counts = [epoch[symbol_idx] for epoch in melody_class_total]
            
            plt.figure()
            plt.plot(range(1, len(correct_counts) + 1), correct_counts, label="Correct Predictions")
            plt.plot(range(1, len(total_counts) + 1), total_counts, label="Total Elements")
            plt.title(f"Melody Symbol: {symbol}")
            plt.xlabel("Epoch")
            plt.ylabel("Count")
            plt.legend()
            plt.savefig(f"metrics/melody_symbol_{symbol.replace('/', '_')}.png")
            plt.close()

        for symbol_idx, symbol in chords_labels.items():
            correct_counts = [epoch[symbol_idx] for epoch in chords_class_correct]
            total_counts = [epoch[symbol_idx] for epoch in chords_class_total]
            
            plt.figure()
            plt.plot(range(1, len(correct_counts) + 1), correct_counts, label="Correct Predictions")
            plt.plot(range(1, len(total_counts) + 1), total_counts, label="Total Elements")
            plt.title(f"Chord Symbol: {symbol}")
            plt.xlabel("Epoch")
            plt.ylabel("Count")
            plt.legend()
            plt.savefig(f"metrics/chord_symbol_{symbol.replace('/', '_')}.png")
            plt.close()

        print("Plots saved for all symbols.")