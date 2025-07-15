import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, chords_model, melody_model, model_settings, constants, starting_weights_idx=-1):
        self.save_weight_idx = starting_weights_idx
        self.model_settings = model_settings
        self.constants = constants
        
        self.chords_model = chords_model
        self.melody_model = melody_model
        
        self.chords_metrics_log = {
            "chords_accuracy": [],
            "chords_loss": [],
            "chords_class_correct": [],
            "chords_class_total": [],
        }
        
        self.melody_metrics_log = {
            "melody_accuracy": [],
            "melody_loss": [],
            "melody_class_correct": [],
            "melody_class_total": []
        }
        
        self._load_model_weights_if_needed()
        self._load_mappings()
        self._init_class_weights()

    def _load_model_weights_if_needed(self):
        if self.save_weight_idx != -1:
            chords_weights_path = self.constants.DEFAULT_CHORDS_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(self.save_weight_idx)
            melody_weights_path = self.constants.DEFAULT_MELODY_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(self.save_weight_idx)
            
            if self.chords_model:
                self.chords_model.load_state_dict(torch.load(chords_weights_path))
            
            if self.melody_model:
                self.melody_model.load_state_dict(torch.load(melody_weights_path))

    def _load_mappings(self):
        with open(self.constants.CHORDS_MAPPINGS_PATH, "r") as fp:
            self.chords_mappings = json.load(fp)
            
        with open(self.constants.MELODY_MAPPINGS_PATH, "r") as fp:
            self.melody_mappings = json.load(fp)
            
    def _init_class_weights(self):
        self.chords_class_weights = self._compute_class_weights(self.chords_mappings)
        self.melody_class_weights = self._compute_class_weights(self.melody_mappings)
        
        self._print_class_weights(self.chords_class_weights, self.chords_mappings["mappings"], "Initial chords_class_weights weights")
        self._print_class_weights(self.melody_class_weights, self.melody_mappings["mappings"], "Initial melody_class_weights weights")
    
    def _compute_class_weights(self, mappings):
        symbol_counts = np.array(list(mappings['counter']['mapped_symbols'].values()))
        class_weights = torch.tensor(sum(symbol_counts) / (len(symbol_counts) * symbol_counts), dtype=torch.float32)
        return class_weights

    def _print_class_weights(self, weights, mappings, message):
        print(message)
        for idx, weight in enumerate(weights):
            symbol = list(mappings.values())[idx]
            print(f"Symbol {symbol}: {weight}")

    def update_melody_class_weights(self, updates):
        updated_weights = self.melody_class_weights.clone()
        for symbol, percentage in updates.items():
            mapped_value = self.melody_mappings['mappings'][symbol]
            updated_weights[mapped_value] *= (0.01 * percentage)
        self._print_class_weights(updated_weights, self.melody_mappings["mappings"], "Updated melody class weights")
        self.melody_class_weights = updated_weights
        
    def update_chords_class_weights(self, updates):
        updated_weights = self.chords_class_weights.clone()
        for symbol, percentage in updates.items():
            mapped_value = self.chords_mappings['mappings'][symbol]
            updated_weights[mapped_value] *= (0.01 * percentage)
        self._print_class_weights(updated_weights, self.chords_mappings["mappings"], "Updated chords class weights")
        self.chords_class_weights = updated_weights
    
    def train_melody(self, data_loader):
        melody_criterion = torch.nn.CrossEntropyLoss(weight=self.melody_class_weights).to(self.constants.DEVICE)
        melody_optimizer = torch.optim.Adam(self.melody_model.parameters(), lr=self.model_settings["LR"])
        
        num_epochs = self.model_settings['num_epochs']

        for epoch in range(num_epochs):
            melody_accuracy, melody_loss, melody_class_correct, melody_class_total = self._train_epoch_melody(data_loader, melody_criterion, melody_optimizer)
            
            print(f"Epoch - {epoch}: Melody accuracy - {melody_accuracy}")
            
            self.melody_metrics_log["melody_accuracy"].append(melody_accuracy)
            self.melody_metrics_log["melody_loss"].append(melody_loss)
            self.melody_metrics_log["melody_class_correct"].append(melody_class_correct)
            self.melody_metrics_log["melody_class_total"].append(melody_class_total)
            
            self._save_metrics(self.melody_metrics_log, is_chords=False)
            self._save_melody_model_weights()
    
    def train_chords(self, data_loader):
        chords_criterion = torch.nn.CrossEntropyLoss(weight=self.chords_class_weights).to(self.constants.DEVICE)
        chords_optimizer = torch.optim.Adam(self.chords_model.parameters(), lr=self.model_settings["LR"])
              
        num_epochs = self.model_settings['num_epochs']

        for epoch in range(num_epochs):     
            chords_accuracy, chords_loss, chords_class_correct, chords_class_total = self._train_epoch_chords(data_loader, chords_criterion, chords_optimizer)
            
            print(f"Epoch - {epoch}: Chords accuracy - {chords_accuracy}")
            
            self.chords_metrics_log["chords_accuracy"].append(chords_accuracy)
            self.chords_metrics_log["chords_loss"].append(chords_loss)
            self.chords_metrics_log["chords_class_correct"].append(chords_class_correct)
            self.chords_metrics_log["chords_class_total"].append(chords_class_total)
            
            self._save_metrics(self.chords_metrics_log, is_chords=True)
            self._save_chords_model_weights()

    def _train_epoch_chords(self, data_loader, chords_criterion, chords_optimizer):
        total_chords_correct, total_chords_samples = 0, 0
        total_chords_loss = 0.0
        
        chords_class_correct = np.zeros(len(self.chords_mappings['mappings']))
        chords_class_total = np.zeros(len(self.chords_mappings['mappings']))
        
        for batch_idx, (melody_batches, chords_batches) in enumerate(data_loader):
            print(f"Processing batch {batch_idx}")

            for i in range(chords_batches.shape[0]):
                chords = self._to_device(chords_batches[i])[0]
                # video = video.permute(0, 4, 1, 2, 3)
                video = 0

                chords_optimizer.zero_grad()

                chords_output = self.chords_model(chords, video)

                chords_output = chords_output.reshape(-1, chords_output.size(-1))
                chords_target = chords.reshape(-1, chords.size(-1))

                # Compute loss
                chords_loss = chords_criterion(chords_output, chords_target)
                chords_loss.backward()
                chords_optimizer.step()
                total_chords_loss += chords_loss.item()

                # Compute accuracy
                total_chords_samples += chords_target.size(0)
                total_chords_correct += self._compute_accuracy(chords_output, chords_target, chords_class_correct, chords_class_total)
                
        chords_accuracy = (total_chords_correct / total_chords_samples) * 100
        return chords_accuracy, total_chords_loss, chords_class_correct, chords_class_total

    def _train_epoch_melody(self, data_loader, melody_criterion, melody_optimizer):
        total_melody_correct, total_melody_samples = 0, 0
        total_melody_loss = 0.0
        
        melody_class_correct = np.zeros(len(self.melody_mappings['mappings']))
        melody_class_total = np.zeros(len(self.melody_mappings['mappings']))
        
        for batch_idx, (melody_batches, chords_batches, chords_context_batches) in enumerate(data_loader):
            print(f"Processing batch {batch_idx}")

            for i in range(melody_batches.shape[0]):
                melody, chords_context = self._to_device(melody_batches[i], chords_context_batches[i])
                # video = video.permute(0, 4, 1, 2, 3)
                video = 0

                melody_optimizer.zero_grad()
            
                melody_output = self.melody_model(melody, chords_context, video)
                
                melody_output = melody_output.reshape(-1, melody_output.size(-1))
                melody_target = melody.reshape(-1, melody.size(-1))
                
                melody_loss = melody_criterion(melody_output, melody_target)
                melody_loss.backward()
                melody_optimizer.step()
                total_melody_loss += melody_loss.item()

                # Compute accuracy        
                total_melody_samples += melody_target.size(0)
                total_melody_correct += self._compute_accuracy(melody_output, melody_target, melody_class_correct, melody_class_total)
        
        melody_accuracy = (total_melody_correct / total_melody_samples) * 100
        return melody_accuracy, total_melody_loss, melody_class_correct, melody_class_total

        

    def _compute_accuracy(self, output, target, class_correct, class_total):
        if target.ndimension() > 1:
            target = torch.argmax(target, dim=1)
              
        _, predicted = torch.max(output, dim=1)
        
        correct = (predicted == target)
        for i, t in enumerate(target):
            class_total[t.item()] += 1
            class_correct[t.item()] += 1 if correct[i].item() else 0
        
        return correct.sum().item()

    def _save_metrics(self, metrics, is_chords):
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
        
        with open(self.constants.CHORDS_METRICS_SAVE_FILE_PATH if is_chords else self.constants.MELODY_METRICS_SAVE_FILE_PATH, "w") as f:
            json.dump(serializable_metrics_log, f)

    def _to_device(self, *args):
        return [arg.to(self.constants.DEVICE) for arg in args]

    def _save_chords_model_weights(self):
        self.save_weight_idx += 1
        weights_folder = self.constants.DEFAULT_MODEL_WEIGHTS_FOLDER_NAME(idx=self.save_weight_idx)
        os.makedirs(weights_folder, exist_ok=True)
        
        chords_weights_path = self.constants.DEFAULT_CHORDS_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(idx=self.save_weight_idx)
        torch.save(self.chords_model.state_dict(), chords_weights_path)
        
    def _save_melody_model_weights(self):
        self.save_weight_idx += 1
        weights_folder = self.constants.DEFAULT_MODEL_WEIGHTS_FOLDER_NAME(idx=self.save_weight_idx)
        os.makedirs(weights_folder, exist_ok=True)
         
        melody_weights_path = self.constants.DEFAULT_MELODY_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(idx=self.save_weight_idx)
        torch.save(self.melody_model.state_dict(), melody_weights_path)

    def _plot_metrics_from_file(self):
        with open(self.constants.MELODY_METRICS_SAVE_FILE_PATH, "r") as f:
            melody_metrics = json.load(f)
            
        with open(self.constants.CHORDS_METRICS_SAVE_FILE_PATH, "r") as f:
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

        self._plot_class_correctness(melody_metrics, chords_metrics)

        print("Plots saved as files.")

    def _plot_class_correctness(self, melody_metrics, chords_metrics):
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