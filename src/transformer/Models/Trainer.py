import json
import os
import numpy as np
import torch

class Trainer:
    def __init__(self, model, model_settings, constants, starting_weights_idx=-1):
        self.save_weight_idx = starting_weights_idx
        self.model_settings = model_settings
        self.constants = constants
            
        self.model = model
        self._load_model_weights_if_needed()
        self._load_mappings()
        self._init_class_weights()

    def _load_model_weights_if_needed(self):
        if self.save_weight_idx != -1:
            weights_path = self.constants.DEFAULT_CHORDS_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(self.save_weight_idx)
            self.model.load_state_dict(torch.load(weights_path))

    def _load_mappings(self):
        with open(self.constants.CHORDS_MAPPINGS_PATH, "r") as fp:
            self.chords_mappings = json.load(fp)

    def _init_class_weights(self):
        self.class_weights = self._compute_class_weights(self.chords_mappings)
        self._print_class_weights(self.class_weights, "Initial class weights")
    
    def _compute_class_weights(self, mappings):
        symbol_counts = np.array(list(mappings['counter']['mapped_symbols'].values()))
        class_weights = torch.tensor(sum(symbol_counts) / (len(symbol_counts) * symbol_counts), dtype=torch.float32)
        return class_weights

    def _print_class_weights(self, weights, message):
        print(message)
        for idx, weight in enumerate(weights):
            print(f"Symbol {idx}: {weight}")

    def update_class_weights(self, updates):
        updated_weights = self.class_weights.clone()
        for symbol, percentage in updates.items():
            mapped_value = self.chords_mappings['mappings'][symbol]
            updated_weights[mapped_value] *= (0.01 * percentage)
        self._print_class_weights(updated_weights, "Updated class weights")
        self.class_weights = updated_weights
    
    def train(self, data_loader):
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights).to(self.constants.DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_settings["LR"])
        num_epochs = self.model_settings['num_epochs']

        for epoch in range(num_epochs):
            accuracy = self._train_epoch(data_loader, criterion, optimizer)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy:.2f}%')
            self._save_model_weights()

    def _train_epoch(self, data_loader, criterion, optimizer):
        total_correct, total_samples = 0, 0

        for batch_idx, (melody_batches, chords_batches, chords_context_batches, video_batches) in enumerate(data_loader):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}")

            for i in range(chords_batches.shape[0]):
                chords, video = self._to_device(chords_batches[i], video_batches[i])
                video = video.permute(0, 4, 1, 2, 3)

                optimizer.zero_grad()
                output = self.model(chords, video.float())

                output = output.reshape(-1, output.size(-1)) 
                chords_target = chords.reshape(-1, chords.size(-1))  

                # Compute loss
                loss = criterion(output, chords_target)
                loss.backward()
                optimizer.step()

                total_samples += chords_target.size(0)
                total_correct += self._compute_accuracy(output, chords_target)

        return (total_correct / total_samples) * 100

    def _compute_accuracy(self, output, target):
        _, predicted = torch.max(output, 1)
        output_binary = torch.zeros_like(output)
        output_binary.scatter_(1, predicted.view(-1, 1), 1)
        return (output_binary == target).all(dim=1).sum().item()

    def _to_device(self, *args):
        return [arg.to(self.constants.DEVICE) for arg in args]

    def _save_model_weights(self):
        self.save_weight_idx += 1
        weights_folder = self.constants.DEFAULT_MODEL_WEIGHTS_FOLDER_NAME(idx=self.save_weight_idx)
        os.makedirs(weights_folder, exist_ok=True)
        
        weights_path = self.constants.DEFAULT_CHORDS_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(idx=self.save_weight_idx)
        torch.save(self.model.state_dict(), weights_path)