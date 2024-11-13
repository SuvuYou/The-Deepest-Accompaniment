import json
import os
import numpy as np
import torch

class Trainer:
    def __init__(self, chords_model, melody_model, model_settings, constants, starting_weights_idx=-1):
        self.save_weight_idx = starting_weights_idx
        self.model_settings = model_settings
        self.constants = constants
            
        self.chords_model = chords_model
        self.melody_model = melody_model
        self._load_model_weights_if_needed()
        self._load_mappings()
        self._init_class_weights()

    def _load_model_weights_if_needed(self):
        if self.save_weight_idx != -1:
            chords_weights_path = self.constants.DEFAULT_CHORDS_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(self.save_weight_idx)
            self.chords_model.load_state_dict(torch.load(chords_weights_path))
            
            melody_weights_path = self.constants.DEFAULT_MELODY_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(self.save_weight_idx)
            self.melody_model.load_state_dict(torch.load(melody_weights_path))

    def _load_mappings(self):
        with open(self.constants.CHORDS_MAPPINGS_PATH, "r") as fp:
            self.chords_mappings = json.load(fp)
            
        with open(self.constants.MELODY_MAPPINGS_PATH, "r") as fp:
            self.melody_mappings = json.load(fp)
            
        with open(self.constants.CHORDS_CONTEXT_MAPPINGS_PATH, "r") as fp:
            self.chords_context_mappings = json.load(fp)

    def _init_class_weights(self):
        self.chords_class_weights = self._compute_class_weights(self.chords_mappings)
        self.melody_class_weights = self._compute_class_weights(self.melody_mappings)
        
        self._print_class_weights(self.chords_class_weights, "Initial chords_class_weights weights")
        self._print_class_weights(self.melody_class_weights, "Initial melody_class_weights weights")
    
    def _compute_class_weights(self, mappings):
        symbol_counts = np.array(list(mappings['counter']['mapped_symbols'].values()))
        class_weights = torch.tensor(sum(symbol_counts) / (len(symbol_counts) * symbol_counts), dtype=torch.float32)
        return class_weights

    def _print_class_weights(self, weights, message):
        print(message)
        for idx, weight in enumerate(weights):
            print(f"Symbol {idx}: {weight}")

    # def update_class_weights(self, updates):
    #     updated_weights = self.class_weights.clone()
    #     for symbol, percentage in updates.items():
    #         mapped_value = self.chords_mappings['mappings'][symbol]
    #         updated_weights[mapped_value] *= (0.01 * percentage)
    #     self._print_class_weights(updated_weights, "Updated class weights")
    #     self.class_weights = updated_weights
    
    def train(self, data_loader):
        chords_criterion = torch.nn.CrossEntropyLoss(weight=self.chords_class_weights).to(self.constants.DEVICE)
        chords_optimizer = torch.optim.Adam(self.chords_model.parameters(), lr=self.model_settings["LR"])
        
        melody_criterion = torch.nn.CrossEntropyLoss(weight=self.melody_class_weights).to(self.constants.DEVICE)
        melody_optimizer = torch.optim.Adam(self.melody_model.parameters(), lr=self.model_settings["LR"])
        
        num_epochs = self.model_settings['num_epochs']

        for epoch in range(num_epochs):
            chords_accuracy, melody_accuracy = self._train_epoch(data_loader, chords_criterion, chords_optimizer, melody_criterion, melody_optimizer)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Chords Accuracy: {chords_accuracy:.2f}%, Melody Accuracy: {melody_accuracy:.2f}%')
            self._save_model_weights()

    def _train_epoch(self, data_loader, chords_criterion, chords_optimizer, melody_criterion, melody_optimizer):
        total_chords_correct, total_chords_samples = 0, 0
        total_melody_correct, total_melody_samples = 0, 0

        for batch_idx, (melody_batches, chords_batches, chords_context_batches, video_batches) in enumerate(data_loader):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}")

            for i in range(chords_batches.shape[0]):
                melody, chords, chords_context, video = self._to_device(melody_batches[i], chords_batches[i], chords_context_batches[i], video_batches[i])
                video = video.permute(0, 4, 1, 2, 3)

                chords_optimizer.zero_grad()
                melody_optimizer.zero_grad()
                
                chords_output = self.chords_model(chords, video)
                melody_output = self.melody_model(melody, chords_context, video)

                chords_output = chords_output.reshape(-1, chords_output.size(-1)) 
                chords_target = chords.reshape(-1, chords.size(-1))  
                
                melody_output = melody_output.reshape(-1, melody_output.size(-1)) 
                melody_target = melody.reshape(-1, melody.size(-1))  

                # Compute loss
                chords_loss = chords_criterion(chords_output, chords_target)
                chords_loss.backward()
                chords_optimizer.step()
                
                # Compute loss
                melody_loss = melody_criterion(melody_output, melody_target)
                melody_loss.backward()
                melody_optimizer.step()
        
                total_chords_samples += chords_target.size(0)
                total_chords_correct += self._compute_accuracy(chords_output, chords_target)
                
                total_melody_samples += melody_target.size(0)
                total_melody_correct += self._compute_accuracy(melody_output, melody_target)

        return (total_chords_correct / total_chords_samples) * 100, (total_melody_correct / total_melody_samples) * 100

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
        
        chords_weights_path = self.constants.DEFAULT_CHORDS_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(idx=self.save_weight_idx)
        torch.save(self.chords_model.state_dict(), chords_weights_path)
        
        melody_weights_path = self.constants.DEFAULT_MELODY_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(idx=self.save_weight_idx)
        torch.save(self.melody_model.state_dict(), melody_weights_path)