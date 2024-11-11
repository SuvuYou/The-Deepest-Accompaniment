import json
import os
import numpy as np
import torch

class Trainer:
    def __init__(self, melody_model, chords_model, model_settings, constants, starting_weights_idx=-1):
        self.save_weight_idx = starting_weights_idx
        self.model_settings = model_settings
        self.constants = constants
            
        self.melody_model = melody_model
        self.chords_model = chords_model
        
        self._load_model_weights_if_needed()
        self._load_mappings()
        self._init_class_weights()

    def _load_model_weights_if_needed(self):
        if self.save_weight_idx != -1:
            melody_weights_path = self.constants.DEFAULT_MELODY_MODEL_WEIGHTS_FILE_NAME(self.save_weight_idx)
            chords_weights_path = self.constants.DEFAULT_CHORDS_MODEL_WEIGHTS_FILE_NAME(self.save_weight_idx)
            self.melody_model.load_state_dict(torch.load(melody_weights_path))
            self.chords_model.load_state_dict(torch.load(chords_weights_path))

    def _load_mappings(self):
        with open(self.constants.MELODY_MAPPINGS_PATH, "r") as fp:
            self.melody_mappings = json.load(fp)['mappings']
            
        with open(self.constants.CHORDS_MAPPINGS_PATH, "r") as fp:
            self.chords_mappings = json.load(fp)['mappings']    

    def _init_class_weights(self):
        self.melody_class_weights = self._compute_class_weights(self.melody_mappings)
        self.chords_class_weights = self._compute_class_weights(self.chords_mappings)
        
        self._print_class_weights(self.melody_class_weights, "Initial melody class weights")
        self._print_class_weights(self.chords_class_weights, "Initial chords class weights")
    
    def _compute_class_weights(self, mappings):
        symbol_counts = np.array(list(mappings['counter']['mapped_symbols'].values()))
        class_weights = torch.tensor(sum(symbol_counts) / (len(symbol_counts) * symbol_counts), dtype=torch.float32)
        return class_weights

    def _print_class_weights(self, weights, message):
        print(message)
        for idx, weight in enumerate(weights):
            print(f"Symbol {idx}: {weight}")

    def update_class_weights(self, weights, mappings, updates):
        updated_weights = weights.clone()
        
        for symbol, percentage in updates.items():
            mapped_value = mappings['mappings'][symbol]
            updated_weights[mapped_value] *= (0.01 * percentage)
            
        return updated_weights

    def update_melody_class_weights(self, updates):
        updated_weights = self.update_class_weights(self.melody_class_weights, self.melody_mappings, updates)
        self._print_class_weights(updated_weights, "Updated melody class weights")
        return updated_weights

    def update_chords_class_weights(self, updates):
        updated_weights = self.update_class_weights(self.chords_class_weights, self.chords_mappings, updates)
        self._print_class_weights(updated_weights, "Updated chords class weights")
        return updated_weights
    
    def train(self, data_loader):
        melody_criterion = torch.nn.CrossEntropyLoss(weight=self.melody_class_weights).to(self.constants.DEVICE)
        chords_criterion = torch.nn.CrossEntropyLoss(weight=self.chords_class_weights).to(self.constants.DEVICE)
        
        melody_optimizer = torch.optim.Adam(self.melody_model.parameters(), lr=self.model_settings["LR"])
        chords_optimizer = torch.optim.Adam(self.chords_model.parameters(), lr=self.model_settings["LR"])
        
        num_epochs = self.constants.MODEL_SETTINGS['num_epochs']      

        for epoch in range(num_epochs):
            melody_accuracy, chords_accuracy = self._train_epoch(data_loader, melody_criterion, chords_criterion, melody_optimizer, chords_optimizer)
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], Melody Accuracy: {melody_accuracy:.2f}%, Chords Accuracy: {chords_accuracy:.2f}%')
            
            self._save_model_weights()

    def _train_epoch(self, data_loader, melody_criterion, chords_criterion, melody_optimizer, chords_optimizer):
        total_melody_correct, total_melody_samples = 0, 0
        total_chords_correct, total_chords_samples = 0, 0

        for batch_idx, ((melody_batches, chords_batches, chords_context_batches, video_batches), 
                        (melody_target_batches, chords_target_batches)) in enumerate(data_loader):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}")

            for i in range(melody_batches.shape[0]):
                melody, chords, chords_context, video = self._to_device(melody_batches[i], chords_batches[i], chords_context_batches[i], video_batches[i])
                melody_target, chords_target = self._to_device(melody_target_batches[i], chords_target_batches[i])

                self._optimize_step(melody, chords, chords_context, video, melody_target, chords_target, melody_criterion, chords_criterion, melody_optimizer, chords_optimizer)

                total_melody_samples += melody_target.size(0)
                total_chords_samples += chords_target.size(0)
                total_melody_correct += self._compute_accuracy(melody, melody_target)
                total_chords_correct += self._compute_accuracy(chords, chords_target)
                
        return (total_melody_correct / total_melody_samples) * 100, (total_chords_correct / total_chords_samples) * 100

    def _optimize_step(self, melody, chords, chords_context, video, melody_target, chords_target, melody_criterion, chords_criterion, melody_optimizer, chords_optimizer):
        melody_optimizer.zero_grad()
        chords_optimizer.zero_grad()

        chords_output = self.chords_model(chords, video.float())
        melody_output = self.melody_model(melody, chords_context.float(), video.float())

        chords_loss = chords_criterion(chords_output, chords_target.float())
        melody_loss = melody_criterion(melody_output, melody_target.float())

        chords_loss.backward()
        chords_optimizer.step()
        melody_loss.backward()
        melody_optimizer.step()

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
        
        melody_weights_path = self.constants.DEFAULT_MELODY_MODEL_WEIGHTS_FILE_NAME(idx=self.save_weight_idx)
        chords_weights_path = self.constants.DEFAULT_CHORDS_MODEL_WEIGHTS_FILE_NAME(idx=self.save_weight_idx)
        
        torch.save(self.melody_model.state_dict(), melody_weights_path)
        torch.save(self.chords_model.state_dict(), chords_weights_path)
