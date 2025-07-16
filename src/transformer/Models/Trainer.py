import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from Processing.const import CONSTANTS
from Processing.SongsMapper import SongsMapper
from Models.MetricsManager import MetricsManager
from Models.ClassWeightManager import ClassWeightManager

class Trainer:
    def __init__(self, chords_model, melody_model, model_settings, starting_weights_idx=-1):
        self.save_weight_idx = starting_weights_idx
        self.model_settings = model_settings
        
        self.chords_model = chords_model
        self.melody_model = melody_model
        
        self.metrics_manager = MetricsManager()
        self.class_weight_manager = ClassWeightManager()
        
        self._load_model_weights_if_needed()

    def _load_model_weights_if_needed(self):
        if self.save_weight_idx != -1:
            chords_weights_path = CONSTANTS.DEFAULT_CHORDS_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(self.save_weight_idx)
            melody_weights_path = CONSTANTS.DEFAULT_MELODY_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(self.save_weight_idx)
            
            if self.chords_model:
                self.chords_model.load_state_dict(torch.load(chords_weights_path))
            
            if self.melody_model:
                self.melody_model.load_state_dict(torch.load(melody_weights_path))
            
    def train_melody(self, data_loader):
        melody_criterion = torch.nn.CrossEntropyLoss(weight=self.melody_class_weights).to(CONSTANTS.DEVICE)
        melody_optimizer = torch.optim.Adam(self.melody_model.parameters(), lr=self.model_settings["LR"])
        
        num_epochs = self.model_settings['num_epochs']

        for epoch in range(num_epochs):
            melody_accuracy, melody_loss, melody_class_correct, melody_class_total = self._train_epoch_melody(data_loader, melody_criterion, melody_optimizer)
            
            print(f"Epoch - {epoch}: Melody accuracy - {melody_accuracy}")

            self._log_metrics(self.melody_metrics_log, melody_accuracy, melody_loss, melody_class_correct, melody_class_total)
            
            self._save_metrics(self.melody_metrics_log, is_chords=False)
            self._save_melody_model_weights()
    
    def train_chords(self, data_loader):
        chords_criterion = torch.nn.CrossEntropyLoss(weight=self.chords_class_weights).to(CONSTANTS.DEVICE)
        chords_optimizer = torch.optim.Adam(self.chords_model.parameters(), lr=self.model_settings["LR"])
              
        num_epochs = self.model_settings['num_epochs']

        for epoch in range(num_epochs):     
            chords_accuracy, chords_loss, chords_class_correct, chords_class_total = self._train_epoch_chords(data_loader, chords_criterion, chords_optimizer)
            
            print(f"Epoch - {epoch}: Chords accuracy - {chords_accuracy}")
            
            self._log_metrics(self.chords_metrics_log, chords_accuracy, chords_loss, chords_class_correct, chords_class_total)
            
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

    def _to_device(self, *args):
        return [arg.to(CONSTANTS.DEVICE) for arg in args]

    def _save_chords_model_weights(self):
        self.save_weight_idx += 1
        weights_folder = CONSTANTS.DEFAULT_MODEL_WEIGHTS_FOLDER_NAME(idx=self.save_weight_idx)
        os.makedirs(weights_folder, exist_ok=True)
        
        chords_weights_path = CONSTANTS.DEFAULT_CHORDS_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(idx=self.save_weight_idx)
        torch.save(self.chords_model.state_dict(), chords_weights_path)
        
    def _save_melody_model_weights(self):
        self.save_weight_idx += 1
        weights_folder = CONSTANTS.DEFAULT_MODEL_WEIGHTS_FOLDER_NAME(idx=self.save_weight_idx)
        os.makedirs(weights_folder, exist_ok=True)
         
        melody_weights_path = CONSTANTS.DEFAULT_MELODY_TRANSFORMER_MODEL_WEIGHTS_FILE_NAME(idx=self.save_weight_idx)
        torch.save(self.melody_model.state_dict(), melody_weights_path)
