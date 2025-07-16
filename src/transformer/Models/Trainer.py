import numpy as np
import torch
from Processing.const import CONSTANTS
from Models.MetricsManager import MetricsManager
from Models.ClassWeightManager import ClassWeightManager
from Models.ModelWeightsManager import ModelWeightsManager

class Trainer:
    def __init__(self, chords_model, melody_model, model_settings, starting_weights_idx=-1):
        self.save_weight_idx = starting_weights_idx
        self.model_settings = model_settings
        
        self.chords_model = chords_model
        self.melody_model = melody_model
        
        self.metrics_manager = MetricsManager()
        self.class_weight_manager = ClassWeightManager()
        self.model_weights_manager = ModelWeightsManager(self.chords_model, self.melody_model, self.save_weight_idx)
            
    def train_melody(self, data_loader):
        melody_pitch_class_weights, melody_duration_class_weights = self.class_weight_manager.get_melody_class_weights()
        
        pitch_loss_fn = torch.nn.CrossEntropyLoss(weight=melody_pitch_class_weights).to(CONSTANTS.DEVICE)
        duration_loss_fn = torch.nn.CrossEntropyLoss(weight=melody_duration_class_weights).to(CONSTANTS.DEVICE)
        optimizer = torch.optim.Adam(self.melody_model.parameters(), lr=self.model_settings["LR"])
        num_epochs = self.model_settings["num_epochs"]

        for epoch in range(num_epochs):
            acc_pitch, acc_dur, total_loss = self._train_epoch_model(
                model=self.melody_model,
                data_loader=data_loader,
                pitch_loss_fn=pitch_loss_fn,
                duration_loss_fn=duration_loss_fn,
                optimizer=optimizer,
            )
                        
            # melody_accuracy, melody_loss, melody_class_correct, melody_class_total = self._train_epoch_melody(data_loader, melody_criterion, melody_optimizer)
            
            print(f"Epoch {epoch} - Melody Pitch Acc: {acc_pitch:.2f}%, Duration Acc: {acc_dur:.2f}%")

            # self.metrics_manager.log_melody_metrics(melody_accuracy, melody_loss, melody_class_correct, melody_class_total)
            # self.metrics_manager.save_metrics(is_chords=False)
            # self.model_weights_manager.save_melody_model_weights()
    
    def train_chords(self, data_loader):
        chords_pitch_class_weights, chords_duration_class_weights = self.class_weight_manager.get_chords_class_weights()
        
        pitch_loss_fn = torch.nn.CrossEntropyLoss(weight=chords_pitch_class_weights).to(CONSTANTS.DEVICE)
        duration_loss_fn = torch.nn.CrossEntropyLoss(weight=chords_duration_class_weights).to(CONSTANTS.DEVICE)
        optimizer = torch.optim.Adam(self.melody_model.parameters(), lr=self.model_settings["LR"])
        num_epochs = self.model_settings["num_epochs"]

        for epoch in range(num_epochs):     
            acc_pitch, acc_dur, total_loss = self._train_epoch_model(
                model=self.chords_model,
                data_loader=data_loader,
                pitch_loss_fn=pitch_loss_fn,
                duration_loss_fn=duration_loss_fn,
                optimizer=optimizer,
            )
            
            print(f"Epoch {epoch} - Chords Pitch Acc: {acc_pitch:.2f}%, Duration Acc: {acc_dur:.2f}%")
            
            # self.metrics_manager.log_chords_metrics(chords_accuracy, chords_loss, chords_class_correct, chords_class_total)
            
            # self.metrics_manager.save_metrics(is_chords=True)
            # self.model_weights_manager.save_chords_model_weights()

    def _train_epoch_model(self, model, data_loader, pitch_loss_fn, duration_loss_fn, optimizer):
        model.train()
        total_pitch_correct = total_duration_correct = total_tokens = 0
        total_loss = 0.0

        for batch_idx, batch in enumerate(data_loader):
            print(f"Processing batch {batch_idx}")
            
            pitch_tokens, dur_tokens, video = self._to_device(*batch)
            
            # Flatten the hierarchical structure of batches (first two dimentions are both batches: one from DataSoader and one from DataSaver by chunks)
            pitch_tokens = pitch_tokens.reshape(-1, pitch_tokens.size(2), pitch_tokens.size(3))
            dur_tokens = dur_tokens.reshape(-1, dur_tokens.size(2), dur_tokens.size(3))
            video = video.reshape(-1, video.size(2), video.size(3), video.size(4), video.size(5))

            print(pitch_tokens.shape, dur_tokens.shape, video.shape)
            output_pitch, output_dur = model(pitch_tokens, dur_tokens, video)
            
            pitch_target = torch.argmax(pitch_tokens, dim=-1)
            duration_target = torch.argmax(dur_tokens, dim=-1)

            optimizer.zero_grad()

            output_pitch = output_pitch.reshape(-1, output_pitch.size(-1))  # [B*T, pitch_dim]
            output_dur = output_dur.reshape(-1, output_dur.size(-1))        # [B*T, dur_dim]

            pitch_target = pitch_target.reshape(-1)  # [B*T]
            duration_target = duration_target.reshape(-1)

            loss_pitch = pitch_loss_fn(output_pitch, pitch_target)
            loss_dur = duration_loss_fn(output_dur, duration_target)
            loss = loss_pitch + loss_dur
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_tokens += pitch_target.size(0)
            total_pitch_correct += (output_pitch.argmax(dim=-1) == pitch_target).sum().item()
            total_duration_correct += (output_dur.argmax(dim=-1) == duration_target).sum().item()

        acc_pitch = 100.0 * total_pitch_correct / total_tokens
        acc_dur = 100.0 * total_duration_correct / total_tokens
        
        return acc_pitch, acc_dur, total_loss

    def _to_device(self, *args):
        return [arg.to(CONSTANTS.DEVICE) for arg in args]
