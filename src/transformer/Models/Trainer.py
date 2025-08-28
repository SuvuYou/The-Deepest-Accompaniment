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
        
        pitch_loss_fn = torch.nn.CrossEntropyLoss().to(CONSTANTS.DEVICE)
        duration_loss_fn = torch.nn.CrossEntropyLoss().to(CONSTANTS.DEVICE)
        optimizer = torch.optim.Adam(self.chords_model.parameters(), lr=self.model_settings["LR"])
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
            self.model_weights_manager.save_chords_model_weights()

    def _train_epoch_model(self, model, data_loader, pitch_loss_fn, duration_loss_fn, optimizer):
        model.train()
        total_pitch_correct = total_duration_correct = total_tokens = 0
        total_loss = 0.0
        
        # Track per-class stats
        pitch_preds_hist = []
        pitch_targets_hist = []
        dur_preds_hist = []
        dur_targets_hist = []
        
        for batch_idx, batch in enumerate(data_loader):
            print(f"Processing batch {batch_idx}")
            
            pitch_tokens, dur_tokens, video = self._to_device(*batch)
            
            # Flatten hierarchical structure
            pitch_tokens = pitch_tokens.reshape(-1, pitch_tokens.size(2), pitch_tokens.size(3))
            dur_tokens = dur_tokens.reshape(-1, dur_tokens.size(2), dur_tokens.size(3))
            video = video.reshape(-1, video.size(2), video.size(3), video.size(4), video.size(5))
            
            pitch_tokens = torch.argmax(pitch_tokens, -1)
            dur_tokens = torch.argmax(dur_tokens, -1)

            input_pitch_seq = pitch_tokens[:, :-1]    # [B, T-1]
            target_pitch_seq = pitch_tokens[:, 1:]    # [B, T-1]

            input_duration_seq = dur_tokens[:, :-1]    # [B, T-1]
            target_duration_seq = dur_tokens[:, 1:]    # [B, T-1]

            output_pitch, output_duration = model(input_pitch_seq, input_duration_seq)     # [B, T-1, dim]
            output_pitch = output_pitch.reshape(-1, output_pitch.size(-1))   # [B*(T-1), pitch_dim]
            output_duration = output_duration.reshape(-1, output_duration.size(-1))   # [B*(T-1), dur_dim]

            target_pitch_seq = target_pitch_seq.reshape(-1)    # [B*(T-1)]
            target_duration_seq = target_duration_seq.reshape(-1)

            optimizer.zero_grad()
            loss_pitch = pitch_loss_fn(output_pitch, target_pitch_seq)
            loss_dur   = duration_loss_fn(output_duration, target_duration_seq)
            loss = loss_pitch + loss_dur

            print(f"Pitch loss: {loss_pitch.item():.4f}, Duration loss: {loss_dur.item():.4f}, Total loss: {loss.item():.4f}")
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy for pitch
            preds_pitch = output_pitch.argmax(dim=-1)
            total_pitch_correct += (preds_pitch == target_pitch_seq).sum().item()
            total_tokens += target_pitch_seq.size(0)

            # Accuracy for duration
            preds_dur = output_duration.argmax(dim=-1)
            total_duration_correct += (preds_dur == target_duration_seq).sum().item()
            
            # Collect histograms for per-class stats
            pitch_preds_hist.append(preds_pitch.cpu())
            pitch_targets_hist.append(target_pitch_seq.cpu())
            dur_preds_hist.append(preds_dur.cpu())
            dur_targets_hist.append(target_duration_seq.cpu())
            
            if batch_idx % 50 == 0:  # only print every N batches to avoid spam
                n_show = 10
                print("\nSample predictions:")
                for i in range(n_show):
                    print(f" Pitch pred: {preds_pitch[i].item():3d} | target: {target_pitch_seq[i].item():3d} "
                        f" || Duration pred: {preds_dur[i].item():3d} | target: {target_duration_seq[i].item():3d}")
                print()

        # After loop
        acc_pitch = 100.0 * total_pitch_correct / total_tokens if total_tokens > 0 else 0
        acc_dur   = 100.0 * total_duration_correct / total_tokens if total_tokens > 0 else 0
        
        # ---- per-class stats ----
        pitch_preds_hist = torch.cat(pitch_preds_hist)
        pitch_targets_hist = torch.cat(pitch_targets_hist)
        dur_preds_hist = torch.cat(dur_preds_hist)
        dur_targets_hist = torch.cat(dur_targets_hist)

        num_pitch_classes = output_pitch.size(-1)
        num_dur_classes   = output_duration.size(-1)

        print("\nPitch class distribution (pred vs target %):")
        for c in range(num_pitch_classes):
            pred_count   = (pitch_preds_hist == c).sum().item()
            target_count = (pitch_targets_hist == c).sum().item()
            correct      = ((pitch_preds_hist == c) & (pitch_targets_hist == c)).sum().item()

            pred_pct   = 100.0 * pred_count / len(pitch_preds_hist)
            target_pct = 100.0 * target_count / len(pitch_targets_hist)
            acc_class  = 100.0 * correct / target_count if target_count > 0 else 0

            print(f" Class {c:2d}: Pred {pred_pct:6.2f}% | Target {target_pct:6.2f}% | Acc {acc_class:6.2f}%")

        print("\nDuration class distribution (pred vs target %):")
        for c in range(num_dur_classes):
            pred_count   = (dur_preds_hist == c).sum().item()
            target_count = (dur_targets_hist == c).sum().item()
            correct      = ((dur_preds_hist == c) & (dur_targets_hist == c)).sum().item()

            pred_pct   = 100.0 * pred_count / len(dur_preds_hist)
            target_pct = 100.0 * target_count / len(dur_targets_hist)
            acc_class  = 100.0 * correct / target_count if target_count > 0 else 0

            print(f" Class {c:2d}: Pred {pred_pct:6.2f}% | Target {target_pct:6.2f}% | Acc {acc_class:6.2f}%")

        return acc_pitch, acc_dur, total_loss

    def _to_device(self, *args):
        return [arg.to(CONSTANTS.DEVICE) for arg in args]
