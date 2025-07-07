import torch
import torch.nn as nn
import torchvision.models as models

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        pe = self.pe[:, :x.size(1)].to(x.device)
        return x + pe

class CNNVideoFeatureExtractor(nn.Module):
    def __init__(self, output_dim):
        super(CNNVideoFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.shape
        x = x.reshape(batch_size * num_frames, channels, height, width)
        features = self.cnn(x)
        features = features.reshape(features.size(0), -1)
        features = self.fc(features)
        features = features.reshape(batch_size, num_frames, -1)
        return features

class PretrainedVideoFeatureExtractor(nn.Module):
    def __init__(self, video_out_dim):
        super(PretrainedVideoFeatureExtractor, self).__init__()

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        
        self.fc = nn.Linear(resnet.fc.in_features, video_out_dim)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.shape
        x = x.reshape(batch_size * num_frames, channels, height, width)
        
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        features = self.fc(features)
        
        features = features.view(batch_size, num_frames, -1)
        
        return features

class ChordGeneratorTransformer(nn.Module):
    def __init__(self, chord_dim, video_out_dim, num_encoder_layers=8, num_decoder_layers=16, nhead=8, d_model=128, dim_feedforward=2048, seq_len=24):
        super(ChordGeneratorTransformer, self).__init__()
        
        self.chord_mask = generate_square_subsequent_mask(seq_len)
        self.chord_embedding = nn.Embedding(chord_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.video_feature_extractor = PretrainedVideoFeatureExtractor(video_out_dim)
        
        self.video_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )
        
        self.chord_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_decoder_layers
        )
        
        self.output_layer = nn.Linear(d_model, chord_dim)
        
    def forward(self, chord_sequence_indices, video_frames):
        chord_sequence_indices = torch.argmax(chord_sequence_indices, dim=-1)
        chord_embedded = self.chord_embedding(chord_sequence_indices)
        chord_embedded = self.positional_encoding(chord_embedded)
        
        extracted_video_features = self.video_feature_extractor(video_frames.float())
        video_features = self.positional_encoding(extracted_video_features)
        video_encoded = self.video_encoder(video_features.permute(1, 0, 2))
        # batch_size, seq_len, d_model = chord_embedded.size()
        # dummy_memory = torch.zeros((seq_len, batch_size, d_model), device=chord_embedded.device)

        output = self.chord_decoder(
            chord_embedded.permute(1, 0, 2), 
            video_encoded, # or dummy_memory
            tgt_mask=self.chord_mask
        )
        
        return self.output_layer(output.permute(1, 0, 2))

class MelodyGeneratorTransformer(nn.Module):
    def __init__(self, melody_dim, chord_context_dim, video_out_dim, num_encoder_layers=8, num_decoder_layers=16, nhead=8, d_model=128, dim_feedforward=2048, seq_len=24):
        super(MelodyGeneratorTransformer, self).__init__()
        
        self.melody_mask = generate_square_subsequent_mask(seq_len)

        # Separate embeddings for melody and chords
        self.melody_embedding = nn.Embedding(melody_dim, d_model)
        self.chord_context_embedding = nn.Embedding(chord_context_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.video_feature_extractor = PretrainedVideoFeatureExtractor(video_out_dim)
        
        # Chord context encoder
        self.chord_context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )
        
        # # Video feature transformer encoder
        self.video_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )
        
        # Melody transformer decoder with dual cross-attention
        self.melody_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_decoder_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, melody_dim)
        
    def forward(self, melody_sequence, chord_context_sequence, video_frames):
        # Melody embedding and positional encoding
        melody_sequence_indices = torch.argmax(melody_sequence, dim=-1)
        melody_embedded = self.melody_embedding(melody_sequence_indices)
        melody_embedded = self.positional_encoding(melody_embedded)
        
        # Chord encoding
        chord_context_sequence_indices = torch.argmax(chord_context_sequence, dim=-1)
        chord_context_embedded = self.chord_context_embedding(chord_context_sequence_indices)
        chord_context_embedded = self.positional_encoding(chord_context_embedded)
        
        chord_context_encoded = self.chord_context_encoder(chord_context_embedded.permute(1, 0, 2))
        
        # Video encoding
        extracted_video_features = self.video_feature_extractor(video_frames.float())
        video_features = self.positional_encoding(extracted_video_features)
        video_encoded = self.video_encoder(video_features.permute(1, 0, 2))
        
        # Melody decoding with sequential cross-attention
        melody_decoded = self.melody_decoder(
            tgt=melody_embedded.permute(1, 0, 2),
            memory=video_encoded,
            tgt_mask=self.melody_mask
        )
        
        melody_output = self.melody_decoder(
            tgt=melody_decoded.permute(1, 0, 2),
            memory=chord_context_encoded,
            tgt_mask=self.melody_mask
        )
        
        # Final output layer
        return self.output_layer(melody_output.permute(1, 0, 2))

# Utility to generate causal mask for chords and melody
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
