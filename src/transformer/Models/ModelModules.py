import torch
import torch.nn as nn
import torchvision.models as models

# Positional encoding module for transformer input
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Register `pe` as a buffer so it moves with the model

    def forward(self, x):
        pe = self.pe[:, :x.size(1)].to(x.device)
        return x + pe

class PretrainedVideoFeatureExtractor(nn.Module):
    def __init__(self, output_dim):
        super(PretrainedVideoFeatureExtractor, self).__init__()

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1] 
        self.feature_extractor = nn.Sequential(*modules)
        
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.shape
        x = x.reshape(batch_size * num_frames, channels, height, width)
        
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        features = self.fc(features)
        
        features = features.view(batch_size, num_frames, -1)
        return features
    
class ChordGeneratorTransformer(nn.Module):
    def __init__(self, chord_dim, video_out_dim, num_layers=8, num_decoder_layers=16, nhead=8, d_model=128, dim_feedforward=2048, seq_len=24):
        super(ChordGeneratorTransformer, self).__init__()
        
        self.chord_mask = generate_square_subsequent_mask(seq_len)
        
        # Use embedding layer instead of linear for chord input
        self.chord_embedding = nn.Embedding(chord_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Video feature transformer encoder
        self.video_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        
        # Chord transformer decoder
        self.chord_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_decoder_layers
        )
        
        # Video feature extractor CNN
        self.video_feature_extractor = PretrainedVideoFeatureExtractor(video_out_dim)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, chord_dim)
        
    def forward(self, chord_sequence_indices, video_frames):
        # Convert chord indices to embeddings
        chord_sequence_indices = torch.argmax(chord_sequence_indices, dim=-1)

        chord_embedded = self.chord_embedding(chord_sequence_indices)
        chord_embedded = self.positional_encoding(chord_embedded)
        
        # Extract and encode video features
        video_features = self.video_feature_extractor(video_frames)
        video_features = self.positional_encoding(video_features)
        video_encoded = self.video_encoder(video_features.permute(1, 0, 2))
        
        # Decode chords with cross-attention to video features
        output = self.chord_decoder(
            chord_embedded.permute(1, 0, 2), 
            video_encoded,
            tgt_mask=self.chord_mask
        )
        
        # Final output layer to predict next chord
        return self.output_layer(output.permute(1, 0, 2))

# Utility to generate causal mask for chords
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
