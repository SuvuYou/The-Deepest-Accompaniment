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

class DualTokenEmbedding(nn.Module):
    def __init__(self, pitch_vocab_size, duration_vocab_size, d_model):
        super().__init__()
        self.pitch_embed = nn.Embedding(pitch_vocab_size, d_model)
        self.duration_embed = nn.Embedding(duration_vocab_size, d_model)
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

    def forward(self, pitch_tokens, duration_tokens):
        pitch = self.pitch_embed(pitch_tokens)
        duration = self.duration_embed(duration_tokens)
        return self.scale * (pitch + duration)

class PretrainedVideoFeatureExtractor(nn.Module):
    def __init__(self, video_out_dim):
        super(PretrainedVideoFeatureExtractor, self).__init__()

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        
        self.fc = nn.Linear(resnet.fc.in_features, video_out_dim)

    def forward(self, x):
        print(x.shape)
        batch_size, num_frames, height, width, channels = x.shape
        x = x.reshape(batch_size * num_frames, channels, height, width)
        
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        features = self.fc(features)
        
        features = features.view(batch_size, num_frames, -1)
        
        return features

class ChordGeneratorTransformer(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.mask = generate_square_subsequent_mask(settings["seq_len"])

        self.embedding = DualTokenEmbedding(
            settings["chord_pitch_dim"],
            settings["chord_duration_dim"],
            settings["d_model"]
        )

        self.positional_encoding = PositionalEncoding(settings["d_model"], settings["seq_len"])
        self.video_feature_extractor = PretrainedVideoFeatureExtractor(settings["video_out_dim"])

        self.video_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=settings["d_model"],
                nhead=settings["nhead"],
                dim_feedforward=settings["dim_feedforward"]
            ),
            num_layers=settings["num_encoder_layers"]
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=settings["d_model"],
                nhead=settings["nhead"],
                dim_feedforward=settings["dim_feedforward"]
            ),
            num_layers=settings["num_decoder_layers"]
        )

        self.output_pitch = nn.Linear(settings["d_model"], settings["chord_pitch_dim"])
        self.output_duration = nn.Linear(settings["d_model"], settings["chord_duration_dim"])

    def forward(self, pitch_tokens, duration_tokens, video_frames):
        x = self.embedding(torch.argmax(pitch_tokens, -1), torch.argmax(duration_tokens, -1))
        x = self.positional_encoding(x)

        video_features = self.video_feature_extractor(video_frames.float())
        video_encoded = self.video_encoder(self.positional_encoding(video_features).permute(1, 0, 2))

        decoded = self.decoder(x.permute(1, 0, 2), video_encoded, tgt_mask=self.mask)

        pitch_out = self.output_pitch(decoded.permute(1, 0, 2))
        duration_out = self.output_duration(decoded.permute(1, 0, 2))
        return pitch_out, duration_out

class MelodyGeneratorTransformer(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.mask = generate_square_subsequent_mask(settings["seq_len"])

        self.melody_embedding = DualTokenEmbedding(
            settings["melody_pitch_dim"],
            settings["melody_duration_dim"],
            settings["d_model"]
        )
        self.positional_encoding = PositionalEncoding(settings["d_model"], settings["seq_len"])
        self.video_feature_extractor = PretrainedVideoFeatureExtractor(settings["video_out_dim"])

        self.video_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=settings["d_model"],
                nhead=settings["nhead"],
                dim_feedforward=settings["dim_feedforward"]
            ),
            num_layers=settings["num_encoder_layers"]
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=settings["d_model"],
                nhead=settings["nhead"],
                dim_feedforward=settings["dim_feedforward"]
            ),
            num_layers=settings["num_decoder_layers"]
        )

        self.output_pitch = nn.Linear(settings["d_model"], settings["melody_pitch_dim"])
        self.output_duration = nn.Linear(settings["d_model"], settings["melody_duration_dim"])

    def forward(self, melody_pitch, melody_duration, video_frames):
        melody_emb = self.melody_embedding(torch.argmax(melody_pitch, -1), torch.argmax(melody_duration, -1))
        melody_emb = self.positional_encoding(melody_emb)

        video_features = self.video_feature_extractor(video_frames.float())
        video_encoded = self.video_encoder(self.positional_encoding(video_features).permute(1, 0, 2))

        decoded_video = self.decoder(melody_emb.permute(1, 0, 2), video_encoded, tgt_mask=self.mask)

        pitch_out = self.output_pitch(decoded_video)
        duration_out = self.output_duration(decoded_video)
        return pitch_out, duration_out

# Utility to generate causal mask for chords and melody
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
