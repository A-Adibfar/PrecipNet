import torch
import torch.nn as nn


class PrecipTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.feature_embeddings = nn.Parameter(torch.randn(input_dim, d_model))
        self.feature_bias = nn.Parameter(torch.randn(input_dim, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.token_weights = nn.Linear(d_model, 1)
        self.fc = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x * self.feature_embeddings + self.feature_bias
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)

        weights = self.token_weights(x)
        weights = torch.softmax(weights, dim=1)
        x = (x * weights).sum(dim=1)

        return self.fc(x)
