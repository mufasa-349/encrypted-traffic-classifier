"""
1D-CNN model for binary classification of network traffic.
"""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    Simple 1D-CNN for binary classification.
    Treats feature vector as a 1D signal.
    """
    
    def __init__(self, num_features: int, num_classes: int = 1):
        super(CNN1D, self).__init__()
        
        # Input shape: (batch, 1, num_features)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch, num_features) -> (batch, 1, num_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # Global pooling: (batch, 128, seq_len) -> (batch, 128, 1)
        x = self.global_pool(x)
        x = x.squeeze(-1)  # (batch, 128)
        
        # FC layers
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_model(num_features: int, device: str = 'cuda') -> CNN1D:
    """Create and initialize the model."""
    model = CNN1D(num_features=num_features, num_classes=1)
    model = model.to(device)
    return model

