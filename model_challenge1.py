import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Challenge 1: Cross-Task Transfer Learning Model
    
    Predicts behavioral performance metrics from Contrast Change Detection (CCD) 
    using EEG data from Surround Suppression (SuS) paradigm.
    
    Objectives:
    - Response time prediction (regression)
    - Success rate prediction (classification)
    
    This model leverages cross-task transfer learning from passive (SuS) to active (CCD) paradigms.
    """
    
    def __init__(self):
        super().__init__()
        
        # Feature extraction backbone (can be pretrained on multiple paradigms)
        self.feature_extractor = nn.Sequential(
            # Multi-scale temporal convolutions for EEG processing
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
            nn.Flatten()
        )
        
        # Shared representation layer for cross-task transfer
        self.shared_layer = nn.Sequential(
            nn.Linear(512 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads for multi-objective learning
        # Response time regression head
        self.response_time_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single output for response time
        )
        
        # Success rate classification head  
        self.success_rate_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary classification (success/failure)
        )
        
        # Load pre-trained weights
        self._load_weights()
    
    def _load_weights(self):
        """Load pre-trained model weights"""
        try:
            state_dict = torch.load("weights_challenge1.pt", map_location='cpu')
            self.load_state_dict(state_dict)
            print("Challenge 1 model weights loaded successfully")
        except FileNotFoundError:
            print("No pre-trained weights found for Challenge 1. Using random initialization.")
        except Exception as e:
            print(f"Error loading Challenge 1 weights: {e}")
        
        self.eval()
    
    def forward(self, x):
        """
        Forward pass for Challenge 1
        
        Args:
            x: Input EEG data tensor from SuS paradigm
               Shape: (batch_size, channels=64, time_points)
            
        Returns:
            dict: {
                'response_time': predicted response time (regression),
                'success_rate': predicted success probability (classification)
            }
        """
        # Extract cross-task transferable features
        features = self.feature_extractor(x)
        shared_repr = self.shared_layer(features)
        
        # Task-specific predictions
        response_time = self.response_time_head(shared_repr)
        success_logits = self.success_rate_head(shared_repr)
        
        return {
            'response_time': response_time,
            'success_rate': torch.softmax(success_logits, dim=-1)
        }