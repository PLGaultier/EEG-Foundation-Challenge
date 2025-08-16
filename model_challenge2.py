import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Challenge 2: Psychopathology Factor Prediction (Subject Invariant Representation)
    
    Predicts four continuous psychopathology scores from EEG recordings
    across multiple experimental paradigms with subject invariance.
    
    Objectives:
    - p-factor prediction (regression)
    - internalizing prediction (regression) 
    - externalizing prediction (regression)
    - attention prediction (regression)
    
    This model focuses on creating robust subject-invariant representations.
    """
    
    def __init__(self):
        super().__init__()
        
        # Foundation model backbone for subject-invariant representations
        self.feature_extractor = nn.Sequential(
            # Multi-scale temporal convolutions for robust EEG feature extraction
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(256),
            nn.Flatten()
        )
        
        # Subject invariant representation layer with domain adaptation
        self.subject_invariant_layer = nn.Sequential(
            nn.Linear(1024 * 256, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Psychopathology prediction head with regularization
        self.psychopathology_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)  # 4 CBCL dimensions: p-factor, internalizing, externalizing, attention
        )
        
        # Load pre-trained weights
        self._load_weights()
    
    def _load_weights(self):
        """Load pre-trained model weights"""
        try:
            state_dict = torch.load("weights_challenge2.pt", map_location='cpu')
            self.load_state_dict(state_dict)
            print("Challenge 2 model weights loaded successfully")
        except FileNotFoundError:
            print("No pre-trained weights found for Challenge 2. Using random initialization.")
        except Exception as e:
            print(f"Error loading Challenge 2 weights: {e}")
        
        self.eval()
    
    def forward(self, x):
        """
        Forward pass for Challenge 2
        
        Args:
            x: Input EEG data tensor from multiple paradigms (RS, SuS, MW, CCD, SL, SyS)
               Shape: (batch_size, channels=64, time_points)
            
        Returns:
            torch.Tensor: Predictions for 4 psychopathology dimensions
                         Shape: (batch_size, 4)
                         [p-factor, internalizing, externalizing, attention]
        """
        # Extract subject-invariant features across paradigms
        features = self.feature_extractor(x)
        subject_invariant_repr = self.subject_invariant_layer(features)
        
        # Predict psychopathology dimensions
        psychopathology_scores = self.psychopathology_head(subject_invariant_repr)
        
        return psychopathology_scores