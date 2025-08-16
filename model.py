import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    EEG Foundation Challenge Model
    
    This model is designed for cross-task and cross-subject EEG decoding.
    It should handle both passive tasks (RS, SuS, MW) and active tasks (CCD, SL, SyS)
    while predicting psychopathology dimensions from CBCL.
    """
    
    def __init__(self):
        super().__init__()
        
        # TODO: Define your model architecture here
        # Example architecture - replace with your actual model
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),  # Assuming 64 EEG channels
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(256),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)  # 4 CBCL psychopathology dimensions
        )
        
        # Load pre-trained weights if available
        self._load_weights()
    
    def _load_weights(self):
        """Load pre-trained model weights"""
        try:
            # TODO: Update path to your actual weights file
            state_dict = torch.load("weights.pt", map_location='cpu')
            self.load_state_dict(state_dict)
            print("Model weights loaded successfully")
        except FileNotFoundError:
            print("No pre-trained weights found. Using random initialization.")
        except Exception as e:
            print(f"Error loading weights: {e}")
        
        self.eval()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input EEG data tensor
            
        Returns:
            Predictions for psychopathology dimensions
        """
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output