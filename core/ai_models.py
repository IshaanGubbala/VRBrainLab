
"""
ai_models.py - AI Diagnostics & Prognostics Module

Implements lightweight machine learning models using pure NumPy 
(to avoid heavy dependencies like sklearn/torch).

Models:
1. BrainStateClassifier: Predicts brain state (Healthy, Seizure, etc.)
2. BiomarkerReducer: Autoencoder/PCA for dimensionality reduction
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import os

@dataclass
class BrainState:
    label: str
    confidence: float
    features: Dict[str, float]

class BrainStateClassifier:
    """
    Nearest Centroid Classifier for Brain States.
    Matches current simulation metrics against known prototypes.
    """
    def __init__(self):
        # Prototypes for different states (normalized features)
        # Features: [mean_activity, synchrony, alpha_power, metastability]
        self.centroids = {
            "Healthy (Resting)": np.array([0.2, 0.4, 0.8, 0.3]),
            "Seizure (Ictal)":   np.array([0.9, 0.9, 0.1, 0.1]),
            "Hypoactive (Coma)": np.array([0.05, 0.1, 0.1, 0.0]),
            "Oscillatory (Beta)": np.array([0.4, 0.6, 0.3, 0.4])
        }
        self.feature_names = ["mean", "synchrony", "alpha_power", "metastability"]
        
    def _extract_features(self, metrics: Dict) -> np.ndarray:
        """normalize parameters to 0-1 range roughly"""
        # Heuristic normalization based on typical values
        mean = np.clip(metrics.get('mean_activity', 0) / 1.0, 0, 1)
        sync = np.clip(metrics.get('synchrony', 0) / 1.0, 0, 1)
        alpha = np.clip(metrics.get('alpha_power', 0) / 2.0, 0, 1) # Alpha power can be >1
        meta = np.clip(metrics.get('metastability', 0) / 0.2, 0, 1)
        
        return np.array([mean, sync, alpha, meta])

    def predict(self, metrics: Dict) -> BrainState:
        """Classify the current brain state."""
        features = self._extract_features(metrics)
        
        best_label = "Unknown"
        best_dist = float('inf')
        
        # Find nearest centroid
        distances = {}
        for label, centroid in self.centroids.items():
            dist = np.linalg.norm(features - centroid)
            distances[label] = dist
            if dist < best_dist:
                best_dist = dist
                best_label = label
                
        # Simple confidence metric (1 - normalized distance)
        confidence = max(0.0, 1.0 - best_dist)
        
        return BrainState(
            label=best_label,
            confidence=confidence,
            features={k: float(v) for k,v in zip(self.feature_names, features)}
        )

class BiomarkerReducer:
    """
    PCA-based dimensionality reduction to visualize brain trajectory 
    in 2D latent space ("Principal Modes").
    """
    def __init__(self, n_components=2):
        self.n = n_components
        self.components = None
        self.mean = None
        
    def fit(self, data: np.ndarray):
        """
        Fit PCA to data (Time x Regions).
        """
        # Center data
        self.mean = np.mean(data, axis=0)
        centered = data - self.mean
        
        # SVD
        # U, S, Vt = svd(X)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        
        # Components are rows of Vt
        self.components = vt[:self.n]
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Project data into latent space."""
        if self.components is None:
            return np.zeros((len(data), self.n))
            
        centered = data - self.mean
        return np.dot(centered, self.components.T)

# Global instances
classifier = BrainStateClassifier()
reducer = BiomarkerReducer()
