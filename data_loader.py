"""
data_loader.py - Brain Data Ingestion and Model Setup

This module handles loading brain connectivity data (structural connectomes),
region definitions, and preparing inputs for the brain network simulator.
Supports both template/generic brain models and personalized data from DTI/MRI.
"""

import numpy as np
import os
from typing import Dict, Optional, Tuple
import json


class BrainDataLoader:
    """
    Handles loading and preparation of brain connectivity data and model parameters.
    """

    def __init__(self):
        self.connectivity_matrix = None
        self.region_labels = None
        self.tract_lengths = None
        self.region_centers = None
        self.num_regions = 0

    def load_generic_connectome(self, num_regions: int = 68) -> Dict:
        """
        Create a generic/template brain connectome for prototyping.
        Uses a simplified Desikan-Killiany parcellation approach.

        Args:
            num_regions: Number of brain regions (default 68 for DK atlas)

        Returns:
            Dictionary containing connectivity data
        """
        print(f"Creating generic brain model with {num_regions} regions...")

        self.num_regions = num_regions

        # More biologically plausible sparse small-world network (~10-15% density)
        self.connectivity_matrix = self._generate_small_world_connectivity(num_regions)

        # Generate region labels (simplified anatomical naming)
        self.region_labels = self._generate_region_labels(num_regions)

        # Generate tract lengths (connection delays) in mm with realistic spread
        self.tract_lengths = self._generate_tract_lengths(self.connectivity_matrix)

        # Generate region centers (3D coordinates in mm, roughly brain-shaped)
        self.region_centers = self._generate_region_centers(num_regions)

        print(f"✓ Generic connectome created:")
        print(f"  - {self.num_regions} regions")
        print(f"  - {np.sum(self.connectivity_matrix > 0)} connections")
        print(f"  - Connection density: {self._calculate_density():.2%}")

        return self._package_connectivity_data()

    def load_from_file(self, connectivity_path: str,
                       labels_path: Optional[str] = None,
                       tract_lengths_path: Optional[str] = None) -> Dict:
        """
        Load connectivity data from files (for personalized/real data).

        Args:
            connectivity_path: Path to connectivity matrix file (npy, csv, or txt)
            labels_path: Optional path to region labels file
            tract_lengths_path: Optional path to tract lengths file

        Returns:
            Dictionary containing connectivity data
        """
        print(f"Loading connectivity from: {connectivity_path}")

        # Load connectivity matrix
        if connectivity_path.endswith('.npy'):
            self.connectivity_matrix = np.load(connectivity_path)
        elif connectivity_path.endswith('.csv'):
            self.connectivity_matrix = np.loadtxt(connectivity_path, delimiter=',')
        else:
            self.connectivity_matrix = np.loadtxt(connectivity_path)

        self.num_regions = self.connectivity_matrix.shape[0]

        # Load or generate region labels
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.region_labels = [line.strip() for line in f.readlines()]
        else:
            self.region_labels = self._generate_region_labels(self.num_regions)

        # Load or generate tract lengths
        if tract_lengths_path and os.path.exists(tract_lengths_path):
            if tract_lengths_path.endswith('.npy'):
                self.tract_lengths = np.load(tract_lengths_path)
            else:
                self.tract_lengths = np.loadtxt(tract_lengths_path)
        else:
            self.tract_lengths = self._generate_tract_lengths(self.connectivity_matrix)

        # Generate region centers
        self.region_centers = self._generate_region_centers(self.num_regions)

        print(f"✓ Loaded connectome:")
        print(f"  - {self.num_regions} regions")
        print(f"  - {np.sum(self.connectivity_matrix > 0)} connections")

        return self._package_connectivity_data()

    def save_connectivity(self, output_dir: str, prefix: str = "brain"):
        """
        Save current connectivity data to files.

        Args:
            output_dir: Directory to save files
            prefix: Prefix for filenames
        """
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, f"{prefix}_connectivity.npy"),
                self.connectivity_matrix)
        np.save(os.path.join(output_dir, f"{prefix}_tract_lengths.npy"),
                self.tract_lengths)
        np.save(os.path.join(output_dir, f"{prefix}_centers.npy"),
                self.region_centers)

        with open(os.path.join(output_dir, f"{prefix}_labels.txt"), 'w') as f:
            f.write('\n'.join(self.region_labels))

        # Save metadata
        metadata = {
            'num_regions': self.num_regions,
            'num_connections': int(np.sum(self.connectivity_matrix > 0)),
            'density': float(self._calculate_density()),
            'region_labels': self.region_labels
        }

        with open(os.path.join(output_dir, f"{prefix}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Saved connectivity data to {output_dir}")

    # ========== Private Helper Methods ==========

    def _generate_small_world_connectivity(self, n: int) -> np.ndarray:
        """
        Generate a sparse small-world-like connectivity matrix.
        Target density ~10-15%, mix of local clusters and a few long-range links.
        """
        # Start with ring lattice (local connections)
        k_neighbors = 4  # fewer neighbors for lower density

        conn = np.zeros((n, n))

        # Create ring structure
        for i in range(n):
            for j in range(1, k_neighbors // 2 + 1):
                neighbor_1 = (i + j) % n
                neighbor_2 = (i - j) % n
                # Connection weights with distance falloff
                dist_scale = 1.0 / (1 + j)
                weight = np.random.uniform(0.6, 1.2) * dist_scale
                conn[i, neighbor_1] = weight
                conn[i, neighbor_2] = weight

        # Add long-range connections (small-world rewiring) with lower probability
        rewiring_prob = 0.08
        for i in range(n):
            for j in range(i + 1, n):
                if conn[i, j] == 0 and np.random.random() < rewiring_prob:
                    weight = np.random.uniform(0.4, 1.0)
                    conn[i, j] = weight
                    conn[j, i] = weight

        # Make symmetric
        conn = (conn + conn.T) / 2

        # Zero diagonal (no self-connections)
        np.fill_diagonal(conn, 0)

        # Normalize to keep weights in [0,1.5]
        if np.max(conn) > 0:
            conn = conn / np.max(conn) * 1.0

        return conn

    def _generate_region_labels(self, n: int) -> list:
        """Generate anatomically-inspired region labels."""
        # Simplified hemispheric organization
        labels = []
        regions_per_hemisphere = n // 2

        # Common brain region prefixes
        area_names = [
            "Frontal", "Parietal", "Temporal", "Occipital",
            "Cingulate", "Insula", "Precuneus", "Motor",
            "Sensory", "Auditory", "Visual", "Prefrontal"
        ]

        for i in range(n):
            hemisphere = "L" if i < regions_per_hemisphere else "R"
            area = area_names[i % len(area_names)]
            sub_idx = (i % regions_per_hemisphere) // len(area_names) + 1
            labels.append(f"{hemisphere}_{area}_{sub_idx}")

        return labels

    def _generate_tract_lengths(self, connectivity: np.ndarray) -> np.ndarray:
        """
        Generate realistic tract lengths (axonal delays) between regions.
        """
        n = connectivity.shape[0]
        tract_lengths = np.zeros_like(connectivity)

        for i in range(n):
            for j in range(i + 1, n):
                if connectivity[i, j] > 0:
                    # Distance depends on topological distance
                    # Same hemisphere: shorter; cross-hemisphere: longer
                    same_hemisphere = (i < n//2 and j < n//2) or (i >= n//2 and j >= n//2)

                    if same_hemisphere:
                        # Intra-hemispheric: log-normal around ~60mm
                        length = float(np.clip(np.random.lognormal(mean=4.1, sigma=0.35), 10, 150))
                    else:
                        # Inter-hemispheric (via corpus callosum): longer
                        length = float(np.clip(np.random.lognormal(mean=4.9, sigma=0.25), 80, 300))

                    tract_lengths[i, j] = length
                    tract_lengths[j, i] = length

        return tract_lengths

    def _generate_region_centers(self, n: int) -> np.ndarray:
        """
        Generate 3D coordinates for region centers (roughly brain-shaped).
        Returns array of shape (n, 3) with [x, y, z] coordinates in mm.
        """
        centers = np.zeros((n, 3))
        regions_per_hemisphere = n // 2

        # Left hemisphere (x < 0)
        for i in range(regions_per_hemisphere):
            angle = 2 * np.pi * i / regions_per_hemisphere
            centers[i] = [
                -40 - 20 * np.cos(angle),  # x: left side
                60 * np.sin(angle),         # y: anterior-posterior
                30 + 30 * np.cos(2 * angle)  # z: inferior-superior
            ]

        # Right hemisphere (x > 0) - mirror left
        for i in range(regions_per_hemisphere, n):
            idx = i - regions_per_hemisphere
            centers[i] = centers[idx].copy()
            centers[i, 0] *= -1  # Mirror x coordinate

        return centers

    def _calculate_density(self) -> float:
        """Calculate network connection density."""
        if self.connectivity_matrix is None:
            return 0.0
        n = self.connectivity_matrix.shape[0]
        possible_connections = n * (n - 1)
        actual_connections = np.sum(self.connectivity_matrix > 0)
        return actual_connections / possible_connections

    def _package_connectivity_data(self) -> Dict:
        """Package all connectivity data into a dictionary."""
        return {
            'weights': self.connectivity_matrix,
            'tract_lengths': self.tract_lengths,
            'region_labels': self.region_labels,
            'centres': self.region_centers,
            'num_regions': self.num_regions
        }

    def get_connectivity_info(self) -> Dict:
        """Get summary information about the loaded connectivity."""
        if self.connectivity_matrix is None:
            return {"error": "No connectivity data loaded"}

        return {
            'num_regions': self.num_regions,
            'num_connections': int(np.sum(self.connectivity_matrix > 0)),
            'density': self._calculate_density(),
            'mean_weight': float(np.mean(self.connectivity_matrix[self.connectivity_matrix > 0])),
            'mean_tract_length': float(np.mean(self.tract_lengths[self.tract_lengths > 0])),
            'regions': self.region_labels[:5] + ['...'] if len(self.region_labels) > 5 else self.region_labels
        }


# ========== Convenience Functions ==========

def create_default_brain(num_regions: int = 68) -> Dict:
    """
    Quick function to create a default generic brain model.

    Args:
        num_regions: Number of brain regions

    Returns:
        Dictionary with connectivity data
    """
    loader = BrainDataLoader()
    return loader.load_generic_connectome(num_regions)


def load_brain_from_files(connectivity_path: str, **kwargs) -> Dict:
    """
    Quick function to load brain data from files.

    Args:
        connectivity_path: Path to connectivity matrix
        **kwargs: Additional paths (labels_path, tract_lengths_path)

    Returns:
        Dictionary with connectivity data
    """
    loader = BrainDataLoader()
    return loader.load_from_file(connectivity_path, **kwargs)


if __name__ == "__main__":
    # Demo: Create and save a generic brain
    print("=" * 60)
    print("Brain Data Loader - Demo")
    print("=" * 60)

    loader = BrainDataLoader()
    brain_data = loader.load_generic_connectome(num_regions=68)

    print("\nConnectivity Info:")
    info = loader.get_connectivity_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Save to data directory
    loader.save_connectivity("brain_data", prefix="generic_68")

    print("\n" + "=" * 60)
    print("✓ Demo complete!")
