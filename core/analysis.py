"""
analysis.py - Brain Simulation Analysis and Biomarker Extraction

Analyzes brain simulation results to extract:
- Network connectivity metrics (graph theory)
- Temporal dynamics metrics (synchrony, oscillations)
- Simulated neuroimaging readouts (EEG-like, fMRI-like signals)
- Vulnerability and risk indices
- Comparative analysis between conditions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform


class BrainActivityAnalyzer:
    """
    Analyzes brain simulation results and extracts biomarkers.
    """

    def __init__(self, simulation_results: Dict):
        """
        Initialize analyzer with simulation results.

        Args:
            simulation_results: Results from simulator.run_simulation()
        """
        self.results = simulation_results
        self.time = simulation_results['time']
        self.activity_E = simulation_results['E']  # Excitatory activity
        self.activity_I = simulation_results.get('I')  # Inhibitory (optional)
        self.region_labels = simulation_results['region_labels']
        self.num_regions = simulation_results['num_regions']
        self.connectivity = simulation_results.get('connectivity')

        # Computed metrics (cached)
        self._metrics_cache = {}

    # ========== NETWORK METRICS ==========

    def compute_network_metrics(self) -> Dict:
        """
        Compute graph-theory network metrics.

        Returns:
            Dictionary of network metrics
        """
        if self.connectivity is None:
            return {"error": "No connectivity data available"}

        print("Computing network metrics...")

        # Binarize connectivity for topological measures
        binary_conn = (self.connectivity > 0).astype(int)
        weighted_conn = self.connectivity.copy()

        metrics = {
            'density': self._network_density(binary_conn),
            'clustering_coefficient': self._clustering_coefficient(binary_conn),
            'path_length': self._characteristic_path_length(binary_conn),
            'modularity': self._modularity_estimate(weighted_conn),
            'hub_regions': self._identify_hubs(weighted_conn),
            'degree_distribution': self._degree_distribution(binary_conn)
        }

        self._metrics_cache['network'] = metrics
        print("✓ Network metrics computed")
        return metrics

    def _network_density(self, binary_conn: np.ndarray) -> float:
        """Network connection density."""
        n = binary_conn.shape[0]
        return np.sum(binary_conn) / (n * (n - 1))

    def _clustering_coefficient(self, binary_conn: np.ndarray) -> float:
        """Average clustering coefficient (local interconnectedness)."""
        n = binary_conn.shape[0]
        clustering = np.zeros(n)

        for i in range(n):
            neighbors = np.where(binary_conn[i, :] > 0)[0]
            k = len(neighbors)

            if k >= 2:
                # Count connections among neighbors
                subgraph = binary_conn[np.ix_(neighbors, neighbors)]
                actual_edges = np.sum(subgraph) / 2
                possible_edges = k * (k - 1) / 2
                clustering[i] = actual_edges / possible_edges

        return float(np.mean(clustering))

    def _characteristic_path_length(self, binary_conn: np.ndarray) -> float:
        """Average shortest path length (global integration)."""
        n = binary_conn.shape[0]

        # Floyd-Warshall algorithm for all-pairs shortest paths
        dist = np.where(binary_conn > 0, 1, np.inf)
        np.fill_diagonal(dist, 0)

        for k in range(n):
            dist = np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])

        # Average path length (excluding infinite distances)
        finite_dist = dist[np.isfinite(dist) & (dist > 0)]
        if len(finite_dist) > 0:
            return float(np.mean(finite_dist))
        else:
            return np.inf

    def _modularity_estimate(self, weighted_conn: np.ndarray) -> float:
        """
        Estimate network modularity (community structure).
        Simplified version - just measures connectivity variance.
        """
        # Real modularity requires community detection - this is a proxy
        degree = np.sum(weighted_conn, axis=1)
        variance = np.std(degree) / (np.mean(degree) + 1e-10)
        return float(variance)

    def _identify_hubs(self, weighted_conn: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Identify hub regions (highly connected).

        Args:
            top_k: Number of top hubs to return

        Returns:
            List of hub dictionaries with region info
        """
        # Node strength (sum of connection weights)
        strength = np.sum(weighted_conn, axis=1)

        # Get top hubs
        hub_indices = np.argsort(strength)[-top_k:][::-1]

        hubs = []
        for idx in hub_indices:
            hubs.append({
                'region': self.region_labels[idx],
                'index': int(idx),
                'strength': float(strength[idx]),
                'degree': int(np.sum(weighted_conn[idx, :] > 0))
            })

        return hubs

    def _degree_distribution(self, binary_conn: np.ndarray) -> Dict:
        """Compute degree distribution statistics."""
        degree = np.sum(binary_conn, axis=1)

        return {
            'mean': float(np.mean(degree)),
            'std': float(np.std(degree)),
            'min': int(np.min(degree)),
            'max': int(np.max(degree))
        }

    # ========== TEMPORAL DYNAMICS METRICS ==========

    def compute_temporal_metrics(self) -> Dict:
        """
        Compute metrics from activity time series.

        Returns:
            Dictionary of temporal metrics
        """
        print("Computing temporal dynamics metrics...")

        # Sampling rate
        dt = np.mean(np.diff(self.time))
        fs = 1000.0 / dt  # Hz (time is in ms)

        metrics = {
            'mean_activity': float(np.mean(self.activity_E)),
            'std_activity': float(np.std(self.activity_E)),
            'global_synchrony': self._global_synchrony(),
            'functional_connectivity': self._functional_connectivity(),
            'dominant_frequency': self._dominant_frequency(fs),
            'metastability': self._metastability(),
            'variance_explained': self._variance_explained()
        }

        self._metrics_cache['temporal'] = metrics
        print("✓ Temporal metrics computed")
        return metrics

    def _global_synchrony(self) -> float:
        """
        Compute global synchrony (correlation across regions).
        """
        # Average pairwise correlation
        correlations = []

        for i in range(self.num_regions):
            for j in range(i + 1, self.num_regions):
                corr = np.corrcoef(self.activity_E[:, i], self.activity_E[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 0.0

    def _functional_connectivity(self) -> np.ndarray:
        """
        Compute functional connectivity matrix (correlation-based).

        Returns:
            Correlation matrix (num_regions x num_regions)
        """
        fc = np.corrcoef(self.activity_E.T)
        np.fill_diagonal(fc, 0)
        return fc

    def _dominant_frequency(self, fs: float) -> Dict:
        """
        Find dominant oscillation frequency using FFT.

        Args:
            fs: Sampling frequency (Hz)

        Returns:
            Dictionary with frequency info
        """
        # Average signal across regions
        global_signal = np.mean(self.activity_E, axis=1)

        # Compute power spectrum
        freqs, psd = signal.welch(global_signal, fs=fs, nperseg=min(256, len(global_signal)))

        # Find peak in physiological range (1-50 Hz)
        mask = (freqs >= 1) & (freqs <= 50)
        if np.any(mask):
            peak_idx = np.argmax(psd[mask])
            peak_freq = freqs[mask][peak_idx]
            peak_power = psd[mask][peak_idx]
        else:
            peak_freq = 0
            peak_power = 0

        return {
            'frequency_hz': float(peak_freq),
            'power': float(peak_power),
            'band': self._classify_frequency_band(peak_freq)
        }

    def _classify_frequency_band(self, freq: float) -> str:
        """Classify frequency into EEG bands."""
        if freq < 4:
            return 'delta'
        elif freq < 8:
            return 'theta'
        elif freq < 13:
            return 'alpha'
        elif freq < 30:
            return 'beta'
        else:
            return 'gamma'

    def _metastability(self) -> float:
        """
        Compute metastability (variability of synchrony over time).
        Measures how much synchrony fluctuates.
        """
        # Compute instantaneous synchrony (sliding window)
        window_size = min(100, len(self.time) // 10)
        synchrony_over_time = []

        for i in range(0, len(self.time) - window_size, window_size // 2):
            window_data = self.activity_E[i:i+window_size, :]
            # Correlation within window
            corr_mat = np.corrcoef(window_data.T)
            mean_corr = np.mean(corr_mat[np.triu_indices_from(corr_mat, k=1)])
            synchrony_over_time.append(mean_corr)

        # Metastability = std of synchrony
        return float(np.std(synchrony_over_time))

    def _variance_explained(self) -> float:
        """
        Variance explained by first principal component (dimensionality).
        """
        # Center data
        data_centered = self.activity_E - np.mean(self.activity_E, axis=0)

        # SVD
        U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)

        # Variance explained by PC1
        variance_explained = (S[0]**2) / np.sum(S**2)

        return float(variance_explained)

    # ========== SIMULATED READOUTS ==========

    def generate_simulated_eeg(self, sensor_locations: Optional[np.ndarray] = None) -> Dict:
        """
        Generate simulated EEG-like signal from region activity.

        Uses a simplified forward model: EEG = weighted sum of region activities.

        Args:
            sensor_locations: Optional electrode positions (if None, uses default)

        Returns:
            Dictionary with simulated EEG data
        """
        print("Generating simulated EEG...")

        # Simplified: average activity with spatial weighting
        # In real TVB: uses lead field matrix from source reconstruction

        num_channels = 64  # Standard EEG montage

        # Create random spatial weights (in real case: from head model)
        if sensor_locations is None:
            weights = np.random.randn(num_channels, self.num_regions)
            # Normalize
            weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
        else:
            # Use provided locations
            weights = sensor_locations

        # Generate EEG: linear combination of source activities
        eeg_signals = self.activity_E @ weights.T

        # Add small measurement noise
        eeg_signals += np.random.randn(*eeg_signals.shape) * 0.01

        print(f"✓ Generated {num_channels}-channel EEG")

        return {
            'signals': eeg_signals,
            'time': self.time,
            'num_channels': num_channels,
            'sampling_rate': 1000.0 / np.mean(np.diff(self.time))
        }

    def generate_simulated_fmri(self, tr: float = 2000.0) -> Dict:
        """
        Generate simulated fMRI BOLD signal from neural activity.

        Uses simplified hemodynamic response (not full Balloon model).

        Args:
            tr: Repetition time in ms (standard: 2000ms)

        Returns:
            Dictionary with simulated BOLD data
        """
        print(f"Generating simulated fMRI (TR={tr}ms)...")

        # Simplified hemodynamic response function (HRF)
        # Real BOLD: convolution with canonical HRF

        # Downsample to TR
        tr_steps = int(tr / np.mean(np.diff(self.time)))
        bold_signal = self.activity_E[::tr_steps, :]
        bold_time = self.time[::tr_steps]

        # Apply smoothing (hemodynamic lag)
        from scipy.ndimage import gaussian_filter1d
        bold_signal = gaussian_filter1d(bold_signal, sigma=2, axis=0)

        # Add noise
        bold_signal += np.random.randn(*bold_signal.shape) * 0.05

        print(f"✓ Generated BOLD signal: {bold_signal.shape[0]} timepoints")

        return {
            'bold': bold_signal,
            'time': bold_time,
            'tr': tr,
            'num_regions': self.num_regions
        }

    # ========== VULNERABILITY & RISK ==========

    def compute_vulnerability_map(self) -> Dict:
        """
        Compute vulnerability index for each region.

        Vulnerability combines:
        - Network centrality (hubs are vulnerable)
        - Activity level (overactive regions)
        - Variability (unstable regions)

        Returns:
            Dictionary with vulnerability scores per region
        """
        print("Computing vulnerability map...")

        # Network centrality (if connectivity available)
        if self.connectivity is not None:
            strength = np.sum(self.connectivity, axis=1)
            centrality = strength / (np.max(strength) + 1e-10)
        else:
            centrality = np.ones(self.num_regions)

        # Activity level
        mean_activity = np.mean(self.activity_E, axis=0)
        activity_norm = mean_activity / (np.max(mean_activity) + 1e-10)

        # Variability
        std_activity = np.std(self.activity_E, axis=0)
        variability_norm = std_activity / (np.max(std_activity) + 1e-10)

        # Combined vulnerability score
        vulnerability = (0.4 * centrality +
                        0.3 * activity_norm +
                        0.3 * variability_norm)

        # Sort regions by vulnerability
        sorted_indices = np.argsort(vulnerability)[::-1]

        vulnerable_regions = []
        for idx in sorted_indices[:10]:  # Top 10
            vulnerable_regions.append({
                'region': self.region_labels[idx],
                'index': int(idx),
                'score': float(vulnerability[idx]),
                'centrality': float(centrality[idx]),
                'activity': float(activity_norm[idx]),
                'variability': float(variability_norm[idx])
            })

        print("✓ Vulnerability map computed")

        return {
            'scores': vulnerability,
            'top_vulnerable': vulnerable_regions
        }

    # ========== COMPARISON TOOLS ==========

    def compare_to_baseline(self, baseline_results: Dict) -> Dict:
        """
        Compare current results to baseline simulation.

        Args:
            baseline_results: Results from baseline simulation

        Returns:
            Dictionary with comparison metrics
        """
        print("Comparing to baseline...")

        baseline_E = baseline_results['E']

        # Activity differences
        activity_diff = np.mean(self.activity_E) - np.mean(baseline_E)
        activity_diff_pct = (activity_diff / np.mean(baseline_E)) * 100

        # Region-wise differences
        region_diff = np.mean(self.activity_E, axis=0) - np.mean(baseline_E, axis=0)

        # Most affected regions
        most_affected_idx = np.argsort(np.abs(region_diff))[::-1][:5]
        most_affected = [
            {
                'region': self.region_labels[idx],
                'baseline': float(np.mean(baseline_E[:, idx])),
                'current': float(np.mean(self.activity_E[:, idx])),
                'change': float(region_diff[idx]),
                'change_pct': float((region_diff[idx] / np.mean(baseline_E[:, idx])) * 100)
            }
            for idx in most_affected_idx
        ]

        # Synchrony change
        baseline_analyzer = BrainActivityAnalyzer(baseline_results)
        baseline_sync = baseline_analyzer._global_synchrony()
        current_sync = self._global_synchrony()
        sync_change = current_sync - baseline_sync

        print("✓ Comparison complete")

        return {
            'mean_activity_change': float(activity_diff),
            'mean_activity_change_pct': float(activity_diff_pct),
            'synchrony_change': float(sync_change),
            'most_affected_regions': most_affected
        }

    def compare_to_empirical(self, empirical_fc: np.ndarray) -> Dict:
        """
        Compare simulated FC to empirical (resting-state) FC.

        Args:
            empirical_fc: Observed functional connectivity matrix (n_regions x n_regions)

        Returns:
            Dictionary with similarity metrics
        """
        print("Comparing to empirical data...")

        if empirical_fc.shape != (self.num_regions, self.num_regions):
            print(f"⚠️ Warning: Empirical FC shape {empirical_fc.shape} matches region count {self.num_regions}")
            # Try to resize/trim if needed (omitted for safety)

        simulated_fc = self._functional_connectivity()

        # Extract upper triangles (excluding diagonal)
        sim_vals = simulated_fc[np.triu_indices_from(simulated_fc, k=1)]
        emp_vals = empirical_fc[np.triu_indices_from(empirical_fc, k=1)]

        # Handle NaNs
        valid_mask = np.isfinite(sim_vals) & np.isfinite(emp_vals)
        if not np.any(valid_mask):
            return {'correlation': 0.0, 'euclidean_distance': np.inf}
            
        sim_vals = sim_vals[valid_mask]
        emp_vals = emp_vals[valid_mask]

        # Pearson correlation
        correlation, _ = stats.pearsonr(sim_vals, emp_vals)
        
        # Euclidean distance (normalized)
        distance = np.linalg.norm(sim_vals - emp_vals) / np.sqrt(len(sim_vals))

        print(f"✓ Measured similarity: r={correlation:.3f}")

        return {
            'correlation': float(correlation),
            'euclidean_distance': float(distance),
            'simulated_fc': simulated_fc,  # Return for plotting
            'empirical_fc': empirical_fc
        }

    def compute_power_spectra(self, region_idx: Optional[int] = None) -> Dict:
        """
        Compute Power Spectral Density (PSD) using Welch's method.

        Args:
            region_idx: Optional index to compute for a specific region. 
                        If None, computes global mean field.

        Returns:
            Dictionary with 'freqs', 'psd', and band powers
        """
        print("Computing power spectra...")
        
        dt = np.mean(np.diff(self.time))  # ms
        fs = 1000.0 / dt  # Hz

        if region_idx is not None:
            # Single region
            signal_data = self.activity_E[:, region_idx]
        else:
            # Global mean field
            signal_data = np.mean(self.activity_E, axis=1)

        # Welch's method
        # Use 2-second window for good low-frequency resolution
        nperseg = int(2.0 * fs)
        if nperseg > len(signal_data):
            nperseg = len(signal_data)
        
        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg)

        # Extract band powers
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 80)
        }
        
        band_powers = {}
        total_power = np.sum(psd)
        
        for band, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                band_powers[band] = np.sum(psd[mask]) / total_power
            else:
                band_powers[band] = 0.0

        return {
            'freqs': freqs,
            'psd': psd,
            'band_powers': band_powers,
            'peak_freq': freqs[np.argmax(psd)]
        }

    # ========== REPORT GENERATION ==========

    def generate_report(self) -> str:
        """
        Generate a comprehensive text report of all metrics.

        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 70)
        report.append("BRAIN SIMULATION ANALYSIS REPORT")
        report.append("=" * 70)

        # Basic info
        report.append(f"\nSimulation Info:")
        report.append(f"  Duration: {self.time[-1] - self.time[0]:.1f} ms")
        report.append(f"  Regions: {self.num_regions}")
        report.append(f"  Timepoints: {len(self.time)}")

        # Temporal metrics
        if 'temporal' not in self._metrics_cache:
            self.compute_temporal_metrics()

        temporal = self._metrics_cache['temporal']
        report.append(f"\nTemporal Dynamics:")
        report.append(f"  Mean activity: {temporal['mean_activity']:.3f}")
        report.append(f"  Activity variability: {temporal['std_activity']:.3f}")
        report.append(f"  Global synchrony: {temporal['global_synchrony']:.3f}")
        report.append(f"  Metastability: {temporal['metastability']:.3f}")
        report.append(f"  Dominant frequency: {temporal['dominant_frequency']['frequency_hz']:.1f} Hz ({temporal['dominant_frequency']['band']} band)")

        # Network metrics (if available)
        if self.connectivity is not None:
            if 'network' not in self._metrics_cache:
                self.compute_network_metrics()

            network = self._metrics_cache['network']
            report.append(f"\nNetwork Structure:")
            report.append(f"  Density: {network['density']:.3f}")
            report.append(f"  Clustering: {network['clustering_coefficient']:.3f}")
            report.append(f"  Path length: {network['path_length']:.2f}")

            report.append(f"\n  Top Hub Regions:")
            for hub in network['hub_regions'][:5]:
                report.append(f"    - {hub['region']}: strength={hub['strength']:.2f}")

        # Vulnerability
        vulnerability = self.compute_vulnerability_map()
        report.append(f"\n  Most Vulnerable Regions:")
        for v in vulnerability['top_vulnerable'][:5]:
            report.append(f"    - {v['region']}: score={v['score']:.3f}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


# ========== Convenience Functions ==========

def analyze_simulation(results: Dict, save_report: bool = False) -> Dict:
    """
    Quick function to analyze simulation results.

    Args:
        results: Simulation results
        save_report: Whether to print report

    Returns:
        Dictionary with all metrics
    """
    analyzer = BrainActivityAnalyzer(results)

    metrics = {
        'temporal': analyzer.compute_temporal_metrics(),
        'vulnerability': analyzer.compute_vulnerability_map()
    }

    if results.get('connectivity') is not None:
        metrics['network'] = analyzer.compute_network_metrics()

    if save_report:
        print(analyzer.generate_report())

    return metrics


if __name__ == "__main__":
    # Demo: Run simulation and analyze
    print("=" * 60)
    print("Brain Analysis Module - Demo")
    print("=" * 60)

    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from core.data_loader import create_default_brain
    from core.simulator_fast import BrainNetworkSimulator

    # Create brain and run simulation
    brain_data = create_default_brain(num_regions=68)
    sim = BrainNetworkSimulator(brain_data)
    results = sim.run_simulation()

    # Analyze
    analyzer = BrainActivityAnalyzer(results)

    # Generate full report
    report = analyzer.generate_report()
    print(report)

    # Generate simulated readouts
    print("\nGenerating simulated readouts...")
    eeg = analyzer.generate_simulated_eeg()
    print(f"  EEG shape: {eeg['signals'].shape}")

    fmri = analyzer.generate_simulated_fmri()
    print(f"  fMRI BOLD shape: {fmri['bold'].shape}")

    print("\n" + "=" * 60)
    print("✓ Demo complete!")
