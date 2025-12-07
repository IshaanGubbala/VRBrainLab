"""
vr_interface.py - VR Communication Layer / API Server

Provides a REST API and WebSocket server for communication between
the Python simulation backend and VR frontend (Unity/Unreal).

Endpoints:
- GET /api/brain/info - Get brain model info
- POST /api/simulation/run - Run simulation
- GET /api/simulation/status - Get simulation status
- GET /api/simulation/data - Get activity data
- POST /api/intervention/lesion - Apply lesion
- POST /api/intervention/stimulate - Apply stimulation
- GET /api/analysis/metrics - Get analysis metrics
- GET /api/analysis/vulnerability - Get vulnerability map
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import threading
import time
from typing import Dict, Optional
import base64
import io

# Import our brain modules
from data_loader import BrainDataLoader, create_default_brain
from simulator import BrainNetworkSimulator, SimulationConfig
from intervention import BrainIntervention
from analysis import BrainActivityAnalyzer


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                           np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class BrainVRServer:
    """
    Flask server for VR-brain simulation communication.
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        """
        Initialize VR server.

        Args:
            host: Server host address
            port: Server port
        """
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for Unity WebGL
        self.app.json_encoder = NumpyEncoder

        self.host = host
        self.port = port

        # Server state
        self.brain_data = None
        self.intervention_manager = None
        self.current_simulation = None
        self.current_results = None
        self.analyzer = None
        self.simulation_running = False
        self.simulation_progress = 0.0

        # Setup routes
        self._setup_routes()

        print(f"VR Brain Server initialized on {host}:{port}")

    def _setup_routes(self):
        """Setup Flask API routes."""

        # ===== BRAIN MODEL ENDPOINTS =====

        @self.app.route('/api/brain/info', methods=['GET'])
        def get_brain_info():
            """Get information about loaded brain model."""
            if self.brain_data is None:
                return jsonify({'error': 'No brain model loaded'}), 404

            info = {
                'num_regions': self.brain_data['num_regions'],
                'region_labels': self.brain_data['region_labels'],
                'num_connections': int(np.sum(self.brain_data['weights'] > 0)),
                'connectivity_shape': self.brain_data['weights'].shape,
                'centers': self.brain_data['centres'].tolist()
            }

            return jsonify(info)

        @self.app.route('/api/brain/load', methods=['POST'])
        def load_brain():
            """Load or create brain model."""
            data = request.json
            num_regions = data.get('num_regions', 68)

            print(f"Loading brain model with {num_regions} regions...")
            self.brain_data = create_default_brain(num_regions)
            self.intervention_manager = BrainIntervention(self.brain_data)

            return jsonify({
                'status': 'success',
                'message': f'Brain loaded with {num_regions} regions'
            })

        @self.app.route('/api/brain/connectivity', methods=['GET'])
        def get_connectivity():
            """Get connectivity matrix."""
            if self.brain_data is None:
                return jsonify({'error': 'No brain model loaded'}), 404

            # Option to get current (with interventions) or original
            mode = request.args.get('mode', 'current')

            if mode == 'original' and self.intervention_manager:
                weights = self.intervention_manager.original_data['weights']
            else:
                weights = self.brain_data['weights']

            return jsonify({
                'weights': weights.tolist(),
                'num_regions': self.brain_data['num_regions']
            })

        # ===== SIMULATION ENDPOINTS =====

        @self.app.route('/api/simulation/run', methods=['POST'])
        def run_simulation():
            """Run brain simulation."""
            if self.brain_data is None:
                return jsonify({'error': 'No brain model loaded'}), 400

            if self.simulation_running:
                return jsonify({'error': 'Simulation already running'}), 400

            # Get parameters
            data = request.json or {}
            duration = data.get('duration', 2000.0)
            global_coupling = data.get('global_coupling', 0.5)
            noise = data.get('noise_strength', 0.01)

            # Create config
            config = SimulationConfig(
                duration=duration,
                global_coupling=global_coupling,
                noise_strength=noise
            )

            # Run in background thread
            def run_sim():
                self.simulation_running = True
                self.simulation_progress = 0.0

                try:
                    print(f"Starting simulation (duration={duration}ms)...")

                    # Use intervention manager's current data if available
                    brain_data = (self.intervention_manager.current_data
                                if self.intervention_manager
                                else self.brain_data)

                    sim = BrainNetworkSimulator(brain_data, config)

                    # Progress callback
                    def progress_callback(pct, step, total):
                        self.simulation_progress = pct

                    self.current_results = sim.run_simulation(
                        progress_callback=progress_callback
                    )

                    # Create analyzer
                    self.analyzer = BrainActivityAnalyzer(self.current_results)

                    print("✓ Simulation complete")

                except Exception as e:
                    print(f"Simulation error: {e}")
                    self.current_results = None

                finally:
                    self.simulation_running = False
                    self.simulation_progress = 100.0

            # Start thread
            thread = threading.Thread(target=run_sim)
            thread.start()

            return jsonify({
                'status': 'started',
                'message': 'Simulation started in background'
            })

        @self.app.route('/api/simulation/status', methods=['GET'])
        def get_simulation_status():
            """Get simulation status."""
            return jsonify({
                'running': self.simulation_running,
                'progress': self.simulation_progress,
                'has_results': self.current_results is not None
            })

        @self.app.route('/api/simulation/data', methods=['GET'])
        def get_simulation_data():
            """Get simulation activity data."""
            if self.current_results is None:
                return jsonify({'error': 'No simulation results available'}), 404

            # Get parameters
            region_idx = request.args.get('region', type=int)
            downsample = request.args.get('downsample', 10, type=int)
            time_range = request.args.get('time_range')  # Format: "start,end"

            # Extract data
            time = self.current_results['time'][::downsample]
            activity_E = self.current_results['E'][::downsample, :]

            # Filter time range if specified
            if time_range:
                start, end = map(float, time_range.split(','))
                mask = (time >= start) & (time <= end)
                time = time[mask]
                activity_E = activity_E[mask, :]

            # Return specific region or all
            if region_idx is not None:
                if 0 <= region_idx < self.brain_data['num_regions']:
                    return jsonify({
                        'time': time.tolist(),
                        'activity': activity_E[:, region_idx].tolist(),
                        'region': self.brain_data['region_labels'][region_idx]
                    })
                else:
                    return jsonify({'error': 'Invalid region index'}), 400
            else:
                return jsonify({
                    'time': time.tolist(),
                    'activity': activity_E.tolist(),
                    'num_regions': activity_E.shape[1]
                })

        @self.app.route('/api/simulation/snapshot', methods=['GET'])
        def get_snapshot():
            """Get activity snapshot at specific time."""
            if self.current_results is None:
                return jsonify({'error': 'No results available'}), 404

            # Get time index
            time_idx = request.args.get('time_idx', -1, type=int)

            activity_E = self.current_results['E']

            # Get snapshot
            if -len(activity_E) <= time_idx < len(activity_E):
                snapshot = activity_E[time_idx, :]
                time_val = self.current_results['time'][time_idx]

                return jsonify({
                    'time': float(time_val),
                    'activity': snapshot.tolist(),
                    'region_labels': self.brain_data['region_labels']
                })
            else:
                return jsonify({'error': 'Invalid time index'}), 400

        # ===== INTERVENTION ENDPOINTS =====

        @self.app.route('/api/intervention/lesion', methods=['POST'])
        def apply_lesion():
            """Apply lesion intervention."""
            if self.intervention_manager is None:
                return jsonify({'error': 'No brain model loaded'}), 400

            data = request.json
            region_idx = data.get('region_idx')
            severity = data.get('severity', 1.0)
            lesion_type = data.get('type', 'region')  # 'region', 'stroke'

            if lesion_type == 'region':
                self.intervention_manager.apply_region_lesion(region_idx, severity)
            elif lesion_type == 'stroke':
                radius = data.get('radius', 2)
                self.intervention_manager.apply_stroke_lesion(region_idx, radius, severity)

            # Update brain data reference
            self.brain_data = self.intervention_manager.current_data

            return jsonify({
                'status': 'success',
                'message': 'Lesion applied',
                'interventions': self.intervention_manager.intervention_history
            })

        @self.app.route('/api/intervention/stimulate', methods=['POST'])
        def apply_stimulation():
            """Apply stimulation intervention."""
            if self.intervention_manager is None:
                return jsonify({'error': 'No brain model loaded'}), 400

            data = request.json
            region_idx = data.get('region_idx')
            amplitude = data.get('amplitude', 1.0)

            initial_state = self.intervention_manager.apply_stimulation(
                region_idx, amplitude
            )

            return jsonify({
                'status': 'success',
                'message': 'Stimulation configured',
                'initial_state': initial_state.tolist()
            })

        @self.app.route('/api/intervention/reset', methods=['POST'])
        def reset_interventions():
            """Reset all interventions."""
            if self.intervention_manager is None:
                return jsonify({'error': 'No brain model loaded'}), 400

            self.intervention_manager.reset()
            self.brain_data = self.intervention_manager.current_data

            return jsonify({
                'status': 'success',
                'message': 'Interventions reset'
            })

        @self.app.route('/api/intervention/history', methods=['GET'])
        def get_intervention_history():
            """Get intervention history."""
            if self.intervention_manager is None:
                return jsonify([])

            return jsonify(self.intervention_manager.intervention_history)

        # ===== ANALYSIS ENDPOINTS =====

        @self.app.route('/api/analysis/metrics', methods=['GET'])
        def get_metrics():
            """Get analysis metrics."""
            if self.analyzer is None:
                return jsonify({'error': 'No analysis available'}), 404

            temporal = self.analyzer.compute_temporal_metrics()

            # Network metrics if connectivity available
            network = None
            if self.current_results.get('connectivity') is not None:
                network = self.analyzer.compute_network_metrics()

            return jsonify({
                'temporal': temporal,
                'network': network
            })

        @self.app.route('/api/analysis/vulnerability', methods=['GET'])
        def get_vulnerability():
            """Get vulnerability map."""
            if self.analyzer is None:
                return jsonify({'error': 'No analysis available'}), 404

            vulnerability = self.analyzer.compute_vulnerability_map()

            return jsonify(vulnerability)

        @self.app.route('/api/analysis/report', methods=['GET'])
        def get_report():
            """Get text report."""
            if self.analyzer is None:
                return jsonify({'error': 'No analysis available'}), 404

            report = self.analyzer.generate_report()

            return jsonify({
                'report': report
            })

        @self.app.route('/api/analysis/functional_connectivity', methods=['GET'])
        def get_functional_connectivity():
            """Get functional connectivity matrix."""
            if self.analyzer is None:
                return jsonify({'error': 'No analysis available'}), 404

            fc = self.analyzer._functional_connectivity()

            return jsonify({
                'fc_matrix': fc.tolist(),
                'region_labels': self.brain_data['region_labels']
            })

        # ===== UTILITY ENDPOINTS =====

        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'brain_loaded': self.brain_data is not None,
                'simulation_available': self.current_results is not None
            })

        @self.app.route('/', methods=['GET'])
        def index():
            """Root endpoint with API info."""
            return jsonify({
                'name': 'VR Brain Lab API',
                'version': '1.0',
                'endpoints': {
                    'brain': ['/api/brain/info', '/api/brain/load', '/api/brain/connectivity'],
                    'simulation': ['/api/simulation/run', '/api/simulation/status', '/api/simulation/data'],
                    'intervention': ['/api/intervention/lesion', '/api/intervention/stimulate', '/api/intervention/reset'],
                    'analysis': ['/api/analysis/metrics', '/api/analysis/vulnerability', '/api/analysis/report']
                }
            })

    def run(self, debug: bool = False):
        """
        Start the Flask server.

        Args:
            debug: Run in debug mode
        """
        print(f"\n{'=' * 60}")
        print(f"VR Brain Lab API Server")
        print(f"{'=' * 60}")
        print(f"Server running at: http://{self.host}:{self.port}")
        print(f"API docs: http://{self.host}:{self.port}/")
        print(f"{'=' * 60}\n")

        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)


# ========== Convenience Functions ==========

def start_server(host: str = '0.0.0.0', port: int = 5000, auto_load_brain: bool = True):
    """
    Quick function to start VR server.

    Args:
        host: Server host
        port: Server port
        auto_load_brain: Automatically load default brain
    """
    server = BrainVRServer(host, port)

    if auto_load_brain:
        print("Auto-loading default brain (68 regions)...")
        server.brain_data = create_default_brain(68)
        server.intervention_manager = BrainIntervention(server.brain_data)
        print("✓ Brain loaded")

    server.run()


if __name__ == "__main__":
    # Start server with default brain
    print("=" * 60)
    print("VR Brain Lab - API Server")
    print("=" * 60)

    start_server(host='0.0.0.0', port=5000, auto_load_brain=True)
