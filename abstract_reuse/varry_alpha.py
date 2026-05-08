"""
Neuron Bound Perturbation Analysis
====================================

This module analyzes how neural network neuron bounds change under weight perturbations.
For each α (alpha - maximum weight perturbation), neurons are classified into 6 categories
based on how their abstract bounds [a,b] change to [c,d].

The module integrates with the existing Network data structures and uses .nnet files.
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import sys
import os
import copy

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
cegar_nn_dir = os.path.join(current_dir, '../..')
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))
sys.path.insert(0, cegar_nn_dir)


from CEGAR_NN.core.data_structures.Network import Network
from CEGAR_NN.core.data_structures.Layer import Layer
from CEGAR_NN.core.data_structures.ARNode import ARNode
from CEGAR_NN.core.data_structures.Edge import Edge
from CEGAR_NN.core.configuration import consts
from CEGAR_NN.core.nnet.read_nnet import network_from_nnet_file
from CEGAR_NN.core.pre_process.pre_process import preprocess
NETWORK_STRUCTS_AVAILABLE = True
property_id = consts.PROPERTY_ID

try:
    from maraboupy import MarabouNetworkNNet
    MARABOU_AVAILABLE = True
except ImportError:
    MARABOU_AVAILABLE = False

NETWORK_AVAILABLE = NETWORK_STRUCTS_AVAILABLE and MARABOU_AVAILABLE

if property_id is not None:
    print("Property id = ", property_id)

class BoundChangeType(Enum):
    """Classification of how bounds change under weight perturbation."""
    IDENTICAL = "Identical"
    SHRINKAGE = "Shrinkage"
    EXPANSION = "Expansion"
    PARTIAL_OVERLAP = "Partial Overlap"
    DISJOINT = "Disjoint"
    DEGENERATE = "Degenerate"


@dataclass
class BoundPair:
    """Represents a pair of bounds (lower, upper)."""
    lower: float
    upper: float
    
    def is_degenerate(self) -> bool:
        """Check if bounds collapse to a single point."""
        return abs(self.upper - self.lower) < 1e-10
    
    def contains(self, other: 'BoundPair') -> bool:
        """Check if this bound contains another bound [other ⊆ self]."""
        return self.lower <= other.lower and other.upper <= self.upper
    
    def overlaps(self, other: 'BoundPair') -> bool:
        """Check if bounds have non-empty overlap."""
        return self.lower <= other.upper and other.lower <= self.upper
    
    def __repr__(self) -> str:
        return f"[{self.lower:.4f}, {self.upper:.4f}]"


@dataclass
class NeuronBoundAnalysis:
    """Analysis result for a single neuron under a specific α."""
    neuron_id: int
    original_bound: BoundPair
    updated_bound: BoundPair
    classification: BoundChangeType
    alpha: float


class BoundClassifier:
    """Classifies neuron bounds into the 6 predefined categories."""
    
    @staticmethod
    def classify(original: BoundPair, updated: BoundPair) -> BoundChangeType:
        """
        Classify the relationship between original [a,b] and updated [c,d] bounds.
        
        Args:
            original: BoundPair representing [a,b]
            updated: BoundPair representing [c,d]
        
        Returns:
            BoundChangeType classification
        """
        # Check for degenerate cases first
        if original.is_degenerate() or updated.is_degenerate():
            return BoundChangeType.DEGENERATE
        
        # Check for identical bounds
        if (abs(original.lower - updated.lower) < 1e-10 and 
            abs(original.upper - updated.upper) < 1e-10):
            return BoundChangeType.IDENTICAL
        
        # Check for shrinkage: [c,d] ⊆ [a,b]
        if original.contains(updated):
            return BoundChangeType.SHRINKAGE
        
        # Check for expansion: [a,b] ⊆ [c,d]
        if updated.contains(original):
            return BoundChangeType.EXPANSION
        
        # Check for disjoint bounds
        if not original.overlaps(updated):
            return BoundChangeType.DISJOINT
        
        # Otherwise, partial overlap
        return BoundChangeType.PARTIAL_OVERLAP


class NetworkAnalyzer:
    """
    Loads and manages neural networks from .nnet files.
    Integrates with the existing Network data structures.
    """
    
    def __init__(self, nnet_path: str):
        """
        Initialize network analyzer and load network from .nnet file.
        
        Args:
            nnet_path: Path to .nnet file
        """
        self.nnet_path = nnet_path
        self.network = None
        self.marabou_network = None
        self._load_network()
    
    def _load_network(self) -> None:
        """Load network from .nnet file using Marabou."""
        if not NETWORK_AVAILABLE:
            raise RuntimeError("Network structures not available. Check imports.")
        
        if not os.path.exists(self.nnet_path):
            raise FileNotFoundError(f"Network file not found: {self.nnet_path}")
        
        try:
            # Load using Marabou
            self.marabou_network = MarabouNetworkNNet(self.nnet_path)
            print(f"Loaded network: {self.nnet_path}")
            
            # Extract network dimensions
            self.input_size = self.marabou_network.inputSize
            self.output_size = self.marabou_network.outputSize
            print(f"Network shape: {self.input_size} inputs -> {self.output_size} outputs")
        except Exception as e:
            raise RuntimeError(f"Failed to load network: {e}")
    
    def get_network_info(self) -> Dict:
        """Get basic information about the network."""
        if self.marabou_network is None:
            return {}
        
        return {
            'path': self.nnet_path,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'num_layers': len(self.marabou_network.layers) if hasattr(self.marabou_network, 'layers') else 'unknown'
        }
    
    def get_layer_info(self, layer_idx: int = None) -> Dict:
        """Get information about specific layer or all layers."""
        if self.marabou_network is None:
            return {}
        
        if hasattr(self.marabou_network, 'layers'):
            if layer_idx is not None:
                if 0 <= layer_idx < len(self.marabou_network.layers):
                    layer = self.marabou_network.layers[layer_idx]
                    return {'layer_index': layer_idx, 'neurons': len(layer)}
                else:
                    return {'error': f'Layer index {layer_idx} out of range'}
            else:
                return {
                    'total_layers': len(self.marabou_network.layers),
                    'layer_sizes': [len(layer) for layer in self.marabou_network.layers]
                }
        return {}


class AlphaAnalyzer:
    """
    Analyzes neuron bound perturbations across varying α values.
    
    For each α (maximum allowed weight perturbation), computes:
    - Classification of each neuron into 6 categories
    - Percentage of neurons falling into each category
    
    Can work with:
    - Direct bound lists (generic use)
    - Neural network from .nnet files
    """
    
    def __init__(self, tolerance: float = 1e-10, network: Optional[NetworkAnalyzer] = None):
        """
        Initialize the analyzer.
        
        Args:
            tolerance: Numerical tolerance for floating-point comparisons
            network: Optional NetworkAnalyzer instance for network-based analysis
        """
        self.tolerance = tolerance
        self.classifier = BoundClassifier()
        self.network = network
        self.results: Dict[float, List[NeuronBoundAnalysis]] = {}
        self.statistics: Dict[float, Dict[BoundChangeType, float]] = {}
        self.layer_statistics: Dict[float, Dict[int, Dict[BoundChangeType, float]]] = {}
    
    def analyze_for_alpha(
        self,
        alpha: float,
        original_bounds: List[Tuple[float, float]],
        updated_bounds: List[Tuple[float, float]]
    ) -> Tuple[List[NeuronBoundAnalysis], Dict[BoundChangeType, float]]:
        """
        Analyze neuron bounds for a specific α value.
        
        Args:
            alpha: Maximum allowed weight perturbation
            original_bounds: List of (lower, upper) tuples for original bounds [a,b]
            updated_bounds: List of (lower, upper) tuples for updated bounds [c,d]
        
        Returns:
            Tuple of:
            - List of NeuronBoundAnalysis objects
            - Dictionary of classification statistics (percentage for each type)
        """
        if len(original_bounds) != len(updated_bounds):
            raise ValueError("Number of original and updated bounds must match")
        
        analyses = []
        classification_counts = {bc: 0 for bc in BoundChangeType}
        
        for neuron_id, (orig_bounds, upd_bounds) in enumerate(
            zip(original_bounds, updated_bounds)
        ):
            original = BoundPair(orig_bounds[0], orig_bounds[1])
            updated = BoundPair(upd_bounds[0], upd_bounds[1])
            
            classification = self.classifier.classify(original, updated)
            classification_counts[classification] += 1
            
            analysis = NeuronBoundAnalysis(
                neuron_id=neuron_id,
                original_bound=original,
                updated_bound=updated,
                classification=classification,
                alpha=alpha
            )
            analyses.append(analysis)
        
        # Compute percentages
        total_neurons = len(original_bounds)
        percentages = {
            bc: (count / total_neurons * 100) 
            for bc, count in classification_counts.items()
        }
        
        # Store results
        self.results[alpha] = analyses
        self.statistics[alpha] = percentages
        
        return analyses, percentages
    
    def print_statistics(self, alpha: float, detailed: bool = False) -> None:
        """
        Print statistics for a specific α value.
        
        Args:
            alpha: The α value to print statistics for
            detailed: If True, print individual neuron analyses as well
        """
        if alpha not in self.statistics:
            print(f"No analysis found for α = {alpha}")
            return
        
        print(f"\n{'='*60}")
        print(f"Analysis for α = {alpha}")
        print(f"{'='*60}")
        
        stats = self.statistics[alpha]
        print("\nClassification Percentages:")
        print("-" * 40)
        for bc in BoundChangeType:
            percentage = stats.get(bc, 0)
            print(f"  {bc.value:<20} {percentage:>6.2f}%")
        
        if detailed and alpha in self.results:
            print("\n" + "-" * 40)
            print("Neuron-level Breakdown:")
            print("-" * 40)
            for analysis in self.results[alpha]:
                print(
                    f"  Neuron {analysis.neuron_id:4d}: "
                    f"{analysis.original_bound} → {analysis.updated_bound} "
                    f"[{analysis.classification.value}]"
                )
    
    def print_all_statistics(self, detailed: bool = False) -> None:
        """
        Print statistics for all analyzed α values.
        
        Args:
            detailed: If True, print individual neuron analyses as well
        """
        for alpha in sorted(self.statistics.keys()):
            self.print_statistics(alpha, detailed=detailed)
    
    def get_summary(self, alpha: float) -> Dict:
        """
        Get summary statistics for a specific α value.
        
        Args:
            alpha: The α value to get summary for
        
        Returns:
            Dictionary with summary information
        """
        if alpha not in self.results:
            return {}
        
        analyses = self.results[alpha]
        stats = self.statistics[alpha]
        
        return {
            'alpha': alpha,
            'total_neurons': len(analyses),
            'classifications': {bc.value: stats.get(bc, 0) for bc in BoundChangeType},
            'analyses': analyses
        }
    
    def compare_alphas(self, alphas: List[float]) -> None:
        """
        Print a comparison table across multiple α values.
        
        Args:
            alphas: List of α values to compare
        """
        if not alphas:
            print("No α values provided")
            return
        
        # Filter to only analyzed alphas
        analyzed_alphas = [a for a in alphas if a in self.statistics]
        if not analyzed_alphas:
            print("None of the provided α values have been analyzed")
            return
        
        print(f"\n{'='*80}")
        print("Comparison Across α Values")
        print(f"{'='*80}\n")
        
        # Header
        header = "Classification".ljust(20)
        for alpha in analyzed_alphas:
            header += f"| α={alpha:<8.4f} ".ljust(15)
        print(header)
        print("-" * len(header))
        
        # Rows for each classification
        for bc in BoundChangeType:
            row = bc.value.ljust(20)
            for alpha in analyzed_alphas:
                percentage = self.statistics[alpha].get(bc, 0)
                row += f"| {percentage:>6.2f}%      ".ljust(15)
            print(row)


# Example usage function
def example_usage():
    """Demonstrates how to use the AlphaAnalyzer."""
    
    print("=" * 60)
    print("Neuron Bound Perturbation Analysis - Example")
    print("=" * 60)
    
    # Example: Analyze 3 neurons across 3 different α values
    analyzer = AlphaAnalyzer()
    
    # Original bounds [a,b] for 3 neurons
    original_bounds = [
        (1.0, 5.0),   # Neuron 0
        (2.0, 4.0),   # Neuron 1
        (-2.0, 2.0),  # Neuron 2
    ]
    
    # Simulate analysis for different α values
    alphas = [0.1, 0.5, 1.0]
    
    for alpha in alphas:
        # Simulate updated bounds based on α
        # (In practice, these would come from actual weight perturbation)
        updated_bounds = [
            (1.0, 5.0),           # Identical
            (2.5, 3.5),           # Shrinkage
            (-3.0, 3.0),          # Expansion
        ]
        
        analyses, percentages = analyzer.analyze_for_alpha(
            alpha, original_bounds, updated_bounds
        )
    
    # Print all statistics
    analyzer.print_all_statistics(detailed=True)
    
    # Compare across α values
    analyzer.compare_alphas(alphas)


def example_network_usage():
    """
    Demonstrates how to use the AlphaAnalyzer with neural networks from .nnet files.
    """
    print("\n" + "=" * 60)
    print("Network-Based Neuron Bound Analysis - Example")
    print("=" * 60)
    
    if not NETWORK_AVAILABLE:
        print("Warning: Network structures not available. Skipping network example.")
        return
    
    # Example path to an ACASXU network
    # nnet_path = "/home/subhankar-jana/artifact/nnet/ACASXU_run2a_1_1_batch_2000.nnet"
    BASE_DIR = os.getcwd()
    nnet_path = os.path.join(BASE_DIR, "nnet", "ACASXU_run2a_1_1_batch_2000.nnet")
    
    try:
        # Load the network
        # net_analyzer = NetworkAnalyzer(nnet_path)
        net = network_from_nnet_file(nnet_path)
        print(f"size={len(net.layers)}")
        # Get network info
        print("\nNetwork Information:")
        print("-" * 40)

        orig_net = copy.deepcopy(net)
        
        preprocess(orig_net)


        
    except Exception as e:
        print(f"Error in network analysis: {e}")
        import traceback
        traceback.print_exc()


class NetworkPropagator:
    """
    Loads neural networks from .nnet files and propagates input values forward
    while maintaining all properties of the data structure (Network, Layer, ARNode, Edge).
    
    This allows:
    - Forward propagation of inputs through all layers
    - Access to neuron details (name, bounds, edges, biases)
    - Inspection of individual neuron activations
    - Complete network structure preservation
    """
    
    def __init__(self, nnet_path: str):
        """
        Initialize the propagator by loading network from .nnet file.
        
        Args:
            nnet_path: Path to .nnet file
        """
        if not NETWORK_STRUCTS_AVAILABLE:
            raise RuntimeError("Network structures not available. Check imports.")
        
        self.nnet_path = nnet_path
        self.network = None
        self.layer_activations = {}  # Store activations for each layer
        self.neuron_details = {}  # Store detailed info about each neuron
        self._load_network()
    
    def _load_network(self) -> None:
        """Load network from .nnet file using core functions."""
        if not os.path.exists(self.nnet_path):
            raise FileNotFoundError(f"Network file not found: {self.nnet_path}")
        
        try:
            self.network = network_from_nnet_file(self.nnet_path)
            print(f"✓ Loaded network: {os.path.basename(self.nnet_path)}")
            self._build_neuron_details()
        except Exception as e:
            raise RuntimeError(f"Failed to load network: {e}")
    
    def _build_neuron_details(self) -> None:
        """Build comprehensive details about each neuron."""
        for layer_idx, layer in enumerate(self.network.layers):
            self.neuron_details[layer_idx] = {}
            for node_idx, node in enumerate(layer.nodes):
                self.neuron_details[layer_idx][node_idx] = {
                    'name': node.name,
                    'layer': layer_idx,
                    'index_in_layer': node_idx,
                    'bias': node.bias,
                    'activation_func': node.activation_func.__name__,
                    'num_in_edges': len(node.in_edges),
                    'num_out_edges': len(node.out_edges),
                    'in_edges_info': [
                        {'src': e.src, 'weight': e.weight} for e in node.in_edges
                    ],
                    'out_edges_info': [
                        {'dest': e.dest, 'weight': e.weight} for e in node.out_edges
                    ]
                }
    
    def propagate_forward(self, input_values: np.ndarray) -> Dict:
        """
        Propagate input values forward through the network layer by layer.
        
        Args:
            input_values: 1D numpy array of input values matching input layer size
        
        Returns:
            Dictionary with:
            - 'layer_outputs': outputs of each layer
            - 'neuron_activations': activation values of each neuron
            - 'final_output': final network output
        """
        if len(input_values) != len(self.network.layers[0].nodes):
            raise ValueError(
                f"Input size {len(input_values)} doesn't match "
                f"network input size {len(self.network.layers[0].nodes)}"
            )
        
        self.layer_activations = {}
        neuron_activations = {}
        
        # Initialize input layer
        current_layer_output = np.array(input_values, dtype=float)
        self.layer_activations[0] = current_layer_output.copy()
        
        # Store input neuron activations
        neuron_activations[0] = {}
        for node_idx, value in enumerate(current_layer_output):
            neuron_activations[0][node_idx] = {
                'name': self.network.layers[0].nodes[node_idx].name,
                'value': float(value),
                'value_before_activation': float(value)
            }
        
        # Propagate through hidden and output layers
        for layer_idx in range(1, len(self.network.layers)):
            layer = self.network.layers[layer_idx]
            prev_layer = self.network.layers[layer_idx - 1]
            
            layer_output = np.zeros(len(layer.nodes))
            neuron_activations[layer_idx] = {}
            
            # Process each neuron in current layer
            for node_idx, node in enumerate(layer.nodes):
                # Initialize with bias
                neuron_sum = node.bias
                
                # Sum weighted inputs from previous layer
                for in_edge in node.in_edges:
                    # Find source neuron value
                    src_node_idx = self._find_node_index(layer_idx - 1, in_edge.src)
                    if src_node_idx is not None:
                        # Use the appropriate activation (before activation for hidden layers)
                        src_value = current_layer_output[src_node_idx]
                        neuron_sum += in_edge.weight * src_value
                
                # Store value before activation
                value_before_activation = neuron_sum
                
                # Apply activation function
                activated_value = node.activation_func(neuron_sum)
                layer_output[node_idx] = activated_value
                
                # Store neuron activation details
                neuron_activations[layer_idx][node_idx] = {
                    'name': node.name,
                    'value': float(activated_value),
                    'value_before_activation': float(value_before_activation),
                    'bias': node.bias,
                    'num_inputs': len(node.in_edges)
                }
            
            # Store layer output and prepare for next layer
            self.layer_activations[layer_idx] = layer_output.copy()
            current_layer_output = layer_output
        
        return {
            'layer_outputs': self.layer_activations,
            'neuron_activations': neuron_activations,
            'final_output': current_layer_output
        }
    
    def _find_node_index(self, layer_idx: int, node_name: str) -> Optional[int]:
        """Find the index of a node in a layer by its name."""
        for idx, node in enumerate(self.network.layers[layer_idx].nodes):
            if node.name == node_name:
                return idx
        return None
    
    def print_network_structure(self) -> None:
        """Print comprehensive network structure."""
        print("\n" + "=" * 70)
        print("NETWORK STRUCTURE")
        print("=" * 70)
        
        general_data = self.network.get_general_net_data()
        print(f"\nGeneral Information:")
        print(f"  Total Layers: {general_data['num_layers']}")
        print(f"  Total Neurons: {general_data['num_nodes']}")
        print(f"  Hidden Layers: {general_data['num_hidden_layers']}")
        print(f"  Layer Sizes: {general_data['layer_sizes']}")
        
        print(f"\nDetailed Layer Information:")
        print("-" * 70)
        
        for layer_idx, layer in enumerate(self.network.layers):
            layer_type = layer.type_name.upper()
            num_nodes = len(layer.nodes)
            print(f"\nLayer {layer_idx} ({layer_type}): {num_nodes} neurons")
            
            for node_idx, node in enumerate(layer.nodes):
                print(f"  Neuron {node_idx}: {node.name}")
                print(f"    Bias: {node.bias:.6f}")
                print(f"    Activation: {node.activation_func.__name__}")
                print(f"    Input Edges: {len(node.in_edges)}")
                if node.in_edges:
                    for edge in node.in_edges[:3]:  # Show first 3 edges
                        print(f"      {edge.src} → {edge.dest} (weight: {edge.weight:.6f})")
                    if len(node.in_edges) > 3:
                        print(f"      ... and {len(node.in_edges) - 3} more edges")
                print(f"    Output Edges: {len(node.out_edges)}")
    
    def print_propagation_results(self, detailed: bool = False) -> None:
        """Print results of forward propagation."""
        print("\n" + "=" * 70)
        print("FORWARD PROPAGATION RESULTS")
        print("=" * 70)
        
        if not self.layer_activations:
            print("No propagation results available. Run propagate_forward() first.")
            return
        
        # Summary
        print(f"\nInput Layer Output: {self.layer_activations[0]}")
        print(f"Final Network Output: {self.layer_activations[len(self.layer_activations)-1]}")
        
        if detailed:
            print(f"\nDetailed Layer-by-Layer Output:")
            print("-" * 70)
            for layer_idx in sorted(self.layer_activations.keys()):
                output = self.layer_activations[layer_idx]
                print(f"\nLayer {layer_idx}:")
                print(f"  Shape: {output.shape}")
                print(f"  Min: {np.min(output):.6f}")
                print(f"  Max: {np.max(output):.6f}")
                print(f"  Mean: {np.mean(output):.6f}")
                print(f"  Values: {output}")
    
    def get_neuron_info(self, layer_idx: int, neuron_idx: int) -> Dict:
        """
        Get detailed information about a specific neuron.
        
        Args:
            layer_idx: Layer index
            neuron_idx: Neuron index within layer
        
        Returns:
            Dictionary with neuron information
        """
        if layer_idx not in self.neuron_details:
            return {'error': f'Layer {layer_idx} not found'}
        
        if neuron_idx not in self.neuron_details[layer_idx]:
            return {'error': f'Neuron {neuron_idx} not found in layer {layer_idx}'}
        
        info = self.neuron_details[layer_idx][neuron_idx].copy()
        
        # Add activation if available
        if layer_idx in self.layer_activations and neuron_idx < len(self.layer_activations[layer_idx]):
            info['activation_value'] = float(self.layer_activations[layer_idx][neuron_idx])
        
        return info
    
    def print_neuron_details(self, layer_idx: int, neuron_idx: int) -> None:
        """Print comprehensive details about a specific neuron."""
        info = self.get_neuron_info(layer_idx, neuron_idx)
        
        if 'error' in info:
            print(f"Error: {info['error']}")
            return
        
        print("\n" + "=" * 70)
        print(f"NEURON DETAILS: Layer {layer_idx}, Neuron {neuron_idx}")
        print("=" * 70)
        
        print(f"\nBasic Information:")
        print(f"  Name: {info['name']}")
        print(f"  Bias: {info['bias']:.6f}")
        print(f"  Activation Function: {info['activation_func']}")
        
        if 'activation_value' in info:
            print(f"  Current Activation: {info['activation_value']:.6f}")
        
        print(f"\nConnectivity:")
        print(f"  Input Connections: {info['num_in_edges']}")
        if info['in_edges_info']:
            for edge_info in info['in_edges_info'][:5]:
                print(f"    From {edge_info['src']} (weight: {edge_info['weight']:.6f})")
            if len(info['in_edges_info']) > 5:
                print(f"    ... and {len(info['in_edges_info']) - 5} more")
        
        print(f"  Output Connections: {info['num_out_edges']}")
        if info['out_edges_info']:
            for edge_info in info['out_edges_info'][:5]:
                print(f"    To {edge_info['dest']} (weight: {edge_info['weight']:.6f})")
            if len(info['out_edges_info']) > 5:
                print(f"    ... and {len(info['out_edges_info']) - 5} more")


class BoundPropagator:
    """
    Propagates input bounds through a neural network to compute output bounds.
    Uses interval arithmetic to track lower and upper bounds for each neuron.
    
    Key Features:
    - Maintains complete network structure and properties
    - Tracks bounds for each neuron at each layer
    - Supports ReLU and other activation functions
    - Computes precise bounds based on weight configurations
    """
    
    def __init__(self, network: Optional['Network'] = None, nnet_path: Optional[str] = None):
        """
        Initialize bound propagator.
        
        Args:
            network: Existing Network object
            nnet_path: Path to .nnet file to load
        """
        if network is not None:
            self.network = network
        elif nnet_path is not None:
            if not NETWORK_STRUCTS_AVAILABLE:
                raise RuntimeError("Network structures not available")
            self.network = network_from_nnet_file(nnet_path)
            print(f"Loaded network from: {nnet_path}")
        else:
            raise ValueError("Must provide either network or nnet_path")
        
        # Stores bounds for each neuron: layer_idx -> neuron_idx -> {'lower': float, 'upper': float}
        self.bounds: Dict[int, Dict[int, Dict[str, float]]] = {}
        self.layer_bounds: Dict[int, Dict[str, List[float]]] = {}  # For summary
    
    def set_input_bounds(self, input_bounds: List[Tuple[float, float]]) -> None:
        """
        Set input bounds for the input layer.
        
        Args:
            input_bounds: List of (lower, upper) tuples for each input neuron
        """
        input_layer = self.network.layers[0]
        
        if len(input_bounds) != len(input_layer.nodes):
            raise ValueError(
                f"Number of input bounds ({len(input_bounds)}) must match "
                f"input layer size ({len(input_layer.nodes)})"
            )
        
        self.bounds[0] = {}
        self.layer_bounds[0] = {'lower': [], 'upper': []}
        
        for neuron_idx, (lower, upper) in enumerate(input_bounds):
            if lower > upper:
                raise ValueError(f"Input {neuron_idx}: lower bound > upper bound")
            
            self.bounds[0][neuron_idx] = {'lower': float(lower), 'upper': float(upper)}
            self.layer_bounds[0]['lower'].append(float(lower))
            self.layer_bounds[0]['upper'].append(float(upper))
        
        print(f"Set input bounds for {len(input_bounds)} neurons")
    
    def propagate_bounds_forward(self) -> None:
        """
        Propagate bounds from input layer through all hidden layers to output layer.
        Uses interval arithmetic for precise bound computation.
        """
        if 0 not in self.bounds:
            raise RuntimeError("Input bounds not set. Call set_input_bounds() first.")
        
        num_layers = len(self.network.layers)
        
        # Propagate layer by layer
        for layer_idx in range(1, num_layers):
            print(f"Propagating bounds for layer {layer_idx}...")
            self._compute_layer_bounds(layer_idx)
    
    def _compute_layer_bounds(self, layer_idx: int) -> None:
        """
        Compute bounds for all neurons in a given layer.
        
        Args:
            layer_idx: Index of the layer
        """
        prev_layer_bounds = self.bounds[layer_idx - 1]
        current_layer = self.network.layers[layer_idx]
        
        self.bounds[layer_idx] = {}
        self.layer_bounds[layer_idx] = {'lower': [], 'upper': []}
        
        for neuron_idx, node in enumerate(current_layer.nodes):
            # Compute weighted sum bounds before activation
            weighted_sum_lower, weighted_sum_upper = self._compute_weighted_sum_bounds(
                node, prev_layer_bounds
            )
            
            # Apply bias
            weighted_sum_lower += node.bias
            weighted_sum_upper += node.bias
            
            # Apply activation function
            activated_lower, activated_upper = self._apply_activation_bounds(
                node.activation_func,
                weighted_sum_lower,
                weighted_sum_upper
            )
            
            # Store bounds
            self.bounds[layer_idx][neuron_idx] = {
                'lower': activated_lower,
                'upper': activated_upper,
                'weighted_sum_lower': weighted_sum_lower,
                'weighted_sum_upper': weighted_sum_upper,
                'pre_activation_lower': weighted_sum_lower,
                'pre_activation_upper': weighted_sum_upper
            }
            
            self.layer_bounds[layer_idx]['lower'].append(activated_lower)
            self.layer_bounds[layer_idx]['upper'].append(activated_upper)
    
    def _compute_weighted_sum_bounds(self, node: 'ARNode', 
                                     prev_layer_bounds: Dict) -> Tuple[float, float]:
        """
        Compute bounds of weighted sum: sum(weight_i * input_i).
        
        Args:
            node: The neuron (ARNode)
            prev_layer_bounds: Bounds of neurons in previous layer
        
        Returns:
            (lower_bound, upper_bound) of weighted sum
        """
        lower_bound = 0.0
        upper_bound = 0.0
        
        for in_edge in node.in_edges:
            # Get source neuron index
            src_node_name = in_edge.src
            try:
                # Extract neuron index from source node name
                src_neuron_idx = self._get_neuron_index_from_node_name(
                    src_node_name, self.network.name2node_map
                )
            except:
                # If extraction fails, try to find it in previous layer
                prev_layer = self.network.layers[len([l for l in self.network.layers[:len(self.network.layers)] if l])-1]
                src_neuron_idx = None
                for idx, prev_node in enumerate(prev_layer.nodes):
                    if prev_node.name == src_node_name:
                        src_neuron_idx = idx
                        break
                
                if src_neuron_idx is None:
                    continue
            
            weight = in_edge.weight
            prev_bounds = prev_layer_bounds.get(src_neuron_idx, {})
            
            if not prev_bounds:
                continue
            
            prev_lower = prev_bounds.get('lower', 0.0)
            prev_upper = prev_bounds.get('upper', 0.0)
            
            # Interval arithmetic for multiplication
            if weight >= 0:
                lower_bound += weight * prev_lower
                upper_bound += weight * prev_upper
            else:
                lower_bound += weight * prev_upper
                upper_bound += weight * prev_lower
        
        return lower_bound, upper_bound
    
    def _get_neuron_index_from_node_name(self, node_name: str, name2node_map: Dict) -> int:
        """Extract neuron index from node name."""
        # Try to find in name2node_map
        if node_name in name2node_map:
            # Find the neuron index by searching in layers
            for layer in self.network.layers:
                for idx, node in enumerate(layer.nodes):
                    if node.name == node_name:
                        return idx
        
        # Fallback: parse from name format "L_idx_neuron_idx" or similar
        parts = node_name.split('_')
        for part in parts:
            if part.isdigit():
                return int(part)
        
        return 0  # Default
    
    def _apply_activation_bounds(self, activation_func: Callable, 
                                 lower: float, upper: float) -> Tuple[float, float]:
        """
        Apply activation function to bounds.
        
        Args:
            activation_func: Activation function (e.g., relu)
            lower: Lower bound before activation
            upper: Upper bound before activation
        
        Returns:
            (lower_bound, upper_bound) after activation
        """
        if activation_func.__name__ == 'relu':
            # ReLU: max(0, x)
            return max(0, lower), max(0, upper)
        elif activation_func.__name__ == 'identity' or activation_func is None:
            # Identity function
            return lower, upper
        else:
            # For other activation functions, apply piecewise
            # Sample key points and use min/max
            sample_points = [lower, (lower + upper) / 2, upper]
            activated = [activation_func(x) for x in sample_points]
            return min(activated), max(activated)
    
    def print_bounds_summary(self) -> None:
        """Print summary of bounds for all layers."""
        print("\n" + "=" * 70)
        print("BOUNDS PROPAGATION SUMMARY")
        print("=" * 70)
        
        for layer_idx in sorted(self.layer_bounds.keys()):
            bounds = self.layer_bounds[layer_idx]
            print(f"\nLayer {layer_idx}:")
            print(f"  Number of neurons: {len(bounds['lower'])}")
            
            if bounds['lower']:
                print(f"  Lower bounds: min={min(bounds['lower']):.6f}, "
                      f"max={max(bounds['lower']):.6f}, "
                      f"mean={np.mean(bounds['lower']):.6f}")
                print(f"  Upper bounds: min={min(bounds['upper']):.6f}, "
                      f"max={max(bounds['upper']):.6f}, "
                      f"mean={np.mean(bounds['upper']):.6f}")
    
    def print_detailed_bounds(self, layer_idx: Optional[int] = None) -> None:
        """
        Print detailed bounds for specific layer or all layers.
        
        Args:
            layer_idx: If specified, only print bounds for this layer
        """
        print("\n" + "=" * 70)
        print("DETAILED BOUNDS INFORMATION")
        print("=" * 70)
        
        if layer_idx is not None:
            if layer_idx not in self.bounds:
                print(f"Layer {layer_idx} not found in computed bounds")
                return
            
            self._print_layer_detailed_bounds(layer_idx)
        else:
            for layer_idx in sorted(self.bounds.keys()):
                self._print_layer_detailed_bounds(layer_idx)
    
    def _print_layer_detailed_bounds(self, layer_idx: int) -> None:
        """Print detailed bounds for a specific layer."""
        layer = self.network.layers[layer_idx]
        layer_bounds = self.bounds[layer_idx]
        
        print(f"\nLayer {layer_idx} ({layer.type_name} layer):")
        print("-" * 70)
        
        for neuron_idx, node in enumerate(layer.nodes):
            bounds = layer_bounds.get(neuron_idx, {})
            
            if not bounds:
                print(f"  Neuron {neuron_idx} ({node.name}): No bounds computed")
                continue
            
            lower = bounds.get('lower', 0.0)
            upper = bounds.get('upper', 0.0)
            
            print(f"  Neuron {neuron_idx} ({node.name}):")
            print(f"    Final bounds: [{lower:.6f}, {upper:.6f}]")
            print(f"    Range width: {upper - lower:.6f}")
            
            if 'pre_activation_lower' in bounds:
                pre_lower = bounds['pre_activation_lower']
                pre_upper = bounds['pre_activation_upper']
                print(f"    Pre-activation bounds: [{pre_lower:.6f}, {pre_upper:.6f}]")
            
            print(f"    Bias: {node.bias:.6f}")
            print(f"    Input connections: {len(node.in_edges)}")
    
    def get_output_bounds(self) -> Tuple[List[float], List[float]]:
        """
        Get bounds for output layer.
        
        Returns:
            (lower_bounds, upper_bounds) lists for output layer
        """
        output_layer_idx = len(self.network.layers) - 1
        
        if output_layer_idx not in self.bounds:
            raise RuntimeError("Bounds not computed for output layer")
        
        output_bounds = self.bounds[output_layer_idx]
        lower_bounds = [output_bounds[i].get('lower', 0.0) 
                       for i in range(len(output_bounds))]
        upper_bounds = [output_bounds[i].get('upper', 0.0) 
                       for i in range(len(output_bounds))]
        
        return lower_bounds, upper_bounds
    
    def compare_neuron_bounds_across_layers(self, neuron_idx: int) -> None:
        """
        Track how bounds change for a specific neuron across layers.
        Useful for understanding bottlenecks in the network.
        
        Args:
            neuron_idx: Index of neuron to track
        """
        print("\n" + "=" * 70)
        print(f"BOUND EVOLUTION FOR NEURON {neuron_idx}")
        print("=" * 70)
        
        for layer_idx in sorted(self.bounds.keys()):
            if neuron_idx not in self.bounds[layer_idx]:
                continue
            
            bounds = self.bounds[layer_idx][neuron_idx]
            lower = bounds.get('lower', 0.0)
            upper = bounds.get('upper', 0.0)
            
            print(f"Layer {layer_idx}: [{lower:.6f}, {upper:.6f}] (width: {upper - lower:.6f})")


class PerturbedNetworkAnalyzer:
    """
    Analyzes how neural network bounds change when weights are perturbed by alpha.
    
    For a fixed alpha (α):
    - Creates a perturbed copy of the network where each weight can vary by ±α
    - Computes bounds for both original and worst-case perturbed networks
    - Classifies neurons into the 6 categories based on bound changes
    - Compares original vs. perturbed bounds
    
    Key Features:
    - Maintains complete network structure for both original and perturbed networks
    - Supports worst-case (max expansion) and best-case (min expansion) perturbations
    - Tracks how different perturbation strategies affect output bounds
    - Provides detailed analysis of affected neurons
    """
    
    def __init__(self, nnet_path: str, input_bounds: List[Tuple[float, float]]):
        """
        Initialize the perturbed network analyzer.
        
        Args:
            nnet_path: Path to .nnet file
            input_bounds: List of (lower, upper) tuples for input layer
        """
        if not NETWORK_STRUCTS_AVAILABLE:
            raise RuntimeError("Network structures not available")
        
        self.nnet_path = nnet_path
        self.input_bounds = input_bounds
        
        # Load original network
        self.original_network = network_from_nnet_file(nnet_path)
        self.original_propagator = BoundPropagator(network=self.original_network)
        self.original_propagator.set_input_bounds(input_bounds)
        self.original_propagator.propagate_bounds_forward()
        
        # Store for perturbed networks
        self.perturbed_networks: Dict[float, Dict] = {}
        self.perturbation_analysis: Dict[float, Dict] = {}
    
    def perturb_network_weights(self, alpha: float, strategy: str = 'worst-case') -> 'Network':
        """
        Create a perturbed copy of the network with weight perturbations.
        
        Args:
            alpha: Maximum perturbation amount for each weight
            strategy: 'worst-case' (expand bounds maximally) or 'best-case' (shrink bounds minimally)
        
        Returns:
            Perturbed network copy
        """
        # Deep copy the original network
        perturbed_net = copy.deepcopy(self.original_network)
        
        # Perturb all weights in all edges
        for layer_idx, layer in enumerate(perturbed_net.layers[1:], 1):
            for node in layer.nodes:
                for in_edge in node.in_edges:
                    if strategy == 'worst-case':
                        # For worst-case: maximize positive weights, minimize negative weights
                        # This typically expands bounds
                        if in_edge.weight >= 0:
                            in_edge.weight += alpha
                        else:
                            in_edge.weight -= alpha
                    elif strategy == 'best-case':
                        # For best-case: minimize positive weights, maximize negative weights
                        # This typically shrinks bounds
                        if in_edge.weight >= 0:
                            in_edge.weight -= alpha
                        else:
                            in_edge.weight += alpha
                    elif strategy == 'random':
                        # Step 1: decide whether to perturb (Bernoulli with p=0.5)
                        if np.random.rand() < 0.5:
                            # Step 2: apply random perturbation
                            perturbation = np.random.uniform(-alpha, alpha)
                            in_edge.weight += perturbation
                        # else: do nothing (weight unchanged)
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
        
        # Regenerate weights and biases
        perturbed_net.weights = perturbed_net.generate_weights()
        perturbed_net.biases = perturbed_net.generate_biases()
        
        return perturbed_net
    
    def analyze_for_alpha(self, alpha: float, strategies: Optional[List[str]] = None) -> None:
        """
        Analyze bound changes for a specific alpha value with different perturbation strategies.
        
        Args:
            alpha: Maximum perturbation amount
            strategies: List of strategies to analyze ('worst-case', 'best-case', 'random')
        """
        if strategies is None:
            strategies = ['worst-case', 'best-case']
        
        print(f"\n{'='*70}")
        print(f"PERTURBATION ANALYSIS FOR α = {alpha}")
        print(f"{'='*70}")
        
        self.perturbation_analysis[alpha] = {}
        self.perturbed_networks[alpha] = {}
        
        for strategy in strategies:
            print(f"\nStrategy: {strategy.upper()}")
            print("-" * 70)
            
            # Create perturbed network
            perturbed_net = self.perturb_network_weights(alpha, strategy)
            self.perturbed_networks[alpha][strategy] = perturbed_net
            
            # Propagate bounds for perturbed network
            perturbed_propagator = BoundPropagator(network=perturbed_net)
            perturbed_propagator.set_input_bounds(self.input_bounds)
            perturbed_propagator.propagate_bounds_forward()
            
            # Compare bounds
            analysis = self._compare_bounds(
                alpha, strategy,
                self.original_propagator,
                perturbed_propagator
            )
            
            self.perturbation_analysis[alpha][strategy] = analysis
    
    def _compare_bounds(self, alpha: float, strategy: str,
                       original_prop: BoundPropagator,
                       perturbed_prop: BoundPropagator) -> Dict:
        """
        Compare bounds between original and perturbed networks.
        
        Returns:
            Analysis dictionary with classification statistics
        """
        num_layers = len(original_prop.network.layers)
        analysis = {
            'alpha': alpha,
            'strategy': strategy,
            'layer_analysis': {},
            'total_stats': {bc: 0 for bc in BoundChangeType}
        }
        
        # Get output bounds
        orig_lower, orig_upper = original_prop.get_output_bounds()
        pert_lower, pert_upper = perturbed_prop.get_output_bounds()
        
        analysis['original_output_bounds'] = {
            'lower': orig_lower,
            'upper': orig_upper
        }
        analysis['perturbed_output_bounds'] = {
            'lower': pert_lower,
            'upper': pert_upper
        }
        
        # Analyze each layer
        for layer_idx in range(1, num_layers):
            if layer_idx not in original_prop.bounds:
                continue
            
            layer_analysis = self._analyze_layer(
                layer_idx, original_prop, perturbed_prop
            )
            analysis['layer_analysis'][layer_idx] = layer_analysis
            
            # Accumulate statistics
            for bc, count in layer_analysis['classification_counts'].items():
                analysis['total_stats'][bc] += count
        
        # Print analysis
        self._print_comparison_analysis(analysis)
        
        return analysis
    
    def _analyze_layer(self, layer_idx: int,
                      original_prop: BoundPropagator,
                      perturbed_prop: BoundPropagator) -> Dict:
        """Analyze bound changes for a specific layer."""
        orig_bounds = original_prop.bounds[layer_idx]
        pert_bounds = perturbed_prop.bounds[layer_idx]
        
        classification_counts = {bc: 0 for bc in BoundChangeType}
        classifications = {}
        
        for neuron_idx in orig_bounds.keys():
            if neuron_idx not in pert_bounds:
                continue
            
            orig = BoundPair(
                orig_bounds[neuron_idx]['lower'],
                orig_bounds[neuron_idx]['upper']
            )
            pert = BoundPair(
                pert_bounds[neuron_idx]['lower'],
                pert_bounds[neuron_idx]['upper']
            )
            
            classification = BoundClassifier.classify(orig, pert)
            classification_counts[classification] += 1
            classifications[neuron_idx] = {
                'original': orig,
                'perturbed': pert,
                'classification': classification
            }
        
        # Compute percentages
        total = sum(classification_counts.values())
        percentages = {
            bc: (count / total * 100) if total > 0 else 0
            for bc, count in classification_counts.items()
        }
        
        return {
            'classification_counts': classification_counts,
            'percentages': percentages,
            'classifications': classifications,
            'total_neurons': total
        }
    
    def _print_comparison_analysis(self, analysis: Dict) -> None:
        """Print detailed comparison analysis."""
        alpha = analysis['alpha']
        strategy = analysis['strategy']
        
        # Output bounds
        print(f"\nOutput Layer Bounds:")
        print("-" * 70)
        orig_bounds = analysis['original_output_bounds']
        pert_bounds = analysis['perturbed_output_bounds']
        
        print(f"{'Output':<10} {'Original':<25} {'Perturbed':<25}")
        print("-" * 60)
        for i in range(len(orig_bounds['lower'])):
            orig_lower = orig_bounds['lower'][i]
            orig_upper = orig_bounds['upper'][i]
            pert_lower = pert_bounds['lower'][i]
            pert_upper = pert_bounds['upper'][i]
            
            orig_str = f"[{orig_lower:.2f}, {orig_upper:.2f}]"
            pert_str = f"[{pert_lower:.2f}, {pert_upper:.2f}]"
            
            print(f"{i:<10} {orig_str:<25} {pert_str:<25}")
        
        # Total statistics
        print(f"\nOverall Classification Statistics:")
        print("-" * 70)
        total_stats = analysis['total_stats']
        total_neurons = sum(total_stats.values())
        
        for bc in BoundChangeType:
            count = total_stats[bc]
            percentage = (count / total_neurons * 100) if total_neurons > 0 else 0
            print(f"  {bc.value:<20} {count:>6} neurons ({percentage:>6.2f}%)")
    
    def print_detailed_layer_comparison(self, alpha: float, strategy: str, 
                                       layer_idx: int, top_n: int = 10) -> None:
        """
        Print detailed neuron-by-neuron comparison for a layer.
        
        Args:
            alpha: Alpha value
            strategy: Perturbation strategy
            layer_idx: Layer index
            top_n: Number of most changed neurons to show
        """
        if alpha not in self.perturbation_analysis:
            print(f"No analysis for α = {alpha}")
            return
        
        if strategy not in self.perturbation_analysis[alpha]:
            print(f"No analysis for strategy = {strategy}")
            return
        
        if layer_idx not in self.perturbation_analysis[alpha][strategy]['layer_analysis']:
            print(f"No analysis for layer {layer_idx}")
            return
        
        layer_analysis = self.perturbation_analysis[alpha][strategy]['layer_analysis'][layer_idx]
        classifications = layer_analysis['classifications']
        
        # Calculate bound changes
        bound_changes = []
        for neuron_idx, class_info in classifications.items():
            orig = class_info['original']
            pert = class_info['perturbed']
            
            orig_width = orig.upper - orig.lower
            pert_width = pert.upper - pert.lower
            width_change = pert_width - orig_width
            
            bound_changes.append({
                'neuron_idx': neuron_idx,
                'classification': class_info['classification'],
                'original': orig,
                'perturbed': pert,
                'width_change': width_change
            })
        
        # Sort by width change (descending)
        bound_changes.sort(key=lambda x: abs(x['width_change']), reverse=True)
        
        print(f"\n{'='*70}")
        print(f"LAYER {layer_idx} DETAILED COMPARISON")
        print(f"α = {alpha}, Strategy = {strategy}")
        print(f"{'='*70}")
        
        print(f"\nTop {min(top_n, len(bound_changes))} neurons with largest bound changes:")
        print("-" * 70)
        print(f"{'Neuron':<8} {'Classification':<20} {'Original':<20} {'Perturbed':<20} {'Δ Width':<12}")
        print("-" * 70)
        
        for i, change in enumerate(bound_changes[:top_n]):
            neuron_idx = change['neuron_idx']
            classification = change['classification'].value
            orig_str = f"[{change['original'].lower:.4f}, {change['original'].upper:.4f}]"
            pert_str = f"[{change['perturbed'].lower:.4f}, {change['perturbed'].upper:.4f}]"
            width_change = change['width_change']
            
            print(f"{neuron_idx:<8} {classification:<20} {orig_str:<20} {pert_str:<20} {width_change:>+.6f}")
    
    def print_comparison_summary(self) -> None:
        """Print summary of all perturbation analyses."""
        print(f"\n{'='*70}")
        print("PERTURBATION ANALYSIS SUMMARY")
        print(f"{'='*70}")
        
        for alpha in sorted(self.perturbation_analysis.keys()):
            print(f"\nα = {alpha}")
            print("-" * 70)
            
            for strategy in sorted(self.perturbation_analysis[alpha].keys()):
                analysis = self.perturbation_analysis[alpha][strategy]
                total_stats = analysis['total_stats']
                total_neurons = sum(total_stats.values())
                
                print(f"\n  Strategy: {strategy}")
                for bc in BoundChangeType:
                    count = total_stats[bc]
                    percentage = (count / total_neurons * 100) if total_neurons > 0 else 0
                    print(f"    {bc.value:<20} {percentage:>6.2f}%")


def demo_all():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE NETWORK ANALYSIS DEMO")
    print("=" * 70)
    
    if not NETWORK_STRUCTS_AVAILABLE:
        print("Warning: Network structures not available. Skipping demo.")
        return
    
    BASE_DIR = os.getcwd()
    nnet_path = os.path.join(BASE_DIR, "nnet", "ACASXU_run2a_1_1_batch_2000.nnet")
    # nnet_path = "/home/subhankar-jana/artifact/nnet/ACASXU_run2a_1_1_batch_2000.nnet"
    
    try:
        # ===== PART 1: Forward Propagation with Value Propagation =====
        print("\n" + "=" * 70)
        print("PART 1: FORWARD VALUE PROPAGATION")
        print("=" * 70)
        
        propagator = NetworkPropagator(nnet_path)
        propagator.print_network_structure()
        
        num_inputs = len(propagator.network.layers[0].nodes)
        input_values = np.random.randn(num_inputs)
        
        print(f"\nInput Values (first 10): {input_values[:10]}")
        print(f"\nPropagating forward...")
        results = propagator.propagate_forward(input_values)
        propagator.print_propagation_results(detailed=True)
        
        # Print specific neuron details
        propagator.print_neuron_details(layer_idx=1, neuron_idx=0)
        
        # ===== PART 2: Bound Propagation =====
        print("\n\n" + "=" * 70)
        print("PART 2: INTERVAL BOUND PROPAGATION")
        print("=" * 70)
        
        bound_propagator = BoundPropagator(network=propagator.network)
        
        # Set fixed input ranges
        num_inputs = len(propagator.network.layers[0].nodes)
        input_bounds = [(-0.5, 0.5) for _ in range(num_inputs)]
        
        print(f"\nSetting input bounds:")
        for i, (lower, upper) in enumerate(input_bounds[:5]):
            print(f"  Input {i}: [{lower}, {upper}]")
        if num_inputs > 5:
            print(f"  ... and {num_inputs - 5} more inputs")
        
        bound_propagator.set_input_bounds(input_bounds)
        
        print(f"\nPropagating bounds forward...")
        bound_propagator.propagate_bounds_forward()
        
        bound_propagator.print_bounds_summary()
        bound_propagator.print_detailed_bounds(layer_idx=1)
        
        # Print output bounds
        lower_bounds, upper_bounds = bound_propagator.get_output_bounds()
        print(f"\n" + "=" * 70)
        print("OUTPUT BOUNDS (Original Network)")
        print("=" * 70)
        print(f"Lower bounds (first 5): {lower_bounds[:5]}")
        print(f"Upper bounds (first 5): {upper_bounds[:5]}")
        
        # Track bounds evolution for specific neuron
        if num_inputs > 0:
            bound_propagator.compare_neuron_bounds_across_layers(neuron_idx=0)
        
        # ===== PART 3: Perturbation Analysis =====
        print("\n\n" + "=" * 70)
        print("PART 3: WEIGHT PERTURBATION ANALYSIS")
        print("=" * 70)
        
        # Create perturbed network analyzer
        perturbed_analyzer = PerturbedNetworkAnalyzer(nnet_path, input_bounds)
        
        print(f"\nAnalyzing weight perturbations for multiple α values...")
        print(f"Input bounds: {input_bounds}")
        
        # Analyze for different alpha values
        alphas = [0.1, 0.5, 1.0]
        
        for alpha in alphas:
            perturbed_analyzer.analyze_for_alpha(alpha, strategies=['worst-case', 'best-case'])
        
        # Print detailed comparison for specific layer
        print("\n\n" + "=" * 70)
        print("DETAILED LAYER COMPARISON")
        print("=" * 70)
        
        perturbed_analyzer.print_detailed_layer_comparison(
            alpha=0.5, strategy='worst-case', layer_idx=1, top_n=5
        )
        
        # Print summary
        perturbed_analyzer.print_comparison_summary()
        
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_all()
