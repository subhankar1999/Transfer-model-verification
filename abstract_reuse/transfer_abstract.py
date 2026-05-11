#!/usr/bin/env python3

"""
Transfer Abstract Network Program
==================================

This program transfers an abstract neural network through step-by-step process.

Step 1: Transfer a network by changing edge weights by maximum alpha using random strategy
Step 2: Verify source network by CEGAR-NN
Step 3: Create Abstraction of the transfer network by using abstraction of the source network
Step 4: Verify transfer network using transfer abstract
Step 5: Verify transfer network from Scratch by CEGAR-NN
"""

import sys
import os
import copy
import pickle
import json
import glob
import time
import argparse
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

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
from CEGAR_NN.core.nnet.read_nnet import network_from_nnet_file
from CEGAR_NN.core.utils.verification_properties_utils import is_satisfying_assignment, get_test_property_acas
from CEGAR_NN.core.utils.marabou_query_utils import get_query
from CEGAR_NN.core.refinement.refine import refine
from CEGAR_NN.experiments.run.one_experiment_cegarabou import one_experiment

try:
    from maraboupy import MarabouNetworkNNet
except ImportError:
    print("Error: MarabouNetworkNNet not available")
    sys.exit(1)

# Import logging utility
from abstract_reuse.transfer_logging import TransferExecutionLogger


import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Step timed out")


def run_with_timeout(func, timeout_sec, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)

    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # disable alarm
        return result, False  # success
    except TimeoutException:
        return None, True  # timeout occurred
    except Exception as e:
        signal.alarm(0)
        raise e


class NetworkTransferer:
    """
    Handles the step-by-step transfer process of neural networks.
    """
    
    def __init__(self, nnet_path: str):
        """
        Initialize the NetworkTransferer with a network file.
        
        Args:
            nnet_path: Path to the .nnet file
        """
        self.nnet_path = nnet_path
        self.original_network = None
        self.transferred_network = None
        self.alpha = None
        self.strategy = None
        self.abstract_network = None
        self.node_mapping = None
        self.reverse_mapping = None
        
        print("=" * 70)
        print("NETWORK TRANSFERER INITIALIZATION")
        print("=" * 70)
        print(f"Loading network from: {nnet_path}")
        self._load_network()
        print(f"✓ Network loaded successfully")
        
    def _load_network(self) -> None:
        """Load the network from the nnet file."""
        if not os.path.exists(self.nnet_path):
            raise FileNotFoundError(f"Network file not found: {self.nnet_path}")
        
        self.original_network = network_from_nnet_file(self.nnet_path)
    
    def _zero_biases(self, network: Network) -> None:
        """
        Zero out all biases in the network.
        
        Args:
            network: Network whose biases should be zeroed
        """
        for layer in network.layers[1:]:  # Skip input layer
            for node in layer.nodes:
                node.bias = 0.0
        
        # Regenerate biases array
        network.biases = network.generate_biases()
    
    def step1_transfer_weights_by_alpha(self, alpha: float, 
                                       strategy: str = 'random-discrete',
                                       save_to_file: bool = True,
                                       output_dir: str = 'transfer_network', l: int = None, prob: float = 0.5) -> Network:
        """
        STEP 1: Transfer a network by changing edge weights by maximum alpha.
        
        Uses the random strategy to perturb weights randomly within [-alpha, +alpha].
        Zeros out biases in both original and transferred networks.
        
        Args:
            alpha: Maximum perturbation amount for each weight
            strategy: Perturbation strategy ('random', 'worst-case', 'best-case', 'random-discrete')
            save_to_file: If True, save the transferred network to .nnet file
            output_dir: Directory to save the transferred network (default: 'transfer_network')
        
        Returns:
            Transferred network with perturbed weights and zero biases
        
        Raises:
            ValueError: If strategy is not recognized
        """
        print("\n" + "=" * 70)
        print("STEP 1: TRANSFER NETWORK BY WEIGHT PERTURBATION")
        print("=" * 70)
        print(f"Parameters:")
        print(f"  - Alpha (max perturbation): {alpha}")
        print(f"  - Strategy: {strategy}")
        print(f"  - Save to file: {save_to_file}")
        if save_to_file:
            print(f"  - Output directory: {output_dir}")
        
        # Validate strategy
        valid_strategies = ['random', 'worst-case', 'best-case', 'random-discrete']
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Valid options: {', '.join(valid_strategies)}"
            )
        
        # STEP 1a: Zero out biases in original network
        print("\n[1a] Zeroing biases in original network...")
        self._zero_biases(self.original_network)
        # print(" Original network biases set to zero")
        
        # Deep copy the original network
        print("\n[1b] Creating deep copy of original network...")
        transferred_net = copy.deepcopy(self.original_network)
        print("Deep copy created")
        
        # Perturb all weights in all edges
        print(f"\n[1c] Applying {strategy} perturbation to all edges...")
        
        perturbation_stats = {
            'total_edges': 0,
            'perturbed_edges': 0,
            'total_weight_change': 0.0,
            'max_weight_change': 0.0
        }

        total_layers = len(transferred_net.layers)

        if l is None or l > total_layers:
            start_layer = 1
        else:
            start_layer = max(1, total_layers - l)
        if strategy == 'random-discrete':
            for layer_idx, layer in enumerate(transferred_net.layers[start_layer:], start_layer):
                print(f"\n  Processing layer {layer_idx}...")

                layer_total_edges = 0
                layer_changed_edges = 0


                distance_from_last = total_layers - 1 - layer_idx

                
                n_active = max(50 - 10 * distance_from_last, 10)  # minimum safeguard

                
                n_active = min(n_active, len(layer.nodes))

                print(f"    Active neurons (cone): {n_active} / {len(layer.nodes)}")

                
                active_nodes = layer.nodes[:n_active]

                # for node in active_nodes:

                for node in layer.nodes:
                    for in_edge in node.in_edges:

                        perturbation_stats['total_edges'] += 1
                        layer_total_edges += 1

                        # Random selection
                        if np.random.rand() < prob:
                            sign = 1 if np.random.rand() < 0.5 else -1
                            in_edge.weight += sign * alpha

                            # Stats
                            weight_change = abs(sign * alpha)
                            perturbation_stats['perturbed_edges'] += 1
                            perturbation_stats['total_weight_change'] += weight_change
                            perturbation_stats['max_weight_change'] = max(
                                perturbation_stats['max_weight_change'],
                                weight_change
                            )

                            layer_changed_edges += 1

               
                percentage = (layer_changed_edges / max(1, layer_total_edges)) * 100
                print(f"     Changed edges: {layer_changed_edges} / {layer_total_edges} ({percentage:.2f}%)")

        else:
            for layer_idx, layer in enumerate(transferred_net.layers[1:], 1):
                print(f"  Processing layer {layer_idx}...")
                
                for node_idx, node in enumerate(layer.nodes):
                    for edge_idx, in_edge in enumerate(node.in_edges):
                        perturbation_stats['total_edges'] += 1
                        original_weight = in_edge.weight
                        
                        if strategy == 'random':
                            # Random perturbation within [-alpha, +alpha]
                            perturbation = np.random.uniform(-alpha, alpha)
                            in_edge.weight += perturbation
                            
                        elif strategy == 'worst-case':
                            # For worst-case: maximize positive weights, minimize negative weights
                            if in_edge.weight >= 0:
                                in_edge.weight += alpha
                            else:
                                in_edge.weight -= alpha
                                
                        elif strategy == 'best-case':
                            # For best-case: minimize positive weights, maximize negative weights
                            if in_edge.weight >= 0:
                                in_edge.weight -= alpha
                            else:
                                in_edge.weight += alpha
                        
                        # Track statistics
                        weight_change = abs(in_edge.weight - original_weight)
                        perturbation_stats['perturbed_edges'] += 1
                        perturbation_stats['total_weight_change'] += weight_change
                        perturbation_stats['max_weight_change'] = max(
                            perturbation_stats['max_weight_change'],
                            weight_change
                        )
        

        transferred_net.weights = transferred_net.generate_weights()
        transferred_net.biases = transferred_net.generate_biases()

    
        self.transferred_network = transferred_net
        self.alpha = alpha
        self.strategy = strategy

        
        # Print statistics
        print("\n" + "-" * 70)
        print("PERTURBATION STATISTICS")
        print("-" * 70)
        print(f"Total edges perturbed: {perturbation_stats['perturbed_edges']}")
        print(f"Average weight change: {perturbation_stats['total_weight_change'] / max(1, perturbation_stats['perturbed_edges']):.6f}")
        print(f"Maximum weight change: {perturbation_stats['max_weight_change']:.6f}")
        print("\n" + "-" * 70)
        print("NETWORK BIAS STATUS")
        print("-" * 70)
        print(f"Original network: All biases = 0.0")
        print(f"Transferred network: All biases = 0.0")
        
        # Save to file if requested
        if save_to_file:
            print("\n" + "-" * 70)
            print("SAVING TRANSFERRED NETWORK")
            print("-" * 70)
            
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f" Created output directory: {output_dir}")
            
            # Extract original filename without extension
            original_filename = os.path.basename(self.nnet_path)
            filename_without_ext = os.path.splitext(original_filename)[0]
            
            # Create output filename
            output_filename = f"{filename_without_ext}_transfer.nnet"
            output_path = os.path.join(output_dir, output_filename)
            
            # Write the network to file
            try:
                self._write_nnet_file(output_path, transferred_net, alpha, strategy)
                print(f" Transferred network saved to: {output_path}")
                print(f"  File size: {os.path.getsize(output_path) / 1024:.2f} KB")
            except Exception as e:
                print(f"Error saving network: {str(e)}")
                raise
        
        print("\n Transfer Network created successfully")
        
        return transferred_net
    
    def step2_read_abstract_network_and_mapping(self, 
                                               nnet_id,
                                               property_id , results_dir: str = 'CEGAR_NN/experiments/dev_exp_results_archive',
                                               run_experiment: bool = True, model: str = 'original') -> tuple:
        """
        STEP 2: Get abstract network and extract node mappings.
        
        This step can either:
        1. Run one_experiment_cegarabou.py to generate a fresh abstract network
        2. Load the abstract network from a previously saved pickle file if stored
        
        The abstract network comes from the CEGAR process and has all proper
        metadata regenerated for Marabou compatibility.
        
        Args:
            results_dir: Directory for storing/loading CEGAR results
            nnet_id: ACAS Xu network ID (e.g., '1_6')
            property_id: Property ID to use in CEGAR
            run_experiment: If True, run CEGAR experiment; if False, load from file
        
        Returns:
            Tuple of (abstract_network, node_mapping, reverse_mapping, cegar_results)
        
        Raises:
            FileNotFoundError: If no abstract network files are found
            RuntimeError: If unable to load/generate the abstract network
        """
        print("\n" + "=" * 70)
        print("STEP 2: GET ABSTRACT NETWORK AND NODE MAPPINGS")
        print("=" * 70)
        
        cegar_results = None  # Store CEGAR experiment results for logging
        NNET_BASE = ""
        
        if run_experiment:
            print("\n[2a] Running CEGAR experiment to generate abstract network...")
            print(f"  Network: ACASXU_run2a_{nnet_id}_batch_2000.nnet")
            print(f"  Property: {property_id}")
            
            try:
                # Import one_experiment function
                import sys
                exp_run_path = os.path.join(os.path.dirname(__file__), '../../experiments/run')
                if exp_run_path not in sys.path:
                    sys.path.insert(0, exp_run_path)
                
                
                # Run the CEGAR experiment
                print("\n  Starting experiment (this may take couple of minutes)...")
                t_start = time.time()
                if model == "original":
                    print("Verifying source network  . .. . ")
                    experiment_results = one_experiment(
                        nnet_filename=f"ACASXU_run2a_{nnet_id}_batch_2000.nnet",
                        is_tiny=False,
                        refinement_type="cegar",
                        abstraction_type="complete",
                        epsilon=1e-5,
                        lower_bound=25000,
                        preprocess_orig_net=False,
                        refinement_sequence_length=100,
                        abstraction_sequence_length=100,
                        results_directory=results_dir,
                        property_id=property_id,
                        verbose=False
                    )
                else:
                    print("Verifying Transfer network  . .. . ")
                    print(type(self.transferred_network))
                    experiment_results = one_experiment(
                        nnet_filename=f"ACASXU_run2a_{nnet_id}_batch_2000.nnet",
                        is_tiny=False,
                        refinement_type="cegar",
                        abstraction_type="complete",
                        epsilon=1e-5,
                        lower_bound=25000,
                        preprocess_orig_net=False,
                        refinement_sequence_length=100,
                        abstraction_sequence_length=100,
                        results_directory=results_dir,
                        property_id=property_id,
                        verbose=False,
                        net = self.transferred_network,
                    )
                
                t_elapsed = time.time() - t_start
                print(f" CEGAR experiment completed in {t_elapsed:.2f}s")
                
                # Extract query result and abstract network from experiment results
                # one_experiment returns (res_dict, abstract_net) tuple
                experiment_res_dict, abstract_network = experiment_results
                query_result = [v for k, v in experiment_res_dict if k == 'query_result'][0]
                print(f"  Query result: {query_result}")
                print(" Abstract network retrieved directly from experiment")
                
                # Store CEGAR results for logging
                cegar_results = experiment_res_dict
                self.cegar_results = cegar_results
                
            except Exception as e:
                print(f" Error running CEGAR experiment: {str(e)}")
                print(f"  Falling back to loading from file...")
                run_experiment = False
                abstract_network = None
        
        if not run_experiment:
            print("\n Loading abstract network from saved pickle file...")
            abstract_network = None
        
        # Load the abstract network from pickle file if not already loaded
        if abstract_network is None:
            print("\n Searching for abstract network pickle files...")
            pkl_files = glob.glob(os.path.join(results_dir, 'final_network_experiment__*.pkl'))
            
            if not pkl_files:
                raise FileNotFoundError(
                    f"No pickle files found in {results_dir}. "
                    f"Please ensure CEGAR experiments have been run."
                )
            
            # Sort by modification time and get the latest
            latest_pkl = max(pkl_files, key=os.path.getmtime)
            print(f" Found {len(pkl_files)} pickle files")
            print(f" Loading latest: {os.path.basename(latest_pkl)}")
            
            # Load the abstract network
            print("\n Loading abstract network from pickle file...")
            try:
                with open(latest_pkl, 'rb') as f:
                    abstract_network = pickle.load(f)
                
                print(" Abstract network loaded successfully")
                
            except Exception as e:
                raise RuntimeError(f"Failed to load abstract network: {str(e)}")
        else:
            print("\n Abstract network already loaded from experiment")
            print(" Skipping pickle file load (using experiment result)")
        
        # Extract node mapping information
        print("\n Extracting node mappings...")
        
        # Forward mapping: abstract node -> original nodes
        forward_mapping = {}
        
        # Reverse mapping: original node -> abstract node(s)
        reverse_mapping = {}
        
        for layer_idx, layer in enumerate(abstract_network.layers):
            for abstract_node in layer.nodes:
                node_name = abstract_node.name
                # Node names may be formatted as: "x_1_5" or "x_1_5_inc" or merged nodes
                original_nodes = [node_name.split('_pos')[0].split('_neg')[0].split('_inc')[0].split('_dec')[0]]
                
                # Build forward mapping
                forward_mapping[node_name] = {
                    "original_nodes": original_nodes,
                    "layer": layer_idx,
                    "num_merged": len(original_nodes),
                    "bias": abstract_node.bias if hasattr(abstract_node, 'bias') else None,
                    "in_degree": len(abstract_node.in_edges) if hasattr(abstract_node, 'in_edges') else 0,
                    "out_degree": len(abstract_node.out_edges) if hasattr(abstract_node, 'out_edges') else 0
                }
                
                # Build reverse mapping
                for orig_node in original_nodes:
                    if orig_node not in reverse_mapping:
                        reverse_mapping[orig_node] = []
                    reverse_mapping[orig_node].append({
                        "abstract_node": node_name,
                        "layer": layer_idx
                    })
        
        # print("✓ Node mappings extracted successfully")
        
        # Store for future steps
        # print("\n[2e] Storing abstract network and mappings...")
        self.abstract_network = abstract_network
        self.node_mapping = forward_mapping
        self.reverse_mapping = reverse_mapping
        # print("✓ Stored in instance variables")
        
        # Print statistics
        print("\n" + "-" * 70)
        print("ABSTRACT NETWORK STATISTICS")
        print("-" * 70)
        print(f"Number of layers: {len(abstract_network.layers)}")
        
        total_original_nodes = 0
        total_abstract_nodes = 0
        merged_info = {}
        
        for layer_idx, layer in enumerate(abstract_network.layers):
            num_abstract = len(layer.nodes)
            total_abstract_nodes += num_abstract
            layer_type = layer.type_name if hasattr(layer, 'type_name') else 'hidden'
            print(f"Layer {layer_idx} ({layer_type}): {num_abstract} nodes")
            
            for node in layer.nodes:
                # Count merged nodes from node naming
                num_merged = 1  # Default to single node
                original_nodes = forward_mapping[node.name]["original_nodes"]
                num_merged = len(original_nodes)
                total_original_nodes += num_merged
                
                if num_merged not in merged_info:
                    merged_info[num_merged] = 0
                merged_info[num_merged] += 1
        
        print(f"\nTotal original nodes (before abstraction): {total_original_nodes}")
        print(f"Total abstract nodes (after abstraction): {total_abstract_nodes}")
        if total_abstract_nodes > 0:
            print(f"Compression ratio: {total_original_nodes / total_abstract_nodes:.2f}x")
        
        print("\nNode merging statistics:")
        for merge_count in sorted(merged_info.keys()):
            print(f"  {merge_count} nodes merged: {merged_info[merge_count]} abstract nodes")
        
        print("\n" + "-" * 70)
        print("NODE MAPPING STATISTICS")
        print("-" * 70)
        print(f"Forward mappings stored: {len(forward_mapping)}")
        print(f"Reverse mappings stored: {len(reverse_mapping)}")
        print("\n VERIFICATION IS COMPLETED")
        
        return (abstract_network, forward_mapping, reverse_mapping, cegar_results)
    
    def step3_abstract_transfer_network(self) -> 'Network':
        """
        STEP 3: Abstraction of the Transfer Network (Abstract Reuse Algorithm).
        
        Create an abstract transfer network by combining the transfer network weights
        with the abstract network structure of source network.
        
        Returns:
            Abstract transfer network 
            
        Raises:
            RuntimeError: If abstract network or mappings not available
        """
        print("\n" + "=" * 70)
        print("STEP 3: ABSTRACTION OF TRANSFER NETWORK")
        print("=" * 70)
        
        # Validate that required data is available
        if self.abstract_network is None:
            raise RuntimeError(
                "Abstract network not available. "
                "Run step2_read_abstract_network_and_mapping first."
            )
        if self.transferred_network is None:
            raise RuntimeError(
                "Transferred network not available. "
                "Run step1_transfer_weights_by_alpha first."
            )
        if self.reverse_mapping is None:
            raise RuntimeError(
                "Reverse mapping not available. "
                "Run step2_read_abstract_network_and_mapping first."
            )
        
        print("\n Initializing abstract transfer network...")
        # Deep copy abstract network to create Ta
        abstract_transfer_net = copy.deepcopy(self.abstract_network)
        # print(" Deep copy of abstract network created")
        
        # Create name2node map for quick lookups
        print("\n Building node lookup map...")
        name2node_map = {}
        for layer_idx, layer in enumerate(abstract_transfer_net.layers):
            for node in layer.nodes:
                name2node_map[node.name] = node
        print(f" Created name2node map with {len(name2node_map)} nodes")
        
        # Process layers backward (from L-1 to 1)
        print("\n Processing layers backward...")
        split_stats = {
            'total_nodes_processed': 0,
            'sign_consistent': 0,
            'sign_inconsistent': 0,
            'nodes_split': 0
        }
        
        # Track nodes that have been replaced during splitting
        # Map from old_node_name -> [new_node1, new_node2, ...]
        replaced_nodes = {}
        
        num_layers = len(abstract_transfer_net.layers)
        
        # Process from second-to-last layer down to layer 1
        for layer_idx in range(num_layers - 2, 0, -1):  # Traverse backward, skip input layer (0)
            # print(f"  Processing Layer {layer_idx}...")
            layer = abstract_transfer_net.layers[layer_idx]
            layer_nodes_to_process = list(layer.nodes)  # Create copy to iterate safely
            
            nodes_to_add = []  # Nodes created by splitting
            nodes_to_remove = []  # Nodes that were split
            
            for abstract_node in layer_nodes_to_process:
                split_stats['total_nodes_processed'] += 1
                abstract_node_name = abstract_node.name
                
                # Get concrete neurons mapped to this abstract neuron (from reverse mapping)
                # The reverse mapping value is a list of dicts, extract abstract node name
                reverse_map_value = self.reverse_mapping.get(abstract_node_name)
                if reverse_map_value is None:
                    continue
                
                # Get all transfer network weights from this abstract node to next layer
                transfer_weights_info = self._get_outgoing_weights(
                    abstract_node_name,
                    abstract_transfer_net
                )
                
                if not transfer_weights_info['weights']:
                    continue
                
                # Check sign consistency
                is_sign_consistent = self._is_sign_consistent(transfer_weights_info['weights'])
                
                if is_sign_consistent:
                    split_stats['sign_consistent'] += 1
                    self._process_sign_consistent_node(
                        abstract_node,
                        abstract_node_name,
                        abstract_transfer_net,
                        name2node_map
                    )
                else:
                    split_stats['sign_inconsistent'] += 1
                    split_nodes = self._process_sign_inconsistent_node(
                        abstract_node,
                        abstract_node_name,
                        abstract_transfer_net,
                        name2node_map,
                        layer_idx
                    )
                    nodes_to_add.extend(split_nodes)
                    nodes_to_remove.append(abstract_node)
                    split_stats['nodes_split'] += 1
                    
                    # Track this replacement for updating downstream edges
                    replaced_nodes[abstract_node_name] = [n.name for n in split_nodes]
            
            # Update layer with new nodes
            for node in nodes_to_remove:
                if node in layer.nodes:
                    layer.nodes.remove(node)
            
            for node in nodes_to_add:
                layer.nodes.append(node)
                name2node_map[node.name] = node
            
            # Update edges in layers BELOW (higher indices) that reference removed nodes
            if replaced_nodes:
                for lower_layer_idx in range(layer_idx + 1, num_layers):
                    lower_layer = abstract_transfer_net.layers[lower_layer_idx]
                    for lower_node in lower_layer.nodes:
                        # Check if any incoming edges reference replaced nodes
                        new_in_edges = []
                        for in_edge in lower_node.in_edges:
                            if in_edge.src in replaced_nodes:
                                # This edge comes from a removed node
                                # We need to keep it as is - the replacement nodes should have been
                                # created with edges to this node already
                                # So we skip old edges pointing to replaced nodes
                                pass
                            else:
                                new_in_edges.append(in_edge)
                        lower_node.in_edges = new_in_edges
        
        # print(f"   All layers processed")
        
        # Store for future use
        # print("\n Storing abstract transfer network...")
        
        # Regenerate network metadata for Marabou compatibility
        # print("[3d-i] Regenerating name2node_map...")
        abstract_transfer_net.generate_name2node_map()
        # print("  ✓ name2node_map regenerated")
        
        # print("[3d-ii] Regenerating weights matrix...")
        try:
            abstract_transfer_net.weights = abstract_transfer_net.generate_weights()
            print("  weights matrix regenerated")
        except KeyError as e:
            print(f"    Warning: Could not regenerate weights matrix")
            print(f"     This is expected when split nodes create missing edge sources")
            print(f"     Error: {e}")
            print(f"     Using previous weights matrix")
        
        # print("[3d-iii] Regenerating biases matrix...")
        try:
            abstract_transfer_net.biases = abstract_transfer_net.generate_biases()
            print("  biases matrix regenerated")
        except Exception as e:
            print(f"   Warning: Could not regenerate biases matrix")
            print(f"     Error: {e}")
            print(f"     Using previous biases matrix")
        
        self.abstract_transfer_network = abstract_transfer_net
        # print(" Abstract transfer network stored")
        # print(" Metadata regenerated for Marabou compatibility (with fallbacks)")
        
        # Print statistics
        print("\n" + "-" * 70)
        print("ABSTRACTION STATISTICS")
        print("-" * 70)
        print(f"Total nodes processed: {split_stats['total_nodes_processed']}")
        print(f"Sign-consistent nodes: {split_stats['sign_consistent']}")
        print(f"Sign-inconsistent nodes: {split_stats['sign_inconsistent']}")
        print(f"Nodes split: {split_stats['nodes_split']}")
        
        # # Verify transfer abstract network consistency
        # print("\n" + "-" * 70)
        # print("VERIFICATION: Checking Transfer Abstract Network Consistency")
        # print("-" * 70)
        # self._verify_network_consistency(abstract_transfer_net)
        
        print("\nABSTRACT NETWORK CREATION COMPLETED FOR TRANSFER NETWORK")
        
        return abstract_transfer_net
    
    # def _verify_network_consistency(self, network) -> bool:
    #     """
    #     Verify that the transfer abstract network is internally consistent
    #     and matches the properties of the abstract network.
        
    #     Checks:
    #     1. All nodes in layers are in name2node_map
    #     2. weights/biases matrix dimensions match layer structure
    #     3. All split nodes have preserved bias values (not 0.0)
    #     4. All edges have valid source/destination nodes
        
    #     Args:
    #         network: Network object to verify
            
    #     Returns:
    #         True if all checks pass, False otherwise
    #     """
    #     print("[Consistency Check 1] Verifying name2node_map coverage...")
    #     all_node_count = 0
    #     for layer_idx, layer in enumerate(network.layers):
    #         for node in layer.nodes:
    #             all_node_count += 1
    #             if node.name not in network.name2node_map:
    #                 print(f"  ✗ Node {node.name} (layer {layer_idx}) not in name2node_map!")
    #                 return False
    #     print(f"  ✓ All {all_node_count} nodes found in name2node_map")
        
    #     print("[Consistency Check 2] Verifying weights matrix dimensions...")
    #     try:
    #         for layer_idx, layer_weights in enumerate(network.weights):
    #             expected_nodes = len(network.layers[layer_idx + 1].nodes)
    #             actual_rows = len(layer_weights)
    #             if actual_rows != expected_nodes:
    #                 print(f"  ✗ weights[{layer_idx}]: {actual_rows} rows, but layer {layer_idx+1} has {expected_nodes} nodes!")
    #                 return False
    #         print(f"  ✓ weights matrix has correct dimensions")
    #     except Exception as e:
    #         print(f"  ✗ Error checking weights: {e}")
    #         return False
        
    #     print("[Consistency Check 3] Verifying biases matrix dimensions...")
    #     try:
    #         for layer_idx, layer_biases in enumerate(network.biases):
    #             expected_nodes = len(network.layers[layer_idx + 1].nodes)
    #             actual_values = len(layer_biases)
    #             if actual_values != expected_nodes:
    #                 print(f"  ✗ biases[{layer_idx}]: {actual_values} values, but layer {layer_idx+1} has {expected_nodes} nodes!")
    #                 return False
    #         print(f"  ✓ biases matrix has correct dimensions")
    #     except Exception as e:
    #         print(f"  ✗ Error checking biases: {e}")
    #         return False
        
    #     print("[Consistency Check 4] Verifying bias values (no zero biases in split nodes)...")
    #     zero_bias_count = 0
    #     split_node_count = 0
    #     for layer_idx, layer in enumerate(network.layers[1:], start=1):  # Skip input layer
    #         for node in layer.nodes:
    #             if "_pos" in node.name or "_neg" in node.name:
    #                 split_node_count += 1
    #                 if node.bias == 0.0:
    #                     zero_bias_count += 1
        
    #     if zero_bias_count > 0:
    #         print(f"    {zero_bias_count}/{split_node_count} split nodes have zero bias!")
    #         return False
    #     else:
    #         print(f"  ✓ All {split_node_count} split nodes have non-zero biases")
        
    #     print("[Consistency Check 5] Verifying edge validity...")
    #     try:
    #         for layer_idx, layer in enumerate(network.layers):
    #             prev_nodes = {n.name for n in network.layers[layer_idx-1].nodes} if layer_idx > 0 else set()
    #             next_nodes = {n.name for n in network.layers[layer_idx+1].nodes} if layer_idx < len(network.layers) - 1 else set()
                
    #             for node in layer.nodes:
    #                 # Check incoming edges
    #                 if layer_idx > 0:
    #                     for in_edge in node.in_edges:
    #                         if in_edge.src not in prev_nodes:
    #                             print(f"  ✗ Edge source {in_edge.src} not in previous layer for {node.name}!")
    #                             return False
                    
    #                 # Check outgoing edges
    #                 if layer_idx < len(network.layers) - 1:
    #                     for out_edge in node.out_edges:
    #                         if out_edge.dest not in next_nodes:
    #                             print(f"  ✗ Edge destination {out_edge.dest} not in next layer for {node.name}!")
    #                             return False
            
    #         print(f"  ✓ All edges point to valid nodes")
    #     except Exception as e:
    #         print(f"  ✗ Error checking edges: {e}")
    #         return False
        
    #     print("\n✓ ALL CONSISTENCY CHECKS PASSED!")
    #     print("  Transfer abstract network is consistent with abstract network properties")
    #     return True
    
    def _get_outgoing_weights(self, abstract_node_name: str,
                             abstract_transfer_net) -> Dict[str, Any]:
        """
        Get all outgoing weights from an abstract node using the transfer network.
        Maps abstract node to its concrete neurons, then gets their transfer weights.
        
        Args:
            abstract_node_name: Name of abstract node
            abstract_transfer_net: Abstract transfer network
            
        Returns:
            Dictionary with 'weights' list and 'info' dict
        """
        weights = []
        info = {'weights': weights, 'edges': []}
        
        # Get concrete neurons corresponding to this abstract node (from node_mapping)
        concrete_neurons_list = self.node_mapping.get(abstract_node_name, {}).get('original_nodes', [])
        
        if not concrete_neurons_list:
            return info
        
        # Build node map for transfer network
        transfer_node_map = {}
        for layer in self.transferred_network.layers:
            for node in layer.nodes:
                transfer_node_map[node.name] = node
        
        # Get all outgoing weights from concrete neurons in transfer network
        for concrete_neuron_name in concrete_neurons_list:
            if concrete_neuron_name in transfer_node_map:
                concrete_node = transfer_node_map[concrete_neuron_name]
                for out_edge in concrete_node.out_edges:
                    weights.append(out_edge.weight)
                    info['edges'].append({
                        'src': out_edge.src,
                        'dest': out_edge.dest,
                        'weight': out_edge.weight
                    })
        
        return info
    
    def _is_sign_consistent(self, weights: List[float]) -> bool:
        """
        Check if all weights have the same sign (all positive or all negative).
        Zero is considered compatible with both positive and negative.
        
        Args:
            weights: List of weights
            
        Returns:
            True if all weights have consistent sign, False otherwise
        """
        if not weights:
            return True
        
        # Check if all weights are non-negative or all non-positive
        all_non_negative = all(w >= 0 for w in weights)
        all_non_positive = all(w <= 0 for w in weights)
        
        return all_non_negative or all_non_positive
    
    def _process_sign_consistent_node(self, abstract_node, abstract_node_name: str,
                                     abstract_transfer_net, name2node_map) -> None:
        src_concrete_neurons = self.node_mapping.get(abstract_node_name, {}).get('original_nodes', [])
        
        if not src_concrete_neurons:
            return
        

        for in_edge in abstract_node.in_edges:
            src_abstract_node_name = in_edge.src
            src_concrete_for_incoming = self.node_mapping.get(src_abstract_node_name, {}).get('original_nodes', [])
            
            if not src_concrete_for_incoming:
                continue
            
            # Get all weights from source concrete neurons to this node's concrete neurons
            incoming_edge_weights = self._get_edge_weights_from_concrete_to_concrete(
                src_concrete_for_incoming,
                src_concrete_neurons
            )
            
            if not incoming_edge_weights:
                continue
            
            # Select max/min based on THIS node's type (since it's receiving this edge)
            node_type = abstract_node.ar_type  # "inc" or "dec"
            if node_type == "inc":
                selected_weight = max(incoming_edge_weights)
            elif node_type == "dec":
                selected_weight = min(incoming_edge_weights)
            else:
                selected_weight = max(incoming_edge_weights)  # Default to max
            
            in_edge.weight = selected_weight
        

        for out_edge in abstract_node.out_edges:
            target_abstract_node_name = out_edge.dest
            tgt_concrete_neurons = self.node_mapping.get(target_abstract_node_name, {}).get('original_nodes', [])
            
            if not tgt_concrete_neurons:
                continue
            
            
            edge_weights = self._get_edge_weights_from_concrete_to_concrete(
                src_concrete_neurons,
                tgt_concrete_neurons
            )
            
            if not edge_weights:
                continue
            
            # Sum all weights (all same sign because sign-consistent)
            summed_weight = sum(edge_weights)
            
            # Update edge weight
            out_edge.weight = summed_weight
            
            # Update abstract node type based on weight sign
            if summed_weight > 0:
                abstract_node.ar_type = "positive"
            elif summed_weight < 0:
                abstract_node.ar_type = "negative"
            else:
                abstract_node.ar_type = "zero"
    
    def _process_sign_inconsistent_node(self, abstract_node, abstract_node_name: str,
                                       abstract_transfer_net, name2node_map,
                                       layer_idx: int) -> List:

        split_nodes = []
        
    
        node_pos = ARNode(
            name=abstract_node.name + "_pos",
            ar_type="positive",
            in_edges=[],
            out_edges=[],
            bias=abstract_node.bias,
            activation_func=abstract_node.activation_func
        )
        
        node_neg = ARNode(
            name=abstract_node.name + "_neg",
            ar_type="negative",
            in_edges=[],
            out_edges=[],
            bias=abstract_node.bias,
            activation_func=abstract_node.activation_func
        )
        

        src_concrete_neurons = self.node_mapping.get(abstract_node_name, {}).get('original_nodes', [])
        
        if not src_concrete_neurons:
            return split_nodes
        

        if layer_idx > 0:
            for in_edge in abstract_node.in_edges:
                src_abstract_node_name = in_edge.src
                src_concrete_for_incoming = self.node_mapping.get(src_abstract_node_name, {}).get('original_nodes', [])
                
                if not src_concrete_for_incoming:
                    continue
                
                
                incoming_edge_weights = self._get_edge_weights_from_concrete_to_concrete(
                    src_concrete_for_incoming,
                    src_concrete_neurons
                )
                
                if not incoming_edge_weights:
                    continue
                
                
                selected_weight_pos = max(incoming_edge_weights)
                new_edge_pos = Edge(
                    src=in_edge.src,
                    dest=node_pos.name,
                    weight=selected_weight_pos
                )
                node_pos.in_edges.append(new_edge_pos)
                
                
                selected_weight_neg = max(incoming_edge_weights)
                new_edge_neg = Edge(
                    src=in_edge.src,
                    dest=node_neg.name,
                    weight=selected_weight_neg
                )
                node_neg.in_edges.append(new_edge_neg)
                
                
                src_node = name2node_map.get(in_edge.src)
                if src_node:
                    
                    src_node.out_edges = [e for e in src_node.out_edges if e.dest != abstract_node.name]
                    
                    src_node.out_edges.append(new_edge_pos)
                    src_node.out_edges.append(new_edge_neg)
        

        for out_edge in abstract_node.out_edges:
            target_abstract_node_name = out_edge.dest
            tgt_concrete_neurons = self.node_mapping.get(target_abstract_node_name, {}).get('original_nodes', [])
            
            if not tgt_concrete_neurons:
                continue
            
            
            all_edge_weights = self._get_edge_weights_from_concrete_to_concrete(
                src_concrete_neurons,
                tgt_concrete_neurons
            )
            
            if not all_edge_weights:
                continue
            
            
            pos_weights = []
            neg_weights = []
            
            for weight in all_edge_weights:
                if weight > 0:
                    pos_weights.append(weight)
                elif weight < 0:
                    neg_weights.append(weight)
            
            
            if pos_weights:
                summed_weight_pos = sum(pos_weights)
                new_edge_pos = Edge(
                    src=node_pos.name,
                    dest=target_abstract_node_name,
                    weight=summed_weight_pos
                )
                node_pos.out_edges.append(new_edge_pos)
            
            
            if neg_weights:
                summed_weight_neg = sum(neg_weights)
                new_edge_neg = Edge(
                    src=node_neg.name,
                    dest=target_abstract_node_name,
                    weight=summed_weight_neg
                )
                node_neg.out_edges.append(new_edge_neg)
        
        
        if node_pos.in_edges or node_pos.out_edges:
            split_nodes.append(node_pos)
        if node_neg.in_edges or node_neg.out_edges:
            split_nodes.append(node_neg)
        
        return split_nodes
    
    def _get_edge_weights_from_concrete_to_concrete(self, src_concrete_neurons: List[str],
                                                   dst_concrete_neurons: List[str]) -> List[float]:
        """
        Get all transfer network weights from source concrete neurons to destination concrete neurons.
        
        Args:
            src_concrete_neurons: Source concrete neuron identifiers
            dst_concrete_neurons: Destination concrete neuron identifiers
            
        Returns:
            List of weights between the concrete neurons
        """
        weights = []
        
        # Build node name map for transfer network
        node_map = {}
        for layer in self.transferred_network.layers:
            for node in layer.nodes:
                node_map[node.name] = node
        
        # For each source concrete neuron, find edges to destination concrete neurons
        for src_neuron in src_concrete_neurons:
            if src_neuron not in node_map:
                continue
            
            src_node = node_map[src_neuron]
            
            for out_edge in src_node.out_edges:
                if out_edge.dest in dst_concrete_neurons:
                    weights.append(out_edge.weight)
        
        return weights
    
    def step4_verify_with_cegar_refinement(self, test_property: Dict[str, Any],
                                          max_refinement_steps: int = 10,
                                          refinement_sequence_length: int = 100) -> Dict[str, Any]:
        """
        STEP 4: Verify Transfer Abstract Network with CEGAR-like Refinement.
        
        Verification loop:
        1. Verify abstract transfer network against property
        2. If UNSAT: Property verified
        3. If SAT: Check counterexample against transfer network
           - If valid: Real counterexample found
           - If spurious: Refine and loop
        
        Args:
            test_property: The property to verify
            max_refinement_steps: Maximum refinement iterations
            refinement_sequence_length: Sequence length for refinement
            
        Returns:
            Dictionary with verification results
        """
        print("\n" + "=" * 70)
        print("STEP 4: VERIFICATION OF TRANSFER ABSTRACTION WITH CEGAR-LIKE REFINEMENT")
        print("=" * 70)
        
        # Validate required data
        if not hasattr(self, 'abstract_transfer_network') or self.abstract_transfer_network is None:
            raise RuntimeError(
                "Abstract transfer network not available. "
                "Run step3_abstract_transfer_network first."
            )
        
        if self.transferred_network is None:
            raise RuntimeError(
                "Transfer network not available. "
                "Run step1_transfer_weights_by_alpha first."
            )
        
        # print(f"\n[4a] Initializing verification...")
        print(f"  - Test property: {test_property}")
        print(f"  - Max refinement steps: {max_refinement_steps}")
        
        # Initialize verification state
        verification_state = {
            'result': None,
            'counterexample': None,
            'num_refine_steps': 0,
            'refine_times': [],
            'query_times': [],
            'is_spurious': False,
            'refinement_history': []
        }
        
        current_abstract_transfer_net = copy.deepcopy(self.abstract_transfer_network)
        
        # Main CEGAR loop
        print(f"\nStarting CEGAR verification loop...")
        step_count = 0
        
        while step_count <= max_refinement_steps:
            print(f"\n  Iteration {step_count}:")
            
            # Query abstract transfer network
            print(f"    - Querying abstract transfer network...")
            t_query_start = time.time()
            
            try:
                stats, vars1, query_result = get_query(
                    network=current_abstract_transfer_net,
                    test_property=test_property,
                    verbose=False
                )
            except Exception as e:
                print(f"   Error during query: {str(e)}")
                verification_state['result'] = 'ERROR'
                verification_state['error'] = str(e)
                return verification_state
            
            t_query_end = time.time()
            query_time = t_query_end - t_query_start
            verification_state['query_times'].append(query_time)
            
            print(f"    - Query result: {query_result} (time: {query_time:.4f}s)")
            
            # Case 1: UNSAT - property verified
            if query_result == "UNSAT":
                print(f"     UNSAT - Property verified!")
                verification_state['result'] = 'UNSAT'
                verification_state['num_refine_steps'] = step_count
                break
            
            # Case 2: SAT - check counterexample
            if query_result == "SAT":
                print(f"    - SAT: Got counterexample, checking against transfer network...")
                
                # Evaluate counterexample on transfer network
                try:
                    transfer_net_output = self.transferred_network.evaluate(vars1)
                except Exception as e:
                    print(f"     Error evaluating transfer network: {str(e)}")
                    verification_state['result'] = 'ERROR'
                    verification_state['error'] = str(e)
                    return verification_state
                
                # Get variable mappings
                try:
                    nodes2variables, variables2nodes = self.transferred_network.get_variables()
                except Exception as e:
                    print(f"     Error getting variables: {str(e)}")
                    nodes2variables, variables2nodes = {}, {}
                
                # Check if counterexample is satisfying
                is_satisfying = is_satisfying_assignment(
                    network=self.transferred_network,
                    test_property=test_property,
                    output=transfer_net_output,
                    variables2nodes=variables2nodes
                )
                
                # Case 2a: Non-spurious counterexample
                if is_satisfying:
                    print(f"     Counterexample valid on transfer network - Property violated!")
                    verification_state['result'] = 'SAT'
                    verification_state['counterexample'] = vars1
                    verification_state['num_refine_steps'] = step_count
                    verification_state['is_spurious'] = False
                    break
                
                # Case 2b: Spurious counterexample - refine
                else:
                    print(f"    - Spurious counterexample detected - refining...")
                    
                    if step_count >= max_refinement_steps:
                        print(f"     Max refinement steps reached")
                        verification_state['result'] = 'MAX_REFINEMENTS'
                        verification_state['num_refine_steps'] = step_count
                        break
                    
                    # Refine using Algorithm Y
                    print(f"    - Applying maximal splitting refinement...")
                    t_refine_start = time.time()
                    
                    try:
                        current_abstract_transfer_net = self._refine_with_extremal_splitting(
                            abstract_transfer_net=current_abstract_transfer_net,
                            transfer_net=self.transferred_network,
                            counterexample=vars1,
                            test_property=test_property
                        )
                    except Exception as e:
                        print(f"     Error during refinement: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        verification_state['result'] = 'REFINEMENT_ERROR'
                        verification_state['error'] = str(e)
                        return verification_state
                    
                    t_refine_end = time.time()
                    refine_time = t_refine_end - t_refine_start
                    verification_state['refine_times'].append(refine_time)
                    verification_state['refinement_history'].append({
                        'step': step_count,
                        'time': refine_time,
                        'num_nodes': sum(len(layer.nodes) for layer in current_abstract_transfer_net.layers)
                    })
                    
                    print(f"     Refinement complete (time: {refine_time:.4f}s)")
                    print(f"      New network size: {verification_state['refinement_history'][-1]['num_nodes']} nodes")
            
            step_count += 1
        
        # Summary
        print(f"\n" + "-" * 70)
        print("VERIFICATION SUMMARY")
        print("-" * 70)
        print(f"Result: {verification_state['result']}")
        print(f"Refinement steps: {verification_state['num_refine_steps']}")
        if verification_state['query_times']:
            print(f"Total query time: {sum(verification_state['query_times']):.4f}s")
        if verification_state['refine_times']:
            print(f"Total refinement time: {sum(verification_state['refine_times']):.4f}s")
        print(f"Spurious counterexample: {verification_state['is_spurious']}")
        
        # Store results
        self.verification_result = verification_state
        self.final_abstract_transfer_network = current_abstract_transfer_net
        
        print("\n VERIFICATION OF TRANSFER NETWORK COMPLETED USING REUSED ABSTRACT")
        
        return verification_state
    
    def _refine_with_extremal_splitting(self, abstract_transfer_net, transfer_net,
                                       counterexample: Dict, test_property: Dict) -> 'Network':
        """
        Algorithm Y: Counterexample-Guided Extremal Splitting Refinement.
        
        Refines the abstract transfer network by splitting neurons that cause spurious counterexamples.
        
        Args:
            abstract_transfer_net: The abstract transfer network to refine
            transfer_net: The transfer network for evaluation
            counterexample: The spurious counterexample input
            test_property: The test property
            
        Returns:
            Refined abstract transfer network
        """
        # Step 1: Evaluate concrete transfer network on counterexample
        concrete_activations = transfer_net.evaluate(counterexample)
        
        # Step 2: Evaluate abstract transfer network on counterexample
        abstract_activations = abstract_transfer_net.evaluate(counterexample)
        
        # Step 3: Identify mismatched neurons
        mismatched_neurons = self._identify_mismatched_neurons(
            abstract_transfer_net,
            transfer_net,
            counterexample,
            concrete_activations,
            abstract_activations
        )
        
        if not mismatched_neurons:
            print(f"      Warning: No mismatched neurons found")
            return abstract_transfer_net
        
        print(f"      Mismatched neurons found: {len(mismatched_neurons)}")
        
        # Create refined network as copy
        refined_net = copy.deepcopy(abstract_transfer_net)
        
        # Build node maps for refined network
        refined_name2node = {}
        for layer_idx, layer in enumerate(refined_net.layers):
            for node in layer.nodes:
                refined_name2node[node.name] = node
        
        # Step 4-12: For each mismatched neuron, perform extremal splitting
        nodes_to_remove = []
        nodes_to_add = []
        mapping_updates = {}  # Track neuron name changes for updating reverse mapping
        
        for layer_idx, layer in enumerate(refined_net.layers):
            layer_mismatched = [n for n in mismatched_neurons if n['layer'] == layer_idx]
            
            for mismatch_info in layer_mismatched:
                abstract_neuron_name = mismatch_info['name']
                abstract_neuron = refined_name2node.get(abstract_neuron_name)
                
                if abstract_neuron is None:
                    continue
                
                # Get mapped concrete neurons
                mapped_concrete = self.node_mapping.get(abstract_neuron_name, {}).get('original_nodes', [])
                
                if not mapped_concrete:
                    continue
                
                print(f"      Splitting neuron {abstract_neuron_name} ({len(mapped_concrete)} mapped)")
                
                # Compute contributions of each concrete neuron
                contributions = self._compute_contributions(
                    mapped_concrete,
                    transfer_net,
                    counterexample,
                    abstract_neuron_name,
                    layer_idx
                )
                
                if not contributions:
                    continue
                
                # Find extremal contributors
                max_contrib = max(contributions.values())
                min_contrib = min(contributions.values())
                threshold = (max_contrib + min_contrib) / 2.0
                
                print(f"        Contributions: max={max_contrib:.6f}, min={min_contrib:.6f}, threshold={threshold:.6f}")
                
                # Partition concrete neurons
                m_plus = [u for u in mapped_concrete if contributions.get(u, 0) >= threshold]
                m_minus = [u for u in mapped_concrete if contributions.get(u, 0) < threshold]
                
                if not m_plus or not m_minus:
                    print(f"        Warning: Empty partition, skipping split")
                    continue
                
                # Create new split neurons
                split_nodes = self._create_split_neurons(
                    abstract_neuron,
                    m_plus,
                    m_minus,
                    refined_net,
                    refined_name2node,
                    transfer_net,
                    test_property
                )
                
                if split_nodes:
                    nodes_to_remove.append(abstract_neuron)
                    nodes_to_add.extend(split_nodes)
                    
                    # Track mapping updates
                    for split_node in split_nodes:
                        mapping_updates[abstract_neuron_name] = split_node.name
        
        # Apply node updates to refined network
        for layer in refined_net.layers:
            for node_to_remove in nodes_to_remove:
                if node_to_remove in layer.nodes:
                    layer.nodes.remove(node_to_remove)
            
            for node_to_add in nodes_to_add:
                if node_to_add not in layer.nodes:
                    layer.nodes.append(node_to_add)
        
        return refined_net
    
    def _identify_mismatched_neurons(self, abstract_net, transfer_net, counterexample,
                                    concrete_acts, abstract_acts) -> List[Dict]:
        """
        Identify abstract neurons whose activations differ from their concrete neurons.
        
        Args:
            abstract_net: Abstract transfer network
            transfer_net: Transfer network
            counterexample: Input counterexample
            concrete_acts: Concrete network activations
            abstract_acts: Abstract network activations
            
        Returns:
            List of mismatched neuron info
        """
        mismatched = []
        
        # Build concrete node map
        concrete_node_map = {}
        for layer in transfer_net.layers:
            for node in layer.nodes:
                concrete_node_map[node.name] = node
        
        # Check each abstract neuron
        for layer_idx, layer in enumerate(abstract_net.layers):
            for abstract_node in layer.nodes:
                abstract_node_name = abstract_node.name
                
                # Get mapped concrete neurons
                mapped_concrete = self.node_mapping.get(abstract_node_name, {}).get('original_nodes', [])
                
                # Get abstract node activation
                try:
                    abstract_act = abstract_acts[abstract_node_name]
                except (KeyError, TypeError):
                    continue
                
                # Check if any mapped concrete neuron has different activation
                mismatch = False
                for concrete_name in mapped_concrete:
                    try:
                        concrete_act = concrete_acts.get(concrete_name, abstract_act)
                        # Compare activations (with small tolerance)
                        if abs(concrete_act - abstract_act) > 1e-6:
                            mismatch = True
                            break
                    except:
                        continue
                
                if mismatch:
                    mismatched.append({
                        'name': abstract_node_name,
                        'layer': layer_idx,
                        'mapped_concrete': mapped_concrete
                    })
        
        return mismatched
    
    def _compute_contributions(self, mapped_concrete: List[str], transfer_net,
                              counterexample, abstract_neuron_name, layer_idx) -> Dict[str, float]:
        """
        Compute contribution c_{u_i} = w_{u_i,v} · z_{u_i}(x) for each concrete neuron.
        
        Args:
            mapped_concrete: Concrete neurons mapped to abstract neuron
            transfer_net: Transfer network
            counterexample: Input counterexample
            abstract_neuron_name: Name of abstract neuron
            layer_idx: Layer index
            
        Returns:
            Dictionary mapping concrete neuron names to their contributions
        """
        contributions = {}
        
        # Evaluate transfer network on counterexample to get neuron activations
        try:
            activations = transfer_net.evaluate(counterexample)
        except:
            return contributions
        
        # Build node map for transfer network
        node_map = {}
        for layer in transfer_net.layers:
            for node in layer.nodes:
                node_map[node.name] = node
        
        # For each mapped concrete neuron
        for concrete_neuron_name in mapped_concrete:
            if concrete_neuron_name not in node_map:
                continue
            
            concrete_node = node_map[concrete_neuron_name]
            
            # Get activation z_{u_i}(x)
            z_ui = activations.get(concrete_neuron_name, 0.0)
            
            # Sum contributions to next layer: c_{u_i} = sum of (w_{u_i,v} · z_{u_i})
            contribution = 0.0
            for out_edge in concrete_node.out_edges:
                contribution += out_edge.weight * z_ui
            
            contributions[concrete_neuron_name] = contribution
        
        return contributions
    
    def _create_split_neurons(self, original_node, m_plus: List[str], m_minus: List[str],
                             refined_net, name2node_map, transfer_net,
                             test_property) -> List['ARNode']:
        """
        Create two new split neurons a^+ and a^- for the partitioned concrete neurons.
        
        Args:
            original_node: The original abstract neuron to split
            m_plus: Concrete neurons in positive partition
            m_minus: Concrete neurons in negative partition
            refined_net: The refined network
            name2node_map: Node name to node mapping
            transfer_net: Transfer network for weight computation
            test_property: Test property
            
        Returns:
            List of newly created split nodes
        """
        split_nodes = []
        
        # Create positive split node
        node_plus = ARNode(
            name=original_node.name + "_plus",
            ar_type="",
            in_edges=[],
            out_edges=[],
            bias=0.0,
            activation_func=original_node.activation_func
        )
        
        # Create negative split node
        node_minus = ARNode(
            name=original_node.name + "_minus",
            ar_type="",
            in_edges=[],
            out_edges=[],
            bias=0.0,
            activation_func=original_node.activation_func
        )
        
        # Copy incoming edges from original node
        for in_edge in original_node.in_edges:
            new_edge_plus = Edge(src=in_edge.src, dest=node_plus.name, weight=in_edge.weight)
            new_edge_minus = Edge(src=in_edge.src, dest=node_minus.name, weight=in_edge.weight)
            
            node_plus.in_edges.append(new_edge_plus)
            node_minus.in_edges.append(new_edge_minus)
            
            # Update source node's outgoing edges
            src_node = name2node_map.get(in_edge.src)
            if src_node:
                src_node.out_edges = [e for e in src_node.out_edges if e.dest != original_node.name]
                src_node.out_edges.append(new_edge_plus)
                src_node.out_edges.append(new_edge_minus)
        
        # Process outgoing edges using extremal aggregation
        for out_edge in original_node.out_edges:
            target_node_name = out_edge.dest
            target_node = name2node_map.get(target_node_name)
            
            if not target_node:
                continue
            
            # Get target concrete neurons
            target_concrete = self.node_mapping.get(target_node_name, {}).get('original_nodes', [])
            
            # Get weights from m_plus to target concrete neurons
            weights_plus = self._get_edge_weights_from_concrete_to_concrete(m_plus, target_concrete)
            weights_minus = self._get_edge_weights_from_concrete_to_concrete(m_minus, target_concrete)
            
            # Select extremal weights based on target neuron type
            target_type = target_node.ar_type
            
            if weights_plus:
                if target_type == "inc":
                    selected_weight_plus = max(weights_plus)
                    node_plus.ar_type = "positive_increment"
                else:
                    selected_weight_plus = min(weights_plus)
                    node_plus.ar_type = "positive_decrement"
                
                new_edge_plus = Edge(src=node_plus.name, dest=target_node_name, weight=selected_weight_plus)
                node_plus.out_edges.append(new_edge_plus)
            
            if weights_minus:
                if target_type == "inc":
                    selected_weight_minus = max(weights_minus)
                    node_minus.ar_type = "negative_increment"
                else:
                    selected_weight_minus = min(weights_minus)
                    node_minus.ar_type = "negative_decrement"
                
                new_edge_minus = Edge(src=node_minus.name, dest=target_node_name, weight=selected_weight_minus)
                node_minus.out_edges.append(new_edge_minus)
        
        # Add nodes to result if they have outgoing edges
        if node_plus.out_edges:
            split_nodes.append(node_plus)
        if node_minus.out_edges:
            split_nodes.append(node_minus)
        
        return split_nodes
    
    def save_transferred_network(self, output_path: str) -> None:
        """
        Save the transferred network to a file.
        
        Args:
            output_path: Path where the network should be saved
        
        Raises:
            RuntimeError: If no transferred network exists
        """
        if self.transferred_network is None:
            raise RuntimeError(
                "No transferred network available. "
                "Run step1_transfer_weights_by_alpha first."
            )
        
        print("\n" + "=" * 70)
        print("SAVING TRANSFERRED NETWORK")
        print("=" * 70)
        print(f"Output path: {output_path}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        
        try:
            # Get the Marabou network object from the transferred network
            marabou_net = self.transferred_network.acasxu_net
            
            if marabou_net is None:
                raise RuntimeError(
                    "Cannot save: Marabou network object not available"
                )
            
            # Update the Marabou network with transferred weights
            self._update_marabou_weights(marabou_net, self.transferred_network)
            
            # Save using Marabou's write capability
            # Note: MarabouNetworkNNet may not have a direct write method,
            # so we write using a custom method or print weights to file
            self._write_nnet_file(output_path, self.transferred_network)
            
            print(f" Network saved successfully to: {output_path}")
            
        except Exception as e:
            print(f" Error saving network: {str(e)}")
            raise
    
    def _update_marabou_weights(self, marabou_net, network: Network) -> None:
        """
        Update the Marabou network object with weights from our network.
        
        Args:
            marabou_net: Marabou network object
            network: Our Network object with updated weights
        """
        # Update weights in Marabou network
        for layer_idx, layer_weights in enumerate(network.weights):
            for node_idx, node_weights in enumerate(layer_weights):
                for src_idx, weight in enumerate(node_weights):
                    marabou_net.weights[layer_idx][node_idx][src_idx] = weight
        
        # Update biases in Marabou network
        for layer_idx, layer_biases in enumerate(network.biases):
            for node_idx, bias in enumerate(layer_biases):
                marabou_net.biases[layer_idx][node_idx] = bias
    
    def _write_nnet_file(self, output_path: str, network: Network, alpha: float = None, 
                        strategy: str = None) -> None:
        """
        Write network to .nnet file in standard ACAS Xu format.
        
        The .nnet format is:
        Line 0: num_layers, size_of_last_layer
        Line 1: All layer sizes (comma-separated)
        Line 2: 0 (unused)
        Lines 3-4: Input bounds (mins on line 3, maxs on line 4)
        Lines 5-6: Output bounds (mins on line 5, maxs on line 6)
        Lines 7+: Weights (organized by layer and node) then Biases
        
        Args:
            output_path: Output file path
            network: Network to save
            alpha: Maximum perturbation amount (for header info)
            strategy: Perturbation strategy (for header info)
        """
        # Make sure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Get layer sizes
            layer_sizes = [len(layer.nodes) for layer in network.layers]
            num_layers = len(layer_sizes) - 1  # Number of weight layers (not including input layer)
            input_size = layer_sizes[0]
            output_size = layer_sizes[-1]
            max_layer_size = max(layer_sizes)

            f.write(f"{num_layers},{input_size},{output_size},{max_layer_size},\n")

            
            # Line 1: all layer sizes
            f.write(",".join(map(str, layer_sizes)) + ",\n")
            
            # Line 2: unused (set to 0)
            f.write("0,\n")
            
            # Get input and output bounds from original network
            num_inputs = layer_sizes[0]
            num_outputs = layer_sizes[-1]

            acas = self.original_network.acasxu_net

            input_means = [0.5]*(num_inputs + 2)
            input_ranges = [0.5]*(num_inputs + 2)
            output_means = [0.5]*(num_outputs + 3)
            output_ranges = [0.5]*(num_outputs + 3)
            
            # Line 3: input mins
            for i in range(len(input_means)):
                f.write(f"{input_means[i]},")
            f.write("\n")
            
            # Line 4: input maxs
            for i in range(len(input_ranges)):
                f.write(f"{input_ranges[i]},")
            f.write("\n")
            
            # Line 5: output mins
            for i in range(len(output_means)):
                f.write(f"{output_means[i]},")
            f.write("\n")
            
            # Line 6: output maxs
            for i in range(len(output_ranges)):
                f.write(f"{output_ranges[i]},")
            f.write("\n")

            for layer_idx in range(len(network.weights)):
                layer_weights = network.weights[layer_idx]
                layer_biases = network.biases[layer_idx]

                # 🔹 Write weights (one line per neuron)
                for node_weights in layer_weights:
                    for weight in node_weights:
                        f.write(f"{weight},")
                    f.write("\n")

                # 🔹 Write biases (one per neuron)
                for bias in layer_biases:
                    f.write(f"{bias},\n")
    
    
    def get_transfer_info(self) -> dict:
        """
        Get information about the transfer process.
        
        Returns:
            Dictionary with transfer information
        """
        if self.transferred_network is None:
            return {
                'status': 'Network not transferred yet',
                'alpha': None,
                'strategy': None
            }
        
        return {
            'status': 'Network transferred',
            'alpha': self.alpha,
            'strategy': self.strategy,
            'original_network_path': self.nnet_path,
            'transferred_network': self.transferred_network is not None
        }
    
    def get_networks(self) -> tuple:
        """
        Get both original and transferred networks.
        
        Returns:
            Tuple of (original_network, transferred_network)
        
        Raises:
            RuntimeError: If networks are not available
        """
        if self.original_network is None:
            raise RuntimeError("Original network not available")
        
        if self.transferred_network is None:
            raise RuntimeError(
                "Transferred network not available. "
                "Run step1_transfer_weights_by_alpha first."
            )
        
        return (self.original_network, self.transferred_network)
    
    def get_original_network(self) -> Network:
        """Get the original network with zeroed biases."""
        if self.original_network is None:
            raise RuntimeError("Original network not available")
        return self.original_network
    
    def get_transferred_network(self) -> Network:
        """Get the transferred network with perturbed weights and zeroed biases."""
        if self.transferred_network is None:
            raise RuntimeError(
                "Transferred network not available. "
                "Run step1_transfer_weights_by_alpha first."
            )
        return self.transferred_network
    
    def get_abstract_network(self):
        """
        Get the abstract network.
        
        Returns:
            Abstract network object
        
        Raises:
            RuntimeError: If abstract network not available
        """
        if self.abstract_network is None:
            raise RuntimeError(
                "Abstract network not available. "
                "Run step2_read_abstract_network_and_mapping first."
            )
        return self.abstract_network
    
    def get_node_mapping(self) -> Dict[str, Any]:
        """
        Get the forward node mapping (abstract -> original).
        
        Returns:
            Dictionary mapping abstract node names to their original nodes
        
        Raises:
            RuntimeError: If mapping not available
        """
        if self.node_mapping is None:
            raise RuntimeError(
                "Node mapping not available. "
                "Run step2_read_abstract_network_and_mapping first."
            )
        return self.node_mapping
    
    def get_reverse_mapping(self) -> Dict[str, Any]:
        """
        Get the reverse node mapping (original -> abstract).
        
        Returns:
            Dictionary mapping original node names to their abstract nodes
        
        Raises:
            RuntimeError: If reverse mapping not available
        """
        if self.reverse_mapping is None:
            raise RuntimeError(
                "Reverse mapping not available. "
                "Run step2_read_abstract_network_and_mapping first."
            )
        return self.reverse_mapping
    
    def get_all_mappings(self) -> tuple:
        """
        Get both forward and reverse mappings.
        
        Returns:
            Tuple of (abstract_network, node_mapping, reverse_mapping)
        
        Raises:
            RuntimeError: If mappings not available
        """
        return (self.get_abstract_network(), self.get_node_mapping(), self.get_reverse_mapping())
    
    def get_abstract_transfer_network(self):
        """
        Get the abstract transfer network.
        
        Returns:
            Abstract transfer network object
        
        Raises:
            RuntimeError: If abstract transfer network not available
        """
        if not hasattr(self, 'abstract_transfer_network') or self.abstract_transfer_network is None:
            raise RuntimeError(
                "Abstract transfer network not available. "
                "Run step3_abstract_transfer_network first."
            )
        return self.abstract_transfer_network
    
    def save_abstract_transfer_network(self, output_path: str) -> None:
        """
        Save the abstract transfer network to a pickle file.
        
        Args:
            output_path: Path where the pickle file should be saved
            
        Raises:
            RuntimeError: If abstract transfer network not available
        """
        if not hasattr(self, 'abstract_transfer_network') or self.abstract_transfer_network is None:
            raise RuntimeError(
                "Abstract transfer network not available. "
                "Run step3_abstract_transfer_network first."
            )
        
        print("\n" + "=" * 70)
        print("SAVING ABSTRACT TRANSFER NETWORK")
        print("=" * 70)
        print(f"Output path: {output_path}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f" Created output directory: {output_dir}")
        
        try:
            # Save to pickle file
            with open(output_path, 'wb') as f:
                pickle.dump(self.abstract_transfer_network, f)
            
            print(f" Abstract transfer network saved successfully to: {output_path}")
            
            # Print network info
            net = self.abstract_transfer_network
            print(f"\nNetwork info:")
            print(f"  - Number of layers: {len(net.layers)}")
            print(f"  - Total nodes: {sum(len(layer.nodes) for layer in net.layers)}")
            
        except Exception as e:
            print(f" Error saving abstract transfer network: {str(e)}")
            raise
    
    def save_all_networks(self, output_dir: str = "./transferred_networks") -> None:
        """
        Save all networks (original, transferred, abstract, abstract_transfer) to output directory.
        
        Args:
            output_dir: Directory where all networks will be saved
        """
        print("\n" + "=" * 70)
        print("SAVING ALL NETWORKS")
        print("=" * 70)
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f" Created output directory: {output_dir}")
        
        # Save original network
        if self.original_network is not None:
            original_path = os.path.join(output_dir, "original_network.pkl")
            try:
                with open(original_path, 'wb') as f:
                    pickle.dump(self.original_network, f)
                print(f" Original network saved: {original_path}")
            except Exception as e:
                print(f" Error saving original network: {str(e)}")
        
        # Save transferred network
        if self.transferred_network is not None:
            transferred_path = os.path.join(output_dir, "transferred_network.pkl")
            try:
                with open(transferred_path, 'wb') as f:
                    pickle.dump(self.transferred_network, f)
                print(f" Transferred network saved: {transferred_path}")
            except Exception as e:
                print(f" Error saving transferred network: {str(e)}")
        
        # Save abstract network
        if self.abstract_network is not None:
            abstract_path = os.path.join(output_dir, "abstract_network.pkl")
            try:
                with open(abstract_path, 'wb') as f:
                    pickle.dump(self.abstract_network, f)
                print(f" Abstract network saved: {abstract_path}")
            except Exception as e:
                print(f" Error saving abstract network: {str(e)}")
        
        # Save abstract transfer network
        if hasattr(self, 'abstract_transfer_network') and self.abstract_transfer_network is not None:
            abstract_transfer_path = os.path.join(output_dir, "abstract_transfer_network.pkl")
            try:
                with open(abstract_transfer_path, 'wb') as f:
                    pickle.dump(self.abstract_transfer_network, f)
                print(f" Abstract transfer network saved: {abstract_transfer_path}")
            except Exception as e:
                print(f" Error saving abstract transfer network: {str(e)}")
        
        # Save mappings as JSON
        if self.node_mapping is not None:
            mapping_path = os.path.join(output_dir, "node_mapping.json")
            try:
                # Convert to JSON-serializable format
                json_mapping = {}
                for key, value in self.node_mapping.items():
                    json_mapping[key] = {
                        "original_nodes": value.get("original_nodes", []),
                        "layer": value.get("layer", -1),
                        "num_merged": value.get("num_merged", 0)
                    }
                
                with open(mapping_path, 'w') as f:
                    json.dump(json_mapping, f, indent=2)
                print(f" Node mapping saved: {mapping_path}")
            except Exception as e:
                print(f" Error saving node mapping: {str(e)}")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        try:
            metadata = {
                "nnet_file": self.nnet_path,
                "alpha": self.alpha,
                "strategy": self.strategy,
                "networks_saved": {
                    "original": self.original_network is not None,
                    "transferred": self.transferred_network is not None,
                    "abstract": self.abstract_network is not None,
                    "abstract_transfer": hasattr(self, 'abstract_transfer_network') and self.abstract_transfer_network is not None
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f" Metadata saved: {metadata_path}")
        except Exception as e:
            print(f" Error saving metadata: {str(e)}")
        
        print("\n All networks saved successfully")


def parse_args():
    """
    Parse command-line arguments for the network transfer program.
    """
    parser = argparse.ArgumentParser(
        description="Neural Network Transfer Abstraction with CEGAR Verification"
    )
    parser.add_argument(
        "-nn", "--nnet_file",
        dest="nnet_file",
        default="1_1",
        type=str,
        help="ACAS Xu network identifier (e.g., '1_6' for ACASXU_run2a_1_6_batch_2000.nnet)"
    )
    parser.add_argument(
        "-pid", "--property_id",
        dest="property_id",
        default="property_1",
        type=str,
        help="Property ID to verify (e.g., 'property_1')"
    )
    parser.add_argument(
        "-alpha", "--alpha",
        dest="alpha",
        default=0.05,
        type=float,
        help="Alpha value for weight perturbation in Step 1"
    )
    parser.add_argument(
        "-strategy",
        dest="strategy",
        default='random-discrete',
        choices=["random", "worst_case", "best_case",'random-discrete'],
        help="Strategy for weight perturbation"
    )
    parser.add_argument(
        "-r", "--results_dir",
        dest="results_dir",
        default="CEGAR_NN/experiments/dev_exp_results_archive",
        type=str,
        help="Directory containing abstract network pickle files"
    )
    parser.add_argument(
        "-o", "--output_dir",
        dest="output_dir",
        default="abstract_reuse/transferred_networks",
        type=str,
        help="Directory where to save all networks"
    )
    parser.add_argument(
        "-max_refine", "--max_refinement_steps",
        dest="max_refinement_steps",
        default=10,
        type=int,
        help="Maximum CEGAR refinement steps in Step 4"
    )
    parser.add_argument(
        "-log", "--log_csv",
        dest="log_csv",
        default="sample_result.csv",
        type=str,
        help="Path to the CSV file for logging execution metrics"
    )
    parser.add_argument(
        "-layer", "--last_layers",
        dest="last_layers",
        default=10,
        type=int,
        help="Number of layers has been changed in transfer network"
    )
    parser.add_argument(
        "-v", "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose output"
    )
    return parser.parse_args()


def main():
    """
    Main function demonstrating all 4 steps of the network transfer process.
    Step 1: Transfer weights by alpha using specified strategy
    Step 2: Load abstract network and node mappings
    Step 3: Create abstract transfer network using Abstract reuse
    Step 4: Verify with CEGAR-like refinement using new refinement strategy
    """
    # Parse command-line arguments
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("NEURAL NETWORK TRANSFER ABSTRACT PROGRAM")
    print("=" * 70)
    


    BASE_DIR = os.getcwd()
    NNET_BASE = os.path.join(BASE_DIR, "nnet", "ACASXU_run2a_{}_batch_2000.nnet")


    
    # Handle both full paths and network IDs
    if os.path.exists(args.nnet_file):
        # If full path exists, use it directly
        NNET_FILE = args.nnet_file
    elif "/" not in args.nnet_file and "\\" not in args.nnet_file:
        # If it's just an ID (like "1_6" or "random"), format it with NNET_BASE
        NNET_FILE = NNET_BASE.format(args.nnet_file)
    else:
        # Otherwise treat as-is (might be a path that doesn't exist yet)
        NNET_FILE = args.nnet_file
    ALPHA = args.alpha
    STRATEGY = args.strategy
    RESULTS_DIR = args.results_dir
    OUTPUT_DIR = args.output_dir
    PROPERTY_ID = args.property_id
    MAX_REFINE_STEPS = args.max_refinement_steps
    LOG_CSV = args.log_csv
    VERBOSE = args.verbose
    LAYERS = args.last_layers
    
    print(f"\nConfiguration:")
    print(f"  NNET file: {NNET_FILE}")
    print(f"  Property ID: {PROPERTY_ID}")
    print(f"  Alpha: {ALPHA}")
    print(f"  Strategy: {STRATEGY}")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Log CSV: {LOG_CSV}")
    
    # Initialize logger
    logger = None
    if TransferExecutionLogger is not None:
        logger = TransferExecutionLogger(csv_file_path=LOG_CSV)
        logger.start_execution(original_nnet_file=NNET_FILE, property_id=PROPERTY_ID)
    
    try:
        # Initialize transferer
        transferer = NetworkTransferer(NNET_FILE)
        step_timeout = 8000 #seconds
        # STEP 1: Transfer weights by alpha using random strategy
        t1_start = time.time()
        transferred_net, timeout_flag = run_with_timeout(transferer.step1_transfer_weights_by_alpha,
            step_timeout,
            alpha=ALPHA,
            strategy=STRATEGY,
            l = LAYERS
        )
        t1_elapsed = time.time() - t1_start
        
        print(f"\n Transferred network ready")
        
        # Log Step 1
        if logger:
            transferred_net_name = f"{args.nnet_file}_transfer_a{ALPHA}_s{STRATEGY}.nnet"
            if timeout_flag:
                # print("Creating transfer model timed out")
                logger.log_step1(transferred_network_name="N/A", step1_time="N/A")
            else:
                logger.log_step1(transferred_network_name=transferred_net_name, step1_time=t1_elapsed)
        
        # STEP 2: Read abstract network and node mappings
        t2_start = time.time()
        step2_result, timeout_flag = run_with_timeout(transferer.step2_read_abstract_network_and_mapping,
            step_timeout,
            args.nnet_file, PROPERTY_ID, results_dir=RESULTS_DIR
        )
        t2_elapsed = time.time() - t2_start
        
        if timeout_flag:
            # print("verify the original model- timed out")
            if logger:
                logger.log_step2(step2_time="N/A", cegar_results=None)
        else:
            # Unpack results - now includes cegar_results as 4th element
            if len(step2_result) == 4:
                abstract_net, node_map, reverse_map, cegar_results = step2_result
            else:
                abstract_net, node_map, reverse_map = step2_result
                cegar_results = None
            
            print(f"\n Abstract network for source model created by CEGAR-NN")
        
            # Log Step 2
            if logger:
                logger.log_step2(step2_time=t2_elapsed, cegar_results=cegar_results)
        

        # STEP 3: Abstraction of transfer network
        t3_start = time.time()
        abstract_transfer_net, timeout_flag = run_with_timeout(transferer.step3_abstract_transfer_network,step_timeout)
        t3_elapsed = time.time() - t3_start
        
        print(f"\n Abstract transfer network created")
        
        # Log Step 3
        if timeout_flag:
            print("abstract reuse - timed out")
            logger.log_step3(step3_time="N/A")
        else:
            if logger:
                logger.log_step3(step3_time=t3_elapsed)
        
        # Save all networks
        transferer.save_all_networks(output_dir=OUTPUT_DIR)
        
        # STEP 4: VERIFY WITH CEGAR-LIKE REFINEMENT (USING ALGORITHM Y)
        print(f"\n" + "=" * 70)
        print("STEP 4: VERIFY WITH CEGAR-LIKE REFINEMENT (new refinement approach)")
        print("=" * 70)
        

        # Get test property
        try:
            test_property = get_test_property_acas(PROPERTY_ID)
            print(f"\n Test property '{PROPERTY_ID}' loaded successfully")
        except Exception as e:
            print(f"\n Error loading test property: {str(e)}")
            print(f"  Skipping Transfer Abstract verification")
            test_property = None
        
        # Run verification with refinement if property is available
        if test_property is not None:
            t4_start = time.time()
            try:
                # Call step4_verify_with_cegar_refinement function
                # This uses Algorithm Y (extremal splitting) for refinement
                verification_result, timeout_flag = run_with_timeout( transferer.step4_verify_with_cegar_refinement,
                    step_timeout,
                    test_property=test_property,
                    max_refinement_steps=MAX_REFINE_STEPS,
                    refinement_sequence_length=100
                )
            
            except Exception as e:
                print(f"\n Error during Transfer Abstract verification: {str(e)}")
                if VERBOSE:
                    import traceback
                    traceback.print_exc()
                verification_result = {
                    'result': 'ERROR',
                    'error': str(e),
                    'message': 'Verification error',
                    'num_refine_steps': 0
                }
            
            t4_elapsed = time.time() - t4_start
        
            # Log Step 4
            if timeout_flag:
                logger.log_step4(step4_time="N/A", verification_result=None)
                print("Verifying the transfer abstract- timed out")
            else:
                if logger:
                    logger.log_step4(step4_time=t4_elapsed, verification_result=verification_result)
        else:
            verification_result = None
            t4_elapsed = 0
        
        #step 5: verify the transfer model without reusing abstract
        print("Verifying transfer model from scratch by CEGAR-NN .. ")
        t5_start = time.time()
        step5_result, timeout_flag = run_with_timeout(transferer.step2_read_abstract_network_and_mapping,
            step_timeout,
            args.nnet_file, PROPERTY_ID, results_dir=RESULTS_DIR, model = 'transfer'
        )
        t5_elapsed = time.time() - t5_start
                # Unpack results - now includes cegar_results as 4th element

        if timeout_flag:
            print("Verifying transfer network -> timed out")
            if logger:
                logger.log_step5(step5_time="N/A", cegar_results=None)
        else:
            # Unpack results - now includes cegar_results as 4th element
            if len(step5_result) == 4:
                abstract_net, node_map, reverse_map, cegar_results = step5_result
            else:
                abstract_net, node_map, reverse_map = step5_result
                cegar_results = None
            
            print(f"\n Abstract network and mappings loaded")
        
            # Log Step 5
            if logger:
                logger.log_step5(step5_time=t5_elapsed, cegar_results=cegar_results)
            print("verification result --> ")
            print(cegar_results)
        
        print(f"\n Transfer network verifiction has been completed without reusing abstract")


        print("\n" + "=" * 70)
        print("FINAL TRANSFER INFORMATION")
        print("=" * 70)
        info = transferer.get_transfer_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        if test_property:
            print(f"  - Verification result: {verification_result['result']}")
        print(f"\nAll networks saved to: {OUTPUT_DIR}")
        
        # Finish logging
        if logger:
            logger.finish_execution(status='completed', notes='All steps completed successfully')
        
        return transferer
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Log error
        if logger:
            logger.log_error(error_message=str(e))
        
        sys.exit(1)

if __name__ == "__main__":
    transferer = main()
