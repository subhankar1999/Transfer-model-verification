#!/usr/bin/env python3

"""
Extract node mapping from abstract network to original network.
This script analyzes node names to determine which original nodes were merged 
to create each abstract node.
"""

import pickle
import glob
import os
import sys
import json

# Change to artifact directory first
artifact_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
os.chdir(artifact_dir)

# Add the artifact directory to the path
sys.path.insert(0, artifact_dir)
sys.path.insert(0, os.path.join(artifact_dir, 'CEGAR_NN'))

print(f"Working directory: {os.getcwd()}")

# Import the necessary classes for unpickling
try:
    from CEGAR_NN.core.data_structures.ARNode import ARNode
    from CEGAR_NN.core.data_structures.Layer import Layer
    from CEGAR_NN.core.data_structures.Edge import Edge
    from CEGAR_NN.core.data_structures.Network import Network
    print("Successfully imported required data structure classes")
except ImportError as e:
    print(f"Warning: Could not import all classes: {e}")

# Create a custom unpickler to handle module path changes
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module, name)

# Find the latest final_network pickle file
results_dir = 'CEGAR_NN/experiments/dev_exp_results_archive'
pkl_files = glob.glob(os.path.join(results_dir, 'final_network_*.pkl'))

if not pkl_files:
    print(f"No pickle files found in {results_dir}")
else:
    # Sort by modification time and get the latest
    latest_pkl = max(pkl_files, key=os.path.getmtime)
    print(f"\nLoading: {latest_pkl}")
    
    try:
        # Load the final refined network
        with open(latest_pkl, 'rb') as f:
            unpickler = CustomUnpickler(f)
            abstract_network = unpickler.load()
        
        print("\n=== Node Mapping Analysis ===")
        print("Extracting mapping from abstract nodes to original nodes...\n")
        
        # Create mapping structure
        node_mapping = {}
        
        # Parse node names to extract original nodes that were merged
        for layer_idx, layer in enumerate(abstract_network.layers):
            layer_mapping = {}
            print(f"Layer {layer_idx} ({layer.type_name}):")
            print("-" * 80)
            
            for node in layer.nodes:
                node_name = node.name
                # Node names are formatted as:
                # - Single nodes: "x_layer_id_node_id_type" (e.g., "x_1_5_inc")
                # - Merged nodes: "x_1_5_inc+x_1_6_inc+..." 
                
                original_nodes = node_name.split("+")
                layer_mapping[node_name] = original_nodes
                
                print(f"\nAbstract Node: {node_name}")
                print(f"  Original Nodes ({len(original_nodes)}):")
                for i, orig_node in enumerate(original_nodes, 1):
                    print(f"    {i}. {orig_node}")
                
                if hasattr(node, 'bias'):
                    print(f"  Bias: {node.bias}")
                if hasattr(node, 'in_edges'):
                    print(f"  In-degree: {len(node.in_edges)}")
                if hasattr(node, 'out_edges'):
                    print(f"  Out-degree: {len(node.out_edges)}")
            
            node_mapping[f"layer_{layer_idx}"] = layer_mapping
            print("\n")
        
        # Save mapping to JSON file
        print("\n=== Saving Mapping to File ===")
        
        # Convert the mapping to a more readable format
        readable_mapping = {}
        for layer_idx, layer in enumerate(abstract_network.layers):
            readable_mapping[f"Layer_{layer_idx}_{layer.type_name}"] = {}
            for node in layer.nodes:
                original_nodes = node.name.split("+")
                readable_mapping[f"Layer_{layer_idx}_{layer.type_name}"][node.name] = {
                    "original_nodes": original_nodes,
                    "num_merged": len(original_nodes),
                    "bias": node.bias if hasattr(node, 'bias') else None,
                    "in_degree": len(node.in_edges) if hasattr(node, 'in_edges') else 0,
                    "out_degree": len(node.out_edges) if hasattr(node, 'out_edges') else 0
                }
        
        # Save as JSON
        mapping_file = "node_mapping.json"
        with open(mapping_file, "w") as f:
            json.dump(readable_mapping, f, indent=2)
        print(f"Saved node mapping to: {mapping_file}")
        
        # Also create a reverse mapping (original node -> abstract node)
        print("\n=== Creating Reverse Mapping (Original -> Abstract) ===")
        reverse_mapping = {}
        for layer_idx, layer in enumerate(abstract_network.layers):
            for abstract_node in layer.nodes:
                original_nodes = abstract_node.name.split("+")
                for orig_node in original_nodes:
                    if orig_node not in reverse_mapping:
                        reverse_mapping[orig_node] = []
                    reverse_mapping[orig_node].append({
                        "abstract_node": abstract_node.name,
                        "layer": layer_idx
                    })
        
        # Save reverse mapping
        reverse_mapping_file = "reverse_node_mapping.json"
        with open(reverse_mapping_file, "w") as f:
            json.dump(reverse_mapping, f, indent=2)
        print(f"Saved reverse mapping to: {reverse_mapping_file}")
        
        # Generate statistics
        print("\n=== Mapping Statistics ===")
        total_original_nodes = 0
        total_abstract_nodes = 0
        merged_info = {}
        
        for layer_idx, layer in enumerate(abstract_network.layers):
            num_abstract = len(layer.nodes)
            total_abstract_nodes += num_abstract
            
            for node in layer.nodes:
                num_merged = len(node.name.split("+"))
                total_original_nodes += num_merged
                
                if num_merged not in merged_info:
                    merged_info[num_merged] = 0
                merged_info[num_merged] += 1
        
        print(f"Total original nodes: {total_original_nodes}")
        print(f"Total abstract nodes: {total_abstract_nodes}")
        print(f"Compression ratio: {total_original_nodes / total_abstract_nodes:.2f}x")
        print("\nMerging statistics:")
        for merge_count in sorted(merged_info.keys()):
            print(f"  {merge_count} nodes merged: {merged_info[merge_count]} abstract nodes")
        
        print("\n=== Files Created ===")
        print(f"1. {mapping_file} - Abstract node -> Original nodes mapping")
        print(f"2. {reverse_mapping_file} - Original node -> Abstract node mapping")
        
    except Exception as e:
        print(f"Error loading or processing network: {e}")
        import traceback
        traceback.print_exc()
