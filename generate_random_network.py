import argparse
from pathlib import Path
import numpy as np


def generate_nnet(
    num_inputs: int,
    num_outputs: int,
    num_hidden: int,
    neurons_per_hidden: int,
    output_file: str,
):
    layer_sizes = (
        [num_inputs] + [neurons_per_hidden] * num_hidden + [num_outputs]
    )

    # In .nnet format, numLayers is the count of weight transformations (total_layers - 1)
    num_layers = len(layer_sizes) - 1
    max_layer_size = max(layer_sizes)

    lines = []

    # Line 1: numLayers, inputSize, outputSize, maxLayerSize
    lines.append(f"{num_layers},{num_inputs},{num_outputs},{max_layer_size},")

    # Line 2: Array of layer sizes
    lines.append(",".join(map(str, layer_sizes)) + ",")

    # Line 3: Flag (0 for standard network)
    lines.append("0,")

    # Line 4: Minimum values for inputs
    lines.append(",".join(["0.0"] * num_inputs) + ",")

    # Line 5: Maximum values for inputs
    lines.append(",".join(["1.0"] * num_inputs) + ",")

    # Line 6: Mean values for inputs and target output
    lines.append(",".join(["0.0"] * (num_inputs + 1)) + ",")

    # Line 7: Range values for inputs and target output
    lines.append(",".join(["1.0"] * (num_inputs + 1)) + ",")

    # Generate Weights and Biases for each layer transformation
    for l in range(num_layers):
        n_in = layer_sizes[l]
        n_out = layer_sizes[l + 1]

        # Kaiming/He normal initialization for ReLU networks
        std = np.sqrt(2.0 / n_in)
        weights = np.random.normal(0.0, std, size=(n_out, n_in))
        biases = np.random.normal(0.0, 0.1, size=(n_out,))

        # Write Weight Matrix (n_out rows x n_in cols)
        for row in weights:
            row_str = ",".join(f"{val:.8e}" for val in row) + ","
            lines.append(row_str)

        # Write Bias Vector (n_out values)
        for val in biases:
            lines.append(f"{val:.8e},")

    # Ensure parent directory structure exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to destination file
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Successfully generated Marabou-verifiable NNet at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a random Marabou-verifiable .nnet model."
    )
    parser.add_argument(
        "num_inputs", type=int, help="Number of neurons in the input layer"
    )
    parser.add_argument(
        "num_outputs", type=int, help="Number of neurons in the output layer"
    )
    parser.add_argument(
        "num_hidden", type=int, help="Number of hidden layers"
    )
    parser.add_argument(
        "neurons_per_hidden",
        type=int,
        help="Number of neurons in each hidden layer",
    )
    parser.add_argument(
        "output_file", type=str, help="Destination .nnet file path"
    )

    args = parser.parse_args()

    generate_nnet(
        args.num_inputs,
        args.num_outputs,
        args.num_hidden,
        args.neurons_per_hidden,
        args.output_file,
    )