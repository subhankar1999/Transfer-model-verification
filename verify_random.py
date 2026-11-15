# from maraboupy import Marabou

# network = Marabou.read_nnet("random_nnet/network1.nnet")
# print("Successfully loaded in Marabou!")

import subprocess
import argparse


parser = argparse.ArgumentParser(
    description="Execute all random .nnet model."
)
parser.add_argument(
    "num_hidden", type=int, help="Number of hidden layers"
)
parser.add_argument(
    "neurons_per_hidden",
    type=int,
    help="Number of neurons in each hidden layer",
)
args = parser.parse_args()

num_inputs = 5
num_outputs = 5
num_hidden = args.num_hidden
neurons_per_hidden = args.neurons_per_hidden

item = 5
for item in range (1,16):
    output_folder = "random_nnet/"
    output_folder += "random"+str(num_hidden)+"x"+str(neurons_per_hidden)+"/"
    file_name = "network"+str(item)+".nnet"
    dest_full_path = output_folder+file_name

    command = [
        "python3",
        "generate_random_network.py",
        str(num_inputs),
        str(num_outputs),
        str(num_hidden),
        str(neurons_per_hidden),
        dest_full_path,
    ]

    result = subprocess.run(command, capture_output=True, text = True)

    print(result.stdout)
    if result.returncode != 0:
        print("Error: ", result.stderr)

    marabou_nnet_file = "nnet/"+file_name

    subprocess.run(["cp", dest_full_path, marabou_nnet_file], check=True)

    result_dir = "random_exp_result/"
    result_file = "random"+str(num_hidden)+"x"+str(neurons_per_hidden)+".csv"
    result_path = result_dir+result_file

    ar_command = [
        "python3",
        "abstract_reuse/transfer_abstract.py",
        "--nnet_file",
        marabou_nnet_file,
        "--log_csv",
        result_path,
    ]
    result = subprocess.run(ar_command, capture_output=True, text = True)

    print(result.stdout)
    if result.returncode != 0:
        print("Error: ", result.stderr)

    subprocess.run(["rm",marabou_nnet_file], check = True)