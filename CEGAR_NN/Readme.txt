*** An Abstraction-Based Framework for DNN Verification Artifact
*** April 2020 

Welcome to the Abstraction-Based Framework for DNN Verification
artifact. This file contains three sections:

(i) technical details about docker image and host where image was
prepared;

(ii) information on the abstraction mechanism code and the various
folders contained within the artifact; and

(iii) instructions for using the tool and for running the experiments
described in the paper.



Section (i): Technical details about docker image and host where image
was prepared
----------------------------------------------------------------------
- OS: 
    Ubuntu 18.04.3 LTS
    Linux 4.15.0-88-generic #88-Ubuntu SMP x86_64 x86_64 x86_64 GNU/Linux
    
- RAM:
             total    used    free  shared  buff/cache   available
    Mem:       15G    7.4G    795M    995M        7.3G        7.0G
    Swap:     2.0G    136M    1.9G

- Number of CPUS: 8
- CPU Frequency: 800.016 MHz (min 400.0000, max 4000.0000)



Section (ii): Information regarding the code
--------------------------------------------

[This part of the document assumes some familiarity with the concepts
described in the paper.]

There are multiple directories under the root dir (/)

1. The directory Cegar_NN contains the code for the abstraction and refinement 
mechanism, including the data structures (under core/data_structures) that are 
used for network manipulation, as well as modules for the main ingredients of 
the mechanism: pre-processing (core/pre_process), abstraction (core/abstraction)
and refinement (core/refinement). 

2. Utilities like read .nnet formatted file and interfacing with the Marabou 
infrastructure are located under the utils directory (core/utils).

3. The code to run an experiment is stored under another CEGAR_NN sub-directory 
(experiments/run)

4. The Marabou directories, Marabou_Reg (regular) and Marabou_Adv
(adversarial). These two directories contain the Marabou verification
engine, which we used as a black-box tool. There are two versions
because we have made slight adjustments to Marabou's interface, to
more easily use within our tool.


There are many comments annotating the code base. Below we provide
pointers to some of the key files within the /CEGAR_NN/core directory,
which the reviewers may wish to focus on:

- The core/data_structures directory includes 4 modules with 4
classes: Edge, ARNode, Layer and Network. Class ARNode, which stands
for "Abstraction-Refinement node", represents a node in an abstract
network, and includes the key "split_pos_neg" and "split_inc_dec" functions.
The Layer class represents a layer of nodes, and includes key
"abstract" and "split" functions.  The Network class, which represents
a full neural network, includes meta-data pertinent to the abstraction
process of its constituent nodes, and this data is heavily manipulated
during the abstraction and refinement process (e.g., the name2node_map
mapping from node name to an ARNode)

- The preprocessing code is actually spread across multiple modules and
classes: the main functions pre_process, split_pos_neg, split_inc_dec
are located under core/pre_process, whereas the split_pos_neg and
split_inc_dec functions reside in the Layer and ARNode classes. Our
code takes a Network and applies the 2 pre-processing stages:
split_pos_neg() is applied to split nodes to positive and negative
nodes, and then split_inc_dec() is applied to split the positive and
negative nodes into increasing and decreasing nodes (thus, the
preprocessing process results in 4 types of nodes, as described in
the paper).

- The abstraction code is located mainly under the core/abstraction
modules, but for technical reasons the naive.abstract_network()
function actually uses the method abstract() of the Layer class. This
method generates 4 abstract nodes, each containing all nodes with a
certain type. The function that implements the abstraction scheme
denoted Algorithm 2 in the paper, alg2.heuristic_abstract(), uses the
abstraction.step.union_couple_of_nodes() function that actually
implements the abstraction step.

- The refinement code is located under core/refinement, and is
responsible for the refinement steps performed after a spurious
example arises. The main function is refine.refine(), which calls
refine.get_refinement_sequence() to get the best refinement sequence
(according to CEGAR or weight-based method, which ever is being used;
CEGAR is used if **kws includes the "example" key, and weight-based is
used otherwise). The refinement steps obtained are then applied by
using the function refinement/step/split_back(), which actually
implements the refinement step.


Section (iii): Instructions for using the tool and for running the
experiments described in the paper
--------------------------------------------------------------------

- Booting the docker image:

Step 1: load image from tar file

  sudo docker load -i ../cegar_nn_artifact_docker_image

Step 2: get the docker image name

  sudo docker images (next, use the relevant image name)

Step 3: execute the image to run commands interactively

  sudo docker run -it <image_name> bash

  #example: sudo docker run -it cegarabou_2020_artifact bash



- Running the experiments reported in the paper:

IMPORTANT NOTE:

Most of the experiments described in the paper take a long time to
run. Running them all sequentially on a laptop will probably take a
few weeks (we ran the experiments in parallel, using several
machines). Below we mention some of the faster experiments - the
reviewers may wish to focus on them. For the longer experiments,
many of them were much long and we stopped them after 20 hours. Here
we supply results for experiments that finished in less than 20 hours,
you can find them (for marabou_with_abstraction) in
/CEGAR_NN/results/marabou_with_ar__basic_properties_outfiles/ directory
and (for vanilla_marabou) in
/CEGAR_NN/results/vanilla_marabou__basic_properties_outfiles/ directory
in the docker image.
The reviewers may compare the prefix of the result files that they've
obtained to those in our files, to see that they match.
Notice that the names include two types of abstraction methods: complete (naive)
and heuristic (alg 2), and two refinement methods cegar (cegar) and cetar
(Weight-Based), and that not familiar parameters in the results filenames
are irrelevant.

The machines we used to run the experiments all had the following
specifications:
- OS: Ubuntu 16.04.1 LTS
- Cores: 8 (although our tool is single-threaded)
- RAM: 32 gigabyte
- CPU: Intel(R) Xeon(R) CPU E5-2637 v4 @ 3.50GHz


To run a single experiment:

  python /CEGAR_NN/experiments/run/one_experiment_1.py  # --help

NOTE: to interrupt a running experiment, use ctrl + \

  # example: run an experiment with vanilla Marabou, on network 1_8,
    with property basic_2

  python /CEGAR_NN/experiments/run/one_experiment_1.py -nn 1_8 -pid basic_2 -m marabou

  # example: run an experiment using Marabou with our
  # abstraction/refinement enhancement, using the naive initial
  # abstraction algorithm and CEGAR refinements:
  
  python /CEGAR_NN/experiments/run/one_experiment_1.py -nn 5_7 -pid basic_2 -m marabou_with_ar -a naive -r cegar


The general format of the command line for running a single experiment is:

  python one_experiment_1.py \
  --net_name <1_1/.../1_9/2_1/.../2_9/.../4_1/4_9> \
  --property_id <basic_1/basic_2/adversarial_1/adversarial_2/.../adversarial_19> \
  --mechanism <marabou/marabou_with_ar> \
  --abstraction naive/alg2 \
  --refinement cegar/weight_based

Properties info:
sanity_UNSAT and sanity_SAT are used for sanity check.

basic_1 and basic_2 are the properties used for comparing:
    - naive to Alg 2 (figure 7 in the paper), on networks 1_1 to 4_9.
    - cegar to Weight-Based (figure 8 in the paper), on networks 1_1 to 4_9.
    - Marabou with and without abstraction (figure 9 in the paper),
on networks 1_1 to 5_9.

[adversarial_0, ..., adversarial_19] are the properties used for
    - comparing Marabou with and without abstraction (figure 10 in the paper),
on networks 1_1 to 5_9.


To view the results:

The results are both printed to screen and saved into
/CEGAR_NN/experiments/results_archive directory, where each result
file is named according to experiment's parameters, time and date.

Two different files are generated for each experiment:

1. A file with an "experiment" prefix: results in a human readable
   format (recommended)
2. A file with a "df_experiment" prefix: results in json format


Thank you for taking the time to review this artifact!

