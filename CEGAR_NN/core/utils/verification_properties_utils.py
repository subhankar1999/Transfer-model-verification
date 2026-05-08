#!/usr/bin/env python3

import json
from typing import Dict

from core.configuration import consts

TEST_PROPERTY_ACAS = {

    "property_1": {
        "type": "basic",
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 3.9911256459}),
            ]
    },
"property_2": {
        "type": "acas_xu_conjunction",
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                # representation of inequasion: +1*y0 -1*y1 <= 0
                ([(1, 0), (-1, 1)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y2 <= 0
                ([(1, 0), (-1, 2)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y3 <= 0
                ([(1, 0), (-1, 3)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y4 <= 0
                ([(1, 0), (-1, 4)], {"Upper": 0}),
            ]
    },
    "property_3": {
        "type": "acas_xu_conjunction",
        "input":
            [
                (0, {"Lower": -0.3035311561, "Upper": -0.2985528119}),
                (1, {"Lower": -0.0095492966, "Upper": 0.0095492966}),
                (2, {"Lower": 0.4933803236, "Upper": 0.5}),
                (3, {"Lower": 0.3, "Upper": 0.5}),
                (4, {"Lower": 0.3, "Upper": 0.5}),
            ],
        "output":
            [
                # representation of inequasion: +1*y0 -1*y1 <= 0
                ([(1, 0), (-1, 1)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y2 <= 0
                ([(1, 0), (-1, 2)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y3 <= 0
                ([(1, 0), (-1, 3)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y4 <= 0
                ([(1, 0), (-1, 4)], {"Upper": 0}),
            ]
    },
    "property_4": {
        "type": "acas_xu_conjunction",
        "input":
            [
                (0, {"Lower": -0.3035311561, "Upper": -0.2985528119}),
                (1, {"Lower": -0.0095492966, "Upper": 0.0095492966}),
                (2, {"Lower": 0.0, "Upper": 0.0}),
                (3, {"Lower": 0.3181818182, "Upper": 0.5}),
                (4, {"Lower": 0.0833333333, "Upper": 0.1666666667}),
            ],
        "output":
            [
                # representation of inequasion: +1*y0 -1*y1 <= 0
                ([(1, 0), (-1, 1)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y2 <= 0
                ([(1, 0), (-1, 2)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y3 <= 0
                ([(1, 0), (-1, 3)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y4 <= 0
                ([(1, 0), (-1, 4)], {"Upper": 0}),
            ]
    }
}


def is_satisfying_assignment(network, test_property, output, variables2nodes) \
        -> bool:
    """
    returns if output is valid a.t. test_property output bounds
    @output - dict from node_name to value
    @test_property - see get_query() method documentation

    # for the reader, e.g:
    output
    {'x_7_0': 2.4388628607018603,
     'x_7_1': 0.1540985927922452,
     'x_7_2': 0.44519896688589594,
     'x_7_3': 1.0787132557294057,
     'x_7_4': 0.49820921502029597}

    output_vars
    [(605, 'x_7_0'),
     (606, 'x_7_1'),
     (607, 'x_7_2'),
     (608, 'x_7_3'),
     (609, 'x_7_4')]

    test_property["output"][0]
    (0, {'Upper': 3.9911256459})

    test_property["output"][0][0]
    0

    output_vars[test_property["output"][0][0]]
    (605, 'x_7_0')

    output_vars[test_property["output"][0][0]][1]
    'x_7_0'

    output[output_vars[test_property["output"][0][0]][1]]
    2.4388628607018603
    """
    output_property = test_property["output"]
    # print(f"output_property={output_property}")
    sorted_vars = sorted(variables2nodes.items(), key=lambda x: x[0])
    # print(f"sorted_vars={sorted_vars}")
    # couples of (variable index, node name) of output layer only
    output_vars = sorted_vars[-len(network.layers[-1].nodes):]
    # print(f"output_vars={output_vars}")
    for variable, bounds in output_property:
        index_name = output_vars[variable]
        # print(f"index_name={index_name}")
        node_name = index_name[1]
        key = node_name.replace("_inc", "")
        # print(f"node_name={node_name}")
        if "Lower" in bounds.keys():
            # print("Lower in bounds.keys")
            # print(node_name)
            lower_bound = bounds["Lower"]
            # print(f"lower_bound = {lower_bound}")
            # print(f"Output : {output}")
            # print(f"Node name : {(node_name)}")
            # print(f"Bounds : {bounds}")
            # if lower_bound > output[int(node_name.replace("_inc", "").split("_")[-1])]:
            if lower_bound>output[key]:
                return False
        if "Upper" in bounds.keys():
            upper_bound = bounds["Upper"]
            if output[int(node_name.replace("_inc", "").split("_")[-1])] > upper_bound:
                return False
    return True


def read_test_property(property_filename) -> Dict:
    with open(property_filename) as pf:
        test_property = json.loads(pf.read())
    return test_property


def write_test_property(test_property, filename) -> None:
    with open(filename, "w") as pf:
        pf.write(json.dumps(test_property))


def get_test_property_acas(property_id:str=consts.PROPERTY_ID) -> Dict:
    test_property = TEST_PROPERTY_ACAS.get(property_id, None)
    if test_property is None:
        raise Exception(f"wrong property_id: {property_id}")
    return test_property


def get_test_property_tiny() -> Dict:
    test_property_tiny = {
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.1, "Upper": 0.1}),
                (2, {"Lower": -0.1, "Upper": 0.1}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 250.0}),
                (1, {"Lower": 250.0}),
                (2, {"Lower": 250.0}),
                (3, {"Lower": 250.0}),
                (4, {"Lower": 250.0})
            ]
    }
    return test_property_tiny


def get_test_property_input_2_output_1() -> Dict:
    """
    :return: test property of random net of order 2,2,1
    """
    test_property_tiny = {
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.1, "Upper": 0.1}),
            ],
        "output":
            [
                (0, {"Lower": 250.0}),
            ]
    }
    return test_property_tiny


def get_test_property_input_3_output_1() -> Dict:
    """
    :return: test property of random net of order 2,2,1
    """
    test_property_tiny = {
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.1, "Upper": 0.1}),
                (2, {"Lower": -0.1, "Upper": 0.1}),
            ],
        "output":
            [
                (0, {"Lower": 250.0}),
            ]
    }
    return test_property_tiny


def get_winner_and_runner_up(adversarial_test_property):
    """
    in acas, the minimal value is chosen as answer
    this function check which nodes are the winner (minimal) and runner_up (the
    second minimal) a.t. specific test_property
    :param adversarial_test_property: test_property
    :return: values+indices of two minimal output nodes
    """
    s = []
    for index, d_val in adversarial_test_property["output"]:
        s.append((index, d_val["Lower"]))
    winner, runner = float("inf"), float("inf")
    w_ind, r_ind = None, None
    for (x,y) in s:
        if winner > y:
            winner = y
            w_ind = x
    for (x,y) in s:
        if x == w_ind:
            continue
        if runner > y:
            runner = y
            r_ind = x
    return winner, w_ind, runner, r_ind


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--property_filename", dest="filename", default=consts.PROPERY_FILNAME_ACAS)
    parser.add_argument("-rw", "--read_write", dest="read_write", default="r", choices=["r", "w"],
                        help="read property from file or write property to file")
    args = parser.parse_args()
    if args.read_write == "r":
        prop = read_test_property(args.filename)
        print(prop)
    else:  # "w"
        prop = get_test_property_acas()
        write_test_property(test_property=prop, filename=args.filename)
