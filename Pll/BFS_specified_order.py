from utils import *
import numpy as np
import getopt
import sys

def fetch_map_name():
    help_msg = "python BFS_specified_order.py -i [input_file]"
    try:
        options, _ = getopt.getopt(sys.argv[1:], "-i:", ["input="])
        map_file_name = options[0][1]
        return map_file_name
    except:
        print(help_msg)
        exit()

if __name__== "__main__":
    map_file_name = fetch_map_name()
    pll_class = PrunedLandmarkLabeling(map_file_name)

    order = pll_class.load_specified_order()
    pll_class.vertex_order = order
    # print(order)
    pll_class.build_index()
    print(pll_class.index)
    print(pll_class.vertex_order)
