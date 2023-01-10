from utils import *
import time
import getopt
import sys

modes = ['degree', 'betweenness', 'hop_count'] 
BFS_time_list = []
query_time_100K_list = []
gen_order_time_list = []
avg_label_size_list = []
# 从命令行提取参数
def fetch_map_name():
    help_msg = "python script.py -i [input_file]"
    try:
        options, _ = getopt.getopt(sys.argv[1:], "-i:", ["input="])
        map_file_name = options[0][1]
        return map_file_name
    except:
        print(help_msg)
        exit()


def BFS_strategy_base(pll_class, mode):
    start = time.perf_counter()
    print(f"\n*************{mode}****************")
    if  mode == "degree":
        pll_class.gen_degree_base_order()
    elif mode == 'betweenness':
        pll_class.gen_betweeness_base_order()
    elif mode ==  'hop_count':
        hop_count = pll_class.gen_2_hop_base_order()
    end = time.perf_counter()
    gen_order_time = end-start
    print(f"finish generating order, time cost: {gen_order_time:.2f}")
    pll_class.write_order(mode)
    # print(f"{mode} :{pll_class.vertex_order}\n")
    BFS_time, avg_label_size = pll_class.build_index()
    query_time_100K = pll_class.query_100K()

    gen_order_time_list.append(float("%.2f"%gen_order_time))
    BFS_time_list.append(float("%.2f"%BFS_time))
    query_time_100K_list.append(float("%.2f"%query_time_100K))
    avg_label_size_list.append(float("%.2f"%avg_label_size))
    
# map_file_name = fetch_map_name()
# pll_class = PrunedLandmarkLabeling(map_file_name = map_file_name)
pll_class = PrunedLandmarkLabeling(map_file_name = 'test')
nNodes = pll_class.G.number_of_nodes()

for mode in modes:
    BFS_strategy_base(pll_class, mode)
pll_class.output_to_excel(gen_order_time_list, BFS_time_list, query_time_100K_list, avg_label_size_list)
