import time
import networkx as nx
import networkit as nk
import queue as Q
import json
import numpy as np
import concurrent.futures
from random import randint
import pandas as pd

max_length = 999999999

class PrunedLandmarkLabeling(object):
    def __init__(self, map_file_name = ""):
        super(PrunedLandmarkLabeling, self).__init__()
        self.map_file_name = map_file_name
        self.G= self.read_graph()
        self.index = {}
        self.vertex_order = []
        self.order2index = []
        self.vertex_order = []

    def write_betweenness(self, betweenness):
        fileName = "dataset/betweenness/"+ self.map_file_name+"_betweenness.txt"
        f = open(fileName, 'w')
        f.write(str(betweenness))
        f.close()

    def write_order(self, mode):
        # order写入单独文件
        with open("dataset/order/" + mode + "/" + self.map_file_name + f"_{mode}_order.txt", 'w') as f:
            write_data = list(self.vertex_order)
            f.write(str(write_data))
        f.close()

    def write_2_hop_count(self, hop_count):
        fileName = "dataset/hop_count/"+ self.map_file_name+"_hop_count.txt"
        f = open(fileName, 'w')
        f.write(str(list(hop_count)))
        f.close()

    # 使用networkx读入图
    def read_graph(self):
        print(f"\n************Read {self.map_file_name}*************")
        G = nx.DiGraph()
        f = open("dataset/map_file/" + self.map_file_name, 'r')
        data = f.readlines()
        f.close()
        for idx, lines in enumerate(data):
            if (idx < 2):
                continue
            src, dest, dist, is_one_way = lines.split(" ")
            G.add_weighted_edges_from([(int(src), int(dest), float(dist))])
            #  is_one_way=0 => 为无向图
            if (int(is_one_way) == 0):
                G.add_weighted_edges_from([(int(dest), int(src), float(dist))])
        # 输出节点和边的个数
        print("Finish Reading Graph!")
        print(f"nodes:{len(G.nodes())}   edges:{len(G.edges())}")
        return G

    def load_specified_order(self):
        fileName = "dataset/order/user_define/specified_order"
        f = open(fileName,"r")
        data = f.readline()

        order = np.array(eval(data), dtype= np.int64)
        return order

    # 查询俩点间的距离
    def query(self, src, dest):
        src_list = self.index[src]["backward"]
        dest_list = self.index[dest]["forward"]
        i = 0
        j = 0
        shortest_dist = max_length
        # 构建好的index里label是按照nodes_list里的节点顺序排序的，所以可以这么写
        while i < len(src_list) and j < len(dest_list):
            if src_list[i][0] == dest_list[j][0]:
                # print(src_list[i][0] == dest_list[j][0])
                curr_dist = src_list[i][1] + dest_list[j][1]
                if(curr_dist == 0 or curr_dist == 1):
                    shortest_dist = curr_dist
                    break
                # print(curr_dist)
                # 当前距离未必为最小，若找到更小的距离，更新最小距离，之前的 hop_nodes作废，用新的代替
                if curr_dist < shortest_dist:
                    shortest_dist = curr_dist
                i += 1
                j += 1
            elif self.order2index[src_list[i][0]] < self.order2index[dest_list[j][0]]:
                    i += 1
            else:
                    j += 1
        return shortest_dist

    def query_100K(self):
        start = time.perf_counter()
        nNodes = self.G.number_of_nodes()
        for i in range(100000):
            src = randint(0,nNodes-1)
            dest = randint(0,nNodes-1)
            _ = self.query(src,dest)
        end = time.perf_counter()
        print(f'finish query_100K, time cost: {(end-start):.2f}')
        return (end-start)

    def gen_degree_base_order(self):
        rank_dict = np.array(sorted(self.G.degree, key=lambda x: x[1], reverse=True))
        self.generate_order_for_BFS(rank_dict)
        return self.vertex_order

    def gen_betweeness_base_order(self):
        g_nkit = self.nx2nkit(self.G)
        bet_raw = nk.centrality.Betweenness(g_nkit,normalized=True).run()
        betweenness_data = bet_raw.scores()
        # nodes_list = nx.betweenness_centrality(G, normalized = True, weight="weight")
        # print(nodes_list)
        self.write_betweenness(betweenness_data)
        rank_dict = bet_raw.ranking()
        self.generate_order_for_BFS(rank_dict)

    def generate_order_for_BFS(self, rank_dict):
        result = np.empty((0,), dtype=np.int64)
        nNodes = self.G.number_of_nodes()
        for i in range(nNodes):
            result = np.append(result, int(rank_dict[i][0]))
        self.vertex_order = result
        
    def nx2nkit(self, g_nx):
        node_num = g_nx.number_of_nodes()
        g_nkit = nk.Graph(directed=True, weighted = True)

        for i in range(node_num):
            g_nkit.addNode()

        for e1,e2 in g_nx.edges():
            g_nkit.addEdge(e1,e2,w=g_nx[e1][e2]['weight'])
        return g_nkit

    def gen_2_hop_base_order(self):
        nNodes = self.G.number_of_nodes()
        hop_count = np.zeros(nNodes, dtype= np.int64)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(self.query_for_2_hop, self.vertex_order,chunksize= 128)
        for sub_list in result:
            hop_count += sub_list
        # 排序
        self.vertex_order = np.argsort(hop_count)[::-1]
        self.write_2_hop_count(hop_count)
        return hop_count
    
    
    def query_for_2_hop(self, src):
        nNodes = self.G.number_of_nodes()
        count_result = np.zeros(nNodes, dtype= np.int64)
        src_list = self.index[src]["backward"]
        for dest in self.vertex_order:
            i = 0
            j = 0
            shortest_dist = max_length
            hop_nodes = []
            dest_list = self.index[dest]["forward"]
            while i < len(src_list) and j < len(dest_list):
                if src_list[i][0] == dest_list[j][0]:
                    # print(src_list[i][0] == dest_list[j][0])
                    curr_dist = src_list[i][1] + dest_list[j][1]
                    if(curr_dist == 0):
                        hop_nodes.clear()
                        shortest_dist = curr_dist
                        break
                    # 当前距离未必为最小，若找到更小的距离，更新最小距离，之前的 hop_nodes作废，用新的代替
                    if curr_dist < shortest_dist:
                        shortest_dist = curr_dist
                        hop_nodes.clear()
                        hop_nodes.append(src_list[i][0])   
                    # 假定当前距离为最小，相等的点被暂时加入到 hop_nodes中
                    elif curr_dist == shortest_dist:
                        hop_nodes.append(src_list[i][0])
                    i += 1
                    j += 1
            
                elif self.order2index[src_list[i][0]] < self.order2index[dest_list[j][0]]:
                        i += 1
                else:
                        j += 1
            for hop_node in hop_nodes:
                count_result[hop_node]+=1   
            # print(count_result)
        return count_result

    # 判断是否需要剪枝
    def need_to_expand(self, src, dest, dist = -1):
        our_result = self.query(src, dest)
        if (our_result <= dist):
            return False
        return True

    # 进行BFS
    def build_index(self):
        start = time.perf_counter()
        nNodes = self.G.number_of_nodes()
        self.index = {}
        pq = Q.PriorityQueue()
        has_process = np.zeros(nNodes)
        self.order2index = np.zeros(nNodes,dtype= np.int64)
        for i in range(nNodes):
            self.order2index[self.vertex_order[i]] = i
        for v in sorted(list(self.G.nodes())):
            self.index[v] = {"backward": [], "forward": []}
        i = 0
        count = 0
        for cur_node in self.vertex_order:
            i += 1
            # Calculate Forward
            if (i%2000 == 0) :
                print("Caculating %s (%d/%d) forward ... " % (cur_node, i, nNodes))
            pq.put((0, cur_node))
            # 把所有点是否剪枝记为0
            has_process[:] = 0
            while (not pq.empty()):
                cur_dist, src = pq.get()
                if (has_process[src] or self.order2index[cur_node] > self.order2index[src]  or not self.need_to_expand(cur_node, src, cur_dist)):
                    has_process[src] = 1
                    continue
                has_process[src] = 1
                self.index[src]["forward"].append((cur_node, cur_dist))
                count += 1
                edges = self.G.out_edges(src)
                # print(src)
                # print(f"edges:{edges}")
                for _, dest in edges:
                    # print(f"dest: {dest}")
                    weight = self.G.get_edge_data(src, dest)['weight']
                    if (has_process[dest]):
                        continue
                    pq.put((cur_dist + weight, dest))

            # Calculate Backward
            pq.put((0, cur_node))
            has_process[:] = 0

            while (not pq.empty()):
                cur_dist, src = pq.get()
                # print("Pop: (%s %d)"%(src,cur_dist))
                if (has_process[src] or self.order2index[cur_node] > self.order2index[src]  or not self.need_to_expand(src, cur_node, cur_dist)):
                    has_process[src] = 1
                    continue
                has_process[src] = 1
                self.index[src]["backward"].append((cur_node, cur_dist))
                count += 1
                edges = self.G.in_edges(src)
                # print(src)
                # print(edges)
                for dest, _ in edges:
                    weight = self.G.get_edge_data(dest, src)['weight']
                    if (has_process[dest]):
                        continue
                    pq.put((cur_dist + weight, dest))
        end = time.perf_counter()
        average_label_size = (count - 2* nNodes)/nNodes
        print(f'finish building index, time cost: {end-start:.4f}')
        print(f'average label size: {average_label_size:.2f}')
        return (end-start), average_label_size

    def output_to_excel(self, gen_order_time_list, BFS_time_list, query_time_100K_list, avg_label_size_list):
        total_time_list = [BFS_time_list[i] + gen_order_time_list[i] for i in range(len(BFS_time_list))]
        df = pd.DataFrame(columns=["Degree","betweenness",'betweenness-2-hop-count'])
        df.loc[len(df.index)] = gen_order_time_list
        df.loc[len(df.index)] = BFS_time_list
        df.loc[len(df.index)] = query_time_100K_list
        df.loc[len(df.index)] = avg_label_size_list
        df.loc[len(df.index)] = total_time_list
        df.rename(index = {0:"gen_order_time",1:"BFS_time",2:"query_time_100K",3:"avg_label_size",4:"each_total_time"},inplace=True)
       
        print(df)

        excel_file_name = "dataset/excel/"+ self.map_file_name+".xlsx"
        writer = pd.ExcelWriter(excel_file_name, engine='xlsxwriter')
        df.to_excel(writer,'Sheet1')
        worksheet = writer.sheets['Sheet1']
        worksheet.set_column('A:G', 25)
        total_time = df.loc['each_total_time',"Degree"] + df.loc['each_total_time',"betweenness"] + df.loc['each_total_time',"betweenness-2-hop-count"]
        print(f"total_time: {total_time}")
        worksheet.write(7,0,total_time)
        writer.save()