# PrunedLandmarkLabeling

This repo implements PrunedLandmarkLabeling that calculates the shortest distance in a map, and we make some optimizations for the algorithm.

该repo实现了PrunedLandmarkLabeling，用于计算地图中的最短距离，并对算法的实现进行了一些优化。

## 说明

### Prerequisites
如果需要运行这个仓库的代码，需要以下包:
- numba
- numpy
- networkx
- pandas
### 模块
本仓库分为以下五个模块：
- utils.py：该文件包含了生成order、BFS构建label、俩跳计数等所有工具函数。
- BFS_specified_order.py: 给定一个order，基于该order进行BFS
- script.py: 将所有流程聚在一起，一键运行
- dataset文件夹： 数据集，包含了几个部分：
  - excel：结果转换为excel表存放在这里面
  - hop_count: 俩跳计数的结果放在里面
  - betweenness: betweenness的值放在里面
  - order：生成的节点order放在里面
  - map_file: OSM的图数据

### 运行指令(以macau.map为例)

**script.py**
```bash
python script.py -i [input_file]
e.g: python script.py -i macau.map
# 最常用，完成从基于各种策略生成order，到BFS，到将结果输出到excel的所有流程
```

**BFS.py**
```bash
python BFS.py -i [input_file]
e.g: python BFS.py -i macau.map
# 基于用户输入的order进行BFS, 在此之前，用户需要先将指定的order输入/Pll/order/user_define/specified_order中
```