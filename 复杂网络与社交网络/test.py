import sys
import os
import pandas as pd
import numpy as np 
import networkx as nx
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 这里保险的就是直接先把绝对路径加入到搜索路径
sys.path.insert(0, os.path.join(BASE_DIR))
sys.path.insert(0, os.path.join(BASE_DIR, 'data'))  # 把data所在的绝对路径加入到了搜索路径，这样也可以直接访问dataset.csv文件了
# 这句代码进行切换目录
os.chdir(BASE_DIR)   # 把目录切换到当前项目，这句话是关键
flights = pd.read_csv('Airline.csv') 

flights = flights[['ORIGIN_AIRPORT','DEST_AIRPORT']].reset_index(drop=True) ## 提取三列且重新建立索引
## 给图添加边和边的属性 
G = nx.Graph()
for i in range(len(flights)):
    G.add_edge(flights.iloc[i,0],flights.iloc[i,1]) ## 添加边和边的属性
## 画出航线的网络图
nx.draw(G, with_labels=True)
plt.show()