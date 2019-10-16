import networkx as nx  
from read_graph import read_graphs_in_networkx
import sys
import pdb
from sklearn.model_selection import train_test_split
data_file = "../data/breast.txt"
(graphs,node_dict,edge_dict) = read_graphs_in_networkx(data_file,False,1000)
print("Number of graphs loaded " , len(graphs))

(train_graphs,val_graphs) = train_test_split(graphs,test_size=0.1, random_state=42)
(train_graphs,test_graphs) = train_test_split(train_graphs,test_size=0.2, random_state=42)





max_num_edges = max([graph.number_of_edges() for graph in graphs])
min_num_edges = min([graph.number_of_edges() for graph in graphs])
max_num_nodes = max([graph.number_of_nodes() for graph in graphs])
min_num_nodes = min([graph.number_of_nodes() for graph in graphs])
max_prev_node = 20  ### To be calculated later during batch training ###

print("Number of training graphs, max nodes, min nodes , max edges , min edges ", len(train_graphs), max_num_nodes, min_num_nodes, max_num_edges, min_num_edges)

