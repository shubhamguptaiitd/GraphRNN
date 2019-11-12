import networkx as nx
import numpy as np
import random
import torch
def calculate_M(graphs,iters=10000):
    M = []
    graphs_idx = range(0,len(graphs))
    for i in range(0,iters):
        idx = random.choice(graphs_idx)
        g = graphs[idx]
        node_id = random.choice(range(0,g.number_of_nodes()))
        M.append(get_M_for_graph(g,node_id))
    return min(max(M), np.mean(M) + 3*np.std(M))
def get_M_for_graph(g,ind):
    
    depths = nx.single_source_shortest_path_length(g,ind)
    nodes_at_d = {}
    for key,value in depths.items():
        if value not in nodes_at_d:
            nodes_at_d[value]= [key]
        else:
            nodes_at_d[value].append(key)
    return max([len(val) for val in nodes_at_d.values()])

def encode_M_matrix(adj_graph,M):
    
    M_matrix = np.zeros((adj_graph.shape[0]-1,M))
    for i in range(1,adj_graph.shape[0]):
        reverse_indexes= list(range(i-1,max(0,i-M)-1,-1))
        M_matrix[i-1,0:len(reverse_indexes)] = adj_graph[i,reverse_indexes]

    return M_matrix

def decode_M_matrix(M_matrix,M):
    adj_mat = np.zeros((M_matrix.shape[0]+1,M_matrix.shape[0]+1))
    for i in range(1,M_matrix.shape[0]+1):
        reverse_indexes= list(range(i-1,max(0,i-M)-1,-1))
        adj_mat[i,reverse_indexes] = M_matrix[i-1,0:len(reverse_indexes)]
        
    for i in range(0,adj_mat.shape[0]):
        for j in range(0,adj_mat.shape[1]):
            if adj_mat[i][j]>=1:
                adj_mat[j][i] =adj_mat[i][j]
                
    return adj_mat


class graphs_db(torch.utils.data.Dataset):
    def __init__(self,graphs,max_nodes,M):
        self.graphs = graphs
        self.max_nodes = max_nodes
        self.adj_graphs = [nx.to_numpy_array(graph) for graph in graphs]
        self.max_nodes = max_nodes
        self.M = M
        return
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self,idx):
        graph = self.graphs[idx]
        adj_graph = self.adj_graphs[idx]
        bfs_start_node = np.random.randint(0,graph.number_of_nodes())
        bfs_seq = list(nx.bfs_tree(graph,bfs_start_node))
        adj_graph = adj_graph[np.ix_(bfs_seq,bfs_seq)]  #### Changing the adjacency matrix in bfs sequence
        M_matrix = encode_M_matrix(adj_graph,self.M)
        X = np.zeros((self.max_nodes,self.M))
        Y = np.zeros((self.max_nodes,self.M))
        X[0,:] = np.ones((1,X.shape[1]))
        X[1:M_matrix.shape[0]+1,:] = M_matrix
        Y[0:M_matrix.shape[0],:] = M_matrix
        node_labels = np.zeros((self.max_nodes))
        #for i in range(0,len(bfs_seq)):
        node_labels[0:len(bfs_seq)] = [graph.node[bfs_seq[i]]['node_label'] for i in bfs_seq]
        
        return (X,Y,graph.number_of_nodes(),node_labels)