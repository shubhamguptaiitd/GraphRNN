import pickle
import networkx as nx

def produce_graphs(inputfile):
    """
    :param inputfile: Path to file containing graphs in aido99 format
    :return: lis of graphs produced in nx format
    """

    lines = []
    with open(inputfile, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            lines.append(line)

    index, l = 0, len(lines)
    graphs_ids = set()
    graphs = []
    while index < l:
        if lines[index][0][1:] not in graphs_ids:
            graph_id = lines[index][0][1:]
            G = nx.Graph(id=graph_id)
            
            index += 1
            vert = int(lines[index][0])
            index += 1
            for i in range(vert):
                G.add_node(i, label=lines[index][0])
                index += 1

            edges = int(lines[index][0])
            index += 1
            for i in range(edges):
                G.add_edge(int(lines[index][0]), int(lines[index][1]), label=lines[index][2])
                index += 1
            if nx.is_connected(G):
                graphs.append(G)
                graphs_ids.add(graph_id)

            index += 1
        else:
            vert = int(lines[index + 1][0])
            edges = int(lines[index + 2 + vert][0])
            index += vert + edges + 4
    
    return graphs