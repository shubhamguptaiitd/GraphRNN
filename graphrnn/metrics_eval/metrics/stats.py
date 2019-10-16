import concurrent.futures
import os
import subprocess as sp
import tempfile
from datetime import datetime
from functools import partial
import numpy as np
import networkx as nx

#import metrics.mmd as mmd

PRINT_TIME = True
MAX_WORKERS = 8

def degree_worker(G):
    return np.array(nx.degree_histogram(G))

def degree_stats(graph_ref_list, graph_pred_list):
    """
    Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    
    sample_ref = []
    sample_pred = []
    
    # in case an empty graph is generated
    graph_pred_list = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for deg_hist in executor.map(degree_worker, graph_ref_list):
            sample_ref.append(deg_hist)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for deg_hist in executor.map(degree_worker, graph_pred_list):
            sample_pred.append(deg_hist)

    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, mmd.gaussian_emd, n_jobs=MAX_WORKERS)
    
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    
    return mmd_dist

def node_label_worker(G, node_map):
    freq = np.zeros(len(node_map))
    for u in G.nodes():
        freq[node_map[G.nodes[u]['label']]] += 1
    
    return freq

def node_label_stats(graph_ref_list, graph_pred_list):
    """
    Compute the distance between the node label distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    
    sample_ref = []
    sample_pred = []
    
    # in case an empty graph is generated
    graph_pred_list = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    
    node_map = {}
    for graph in graph_ref_list + graph_pred_list:
        for u in graph.nodes():
            if graph.nodes[u]['label'] not in node_map:
                node_map[graph.nodes[u]['label']] = len(node_map)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for node_label_hist in executor.map(partial(node_label_worker, node_map=node_map), graph_ref_list):
            sample_ref.append(node_label_hist)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for node_label_hist in executor.map(partial(node_label_worker, node_map=node_map), graph_pred_list):
            sample_pred.append(node_label_hist)

    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, mmd.gaussian_emd, n_jobs=MAX_WORKERS)
    
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing node label mmd: ', elapsed)
    
    return mmd_dist

def edge_label_worker(G, edge_map):
    freq = np.zeros(len(edge_map))
    for u, v in G.edges():
        freq[edge_map[G.edges[u, v]['label']]] += 1
    
    return freq

def edge_label_stats(graph_ref_list, graph_pred_list):
    """
    Compute the distance between the edge label distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    
    sample_ref = []
    sample_pred = []
    
    # in case an empty graph is generated
    graph_pred_list = [G for G in graph_pred_list if not G.number_of_nodes() == 0 and not G.number_of_edges() == 0]

    prev = datetime.now()
    
    edge_map = {}
    for graph in graph_ref_list + graph_pred_list:
        for u, v in graph.edges():
            if graph.edges[u, v]['label'] not in edge_map:
                edge_map[graph.edges[u, v]['label']] = len(edge_map)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for edge_label_hist in executor.map(partial(edge_label_worker, edge_map=edge_map), graph_ref_list):
            sample_ref.append(edge_label_hist)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for edge_label_hist in executor.map(partial(edge_label_worker, edge_map=edge_map), graph_pred_list):
            sample_pred.append(edge_label_hist)

    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, mmd.gaussian_emd, n_jobs=MAX_WORKERS)
    
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing edge label mmd: ', elapsed)
    
    return mmd_dist

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
            clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist

def clustering_stats(graph_ref_list, graph_pred_list, bins=100):
    sample_ref = []
    sample_pred = []
    graph_pred_list = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for clustering_hist in executor.map(clustering_worker, 
                [(G, bins) for G in graph_ref_list]):
            sample_ref.append(clustering_hist)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for clustering_hist in executor.map(clustering_worker, 
                [(G, bins) for G in graph_pred_list]):
            sample_pred.append(clustering_hist)
    
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, metric=partial(mmd.gaussian_emd,
                               sigma=0.1, distance_scaling=bins), n_jobs=MAX_WORKERS)
    
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    
    return mmd_dist

def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges

def orca(graph):
    COUNT_START_STR = 'orbit counts: \n'
    tmp_fname = tempfile.NamedTemporaryFile().name
    
    f = open(tmp_fname, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    output = sp.check_output(['./metrics/orca/orca', 'node', '4', tmp_fname, 'std'])
    output = output.decode('utf8').strip()
    
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
          for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts

def orbits_counts_worker(graph):
    try:
        orbit_counts = orca(graph)
    except:
        return None
    
    orbit_counts_graph = np.sum(orbit_counts, axis=0) / graph.number_of_nodes()
    return orbit_counts_graph

def orbit_stats_all(graph_ref_list, graph_pred_list):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for orbit_counts_graph in executor.map(orbits_counts_worker, graph_ref_list):
            if orbit_counts_graph is not None:
                total_counts_ref.append(orbit_counts_graph)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for orbit_counts_graph in executor.map(orbits_counts_worker, graph_pred_list):
            if orbit_counts_graph is not None:
                total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    
    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, metric=partial(mmd.gaussian,
        sigma=30.0), is_hist=False, n_jobs=MAX_WORKERS)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing orbit mmd: ', elapsed)

    return mmd_dist

def nspdk_stats(graph_ref_list, graph_pred_list):
    graph_pred_list = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()

    mmd_dist = mmd.compute_mmd(graph_ref_list, graph_pred_list, metric='nspdk', 
    is_hist=False, n_jobs=MAX_WORKERS)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing NSPDK mmd: ', elapsed)
    
    return mmd_dist
