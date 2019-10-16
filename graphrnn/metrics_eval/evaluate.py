import sys
from statistics import mean
import metrics.stats
from process_dataset import produce_graphs

LINE_BREAK = '----------------------------------------------------------------------\n'
EVAL_BATCH_SIZE = 256

def print_stats(
    node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, 
    edge_count_avg_pred, degree_mmd, clustering_mmd, 
    orbit_mmd, nspdk_mmd, node_label_mmd, edge_label_mmd
):
    print('Node count avg: Test - {:.6f}, Generated - {:.6f}'.format(mean(node_count_avg_ref), mean(node_count_avg_pred)))
    print('Edge count avg: Test - {:.6f}, Generated - {:.6f}'.format(mean(edge_count_avg_ref), mean(edge_count_avg_pred)))

    print('MMD Degree - {:.6f}, MMD Clustering - {:.6f}, MMD Orbits - {:.6f}'.format(
        mean(degree_mmd), mean(clustering_mmd), mean(orbit_mmd)))
    print('MMD NSPDK - {:.6f}'.format(mean(nspdk_mmd)))
    print('MMD Node label - {:.6f}, MMD Node label - {:.6f}'.format (
        mean(node_label_mmd), mean(edge_label_mmd)
    ))

    print(LINE_BREAK)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python3 evaluate.py <reference_graphs_file> <test_graphs_file>')
        exit()

    graphs_ref_path = sys.argv[1]
    graphs_pred_path = sys.argv[2]

    print('Evaluating {} against {}'.format(graphs_ref_path, graphs_pred_path))

    graphs_ref_total = produce_graphs(graphs_ref_path)
    graphs_pred_total = produce_graphs(graphs_pred_path)

    assert len(graphs_ref_total) == len(graphs_pred_total)
    
    node_count_avg_ref, node_count_avg_pred = [], []
    edge_count_avg_ref, edge_count_avg_pred = [], []

    degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd = [], [], [], []
    node_label_mmd, edge_label_mmd = [], []

    for i in range(0, len(graphs_pred_total), EVAL_BATCH_SIZE):
        batch_size = min(EVAL_BATCH_SIZE, len(graphs_pred_total) - i)

        graphs_ref = graphs_ref_total[i: i + batch_size]
        graphs_pred = graphs_pred_total[i: i + batch_size]

        node_count_avg_ref.append(mean([len(G.nodes()) for G in graphs_ref]))
        node_count_avg_pred.append(mean([len(G.nodes()) for G in graphs_pred]))

        edge_count_avg_ref.append(mean([len(G.edges()) for G in graphs_ref]))
        edge_count_avg_pred.append(mean([len(G.edges()) for G in graphs_pred]))

        degree_mmd.append(metrics.stats.degree_stats(graphs_ref, graphs_pred))
        clustering_mmd.append(metrics.stats.clustering_stats(graphs_ref, graphs_pred))
        orbit_mmd.append(metrics.stats.orbit_stats_all(graphs_ref, graphs_pred))
        nspdk_mmd.append(metrics.stats.nspdk_stats(graphs_ref, graphs_pred))
        node_label_mmd.append(metrics.stats.node_label_stats(graphs_ref, graphs_pred))
        edge_label_mmd.append(metrics.stats.edge_label_stats(graphs_ref, graphs_pred))

        print('Running average of metrics:\n')

        print_stats(
            node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred, degree_mmd, 
            clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd, edge_label_mmd
        )
    
    print('Evaluating {} against {}'.format(graphs_ref_path, graphs_pred_path))

    print_stats(
        node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred, degree_mmd, 
        clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd, edge_label_mmd
    )
