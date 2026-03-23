"""
Unsupervised GNN for 3‑coloring Erdős–Rényi graphs (PyTorch Geometric implementation).

Inspired by "Graph Coloring with Physics‑Inspired Graph Neural Networks".
This script generates random ER graphs, trains a GNN per graph using a physics‑based
loss, and reports the number of remaining conflicts (edges with same color).
Zero conflicts indicates a valid 3‑coloring.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import time
import sys

from graph import erdos_renyi_, UndirectedGraph_


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def er_graph_to_data(ugraph: UndirectedGraph_):
    """
    Convert an UndirectedGraph_ to a PyG Data object.

    Returns a Data instance with:
        - edge_index: directed edges (both directions) suitable for message passing.
        - edge_index_undir: unique undirected edges (src < dst) for conflict counting.
    """
    dir_edges = []      # both (u,v) and (v,u)
    undir_edges = []    # only one (u,v) with u < v
    for u, neighbors in ugraph.graph.items():
        for v in neighbors:
            dir_edges.append([u, v])
            if u < v:
                undir_edges.append([u, v])
    edge_index_dir = torch.tensor(dir_edges, dtype=torch.long).t().contiguous()
    edge_index_undir = torch.tensor(undir_edges, dtype=torch.long).t().contiguous()
    num_nodes = len(ugraph.get_vertices())
    data = Data(num_nodes=num_nodes, edge_index=edge_index_dir)
    data.edge_index_undir = edge_index_undir
    return data


class GNNConv(nn.Module):
    """Simple two‑layer GCN for graph coloring."""
    def __init__(self, num_features: int, hidden_dim: int, num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def loss_physics(probs, edge_index):
    """
    Physics‑inspired loss for graph coloring.

    probs : (N, C) tensor of soft color assignments.
    edge_index : (2, E) directed edges (both directions of each undirected edge).

    Returns L = ½ Σ_{i,j∈E} (probs_i · probs_j).
    """
    src, dst = edge_index
    dot = torch.sum(probs[src] * probs[dst], dim=1)   # (E,)
    # each undirected edge appears twice in edge_index
    loss = dot.sum() / 2.0
    return loss


def count_conflicts(coloring, edge_index_undir):
    """
    Number of edges whose endpoints share the same color.

    coloring : (N,) tensor of hard color assignments (0 … C‑1).
    edge_index_undir : (2, E_undir) unique undirected edges.
    """
    src, dst = edge_index_undir
    return (coloring[src] == coloring[dst]).sum().item()


def train_on_graph(data, num_classes=3, device='cpu', hypers=None):
    """
    Train a GNN (with node embeddings) on a single graph.

    Returns:
        best_coloring (Tensor), best_loss (float),
        best_conflicts (int), epochs_used (int)
    """
    if hypers is None:
        hypers = {
            'dim_embedding': 64,
            'hidden_dim': 64,
            'dropout': 0.1,
            'learning_rate': 1e-3,
            'patience': 500,
            'tolerance': 1e-4,
            'max_epochs': 20000,
            'seed': 42,
        }

    set_seed(hypers['seed'])

    num_nodes = data.num_nodes
    dim_embedding = hypers['dim_embedding']
    hidden_dim = hypers['hidden_dim']
    dropout = hypers['dropout']
    lr = hypers['learning_rate']
    patience = hypers['patience']
    tolerance = hypers['tolerance']
    max_epochs = hypers['max_epochs']

    # Model components
    embed = nn.Embedding(num_nodes, dim_embedding).to(device)
    model = GNNConv(dim_embedding, hidden_dim, num_classes, dropout).to(device)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(embed.parameters()),
        lr=lr,
        weight_decay=1e-2
    )

    edge_index = data.edge_index.to(device)
    # Use undirected edges for conflict counting if they are stored
    edge_index_undir = getattr(data, 'edge_index_undir', edge_index).to(device)

    best_loss = float('inf')
    best_coloring = None
    best_conflicts = num_nodes * 10   # large sentinel
    cnt = 0                            # early‑stopping counter
    prev_loss = 1.0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        inputs = embed.weight                     # (N, dim_embedding)
        logits = model(inputs, edge_index)        # (N, num_classes)
        probs = F.softmax(logits, dim=1)
        loss = loss_physics(probs, edge_index)

        # Track hard assignments
        coloring = torch.argmax(probs, dim=1)
        conflicts = count_conflicts(coloring, edge_index_undir)

        # Keep best solution seen during training
        if conflicts < best_conflicts:
            best_conflicts = conflicts
            best_coloring = coloring.detach().cpu()
            best_loss = loss.item()

        # Early stopping based on soft loss change
        if abs(loss.item() - prev_loss) <= tolerance or loss.item() > prev_loss:
            cnt += 1
        else:
            cnt = 0
        prev_loss = loss.item()

        if cnt >= patience:
            break

        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f'    epoch {epoch:5d} | loss {loss.item():.4f} '
                  f'| conflicts {conflicts}')

    return best_coloring, best_loss, best_conflicts, epoch


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description='Train unsupervised GNN for 3‑coloring Erdős–Rényi graphs.'
    )
    parser.add_argument('--num_nodes', type=int, default=50,
                        help='Number of vertices in each generated graph')
    parser.add_argument('--p_values', type=float, nargs='+',
                        default=[0.01, 0.02, 0.03, 0.04, 0.05,
                                 0.06, 0.07, 0.08, 0.09, 0.10],
                        help='Edge probabilities to test')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for graph generation and training')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run training')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f'Using device: {device}')
    print(f'Number of nodes per graph: {args.num_nodes}')
    print(f'Edge probabilities: {args.p_values}')
    print(f'          c = n* p: {[round(val*args.num_nodes,2) for val in args.p_values]}\n')

    hypers = {
        'dim_embedding': 64,
        'hidden_dim': 64,
        'dropout': 0.1,
        'learning_rate': 1e-3,
        'patience': 100,
        'tolerance': 1e-4,
        'max_epochs': 5000,
        'seed': args.seed,
    }

    results = []
    for p in args.p_values:
        print(f'=== p = {p:.3f} ===')
        # Generate ER graph (weights are irrelevant, set to constant 1)
        ugraph = UndirectedGraph_()
        erdos_renyi_(
            ugraph,
            n=args.num_nodes,
            p=p,
            weight_range=(1, 1),
            seed=args.seed
        )
        data = er_graph_to_data(ugraph)
        data = data.to(device)
        print(f'  Graph has {data.num_nodes} nodes, '
              f'{data.edge_index_undir.size(1)} undirected edges')

        t_start = time.time()
        coloring, loss, conflicts, epochs = train_on_graph(
            data, num_classes=3, device=device, hypers=hypers
        )
        t_elapsed = time.time() - t_start

        print(f'  Result: conflicts = {conflicts}, '
              f'time = {t_elapsed:.2f}s, epochs = {epochs}\n')
        results.append((p, conflicts, epochs, t_elapsed))

    # Summary table
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print('p\tconflicts\tepochs\ttime (s)')
    for p, c, e, t in results:
        print(f'{p:.3f}\t{c}\t\t{e}\t{t:.2f}')
    print('='*60)


