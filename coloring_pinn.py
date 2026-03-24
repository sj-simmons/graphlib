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


class GNNConvDeep(nn.Module):
    """Deeper GCN with residual connections for graph coloring."""
    def __init__(self, num_features: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # First layer
        self.conv_first = GCNConv(num_features, hidden_dim)

        # Intermediate layers with residual connections
        self.conv_hidden = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])

        # Last layer
        self.conv_last = GCNConv(hidden_dim, num_classes)

        # Batch normalization layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])

    def forward(self, x, edge_index):
        # First layer
        x = self.conv_first(x, edge_index)
        x = self.bn_layers[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Hidden layers with residual connections
        for i in range(self.num_layers - 2):
            residual = x
            x = self.conv_hidden[i](x, edge_index)
            x = self.bn_layers[i + 1](x)
            x = F.relu(x + residual)  # Residual connection
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Last layer
        x = self.conv_last(x, edge_index)
        return x


def loss_physics_with_confidence(probs, edge_index_undir, lambda_div=0.01, beta_entropy=0.01):
    """
    Physics‑inspired loss with diversity and confidence regularization.

    probs : (N, C) tensor of soft color assignments.
    edge_index_undir : (2, E) unique undirected edges (src < dst).

    Returns L = Σ_{i,j∈E} (probs_i · probs_j) / E
             + λ * KL(avg_probs || uniform)
             + β * mean_node_entropy
    """
    src, dst = edge_index_undir
    dot = torch.sum(probs[src] * probs[dst], dim=1)   # (E,)
    loss_physics_term = dot.sum() / edge_index_undir.size(1)

    # Diversity regularization: encourage using all colors
    color_distribution = probs.mean(dim=0)  # (C,)
    uniform = torch.ones_like(color_distribution) / color_distribution.size(0)
    loss_diversity = torch.sum(
        color_distribution * (torch.log(color_distribution + 1e-8) - torch.log(uniform))
    )

    # Confidence regularization: minimize per-node entropy to encourage peaked predictions
    # Entropy per node: -Σ_c p_c log p_c
    node_entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    loss_confidence = node_entropies.mean()

    return loss_physics_term + lambda_div * loss_diversity + beta_entropy * loss_confidence


def count_conflicts(coloring, edge_index_undir):
    """
    Number of edges whose endpoints share the same color.

    coloring : (N,) tensor of hard color assignments (0 … C‑1).
    edge_index_undir : (2, E_undir) unique undirected edges.
    """
    src, dst = edge_index_undir
    return (coloring[src] == coloring[dst]).sum().item()

def analyze_coloring(coloring, num_classes=3):
    """
    Analyze color distribution and return statistics.
    """
    unique_colors = torch.unique(coloring)
    num_used = len(unique_colors)
    counts = torch.bincount(coloring, minlength=num_classes)
    distribution = counts.float() / len(coloring)
    entropy = -torch.sum(distribution * torch.log(distribution + 1e-8))

    return {
        'num_used': num_used,
        'counts': counts,
        'distribution': distribution,
        'entropy': entropy.item()
    }


def train_on_graph(data, num_classes=3, device='cpu', hypers=None):
    """
    Train a GNN (with node embeddings) on a single graph.

    Returns:
        best_coloring (Tensor), best_loss (float),
        best_conflicts (int), epochs_used (int)
    """
    if hypers is None:
        hypers = {
            'dim_embedding': 128,
            'hidden_dim': 128,
            'num_layers': 5,
            'dropout': 0.2,
            'learning_rate': 5e-3,
            'patience': 500,
            'max_epochs': 20000,
            'seed': 42,
            'lambda_div': 0.02,
            'beta_entropy': 0.05,
        }

    set_seed(hypers['seed'])

    num_nodes = data.num_nodes
    dim_embedding = hypers['dim_embedding']
    hidden_dim = hypers['hidden_dim']
    dropout = hypers['dropout']
    lr = hypers['learning_rate']
    patience = hypers['patience']
    max_epochs = hypers['max_epochs']
    lambda_div = hypers.get('lambda_div', 0.01)

    # Model components
    embed = nn.Embedding(num_nodes, dim_embedding).to(device)
    model = GNNConvDeep(
        dim_embedding,
        hidden_dim,
        num_classes,
        num_layers=hypers.get('num_layers', 4),
        dropout=dropout
    ).to(device)
    #if hasattr(torch, 'compile'):
    #    model = torch.compile(model)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(embed.parameters()),
        lr=lr,
        weight_decay=1e-2
    )
    # More gradual learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-5
    )

    edge_index = data.edge_index.to(device)
    # Use undirected edges for conflict counting if they are stored
    edge_index_undir = getattr(data, 'edge_index_undir', edge_index).to(device)

    best_loss = float('inf')
    best_coloring = None
    best_conflicts = num_nodes * 10   # large sentinel
    cnt = 0                            # early‑stopping counter
    # Save best model state
    best_model_state = None
    best_embed_state = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        inputs = embed.weight                          # (N, dim_embedding)
        logits = model(inputs, edge_index)             # (N, num_classes)
        probs = F.softmax(logits, dim=1)
        loss = loss_physics_with_confidence(
            probs, edge_index_undir,
            lambda_div=lambda_div,
            beta_entropy=hypers.get('beta_entropy', 0.01)
        )

        # Track hard assignments (no gradients needed)
        with torch.no_grad():
            coloring = torch.argmax(probs, dim=1)
            conflicts = count_conflicts(coloring, edge_index_undir)

        # Keep best solution seen during training
        if conflicts < best_conflicts:
            best_conflicts = conflicts
            best_coloring = coloring.detach().cpu()
            best_loss = loss.item()
            best_model_state = model.state_dict().copy()
            best_embed_state = embed.state_dict().copy()
            cnt = 0  # reset patience on improvement
        else:
            cnt += 1

        if best_conflicts == 0:
            break

        # Early stopping: patience exceeded
        if cnt >= patience:
            break

        loss.backward()
        optimizer.step()

        # Update learning rate (cosine annealing)
        scheduler.step()

        if epoch % 100 == 0:
            analysis = analyze_coloring(coloring, num_classes=3)
            print(f'    epoch {epoch:5d} | loss {loss.item():.4f} '
                  f'| colors used {analysis["num_used"]} '
                  f'| entropy {analysis["entropy"]:.3f} '
                  f'| conflicts {conflicts} ')

    # Restore best model if found
    if best_model_state is not None and best_embed_state is not None:
        model.load_state_dict(best_model_state)
        embed.load_state_dict(best_embed_state)

    return best_coloring, best_loss, best_conflicts, epoch


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description='Train unsupervised GNN for 3‑coloring Erdős–Rényi graphs.'
    )
    parser.add_argument('--num_nodes', type=int, default=80,
                        help='Number of vertices in each generated graph')
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
    min_p, max_p = 4.0/args.num_nodes, 4.6/args.num_nodes
    edge_probabilities = [min_p + (max_p - min_p) * i /10 for i in range(10)]
    print(f'Edge probabilities: {[round(e,5) for e in edge_probabilities]}')

    hypers = {
        'dim_embedding': 128,          # Increased for richer representations
        'hidden_dim': 128,             # Increased capacity
        'num_layers': 14,              # Deeper network for larger receptive field
        'dropout': 0.2,                # Slightly higher dropout for regularization
        'learning_rate': 5e-3,         # Higher learning rate
        'patience': 3000,              # More patience for sparse graphs
        'max_epochs': 15000,           # More epochs
        'seed': args.seed,
        'lambda_div': 0.02,            # Increased diversity weight
        'beta_entropy': 0.05,          # Confidence regularization weight
    }

    results = []
    for p in edge_probabilities:
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

        # Analyze the best coloring
        analysis = analyze_coloring(coloring, num_classes=3)
        print(f'  Result: conflicts = {conflicts}, '
              f'time = {t_elapsed:.2f}s, epochs = {epochs}')
        print(f'  Color distribution: {[round(x,2) for x in analysis["distribution"].tolist()]}, '
              f'Colors used: {analysis["num_used"]}, Entropy: {analysis["entropy"]:.3f}\n')
        results.append((p, conflicts, epochs, t_elapsed, analysis))

    # Summary table
    print('\n' + '='*70)
    print('Summary')
    print('='*70)
    print('p\tconflicts\tepochs\ttime (s)\tcolors used\tentropy')
    for p, c, e, t, analysis in results:
        print(f'{p:.3f}\t{c}\t\t{e}\t{t:.2f}\t\t{analysis["num_used"]}\t\t{analysis["entropy"]:.3f}')
    print('='*70)

    # Count successful colorings
    success_count = sum(1 for _, c, _, _, _ in results if c == 0)
    print(f'\nSuccessful 3-colorings: {success_count}/{len(results)}')


