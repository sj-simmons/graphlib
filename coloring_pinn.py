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
from torch_geometric.nn import GCNConv, SAGEConv
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
    dir_edges = []  # both (u,v) and (v,u)
    undir_edges = []  # only one (u,v) with u < v
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
    """GNN with JumpingKnowledge to combat oversmoothing.

    Uses SAGEConv layers with residual connections and concatenates
    representations from all layers (JumpingKnowledge "cat" mode)
    before the final projection.  This lets the model use both
    local (early-layer) and global (late-layer) information.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # First layer: project input to hidden_dim
        self.conv_first = SAGEConv(num_features, hidden_dim)
        self.bn_first = nn.BatchNorm1d(hidden_dim)

        # Intermediate layers with residual connections
        self.conv_hidden = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
        self.bn_hidden = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)]
        )

        # JumpingKnowledge: project concatenated layer outputs to num_classes
        # All num_layers layers contribute a hidden_dim-sized representation
        self.jk_linear = nn.Linear(hidden_dim * num_layers, num_classes)

    def forward(self, x, edge_index):
        layer_outputs = []

        # First layer
        x = self.conv_first(x, edge_index)
        x = self.bn_first(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_outputs.append(x)

        # Hidden layers with residual connections
        for i in range(self.num_layers - 1):
            residual = x
            x = self.conv_hidden[i](x, edge_index)
            x = self.bn_hidden[i](x)
            x = F.relu(x + residual)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

        # JumpingKnowledge: concatenate all layer representations
        x = torch.cat(layer_outputs, dim=1)
        x = self.jk_linear(x)
        return x


def loss_physics_with_confidence(
    probs, edge_index_undir, lambda_div=0.01, beta_entropy=0.01, edge_weights=None
):
    """
    Physics‑inspired loss with diversity and confidence regularization.

    probs : (N, C) tensor of soft color assignments.
    edge_index_undir : (2, E) unique undirected edges (src < dst).
    edge_weights : optional (E,) tensor of per-edge weights.  When provided,
        high-conflict edges contribute more to the physics loss.

    Returns L = Σ_{i,j∈E} w_ij (probs_i · probs_j) / Σ w_ij
             + λ * KL(avg_probs || uniform)
             + β * mean_node_entropy
    """
    src, dst = edge_index_undir
    dot = torch.sum(probs[src] * probs[dst], dim=1)  # (E,)

    if edge_weights is not None:
        loss_physics_term = (dot * edge_weights).sum() / edge_weights.sum()
    else:
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

    return (
        loss_physics_term + lambda_div * loss_diversity + beta_entropy * loss_confidence
    )


def compute_edge_conflict_weights(probs, edge_index_undir):
    """
    Compute per-edge weights based on how conflicted each edge is.

    Edges whose endpoints have similar soft color assignments get higher
    weight so the loss focuses optimization on the hardest spots.

    probs : (N, C) soft color assignments.
    edge_index_undir : (2, E) unique undirected edges.

    Returns (E,) tensor of weights in [1, 2].
    """
    with torch.no_grad():
        src, dst = edge_index_undir
        similarity = torch.sum(probs[src] * probs[dst], dim=1)  # (E,)
        # Map similarity [0, 1] -> weight [1, 2]: more similar = higher weight
        weights = 1.0 + similarity
    return weights


def count_conflicts(coloring, edge_index_undir):
    """
    Number of edges whose endpoints share the same color.

    coloring : (N,) tensor of hard color assignments (0 … C‑1).
    edge_index_undir : (2, E_undir) unique undirected edges.
    """
    src, dst = edge_index_undir
    return (coloring[src] == coloring[dst]).sum().item()


def _build_adj(edge_index_undir, num_nodes):
    """Build adjacency list from undirected edge index."""
    src, dst = edge_index_undir
    adj = [[] for _ in range(num_nodes)]
    for i in range(src.size(0)):
        u, v = src[i].item(), dst[i].item()
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _find_conflicting_nodes(coloring, adj):
    """Return list of nodes involved in at least one conflict."""
    conflicting = []
    for u in range(len(adj)):
        for v in adj[u]:
            if coloring[u] == coloring[v]:
                conflicting.append(u)
                break
    return conflicting


def _greedy_pass(coloring, adj, num_classes):
    """One greedy pass: re-color each conflicting node to the least-conflicted color.
    Returns True if any node was re-colored."""
    conflicting = _find_conflicting_nodes(coloring, adj)
    random.shuffle(conflicting)
    improved = False
    for node in conflicting:
        current = coloring[node].item()
        neighbor_counts = [0] * num_classes
        for nb in adj[node]:
            neighbor_counts[coloring[nb].item()] += 1
        best = min(range(num_classes), key=lambda c: neighbor_counts[c])
        if best != current:
            coloring[node] = best
            improved = True
    return improved


def _kempe_chain_swap(coloring, adj, node, color_a, color_b):
    """Swap colors along the Kempe chain containing *node* for colors (a, b).

    A Kempe chain is the connected component of nodes colored *a* or *b*
    that contains *node*.  Swapping a↔b within this component preserves
    validity for all edges *except* possibly edges that were already
    conflicting — so it can open up new greedy opportunities.
    """
    if coloring[node].item() not in (color_a, color_b):
        return
    visited = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n in visited:
            continue
        visited.add(n)
        c = coloring[n].item()
        if c == color_a:
            coloring[n] = color_b
        elif c == color_b:
            coloring[n] = color_a
        else:
            continue
        for nb in adj[n]:
            if nb not in visited and coloring[nb].item() in (color_a, color_b):
                stack.append(nb)


def greedy_local_search(coloring, edge_index_undir, num_classes=3, max_iters=1000):
    """
    Tabu search for graph coloring — the standard heuristic that actually
    escapes local minima.

    Each iteration picks the best (node, color) move among conflicting nodes.
    A tabu list prevents cycling by forbidding a node from reverting to its
    old color for *tabu_tenure* iterations.  An aspiration criterion overrides
    the tabu if the move produces a new global best.

    coloring : (N,) tensor of hard color assignments (0 … C‑1).
    edge_index_undir : (2, E_undir) unique undirected edges (src < dst).
    num_classes : number of available colors.
    max_iters : maximum number of iterations.

    Returns (improved_coloring, final_conflicts).
    """
    coloring = coloring.clone()
    num_nodes = coloring.size(0)
    adj = _build_adj(edge_index_undir, num_nodes)

    # Precompute per-node conflict counts for O(1) delta evaluation
    # node_color_adj[u][c] = number of neighbors of u with color c
    node_color_adj = [[0] * num_classes for _ in range(num_nodes)]
    for u in range(num_nodes):
        for v in adj[u]:
            node_color_adj[u][coloring[v].item()] += 1

    def _node_conflicts(u):
        return node_color_adj[u][coloring[u].item()]

    total_conflicts = sum(_node_conflicts(u) for u in range(num_nodes)) // 2

    best_coloring = coloring.clone()
    best_conflicts = total_conflicts

    # Tabu list: tabu[node][color] = iteration number when the ban expires
    tabu_tenure = max(7, num_nodes // 10)
    tabu = [[0] * num_classes for _ in range(num_nodes)]

    for it in range(max_iters):
        if total_conflicts == 0:
            return coloring, 0

        # Find the best move among conflicting nodes
        best_move = None   # (node, new_color)
        best_delta = float("inf")

        for u in range(num_nodes):
            if _node_conflicts(u) == 0:
                continue  # not conflicting
            cur_c = coloring[u].item()
            cur_conf = node_color_adj[u][cur_c]

            for c in range(num_classes):
                if c == cur_c:
                    continue
                new_conf = node_color_adj[u][c]
                delta = new_conf - cur_conf   # change in conflicts for this node

                # Accept if: (not tabu) OR (aspiration: produces new global best)
                is_tabu = tabu[u][c] > it
                would_be_best = (total_conflicts + delta) < best_conflicts

                if delta < best_delta and (not is_tabu or would_be_best):
                    best_delta = delta
                    best_move = (u, c)

        if best_move is None:
            break  # no moves available (shouldn't happen)

        # Apply the move
        node, new_color = best_move
        old_color = coloring[node].item()

        # Update node_color_adj for all neighbors
        for v in adj[node]:
            node_color_adj[v][old_color] -= 1
            node_color_adj[v][new_color] += 1

        coloring[node] = new_color
        tabu[node][old_color] = it + tabu_tenure

        # Update total conflict count
        total_conflicts += best_delta

        if total_conflicts < best_conflicts:
            best_conflicts = total_conflicts
            best_coloring = coloring.clone()

    return best_coloring, best_conflicts


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
        "num_used": num_used,
        "counts": counts,
        "distribution": distribution,
        "entropy": entropy.item(),
    }


def train_on_graph(data, num_classes=3, device="cpu", hypers=None):
    """
    Train a GNN (with node embeddings) on a single graph.

    Returns:
        best_coloring (Tensor), best_loss (float),
        best_conflicts (int), epochs_used (int)
    """
    if hypers is None:
        hypers = {
            "dim_embedding": 128,
            "hidden_dim": 128,
            "num_layers": 5,
            "dropout": 0.2,
            "learning_rate": 5e-3,
            "patience": 500,
            "max_epochs": 20000,
            "seed": 42,
            "lambda_div": 0.02,
            "beta_entropy": 0.05,
            "temp_start": 2.0,
            "temp_end": 0.1,
        }

    set_seed(hypers["seed"])

    num_nodes = data.num_nodes
    dim_embedding = hypers["dim_embedding"]
    hidden_dim = hypers["hidden_dim"]
    dropout = hypers["dropout"]
    lr = hypers["learning_rate"]
    patience = hypers["patience"]
    max_epochs = hypers["max_epochs"]
    lambda_div = hypers.get("lambda_div", 0.01)
    temp_start = hypers.get("temp_start", 2.0)
    temp_end = hypers.get("temp_end", 0.1)

    # Model components
    embed = nn.Embedding(num_nodes, dim_embedding).to(device)
    model = GNNConvDeep(
        dim_embedding,
        hidden_dim,
        num_classes,
        num_layers=hypers.get("num_layers", 4),
        dropout=dropout,
    ).to(device)
    # print(model)
    # if hasattr(torch, 'compile'):
    #    model = torch.compile(model)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(embed.parameters()), lr=lr, weight_decay=1e-2
    )
    # Cosine annealing with warm restarts: periodic LR spikes help escape local minima
    restart_period = hypers.get("restart_period", max_epochs // 5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=restart_period, T_mult=2, eta_min=1e-5
    )

    edge_index = data.edge_index.to(device)
    # Use undirected edges for conflict counting if they are stored
    edge_index_undir = getattr(data, "edge_index_undir", edge_index).to(device)

    best_loss = float("inf")
    best_coloring = None
    best_conflicts = num_nodes * 10  # large sentinel
    cnt = 0  # early‑stopping counter
    # Save best model state
    best_model_state = None
    best_embed_state = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        inputs = embed.weight  # (N, dim_embedding)
        logits = model(inputs, edge_index)  # (N, num_classes)

        # Temperature annealing: exponential decay from temp_start to temp_end
        progress = epoch / max(max_epochs - 1, 1)
        temperature = temp_start * (temp_end / temp_start) ** progress
        probs = F.softmax(logits / temperature, dim=1)

        # Compute conflict-weighted edges: focus loss on high-conflict edges
        edge_weights = compute_edge_conflict_weights(probs, edge_index_undir)
        loss = loss_physics_with_confidence(
            probs,
            edge_index_undir,
            lambda_div=lambda_div,
            beta_entropy=hypers.get("beta_entropy", 0.01),
            edge_weights=edge_weights,
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
            print(
                f"    epoch {epoch:5d} | loss {loss.item():.4f} "
                f"| temp {temperature:.3f} "
                f'| colors used {analysis["num_used"]} '
                f'| entropy {analysis["entropy"]:.3f} '
                f"| conflicts {conflicts} "
            )

    # Restore best model if found
    if best_model_state is not None and best_embed_state is not None:
        model.load_state_dict(best_model_state)
        embed.load_state_dict(best_embed_state)

    # Greedy local search post-processing to fix remaining conflicts
    if best_conflicts > 0 and best_coloring is not None:
        edge_undir_cpu = edge_index_undir.cpu()
        improved_coloring, improved_conflicts = greedy_local_search(
            best_coloring, edge_undir_cpu, num_classes=num_classes
        )
        if improved_conflicts < best_conflicts:
            print(
                f"    local search: {best_conflicts} -> {improved_conflicts} conflicts"
            )
            best_coloring = improved_coloring
            best_conflicts = improved_conflicts

    return best_coloring, best_loss, best_conflicts, epoch


def train_with_restarts(data, num_classes=3, device="cpu", hypers=None, num_restarts=5):
    """
    Run train_on_graph multiple times with different random seeds and keep the best result.

    Returns the same tuple as train_on_graph: (best_coloring, best_loss, best_conflicts, epochs_used)
    """
    base_seed = hypers["seed"] if hypers else 42
    overall_best = (None, float("inf"), float("inf"), 0)

    for restart in range(num_restarts):
        restart_hypers = dict(hypers) if hypers else {}
        restart_hypers["seed"] = base_seed + restart * 1000

        print(f'  [restart {restart+1}/{num_restarts}, seed={restart_hypers["seed"]}]')
        coloring, loss, conflicts, epochs = train_on_graph(
            data, num_classes=num_classes, device=device, hypers=restart_hypers
        )

        if conflicts < overall_best[2]:
            overall_best = (coloring, loss, conflicts, epochs)
            print(f"  [restart {restart+1}] new best: {conflicts} conflicts")

        if conflicts == 0:
            break

    return overall_best


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Train unsupervised GNN for 3‑coloring Erdős–Rényi graphs."
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=200,
        help="Number of vertices in each generated graph",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for graph generation and training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run training",
    )
    args = parser.parse_args()

    num_nodes = args.num_nodes
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Number of nodes per graph: {num_nodes}")
    min_p, max_p = 4.0 / num_nodes, 4.6 / num_nodes
    edge_probabilities = [min_p + (max_p - min_p) * i / 10 for i in range(10)]
    print(f"edge_probs    : {[round(e,5) for e in edge_probabilities]}")
    print(f"edge_probs * n: {[round(e*num_nodes,2) for e in edge_probabilities]}")

    hypers = {
        "dim_embedding": 128,  # Increased for richer representations
        "hidden_dim": 128,  # Increased capacity
        "num_layers": 6,  # Reduced: JumpingKnowledge compensates for fewer layers
        "dropout": 0.2,  # Slightly higher dropout for regularization
        "learning_rate": 5e-3,  # Higher learning rate
        "patience": 500,  # More patience for sparse graphs
        "max_epochs": 15000,  # More epochs
        "seed": args.seed,
        "lambda_div": 0.02,  # Increased diversity weight
        "beta_entropy": 0.05,  # Confidence regularization weight
        "temp_start": 2.0,  # Initial softmax temperature (exploratory)
        "temp_end": 0.1,  # Final softmax temperature (committed)
        "restart_period": 3000,  # Warm restart period for LR scheduler
    }
    num_restarts = 3  # Number of random restarts per graph

    results = []
    for p in edge_probabilities:
        print(f"=== p = {p:.3f} ===")
        # Generate ER graph (weights are irrelevant, set to constant 1)
        ugraph = UndirectedGraph_()
        erdos_renyi_(ugraph, n=num_nodes, p=p, weight_range=(1, 1), seed=args.seed)
        data = er_graph_to_data(ugraph)
        data = data.to(device)
        print(
            f"  Graph has {data.num_nodes} nodes, "
            f"{data.edge_index_undir.size(1)} undirected edges"
        )

        t_start = time.time()
        coloring, loss, conflicts, epochs = train_with_restarts(
            data,
            num_classes=3,
            device=device,
            hypers=hypers,
            num_restarts=num_restarts,
        )
        t_elapsed = time.time() - t_start

        # Analyze the best coloring
        analysis = analyze_coloring(coloring, num_classes=3)
        print(
            f"  Result: conflicts = {conflicts}, "
            f"time = {t_elapsed:.2f}s, epochs = {epochs}"
        )
        print(
            f'  Color distribution: {[round(x,2) for x in analysis["distribution"].tolist()]}, '
            f'Colors used: {analysis["num_used"]}, Entropy: {analysis["entropy"]:.3f}\n'
        )
        results.append((p, conflicts, epochs, t_elapsed, analysis))

    # Summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("p\tconflicts\tepochs\ttime (s)\tcolors used\tentropy")
    for p, c, e, t, analysis in results:
        print(
            f'{p:.3f}\t{c}\t\t{e}\t{t:.2f}\t\t{analysis["num_used"]}\t\t{analysis["entropy"]:.3f}'
        )
    print("=" * 70)

    # Count successful colorings
    success_count = sum(1 for _, c, _, _, _ in results if c == 0)
    print(f"\nSuccessful 3-colorings: {success_count}/{len(results)}")
