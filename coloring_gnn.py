import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
import numpy as np
from typing import Any, List, Dict, Tuple, Optional
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Import the existing graph module
from graph import (
    UndirectedGraph_,
    planar_,
    watts_strogatz_,
    barabasi_albert_,
    complete_,
    erdos_renyi_,
)
from coloring_pinn import greedy_local_search, count_conflicts


class GraphColoringDataset:
    """
    Dataset for graph coloring problems.
    Generates graphs without requiring exact CSP solutions.
    """

    def __init__(
        self,
        graph_type: str = "planar",
        num_samples: int = 1000,
        min_nodes: int = 10,
        max_nodes: int = 50,
        num_colors: int = 3,
        seed: int = 42,
    ):
        """
        Initialize the dataset.

        Args:
            graph_type: Type of graph to generate ('planar', 'watts_strogatz',
                       'barabasi_albert', 'complete', 'erdos_renyi')
            num_samples: Number of graphs to generate
            min_nodes: Minimum number of nodes in each graph
            max_nodes: Maximum number of nodes in each graph
            num_colors: Number of colors to use for coloring
            seed: Random seed for reproducibility
        """
        self.graph_type = graph_type
        self.num_samples = num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.num_colors = num_colors
        self.seed = seed
        self.graphs = []
        # Remove solutions list since we don't need exact colorings

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_graphs(self):
        """Generate graphs without computing exact colorings."""
        print(f"Generating {self.num_samples} {self.graph_type} graphs...")

        for i in tqdm(range(self.num_samples)):
            # Randomly choose number of nodes
            n_nodes = random.randint(self.min_nodes, self.max_nodes)

            # Generate graph based on type
            graph = UndirectedGraph_()

            if self.graph_type == "planar":
                # For planar graphs, use remove_probability to vary difficulty
                remove_prob = random.uniform(0.0, 0.5)
                graph = planar_(
                    graph, n=n_nodes, remove_probability=remove_prob, seed=self.seed + i
                )
            elif self.graph_type == "watts_strogatz":
                k = random.randint(2, min(10, n_nodes - 1))
                if k % 2 != 0:
                    k += 1  # Ensure k is even
                beta = random.uniform(0.1, 0.5)
                graph = watts_strogatz_(
                    graph, n=n_nodes, k=k, beta=beta, seed=self.seed + i
                )
            elif self.graph_type == "barabasi_albert":
                m = random.randint(1, min(5, n_nodes - 1))
                graph = barabasi_albert_(graph, n=n_nodes, m=m, seed=self.seed + i)
            elif self.graph_type == "complete":
                graph = complete_(graph, n=n_nodes, seed=self.seed + i)
            elif self.graph_type == "erdos_renyi":
                # Generate Erdos-Renyi graphs at critical threshold for hardness
                # For 3-colorability, critical threshold is around p = 4.5/n
                # We'll sample p around this critical value to get hard instances

                # Base critical threshold for 3 colors
                base_critical_p = 4.5 / n_nodes

                # Add variation: sample from a range around critical threshold
                # This creates a mix of solvable and challenging instances
                if self.num_colors == 3:
                    # For 3-coloring, use critical threshold
                    p_variation = random.uniform(0.8, 1.2)  # ±20% variation
                    p = base_critical_p * p_variation
                else:
                    # For k-coloring, approximate critical threshold as (k * ln(k)) / n
                    critical_constant = self.num_colors * np.log(self.num_colors + 1)
                    p_variation = random.uniform(0.8, 1.2)
                    p = (critical_constant / n_nodes) * p_variation

                # Ensure p is in valid range [0, 1]
                p = max(0.01, min(0.99, p))

                graph = erdos_renyi_(
                    graph,
                    n=n_nodes,
                    p=p,
                    seed=self.seed + i
                )
            else:
                raise ValueError(f"Unknown graph type: {self.graph_type}")

            self.graphs.append(graph)

        print(f"Generated {len(self.graphs)} graphs")

    def generate_hard_erdos_renyi_graphs(self, num_hard_samples: int = 100):
        """
        Generate specifically hard Erdos-Renyi graphs at the critical threshold.

        Args:
            num_hard_samples: Number of hard graphs to generate

        Returns:
            List of hard Erdos-Renyi graphs
        """
        hard_graphs = []
        print(f"Generating {num_hard_samples} hard Erdos-Renyi graphs at critical threshold...")

        for i in tqdm(range(num_hard_samples)):
            # Generate graphs with nodes in the mid-range where hardness is pronounced
            n_nodes = random.randint(30, 80)  # Larger graphs for harder instances

            graph = UndirectedGraph_()

            # Generate at critical threshold for 3-colorability
            # Use the exact critical value with minimal variation
            base_critical_p = 4.5 / n_nodes

            # Add very small variation (±5%) to stay at critical threshold
            p_variation = random.uniform(0.95, 1.05)
            p = base_critical_p * p_variation

            # Ensure p is reasonable
            p = max(0.01, min(0.99, p))

            graph = erdos_renyi_(
                graph,
                n=n_nodes,
                p=p,
                seed=self.seed + 10000 + i  # Different seed range
            )

            hard_graphs.append(graph)

        return hard_graphs

    def graph_to_pyg_data(self, graph_idx: int) -> Data:
        """
        Convert a graph to PyTorch Geometric Data object.
        No target labels needed since we're not learning exact colorings.
        """
        graph = self.graphs[graph_idx]

        # Get vertices and create mapping to indices
        vertices = graph.get_vertices()
        vertex_to_idx = {v: i for i, v in enumerate(vertices)}

        # Create edge index tensor
        edges = []
        for u in vertices:
            for v in graph.get_neighbors(u):
                # Add edge in both directions for undirected graph
                edges.append([vertex_to_idx[u], vertex_to_idx[v]])

        if not edges:
            # Handle isolated nodes
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Lightweight node features — O(degree) per node, no triangle counting.
        # The GNN learns structural patterns (clustering, etc.) via message passing.
        node_features = []
        n = len(vertices)
        degrees = {v: len(graph.get_neighbors(v)) for v in vertices}
        max_degree = max(degrees.values()) if degrees else 1
        for v in vertices:
            degree = degrees[v]
            neighbors = graph.get_neighbors(v)

            # Average neighbor degree (cheap proxy for local density)
            avg_neighbor_deg = (
                sum(degrees[u] for u in neighbors) / degree if degree > 0 else 0
            )

            features = [
                degree / max(n - 1, 1),               # normalized degree
                degree / max(max_degree, 1),           # degree relative to max
                avg_neighbor_deg / max(n - 1, 1),      # avg neighbor degree (normalized)
                1.0 if degree % 2 == 0 else 0.0,       # parity
                min(degree, self.num_colors) / self.num_colors,  # saturation hint
            ]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        # Graph-level features: size and density
        num_edges = len(graph.get_edges())
        density = (
            (2 * num_edges) / (len(vertices) * (len(vertices) - 1))
            if len(vertices) > 1
            else 0
        )
        graph_features = torch.tensor(
            [[len(vertices) / self.max_nodes, density]], dtype=torch.float
        )

        # Return Data without y (target labels)
        return Data(
            x=x,
            edge_index=edge_index,
            num_nodes=len(vertices),
            graph_features=graph_features,
        )

    def get_dataloaders(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15, batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test dataloaders.

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            batch_size: Batch size for dataloaders

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if not self.graphs:
            self.generate_graphs()

        # Shuffle indices
        indices = list(range(len(self.graphs)))
        random.shuffle(indices)

        # Split indices
        train_size = int(train_ratio * len(indices))
        val_size = int(val_ratio * len(indices))

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        # Create datasets
        train_data = [self.graph_to_pyg_data(i) for i in train_indices]
        val_data = [self.graph_to_pyg_data(i) for i in val_indices]
        test_data = [self.graph_to_pyg_data(i) for i in test_indices]

        # Create dataloaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        print(
            f"Train: {len(train_data)} graphs, Val: {len(val_data)} graphs, Test: {len(test_data)} graphs"
        )

        return train_loader, val_loader, test_loader


class GNNColorPredictor(nn.Module):
    """
    GNN model for predicting node colors in graph coloring problems.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        output_dim: int = 3,
        num_layers: int = 3,
        gnn_type: str = "gcn",
        dropout: float = 0.2,
    ):
        """
        Initialize the GNN model.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Number of colors (output dimension)
            num_layers: Number of GNN layers
            gnn_type: Type of GNN layer ('gcn', 'gat', or 'sage')
            dropout: Dropout rate
        """
        super(GNNColorPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout

        # Graph convolutional layers
        self.convs = nn.ModuleList()

        # First layer
        if gnn_type == "gcn":
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif gnn_type == "gat":
            self.convs.append(GATConv(input_dim, hidden_dim))
        elif gnn_type == "sage":
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Middle layers
        for _ in range(num_layers - 2):
            if gnn_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "gat":
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            elif gnn_type == "sage":
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Last layer
        if gnn_type == "gcn":
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn_type == "gat":
            self.convs.append(GATConv(hidden_dim, hidden_dim))
        elif gnn_type == "sage":
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Graph-level pooling and MLP for final prediction
        self.graph_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),  # +2 for graph features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Final classification layer
        self.fc = nn.Linear(
            hidden_dim * 2, output_dim
        )  # *2 for node and graph features

        # Batch normalization
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, data):
        """
        Forward pass of the GNN.

        Args:
            data: PyG Data object with x, edge_index, batch

        Returns:
            Node color predictions
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x) if i < len(self.bns) else x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Get graph-level representation
        graph_embedding = global_mean_pool(x, batch)

        # Handle graph features - they need to be collected from the batch
        # Since graph_features is not automatically batched by PyG, we need to handle it differently
        # We'll check if graph_features exists and has the right shape
        if hasattr(data, "graph_features"):
            # For batched data, graph_features should be stacked
            # If it's a single graph, it will be 1D, so unsqueeze it
            graph_features = data.graph_features
            if graph_features.dim() == 1:
                graph_features = graph_features.unsqueeze(0)
            # Ensure we have the right number of graph features
            if graph_features.size(0) == graph_embedding.size(0):
                graph_embedding = torch.cat([graph_embedding, graph_features], dim=1)

        # Process graph embedding
        graph_embedding = self.graph_mlp(graph_embedding)

        # Expand graph embedding to match node dimensions
        graph_embedding_expanded = graph_embedding[batch]

        # Combine node and graph features for final prediction
        combined = torch.cat([x, graph_embedding_expanded], dim=1)

        # Final classification
        out = self.fc(combined)

        return out


class GraphColoringGNN:
    """
    Main class for training and evaluating GNN-based graph coloring.
    Trains to produce valid colorings without matching exact color assignments.
    """

    def __init__(
        self,
        num_colors: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        gnn_type: str = "gcn",
        learning_rate: float = 0.001,
        constraint_weight: float = 1.0,
        diversity_weight: float = 0.1,  # New: encourages using different colors
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the GNN coloring system.

        Args:
            num_colors: Number of colors to use
            hidden_dim: Hidden dimension of GNN
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ('gcn', 'gat', or 'sage')
            learning_rate: Learning rate for optimizer
            constraint_weight: Weight for adjacency constraint loss
            diversity_weight: Weight for color diversity loss
            device: Device to run on ('cuda' or 'cpu')
        """
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.learning_rate = learning_rate
        self.constraint_weight = constraint_weight
        self.diversity_weight = diversity_weight
        self.device = torch.device(device)

        # Initialize model
        self.model = GNNColorPredictor(
            input_dim=5,
            hidden_dim=hidden_dim,
            output_dim=num_colors,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=0.3,
        ).to(self.device)

        # Optimizer only - no classification criterion needed
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_conflict_ratios = []  # Track validation conflict ratios
        self.train_constraint_losses = []
        self.train_diversity_losses = []

    def compute_constraint_loss(self, probs, edge_index):
        """Compute loss for adjacent nodes having same color."""
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=self.device)

        src, dst = edge_index[0], edge_index[1]
        src_probs = probs[src]
        dst_probs = probs[dst]

        # Dot product measures similarity - we want to minimize this
        similarity = torch.sum(src_probs * dst_probs, dim=1)
        return similarity.mean()

    def compute_diversity_loss(self, probs):
        """Encourage using all colors to avoid trivial solutions."""
        # Average color distribution across all nodes
        avg_distribution = probs.mean(dim=0)

        # We want uniform distribution to encourage using all colors
        target_uniform = (
            torch.ones(self.num_colors, device=self.device) / self.num_colors
        )

        # KL divergence between average distribution and uniform
        kl_div = F.kl_div(avg_distribution.log(), target_uniform, reduction="batchmean")
        return kl_div

    def compute_entropy_loss(self, probs):
        """Encourage confident predictions (low entropy)."""
        # Negative entropy to encourage confident predictions
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return (
            -entropy.mean()
        )  # Negative because we want to minimize negative entropy = maximize confidence

    def train_epoch(self, train_loader):
        """Train for one epoch using constraint-based losses only."""
        self.model.train()
        total_loss = 0
        total_constraint_loss = 0
        total_diversity_loss = 0
        total_entropy_loss = 0
        total_samples = 0

        for batch in train_loader:
            batch = batch.to(self.device)

            # Forward pass
            outputs = self.model(batch)

            # Convert to probabilities
            probs = F.softmax(outputs, dim=1)

            # Compute constraint loss (adjacent nodes should have different colors)
            constraint_loss = self.compute_constraint_loss(probs, batch.edge_index)

            # Compute diversity loss (encourage using all colors)
            diversity_loss = self.compute_diversity_loss(probs)

            # Compute entropy loss (encourage confident predictions)
            entropy_loss = self.compute_entropy_loss(probs)

            # Total loss - weighted combination
            loss = (
                self.constraint_weight * constraint_loss
                + self.diversity_weight * diversity_loss
                + entropy_loss
            )  # entropy_loss is already negative, so adding it encourages confidence

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate statistics
            total_loss += loss.item() * batch.num_graphs
            total_constraint_loss += constraint_loss.item() * batch.num_graphs
            total_diversity_loss += diversity_loss.item() * batch.num_graphs
            total_entropy_loss += entropy_loss.item() * batch.num_graphs
            total_samples += batch.num_graphs

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_constraint_loss = (
            total_constraint_loss / total_samples if total_samples > 0 else 0
        )
        avg_diversity_loss = (
            total_diversity_loss / total_samples if total_samples > 0 else 0
        )
        avg_entropy_loss = (
            total_entropy_loss / total_samples if total_samples > 0 else 0
        )

        self.train_constraint_losses.append(avg_constraint_loss)
        self.train_diversity_losses.append(avg_diversity_loss)

        return avg_loss, avg_constraint_loss, avg_diversity_loss, avg_entropy_loss

    def validate(self, val_loader):
        """Validate the model - compute conflict ratio and other metrics."""
        self.model.eval()
        total_conflicts = 0
        total_edges = 0
        total_colors_used = 0
        total_graphs = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                # Forward pass
                outputs = self.model(batch)
                predictions = torch.argmax(outputs, dim=1)

                # Count conflicts for each graph in batch
                edge_index = batch.edge_index
                batch_indices = batch.batch

                if edge_index.size(1) > 0:
                    src, dst = edge_index[0], edge_index[1]

                    # Get predictions for source and target nodes
                    src_preds = predictions[src]
                    dst_preds = predictions[dst]

                    # Count conflicts (same color on adjacent nodes)
                    conflicts = (src_preds == dst_preds).sum().item()
                    total_conflicts += conflicts
                    total_edges += (
                        edge_index.size(1) // 2
                    )  # Each edge counted twice in undirected

                # Count colors used
                for graph_idx in range(batch.num_graphs):
                    graph_mask = batch_indices == graph_idx
                    graph_predictions = predictions[graph_mask]
                    colors_used = len(torch.unique(graph_predictions))
                    total_colors_used += colors_used
                    total_graphs += 1

        conflict_ratio = total_conflicts / total_edges if total_edges > 0 else 0
        avg_colors_used = total_colors_used / total_graphs if total_graphs > 0 else 0

        return conflict_ratio, avg_colors_used

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        patience: int = 80,
        save_path: str = "best_model.pth",
    ):
        """
        Train the model to produce valid colorings.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_path: Path to save best model
        """
        print(f"Training GNN model on {self.device}...")
        print(
            f"Model: {self.gnn_type.upper()} with {self.num_layers} layers, hidden_dim={self.hidden_dim}"
        )
        print(f"Training to produce valid {self.num_colors}-colorings")

        best_conflict_ratio = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            (
                train_loss,
                train_constraint_loss,
                train_diversity_loss,
                train_entropy_loss,
            ) = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_conflict_ratio, val_avg_colors = self.validate(val_loader)
            self.val_losses.append(
                val_conflict_ratio
            )  # Using conflict ratio as validation "loss"
            self.val_conflict_ratios.append(val_conflict_ratio)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:3d}/{num_epochs}: "
                    f"Train Loss: {train_loss:.4f} (Constraint: {train_constraint_loss:.4f}, "
                    f"Diversity: {train_diversity_loss:.4f}, Entropy: {train_entropy_loss:.4f}), "
                    f"Val Conflict Ratio: {val_conflict_ratio:.4f}, "
                    f"Val Avg Colors: {val_avg_colors:.2f}"
                )

            # Early stopping based on conflict ratio
            if val_conflict_ratio < best_conflict_ratio:
                best_conflict_ratio = val_conflict_ratio
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(
                    f"  -> Saved best model (conflict ratio: {val_conflict_ratio:.4f})"
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        try:
            self.model.load_state_dict(torch.load(save_path))
        except:
            print("Could not load saved model, using current model")
        print(
            f"Training completed. Best validation conflict ratio: {best_conflict_ratio:.4f}"
        )

    def _graph_to_data(self, graph: UndirectedGraph_) -> Data:
        """Convert a single UndirectedGraph_ to a PyG Data object on self.device."""
        dataset = GraphColoringDataset(num_samples=1, min_nodes=10, max_nodes=10,
                                       num_colors=self.num_colors)
        dataset.graphs = [graph]
        data = dataset.graph_to_pyg_data(0)
        data = data.to(self.device)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
        if not hasattr(data, "graph_features"):
            data.graph_features = torch.zeros(1, 2, device=self.device)
        return data

    def predict(self, graph: UndirectedGraph_, use_local_search: bool = True) -> Dict[Any, int]:
        """
        Predict coloring for a single graph, optionally followed by local search.

        Args:
            graph: Input graph
            use_local_search: If True, apply Kempe-chain local search to fix
                remaining conflicts after the GNN forward pass.

        Returns:
            Dictionary mapping vertices to colors
        """
        self.model.eval()
        data = self._graph_to_data(graph)

        with torch.no_grad():
            outputs = self.model(data)
            predictions = torch.argmax(outputs, dim=1)

        # Local search post-processing
        if use_local_search:
            # Build undirected edge index (src < dst) for local search
            edge_index = data.edge_index.cpu()
            src, dst = edge_index[0], edge_index[1]
            mask = src < dst
            edge_index_undir = torch.stack([src[mask], dst[mask]], dim=0)

            predictions_cpu = predictions.cpu()
            conflicts = count_conflicts(predictions_cpu, edge_index_undir)
            if conflicts > 0:
                predictions_cpu, new_conflicts = greedy_local_search(
                    predictions_cpu, edge_index_undir,
                    num_classes=self.num_colors,
                )
                predictions = predictions_cpu

        vertices = graph.get_vertices()
        coloring = {
            vertex: int(predictions[i]) for i, vertex in enumerate(vertices)
        }
        return coloring

    def fine_tune_predict(
        self,
        graph: UndirectedGraph_,
        num_steps: int = 200,
        lr: float = 1e-3,
        use_local_search: bool = True,
    ) -> Dict[Any, int]:
        """
        Predict coloring using the amortized model as initialisation, then
        fine-tune on this specific graph using the physics constraint loss.

        This combines the best of both approaches:
        - The pretrained GNN provides a strong starting point (fast).
        - Per-instance gradient steps resolve graph-specific conflicts.
        - Local search cleans up any remaining issues.

        Args:
            graph: Input graph
            num_steps: Number of fine-tuning gradient steps.
            lr: Learning rate for fine-tuning.
            use_local_search: Apply local search after fine-tuning.

        Returns:
            Dictionary mapping vertices to colors
        """
        data = self._graph_to_data(graph)

        # Build undirected edge index for loss & local search
        edge_index = data.edge_index
        src, dst = edge_index[0], edge_index[1]
        mask = src < dst
        edge_index_undir = torch.stack([src[mask], dst[mask]], dim=0)

        # Fine-tune a copy of the model so the pretrained weights are preserved
        import copy
        ft_model = copy.deepcopy(self.model)
        ft_model.train()
        optimizer = optim.Adam(ft_model.parameters(), lr=lr)

        best_conflicts = float("inf")
        best_predictions = None

        for step in range(num_steps):
            optimizer.zero_grad()
            outputs = ft_model(data)
            probs = F.softmax(outputs, dim=1)

            # Physics constraint loss: minimise dot product of adjacent soft assignments
            src_u, dst_u = edge_index_undir
            dot = torch.sum(probs[src_u] * probs[dst_u], dim=1)
            loss = dot.mean()

            # Diversity: encourage uniform colour usage
            avg_dist = probs.mean(dim=0)
            uniform = torch.ones_like(avg_dist) / avg_dist.size(0)
            loss = loss + 0.01 * torch.sum(
                avg_dist * (torch.log(avg_dist + 1e-8) - torch.log(uniform))
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(probs, dim=1)
                conflicts = count_conflicts(preds.cpu(), edge_index_undir.cpu())
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_predictions = preds.cpu()
                if conflicts == 0:
                    break

        predictions = best_predictions

        # Local search post-processing
        if use_local_search and best_conflicts > 0:
            edge_undir_cpu = edge_index_undir.cpu()
            predictions, _ = greedy_local_search(
                predictions, edge_undir_cpu,
                num_classes=self.num_colors,
            )

        vertices = graph.get_vertices()
        coloring = {
            vertex: int(predictions[i]) for i, vertex in enumerate(vertices)
        }
        return coloring

    def evaluate_coloring(
        self, graph: UndirectedGraph_, coloring: Dict[Any, int]
    ) -> Dict[str, float]:
        """
        Evaluate the quality of a coloring.

        Args:
            graph: Input graph
            coloring: Coloring assignment

        Returns:
            Dictionary with evaluation metrics
        """
        vertices = graph.get_vertices()

        # Check for conflicts
        conflicts = 0
        total_edges = 0

        for u in vertices:
            for v in graph.get_neighbors(u):
                if u < v:  # Count each edge once
                    total_edges += 1
                    if coloring.get(u) == coloring.get(v):
                        conflicts += 1

        # Number of colors used
        colors_used = len(set(coloring.values()))

        return {
            "conflicts": conflicts,
            "total_edges": total_edges,
            "conflict_ratio": conflicts / total_edges if total_edges > 0 else 0,
            "colors_used": colors_used,
            "is_valid": conflicts == 0,
        }

    def plot_training_history(self):
        """Plot training history with new metrics."""
        if not self.train_losses:
            print("No training history to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot total loss
        axes[0, 0].plot(self.train_losses, label="Train Loss", color="blue")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Total Training Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot constraint loss
        if self.train_constraint_losses:
            axes[0, 1].plot(
                self.train_constraint_losses, label="Constraint Loss", color="red"
            )
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].set_title("Constraint Loss (Adjacency)")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Plot conflict ratio
        if self.val_conflict_ratios:
            axes[1, 0].plot(
                self.val_conflict_ratios, label="Val Conflict Ratio", color="green"
            )
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Conflict Ratio")
            axes[1, 0].set_title("Validation Conflict Ratio")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Plot diversity loss if available
        if hasattr(self, "train_diversity_losses") and self.train_diversity_losses:
            axes[1, 1].plot(
                self.train_diversity_losses, label="Diversity Loss", color="purple"
            )
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].set_title("Color Diversity Loss")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def train_on_hard_graphs(
        self,
        hard_graphs,
        num_epochs: int = 50,
        batch_size: int = 16,
    ):
        """
        Train specifically on hard graphs to improve performance on challenging instances.

        Args:
            hard_graphs: List of hard graphs to train on
            num_epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print(f"\nTraining on {len(hard_graphs)} hard graphs for {num_epochs} epochs...")

        # Convert hard graphs to PyG dataset
        dataset = GraphColoringDataset(num_samples=len(hard_graphs), min_nodes=10, max_nodes=10)
        dataset.graphs = hard_graphs

        # Create dataloader
        hard_data = [dataset.graph_to_pyg_data(i) for i in range(len(hard_graphs))]
        hard_loader = DataLoader(hard_data, batch_size=batch_size, shuffle=True)

        # Train on hard graphs
        for epoch in range(num_epochs):
            train_loss, train_constraint_loss, train_diversity_loss, train_entropy_loss = self.train_epoch(hard_loader)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Hard Training Epoch {epoch+1:3d}/{num_epochs}: "
                    f"Loss: {train_loss:.4f} (Constraint: {train_constraint_loss:.4f}, "
                    f"Diversity: {train_diversity_loss:.4f})"
                )


def compare_with_csp(
    gnn_colorer: GraphColoringGNN, graph: UndirectedGraph_, num_colors: int = 3
):
    """
    Compare GNN predictions with CSP solver results.

    Args:
        gnn_colorer: Trained GNN model
        graph: Graph to color
        num_colors: Number of colors to use
    """
    print("\n" + "=" * 50)
    print("Comparing GNN with CSP Solver")
    print("=" * 50)

    # GNN prediction
    print("\n1. GNN Prediction:")
    gnn_coloring = gnn_colorer.predict(graph)
    gnn_metrics = gnn_colorer.evaluate_coloring(graph, gnn_coloring)

    print(f"   Colors used: {gnn_metrics['colors_used']}")
    print(
        f"   Conflicts: {gnn_metrics['conflicts']}/{gnn_metrics['total_edges']} edges"
    )
    print(f"   Conflict ratio: {gnn_metrics['conflict_ratio']:.4f}")
    print(f"   Valid coloring: {gnn_metrics['is_valid']}")

    # CSP solution
    print("\n2. CSP Solver:")
    try:
        from csp2 import CSP

        csp = CSP(graph, domain=list(range(num_colors)))
        solution_tuple = csp.solve(
            use_mrv=True,
            use_degree=True,
            use_lcv=True,
            use_forward_checking=True,
            use_ac3=False,
            max_backtracks=10000,
        )

        if solution_tuple and solution_tuple[0] is not None:
            csp_solution = solution_tuple[0]
            csp_metrics = gnn_colorer.evaluate_coloring(graph, csp_solution)

            print(f"   Solution found: Yes")
            print(f"   Colors used: {csp_metrics['colors_used']}")
            print(
                f"   Conflicts: {csp_metrics['conflicts']}/{csp_metrics['total_edges']} edges"
            )
            print(f"   Valid coloring: {csp_metrics['is_valid']}")

            # Compare
            print("\n3. Comparison:")
            print(
                f"   Both valid: {gnn_metrics['is_valid'] and csp_metrics['is_valid']}"
            )
            if gnn_metrics["is_valid"] and csp_metrics["is_valid"]:
                print(f"   GNN matches CSP validity: ✓")
            else:
                print(
                    f"   GNN matches CSP validity: ✗ (GNN: {gnn_metrics['is_valid']}, CSP: {csp_metrics['is_valid']})"
                )

        else:
            print(f"   Solution found: No (graph may not be {num_colors}-colorable)")
            print(f"   Note: Some graphs may not be colorable with {num_colors} colors")

    except ImportError as e:
        print(f"   Error: Could not import CSP solver - {e}")
    except Exception as e:
        print(f"   Error running CSP solver: {e}")


def main():
    """Main function to demonstrate GNN-based graph coloring."""
    import argparse

    parser = argparse.ArgumentParser(description="GNN-based Graph Coloring")
    parser.add_argument(
        "--graph_type",
        type=str,
        default="erdos_renyi",
        choices=["planar", "watts_strogatz", "barabasi_albert", "complete", "erdos_renyi"],
        help="Type of graph to generate for training",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of training graphs"
    )
    parser.add_argument(
        "--num_colors", type=int, default=3, help="Number of colors to use"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden dimension of GNN"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of GNN layers"
    )
    parser.add_argument(
        "--gnn_type",
        type=str,
        default="gcn",
        choices=["gcn", "gat", "sage"],
        help="Type of GNN layer",
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--constraint_weight",
        type=float,
        default=2.0,
        help="Weight for constraint loss term",
    )
    parser.add_argument(
        "--compare_csp",
        action="store_true",
        help="Compare with CSP solver on test graphs",
    )
    parser.add_argument(
        "--train_max_nodes", type=int, default=30,
        help="Max nodes for training graphs (train on small for speed)",
    )
    parser.add_argument(
        "--fine_tune_steps", type=int, default=300,
        help="Number of per-instance fine-tuning steps at inference (0 to disable)",
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create dataset — train on small graphs for speed and generalization
    print(f"Creating training dataset (≤{args.train_max_nodes} nodes, "
          f"type={args.graph_type})...")
    dataset = GraphColoringDataset(
        graph_type=args.graph_type,
        num_samples=args.num_samples,
        min_nodes=8,
        max_nodes=args.train_max_nodes,
        num_colors=args.num_colors,
        seed=args.seed,
    )

    # Get dataloaders
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        train_ratio=0.7, val_ratio=0.15, batch_size=args.batch_size
    )

    # Initialize and train GNN
    gnn_colorer = GraphColoringGNN(
        num_colors=args.num_colors,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
        learning_rate=0.001,
        constraint_weight=args.constraint_weight,
    )

    # Train the model
    gnn_colorer.train(train_loader, val_loader, num_epochs=args.epochs)

    # Plot training history
    gnn_colorer.plot_training_history()

    # ── helpers for test graph generation ─────────────────────────────
    def make_graph(gtype, n, seed_offset):
        """Generate a single test graph of the given type and size."""
        g = UndirectedGraph_()
        s = args.seed + seed_offset
        if gtype == "erdos_renyi":
            # Critical threshold for 3-colorability — hardest instances
            p = 4.5 / n * random.uniform(0.95, 1.05)
            erdos_renyi_(g, n=n, p=p, seed=s)
        elif gtype == "planar":
            planar_(g, n=n, remove_probability=random.uniform(0.0, 0.3), seed=s)
        elif gtype == "watts_strogatz":
            k = min(6, n - 1)
            if k % 2 != 0:
                k += 1
            watts_strogatz_(g, n=n, k=k, beta=0.3, seed=s)
        elif gtype == "barabasi_albert":
            barabasi_albert_(g, n=n, m=3, seed=s)
        return g

    def evaluate_graph(label, graph, fine_tune_steps):
        """Run amortized + fine-tuned prediction and return metrics tuple."""
        n_v = len(graph.get_vertices())
        n_e = len(graph.get_edges())

        coloring = gnn_colorer.predict(graph)
        m = gnn_colorer.evaluate_coloring(graph, coloring)

        ft_m = None
        if fine_tune_steps > 0:
            ft_coloring = gnn_colorer.fine_tune_predict(
                graph, num_steps=fine_tune_steps,
            )
            ft_m = gnn_colorer.evaluate_coloring(graph, ft_coloring)

        return n_v, n_e, m, ft_m

    # ── Scaling test: multiple graph types × increasing sizes ─────────
    # Train on small graphs (max_nodes above), now stress-test on larger ones.
    test_sizes = [50, 100, 200, 500, 1000]
    # Graph types that are known to be 3-colorable or very likely 3-colorable:
    #   planar (always ≤4-colorable, almost always 3),
    #   ER at critical threshold (likely 3-colorable below threshold),
    #   watts-strogatz (sparse, easy to color),
    #   barabasi-albert (sparse, tree-like core).
    test_types = ["erdos_renyi", "planar", "watts_strogatz", "barabasi_albert"]
    graphs_per_cell = 3  # graphs per (type, size) combination

    print("\n" + "=" * 70)
    print("SCALING TEST — trained on graphs with ≤{} nodes".format(args.train_max_nodes))
    print("=" * 70)

    # Collect results for summary table
    summary_rows = []  # (type, n, amortized_conflicts, ft_conflicts, edges)

    for gtype in test_types:
        print(f"\n{'─'*60}")
        print(f"  Graph type: {gtype}")
        print(f"{'─'*60}")

        for n_nodes in test_sizes:
            amort_total_conflicts = 0
            ft_total_conflicts = 0
            total_edges = 0
            all_valid_amort = True
            all_valid_ft = True

            for trial in range(graphs_per_cell):
                seed_off = hash((gtype, n_nodes, trial)) % 100000
                graph = make_graph(gtype, n_nodes, seed_off)
                n_v, n_e, m, ft_m = evaluate_graph(
                    f"{gtype} n={n_nodes}", graph, args.fine_tune_steps,
                )
                amort_total_conflicts += m["conflicts"]
                total_edges += m["total_edges"]
                if not m["is_valid"]:
                    all_valid_amort = False
                if ft_m is not None:
                    ft_total_conflicts += ft_m["conflicts"]
                    if not ft_m["is_valid"]:
                        all_valid_ft = False

            avg_e = total_edges / graphs_per_cell
            tag_a = "✓" if all_valid_amort else f"{amort_total_conflicts} conflicts"
            tag_f = "✓" if all_valid_ft else f"{ft_total_conflicts} conflicts"

            print(f"  n={n_nodes:5d}  edges≈{avg_e:6.0f}  "
                  f"amortized: {tag_a:>15s}  fine-tuned: {tag_f:>15s}  "
                  f"({graphs_per_cell} graphs)")

            summary_rows.append((
                gtype, n_nodes, graphs_per_cell,
                amort_total_conflicts, ft_total_conflicts,
                total_edges, all_valid_amort, all_valid_ft,
            ))

    # ── Summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'type':<18s} {'n':>5s}  {'trials':>6s}  "
          f"{'amort_conflicts':>15s}  {'ft_conflicts':>12s}  {'edges':>7s}")
    print("-" * 70)
    for gtype, n, trials, ac, fc, te, va, vf in summary_rows:
        print(f"{gtype:<18s} {n:5d}  {trials:6d}  "
              f"{ac:15d}  {fc:12d}  {te:7d}")
    print("=" * 70)

    total_amort = sum(r[3] for r in summary_rows)
    total_ft = sum(r[4] for r in summary_rows)
    total_e = sum(r[5] for r in summary_rows)
    print(f"Total conflicts — amortized: {total_amort}, fine-tuned: {total_ft} "
          f"(over {total_e} edges)")
    all_zero_ft = all(r[4] == 0 for r in summary_rows)
    if all_zero_ft:
        print("All fine-tuned colorings are valid — zero conflicts across all sizes!")


if __name__ == "__main__":
    # Check if torch_geometric is available
    try:
        import torch_geometric

        main()
    except ImportError as e:
        print("Error: torch_geometric is not installed.")
        print("Please install it using:")
        print("  pip install torch_geometric")
        print("\nYou may also need to install additional dependencies:")
        print(
            "  pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv"
        )
        sys.exit(1)
