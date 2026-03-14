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
)


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
            graph_type: Type of graph to generate ('planar', 'watts_strogatz', 'barabasi_albert', 'complete')
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
            else:
                raise ValueError(f"Unknown graph type: {self.graph_type}")

            self.graphs.append(graph)

        print(f"Generated {len(self.graphs)} graphs")

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

        # Enhanced node features (same as before)
        node_features = []
        for v in vertices:
            neighbors = graph.get_neighbors(v)
            degree = len(neighbors)

            # Compute clustering coefficient approximation
            triangle_count = 0
            neighbor_set = set(neighbors)
            for u in neighbors:
                u_neighbors = set(graph.get_neighbors(u))
                common = neighbor_set.intersection(u_neighbors)
                triangle_count += len(common)
            triangle_count = triangle_count / 3 if degree >= 2 else 0
            clustering_coeff = (
                (2 * triangle_count) / (degree * (degree - 1)) if degree >= 2 else 0
            )

            # Enhanced features
            features = [
                degree / (len(vertices) - 1) if len(vertices) > 1 else 0,
                clustering_coeff,
                1.0 if degree >= 2 else 0.0,
                degree % 2,
                len([n for n in neighbors if len(graph.get_neighbors(n)) > degree])
                / max(degree, 1),
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

    def predict(self, graph: UndirectedGraph_) -> Dict[Any, int]:
        """
        Predict coloring for a single graph.

        Args:
            graph: Input graph

        Returns:
            Dictionary mapping vertices to colors
        """
        self.model.eval()

        # Convert graph to PyG Data
        dataset = GraphColoringDataset(num_samples=1, min_nodes=10, max_nodes=10)
        dataset.graphs = [graph]

        # Generate data without solutions
        data = dataset.graph_to_pyg_data(0)
        data = data.to(self.device)

        with torch.no_grad():
            # Add batch dimension
            data.batch = torch.zeros(
                data.num_nodes, dtype=torch.long, device=self.device
            )
            if not hasattr(data, "graph_features"):
                data.graph_features = torch.zeros(1, 2, device=self.device)

            # Get predictions
            outputs = self.model(data)
            predictions = torch.argmax(outputs, dim=1)

            # Convert to dictionary
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
        default="planar",
        choices=["planar", "watts_strogatz", "barabasi_albert", "complete"],
        help="Type of graph to generate",
    )
    parser.add_argument(
        "--num_samples", type=int, default=500, help="Number of training graphs"
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
        "--epochs", type=int, default=100, help="Number of training epochs"
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

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create dataset
    print(f"Creating dataset with {args.graph_type} graphs...")
    dataset = GraphColoringDataset(
        graph_type=args.graph_type,
        num_samples=args.num_samples,
        min_nodes=10,
        max_nodes=50,
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

    # Test on new graphs
    print("\n" + "=" * 50)
    print("Testing on new graphs...")
    print("=" * 50)

    # Generate test graphs
    test_graphs = []
    for i in range(5):
        n_nodes = random.randint(15, 30)
        graph = UndirectedGraph_()

        if args.graph_type == "planar":
            graph = planar_(
                graph, n=n_nodes, remove_probability=0.2, seed=args.seed + 1000 + i
            )
        elif args.graph_type == "watts_strogatz":
            graph = watts_strogatz_(
                graph, n=n_nodes, k=4, beta=0.3, seed=args.seed + 1000 + i
            )
        elif args.graph_type == "barabasi_albert":
            graph = barabasi_albert_(graph, n=n_nodes, m=2, seed=args.seed + 1000 + i)
        elif args.graph_type == "complete":
            graph = complete_(
                graph, n=min(n_nodes, 10), seed=args.seed + 1000 + i
            )  # Complete graphs get large quickly

        test_graphs.append(graph)

    # Predict and evaluate
    for i, graph in enumerate(test_graphs):
        print(
            f"\nTest Graph {i+1}: {len(graph.get_vertices())} vertices, {len(graph.get_edges())} edges"
        )

        # Predict coloring
        coloring = gnn_colorer.predict(graph)

        # Evaluate
        metrics = gnn_colorer.evaluate_coloring(graph, coloring)

        print(f"  Colors used: {metrics['colors_used']}")
        print(f"  Conflicts: {metrics['conflicts']}/{metrics['total_edges']} edges")
        print(f"  Conflict ratio: {metrics['conflict_ratio']:.4f}")
        print(f"  Valid coloring: {metrics['is_valid']}")

        # Compare with CSP if requested
        if args.compare_csp:
            compare_with_csp(gnn_colorer, graph, args.num_colors)

    print("\n" + "=" * 50)
    print("GNN-based Graph Coloring Complete!")
    print("=" * 50)

    # Demonstrate on a specific planar graph
    print("\n" + "=" * 50)
    print("Demonstration on a specific planar graph:")
    print("=" * 50)

    # Create a planar graph for demonstration
    demo_graph = planar_(
        UndirectedGraph_(), n=20, remove_probability=0.1, seed=args.seed + 999
    )
    print(
        f"Demo graph: {len(demo_graph.get_vertices())} vertices, {len(demo_graph.get_edges())} edges"
    )

    # Get GNN prediction
    demo_coloring = gnn_colorer.predict(demo_graph)
    demo_metrics = gnn_colorer.evaluate_coloring(demo_graph, demo_coloring)

    print(f"\nGNN Result:")
    print(f"  Valid coloring: {demo_metrics['is_valid']}")
    print(f"  Colors used: {demo_metrics['colors_used']}")
    print(f"  Conflicts: {demo_metrics['conflicts']}/{demo_metrics['total_edges']}")

    if args.compare_csp:
        compare_with_csp(gnn_colorer, demo_graph, args.num_colors)


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
