import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Federated graph anomaly detection")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs")

    # Dataset and federated partition
    parser.add_argument("--data_name", type=str, default="Tolokers", help="Dataset name (reddit|Tolokers|Amazon|Amazon_all|YelpChi|Questions|Elliptic)")
    parser.add_argument("--partition", type=str, default="Louvain", help="Graph partition method")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of federated clients")
    # Global model
    parser.add_argument("--gmodel_name", type=str, default="GCN", choices=["GCN", "GraphTransformer"], help="Global model architecture")
    parser.add_argument("--num_rounds", type=int, default=100, help="Number of global communication rounds")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of local epochs in each global round")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--drop", type=float, default=0.5, help="Dropout probability")

    # Personalized model
    parser.add_argument("--normalize_train", type=int, default=3, help="Number of personalized training repetitions")
    parser.add_argument("--prop_steps", type=int, default=3, help="Number of graph propagation steps")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden representation dimension")
    parser.add_argument("--output_dim", type=int, default=32, help="Output representation dimension")
    parser.add_argument("--epochs", type=int, default=200, help="Number of personalized training epochs")

    # Relation learning
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for combining original and learned adjacency matrices")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for combining fixed and learnable relations")

    return parser.parse_args()


args = parse_args()
print(args)