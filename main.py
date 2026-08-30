import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import time
from pathlib import Path
import numpy as np
import torch
from torch.optim import Adam
from config import args
from datasets.graph_fl_dataset import GraphFLDataset
from fedKnowledge_main import step1_main
from models.model import MyModel
from utils import *

SPECIAL_DATASETS = {"Tolokers","Elliptic"}
MODEL_WEIGHTS_DIR = Path("./model_weights")

def configure_dataset_args() -> None:
    if args.data_name in SPECIAL_DATASETS:
        args.lr = 5e-3


def load_global_model(datasets, model_name: str, device: torch.device):
    weight_path = MODEL_WEIGHTS_DIR / (f"{datasets.name}_Client{datasets.num_clients}_"
        f"{datasets.sampling}_{model_name}.pt")
    if not weight_path.exists():
        raise FileNotFoundError(f"Global model checkpoint not found: {weight_path}")
    return torch.load(weight_path, map_location=device)


def evaluate(model, subgraph, device):
    model.eval()
    with torch.no_grad():
        _, _, _, anomaly_score = model.hete_forward(
            device=device, is_train=False)
        val_scores = anomaly_score[subgraph.val_idx]
        test_scores = anomaly_score[subgraph.test_idx]
        val_labels = subgraph.y[subgraph.val_idx]
        test_labels = subgraph.y[subgraph.test_idx]

        val_auc, val_ap = auc_ap(val_labels.detach().cpu().numpy(),val_scores.detach().cpu().numpy(),)
        test_auc, test_ap = auc_ap(test_labels.detach().cpu().numpy(),test_scores.detach().cpu().numpy())

    return val_auc, test_auc, val_ap, test_ap


def train_client(subgraph, datasets, device):
    subgraph.y = subgraph.y.to(device)
    global_model = load_global_model(datasets, args.gmodel_name, device)
    global_model.preprocess(subgraph.adj, subgraph.x)
    global_model = global_model.to(device)
    nodes_embedding, _ = global_model.model_forward(range(subgraph.num_nodes), device)

    model = MyModel(prop_steps=args.prop_steps,feat_dim=datasets.input_dim,hidden_dim=args.hidden_dim,output_dim=args.output_dim,)
    model.non_para_lp(subgraph=subgraph,nodes_embedding=nodes_embedding,x=subgraph.x.to(device),device=device)
    model.preprocess(adj=subgraph.adj, device=device)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metrics = {"auc_val": float("-inf"),"auc_test": 0.0,"ap_val": 0.0,"ap_test": 0.0}
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        original_emb, smooth_emb, propagated_emb, _ = model.hete_forward(device=device,is_train=True)
        normal_mask, abnormal_mask = build_train_class_masks(subgraph.train_idx,subgraph.y,device)
        loss = consistency_loss(original_emb,smooth_emb,propagated_emb,normal_mask,abnormal_mask)
        loss.backward()
        optimizer.step()

        val_auc, test_auc, val_ap, test_ap = evaluate(model, subgraph, device)
        if val_auc > best_metrics["auc_val"]:
            best_metrics.update(auc_val=val_auc,auc_test=test_auc,ap_val=val_ap,ap_test=test_ap)

    return best_metrics


def aggregate_client_metrics(datasets, client_records):
    client_sizes = np.asarray([subgraph.num_nodes for subgraph in datasets.subgraphs], dtype=float)
    weights = client_sizes / client_sizes.sum()

    return {metric: float(sum(weight * record[metric] for weight, record in zip(weights, client_records)))
        for metric in ("auc_val", "auc_test", "ap_val", "ap_test")}


def step2_main(run_idx: int):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    datasets = GraphFLDataset(root="./datasets",name=args.data_name,sampling=args.partition,num_clients=args.num_clients,analysis_local_subgraph=True,analysis_global_graph=False)
    print(f"\n| ★ Start Local Client Personalized Training - Run {run_idx + 1}")
    start_time = time.time()
    client_records = [train_client(subgraph, datasets, device) for subgraph in datasets.subgraphs]
    metrics = aggregate_client_metrics(datasets, client_records)

    print(f"| ★ Run {run_idx + 1} Completed - " f"Val AUC: {metrics['auc_val']:.4f}, " f"Test AUC: {metrics['auc_test']:.4f}, " f"Val AP: {metrics['ap_val']:.4f}, " f"Test AP: {metrics['ap_test']:.4f}")
    print(f"| Total Time Elapsed: {time.time() - start_time:.4f}s")

    return (metrics["auc_val"],metrics["auc_test"],metrics["ap_val"],metrics["ap_test"])


def mean_std(values):
    """Return mean and sample standard deviation; handle a single run safely."""
    ddof = 1 if len(values) > 1 else 0
    return float(np.mean(values)), float(np.std(values, ddof=ddof))


def main() -> None:
    configure_dataset_args()
    results = {"val_auc": [], "test_auc": [], "val_ap": [], "test_ap": []}

    for run_idx in range(args.runs):
        print(f"\n=== Overall Run {run_idx + 1}/{args.runs} ===")
        set_seed(args.seed + run_idx)
        step1_main()

        val_auc, test_auc, val_ap, test_ap = step2_main(run_idx)
        results["val_auc"].append(val_auc)
        results["test_auc"].append(test_auc)
        results["val_ap"].append(val_ap)
        results["test_ap"].append(test_ap)

    summaries = {name: mean_std(values) for name, values in results.items()}
    print("\n★ Multi-run Results Summary")
    print(f"Val AUC: {summaries['val_auc'][0]:.4f} ± {summaries['val_auc'][1]:.4f}")
    print(f"Test AUC: {summaries['test_auc'][0]:.4f} ± {summaries['test_auc'][1]:.4f}")
    print(f"Val AP: {summaries['val_ap'][0]:.4f} ± {summaries['val_ap'][1]:.4f}")
    print(f"Test AP: {summaries['test_ap'][0]:.4f} ± {summaries['test_ap'][1]:.4f}")


if __name__ == "__main__":
    main()