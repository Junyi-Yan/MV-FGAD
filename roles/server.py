import os
import os.path as osp
import random
import time
from collections import OrderedDict

import torch

from config import args
from roles.client import build_model
from node_gad import NodeAnomalyDetection, EvaluateAnomalyClients


class ServerManager:

    def __init__(self, model_name, datasets, num_clients, device, num_rounds, client_sample_ratio):
        if not 0 < client_sample_ratio <= 1:
            raise ValueError("client_sample_ratio must be in (0, 1]")

        self.model_name = model_name
        self.datasets = datasets
        self.input_dim = datasets.input_dim
        self.output_dim = datasets.output_dim
        self.hidden_dim = args.hidden_dim
        self.num_clients = num_clients
        self.device = device
        self.client_sample_ratio = client_sample_ratio
        self.num_rounds = num_rounds
        self.init_model()

    def init_model(self):
        self.model = build_model(model_name=self.model_name,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim)

    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @staticmethod
    def model_aggregation(models, mixing_coefficients):
        if len(models) != len(mixing_coefficients):
            raise ValueError("models and mixing_coefficients must have equal length")
        if not models:
            raise ValueError("At least one client model is required for aggregation")

        state_dicts = [model.state_dict() for model in models]
        aggregated = OrderedDict()

        for key in state_dicts[0]:
            first_value = state_dicts[0][key]
            if torch.is_floating_point(first_value) or torch.is_complex(first_value):
                aggregated[key] = sum(coefficient * state_dict[key]
                    for coefficient, state_dict in zip(mixing_coefficients, state_dicts))
            else:
                aggregated[key] = first_value.clone()

        return aggregated

    def _sample_clients(self):
        client_ids = list(range(self.num_clients))
        random.shuffle(client_ids)
        sample_size = max(1, int(self.num_clients * self.client_sample_ratio))
        return sorted(client_ids[:sample_size])

    @staticmethod
    def _mixing_coefficients(clients, sampled_ids):
        node_counts = [clients[client_id].num_nodes for client_id in sampled_ids]
        total_nodes = sum(node_counts)
        if total_nodes == 0:
            raise ValueError("Sampled clients contain no nodes")
        return [count / total_nodes for count in node_counts]

    def _evaluate_global_model(self, clients):
        metrics = {"global_val_auc": 0.0,"global_test_auc": 0.0,"global_val_ap": 0.0,"global_test_ap": 0.0}
        total_nodes = sum(client.num_nodes for client in clients)

        for client in clients:
            self.model.pre_msg_learnable = client.model.pre_msg_learnable
            self.model.processed_feature = client.model.processed_feature
            self.model.adj = client.model.adj
            val_auc, test_auc, val_ap, test_ap = EvaluateAnomalyClients(dataset=client.local_subgraph,
                model=self.model,
                device=self.device).execute()

            weight = client.num_nodes / total_nodes
            metrics["global_val_auc"] += val_auc * weight
            metrics["global_test_auc"] += test_auc * weight
            metrics["global_val_ap"] += val_ap * weight
            metrics["global_test_ap"] += test_ap * weight

        return metrics

    def collaborative_training_model(self, clients, data_name, num_clients, sampling, model_name, normalize_trains=None, lr=None, weight_decay=None, epochs=None):
        normalize_trains = args.normalize_train if normalize_trains is None else normalize_trains
        lr = args.lr if lr is None else lr
        weight_decay = args.weight_decay if weight_decay is None else weight_decay
        epochs = args.num_epochs if epochs is None else epochs

        print("| ★ Start Training Federated GNN Model...")
        start_time = time.time()
        normalize_record = {"val_auc": [], "test_auc": [], "val_ap": [], "test_ap": []}
        checkpoint_path = osp.join("./model_weights",
            f"{data_name}_Client{num_clients}_{sampling}_{model_name}.pt")
        os.makedirs(osp.dirname(checkpoint_path), exist_ok=True)

        for _ in range(normalize_trains):
            self.init_model()
            for client in clients:
                client.init_model()
                client.model.preprocess(client.local_subgraph.adj, client.local_subgraph.x)

            best_metrics = {"global_val_auc": float("-inf"), "global_test_auc": 0.0,"global_val_ap": 0.0,"global_test_ap": 0.0}

            for _ in range(self.num_rounds):
                sampled_ids = self._sample_clients()
                mixing_coefficients = self._mixing_coefficients(clients, sampled_ids)
                local_models = []

                for client_id in sampled_ids:
                    client = clients[client_id]
                    client.set_state_dict(self.model)
                    _, _, _, _, local_model = NodeAnomalyDetection(dataset=client.local_subgraph,
                        model=client.model,
                        lr=lr,
                        weight_decay=weight_decay,
                        epochs=epochs,
                        device=self.device,
                    ).execute()
                    local_models.append(local_model)

                aggregated_state = self.model_aggregation(local_models, mixing_coefficients)
                self.set_state_dict(aggregated_state)
                current_metrics = self._evaluate_global_model(clients)

                if current_metrics["global_val_auc"] > best_metrics["global_val_auc"]:
                    best_metrics = current_metrics
                    torch.save(self.model, checkpoint_path)

            normalize_record["val_auc"].append(best_metrics["global_val_auc"])
            normalize_record["test_auc"].append(best_metrics["global_test_auc"])
            normalize_record["val_ap"].append(best_metrics["global_val_ap"])
            normalize_record["test_ap"].append(best_metrics["global_test_ap"])

        print(f"| ★ Federated training time: {time.time() - start_time:.4f}s")
        return normalize_record