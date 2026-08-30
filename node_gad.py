import torch
from torch.optim import Adam

from utils import BaseTask, auc_ap, evaluate, train


class NodeAnomalyDetection(BaseTask):
    def __init__(self, dataset, model, lr, weight_decay, epochs, device, show_epoch_info=20):
        super().__init__()
        self.dataset = dataset
        self.labels = dataset.y
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.show_epoch_info = show_epoch_info
        self.device = device

    def execute(self):
        self.model = self.model.to(self.device)
        self.labels = self.labels.to(self.device)

        best_auc_val = 0.0
        best_auc_test = 0.0
        best_ap_val = 0.0
        best_ap_test = 0.0

        for _ in range(self.epochs):
            train(self.model, self.dataset.train_idx, self.labels, self.device, self.optimizer)

            auc_val, auc_test, ap_val, ap_test = evaluate(
                self.model,
                self.dataset.train_idx,
                self.dataset.val_idx,
                self.dataset.test_idx,
                self.labels,
                self.device,
            )

            if auc_val > best_auc_val:
                best_auc_val = auc_val
                best_auc_test = auc_test
                best_ap_val = ap_val
                best_ap_test = ap_test

        auc_val, auc_test, ap_val, ap_test = self.postprocess()

        if auc_val > best_auc_val:
            best_auc_val = auc_val
            best_auc_test = auc_test
            best_ap_val = ap_val
            best_ap_test = ap_test

        return best_auc_val, best_auc_test, best_ap_val, best_ap_test, self.model

    def postprocess(self):
        """Evaluate the final local model on validation and test nodes."""
        self.model.eval()

        with torch.no_grad():
            _, anomaly_score = self.model.model_forward(
                idx=torch.arange(self.dataset.num_nodes, device=self.device),
                device=self.device,
            )

            val_mask = self.dataset.val_idx
            labels_val = self.labels[val_mask].cpu().numpy()
            scores_val = anomaly_score[val_mask].cpu().numpy()
            auc_val, ap_val = auc_ap(labels_val, scores_val)

            test_mask = self.dataset.test_idx
            labels_test = self.labels[test_mask].cpu().numpy()
            scores_test = anomaly_score[test_mask].cpu().numpy()
            auc_test, ap_test = auc_ap(labels_test, scores_test)

        return auc_val, auc_test, ap_val, ap_test


class EvaluateAnomalyClients(BaseTask):
    def __init__(self, dataset, model, device):
        super().__init__()
        self.dataset = dataset
        self.labels = dataset.y
        self.model = model
        self.device = device

    def execute(self):
        """Evaluate the supplied model on a client's validation and test nodes."""
        self.model = self.model.to(self.device)
        self.labels = self.labels.to(self.device)
        self.model.eval()

        with torch.no_grad():
            _, anomaly_score = self.model.model_forward(
                idx=torch.arange(self.dataset.num_nodes, device=self.device),
                device=self.device,
            )

            val_mask = self.dataset.val_idx
            labels_val = self.labels[val_mask].cpu().numpy()
            scores_val = anomaly_score[val_mask].cpu().numpy()
            auc_val, ap_val = auc_ap(labels_val, scores_val)

            test_mask = self.dataset.test_idx
            labels_test = self.labels[test_mask].cpu().numpy()
            scores_test = anomaly_score[test_mask].cpu().numpy()
            auc_test, ap_test = auc_ap(labels_test, scores_test)

        return auc_val, auc_test, ap_val, ap_test