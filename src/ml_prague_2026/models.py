import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier

from ml_prague_2026 import losses as custom_losses
from ml_prague_2026 import gnn as gnn_models


def train_chart(history):
    print()
    fig, axes = plt.subplots(1, 1, figsize=(6, 3))
    axes.plot(history['train_loss'], label='Train loss')
    axes.plot(history['val_loss'], label='Val loss', alpha=0.85)
    axes.set(xlabel='Epoch', ylabel='Loss', title='Training vs validation loss')
    axes.legend()
    plt.tight_layout()
    plt.show();


def get_anomalies_from_embeddings(embeddings, contamination, random_seed=42):
    """Get anomalies from embeddings using Isolation Forest."""
    isolation_forest = IsolationForest(contamination=contamination, random_state=random_seed)
    isolation_forest.fit(embeddings)
    return (isolation_forest.predict(embeddings) < 0).astype(int), -isolation_forest.score_samples(embeddings)

def get_anomalies_with_random_forest(embeddings, labels, test_embeddings=None, random_seed=42):
    """Get anomalies from embeddings using Random Forest Classifier."""
    
    rf = RandomForestClassifier(n_estimators=100, random_state=random_seed)
    rf.fit(embeddings, labels)
    
    if test_embeddings is None:
        test_embeddings = embeddings
    
    predictions = rf.predict(test_embeddings)
    scores = rf.predict_proba(test_embeddings)[:, 1]  # Probability of anomaly class
    
    return predictions, scores


def get_anomalies_with_gradient_boosting(embeddings, labels, test_embeddings=None, random_seed=42):
    """Get anomalies from embeddings using Gradient Boosting Classifier."""
    
    gb = GradientBoostingClassifier(n_estimators=100, random_state=random_seed)
    gb.fit(embeddings, labels)
    
    if test_embeddings is None:
        test_embeddings = embeddings
    
    predictions = gb.predict(test_embeddings)
    scores = gb.predict_proba(test_embeddings)[:, 1]  # Probability of anomaly class
    
    return predictions, scores

class SupervisedGCN:
    def __init__(self, model, lr=0.001, alpha=0.9, gamma=2.0):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = custom_losses.FocalLoss(alpha=alpha, gamma=2.0)
        self.history = {'loss': [], 'train_f1': [], 'val_f1': []}

    def fit(self, data, epochs=100, log_interval=10):
        device = data.x.device
        self.model.to(device)
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            
            out = self.model(data.x, data.edge_index)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
            
            loss.backward()
            self.optimizer.step()
            
            # Record metrics
            _, _, train_f1, val_f1 = self.evaluate(data)
            self.history['loss'].append(loss.item())
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)

            if (epoch == 1) or (epoch % log_interval == 0):
                print(f'Epoch: {epoch:03d} | Loss: {loss.item():.4f} | '
                      f'Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}')

    @torch.no_grad()
    def evaluate(self, data):
        self.model.eval()
        out = self.model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        y_true = data.y.cpu()
        y_pred = pred.cpu()
        
        t_mask, v_mask = data.train_mask.cpu(), data.val_mask.cpu()
        
        train_acc = accuracy_score(y_true[t_mask], y_pred[t_mask])
        val_acc = accuracy_score(y_true[v_mask], y_pred[v_mask])
        train_f1 = f1_score(y_true[t_mask], y_pred[t_mask])
        val_f1 = f1_score(y_true[v_mask], y_pred[v_mask])
        
        return train_acc, val_acc, train_f1, val_f1

    @torch.no_grad()
    def predict(self, data):
        self.model.eval()
        out = self.model(data.x, data.edge_index)
        return out.argmax(dim=1)

    @torch.no_grad()
    def predict_proba(self, data):
        self.model.eval()
        device = next(self.model.parameters()).device
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        
        logits = self.model(x, edge_index)
        
        probabilities = torch.softmax(logits, dim=1)
        return probabilities

    @torch.no_grad()
    def predict(self, data):
        """
        Returns the class with the highest probability.
        """
        proba = self.predict_proba(data)
        return proba.argmax(dim=1)


class UnsupervisedGCN:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = {'loss': [], 'train_acc': [], 'val_acc': []}

    def fit(self, train_data, val_data=None, epochs=100, log_interval=10):
        device = train_data.x.device
        self.model.to(device)

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            z = self.model(train_data.x, train_data.edge_index)
            z = torch.nn.functional.normalize(z, p=2, dim=1)

            src, dst = train_data.edge_label_index
            edge_scores = (z[src] * z[dst]).sum(dim=-1)

            loss = self.criterion(edge_scores, train_data.edge_label)
            loss.backward()
            self.optimizer.step()

            train_acc = self.evaluate(train_data)
            val_acc = self.evaluate(val_data) if val_data else 0.0

            self.history['loss'].append(loss.item())
            self.history['train_acc'].append(train_acc)

            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d} | Loss: {loss.item():.4f} | '
                      f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')

    @torch.no_grad()
    def evaluate(self, data):
        self.model.eval()
        z = self.model(data.x, data.edge_index)
        
        z = torch.nn.functional.normalize(z, p=2, dim=1)

        src, dst = data.edge_label_index
        scores = (z[src] * z[dst]).sum(dim=-1)
        
        probs = torch.sigmoid(scores)
        preds = (probs > 0.5).float()
        
        y_true = data.edge_label
        
        correct = (preds == y_true).sum().item()
        acc = correct / data.edge_label.size(0)
        
        return acc

    @torch.no_grad()
    def get_embeddings(self, data):
        self.model.eval()
        device = next(self.model.parameters()).device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        
        z = self.model(x, edge_index)
        return z.cpu().numpy()


class SupervisedBWGNN:
    def __init__(self, in_feats, h_feats, num_classes, edge_index, num_nodes,
                 d=2, dropout=0.0, lr=0.01, alpha=0.9, gamma=2.0, device='cpu'):
        self.device = device
        norm_adj = gnn_models.precompute_norm_adj(edge_index, num_nodes).to(device)
        self.model = gnn_models.BWGNN(in_feats, h_feats, num_classes, norm_adj, d=d, dropout=dropout).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.to(device)
        self.criterion = custom_losses.FocalLoss(alpha=alpha, gamma=gamma)
        self.history = {'loss': [], 'train_f1': [], 'val_f1': []}

    def fit(self, features, labels, train_mask, val_mask, epochs=100, log_interval=10):
        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model(features)
            loss = self.criterion(logits[train_mask], labels[train_mask])

            loss.backward()
            self.optimizer.step()

            _, _, train_f1, val_f1 = self.evaluate(features, labels, train_mask, val_mask)
            self.history['loss'].append(loss.item())
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)

            if (epoch == 1) or (epoch % log_interval == 0):
                print(f'Epoch: {epoch:03d} | Loss: {loss.item():.4f} | '
                      f'Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}')

    @torch.no_grad()
    def evaluate(self, features, labels, train_mask, val_mask):
        self.model.eval()
        logits = self.model(features)
        pred = logits.argmax(dim=1).cpu()
        y_true = labels.cpu()
        t_mask = train_mask.cpu()
        v_mask = val_mask.cpu()

        train_acc = accuracy_score(y_true[t_mask], pred[t_mask])
        val_acc = accuracy_score(y_true[v_mask], pred[v_mask])
        train_f1 = f1_score(y_true[t_mask], pred[t_mask])
        val_f1 = f1_score(y_true[v_mask], pred[v_mask])

        return train_acc, val_acc, train_f1, val_f1

    @torch.no_grad()
    def predict_proba(self, features):
        self.model.eval()
        logits = self.model(features)
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(self, features):
        proba = self.predict_proba(features)
        return proba.argmax(dim=1)