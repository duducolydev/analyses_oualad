# -*- coding: utf-8 -*-
"""
Modèle récurrent léger (T5), compatible scikit-learn (fit / predict_proba).

Entrées : un DataFrame dont les colonnes seq_cWW (clics) et seq_jWW (jours
actifs) forment une séquence hebdomadaire à 2 canaux ; toutes les autres
colonnes sont des variables statiques (agrégées, tendances, démographie).

Architecture : GRU (2 -> h) sur la séquence ; MLP (statiques -> h) ;
concaténation, dropout, sortie logistique. Perte pondérée (pos_weight =
n_négatifs / n_positifs), équivalente à class_weight="balanced" : le seuil 0,5
a le même sens que pour la forêt. Arrêt précoce sur 10 % des données
d'entraînement (tirage stratifié, graine fixée).
"""
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class _Reseau(torch.nn.Module):
    def __init__(self, d_statique, h, dropout):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=2, hidden_size=h, batch_first=True)
        self.statique = torch.nn.Sequential(torch.nn.Linear(d_statique, h), torch.nn.ReLU())
        self.sortie = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(2 * h, 1))

    def forward(self, seq, stat):
        _, h = self.gru(seq)
        return self.sortie(torch.cat([h[-1], self.statique(stat)], dim=1)).squeeze(1)


class GRUClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=32, epochs=30, batch=256, lr=2e-3, dropout=0.2, patience=4, seed=0, threads=4):
        self.hidden, self.epochs, self.batch, self.lr = hidden, epochs, batch, lr
        self.dropout, self.patience, self.seed, self.threads = dropout, patience, seed, threads

    # -- préparation ----------------------------------------------------------
    def _colonnes(self, X):
        sc = sorted(c for c in X.columns if c.startswith("seq_c"))
        sj = sorted(c for c in X.columns if c.startswith("seq_j"))
        return sc, sj, [c for c in X.columns if c not in sc + sj]

    def _seq(self, X):
        s = np.stack([np.log1p(X[self.sc_].to_numpy(float)), np.log1p(X[self.sj_].to_numpy(float))], axis=-1)
        return torch.tensor((s - self.mu_) / self.sd_, dtype=torch.float32)

    def _stat(self, X):
        return torch.tensor(np.asarray(self.pre_.transform(X[self.st_]), dtype=np.float32))

    def fit(self, X: pd.DataFrame, y):
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        y = np.asarray(y).astype(np.float32)
        self.classes_ = np.array([0, 1])
        self.sc_, self.sj_, self.st_ = self._colonnes(X)
        cat = [c for c in self.st_ if not pd.api.types.is_numeric_dtype(X[c])]
        num = [c for c in self.st_ if c not in cat]
        self.pre_ = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
        itr, iva = train_test_split(np.arange(len(X)), test_size=0.1, stratify=y, random_state=self.seed)
        Xtr = X.iloc[itr]
        self.pre_.fit(Xtr[self.st_])
        brut = np.stack([np.log1p(Xtr[self.sc_].to_numpy(float)), np.log1p(Xtr[self.sj_].to_numpy(float))], axis=-1)
        self.mu_, self.sd_ = brut.mean(axis=(0, 1)), brut.std(axis=(0, 1)) + 1e-6

        seq, stat, cible = self._seq(X), self._stat(X), torch.tensor(y)
        self.reseau_ = _Reseau(stat.shape[1], self.hidden, self.dropout)
        poids = torch.tensor((1 - y[itr]).sum() / max(y[itr].sum(), 1.0))
        perte = torch.nn.BCEWithLogitsLoss(pos_weight=poids)
        opt = torch.optim.Adam(self.reseau_.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(self.seed)
        itr_t, iva_t = torch.tensor(itr), torch.tensor(iva)
        meilleur, etat, attente = np.inf, None, 0
        for _ in range(self.epochs):
            self.reseau_.train()
            for lot in itr_t[torch.randperm(len(itr_t), generator=gen)].split(self.batch):
                opt.zero_grad()
                perte(self.reseau_(seq[lot], stat[lot]), cible[lot]).backward()
                opt.step()
            self.reseau_.eval()
            with torch.no_grad():
                v = perte(self.reseau_(seq[iva_t], stat[iva_t]), cible[iva_t]).item()
            if v < meilleur - 1e-4:
                meilleur, attente = v, 0
                etat = {k: t.clone() for k, t in self.reseau_.state_dict().items()}
            else:
                attente += 1
                if attente >= self.patience:
                    break
        self.reseau_.load_state_dict(etat)
        return self

    def predict_proba(self, X: pd.DataFrame):
        self.reseau_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.reseau_(self._seq(X), self._stat(X))).numpy()
        return np.column_stack([1 - p, p])
