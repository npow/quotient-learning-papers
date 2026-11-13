"""Neural Quotient Encoder: GRU-based contrastive history embedding.

Architecture: GRU(input=d_O+d_A, hidden=128, layers=2) →
              Linear(128→64) → L2-normalise

Training:     InfoNCE contrastive loss on MMD-labeled pairs
              (same-class → attract, different-class → repel)

Cluster extraction: k-means + silhouette auto-k on embeddings → CanonicalContinuousQuotient
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from continuous_quotient.types import (
    CanonicalContinuousQuotient,
    ContinuousEstimatorConfig,
    ContinuousQuotientClass,
    ContinuousTrajectory,
    ContinuousTrajectoryDataset,
    HistoryKey,
    make_history_key,
)


class EncoderNotFittedError(Exception):
    pass


@dataclass
class NeuralEncoderConfig:
    hidden_size: int = 128
    n_layers: int = 2
    embedding_dim: int = 64
    n_epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    temperature: float = 0.1
    k_max: int = 16
    pretrain_env: str | None = None
    seed: int = 0


@dataclass
class TrainingMeta:
    n_epochs: int = 0
    loss_history: list[float] = field(default_factory=list)
    val_rand_index: float = 0.0
    pretrain_env: str | None = None


if HAS_TORCH:
    class _GRUEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden: int, n_layers: int, emb_dim: int):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden, n_layers, batch_first=True)
            self.proj = nn.Linear(hidden, emb_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, seq_len, input_dim)
            _, h = self.gru(x)
            h_last = h[-1]          # (batch, hidden)
            z = self.proj(h_last)   # (batch, emb_dim)
            return F.normalize(z, dim=-1)

    class _PairDataset(Dataset):
        """Dataset of (anchor, positive/negative, label) triples."""

        def __init__(
            self,
            history_seqs: dict[HistoryKey, np.ndarray],  # key → (L, d_in)
            labels: dict[HistoryKey, int],
        ):
            keys = list(history_seqs.keys())
            self.pairs: list[tuple] = []
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    ki, kj = keys[i], keys[j]
                    same = int(labels.get(ki, -1) == labels.get(kj, -1))
                    self.pairs.append((
                        torch.tensor(history_seqs[ki], dtype=torch.float32),
                        torch.tensor(history_seqs[kj], dtype=torch.float32),
                        same,
                    ))

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int):
            return self.pairs[idx]


def _pad_collate(batch):
    """Pad variable-length sequences in a batch."""
    import torch
    ancs, poss, labels = zip(*batch)
    max_len = max(max(a.shape[0] for a in ancs), max(p.shape[0] for p in poss))
    d = ancs[0].shape[1]

    def _pad(seq, L):
        if seq.shape[0] < L:
            pad = torch.zeros(L - seq.shape[0], d)
            return torch.cat([seq, pad], dim=0)
        return seq

    ancs_p = torch.stack([_pad(a, max_len) for a in ancs])
    poss_p = torch.stack([_pad(p, max_len) for p in poss])
    labels_t = torch.tensor(labels, dtype=torch.float32)
    return ancs_p, poss_p, labels_t


def _infonce_loss(z_a: "torch.Tensor", z_b: "torch.Tensor", same: "torch.Tensor", temp: float) -> "torch.Tensor":
    """Supervised contrastive loss: same-class pairs attracted, different repelled."""
    import torch
    sim = (z_a * z_b).sum(dim=-1) / temp        # cosine sim / temperature
    # For same-class: loss = -log(sigmoid(sim)); for diff: -log(sigmoid(-sim))
    loss_same = F.softplus(-sim)                 # -log σ(sim)
    loss_diff = F.softplus(sim)                  # -log σ(-sim)
    loss = torch.where(same.bool(), loss_same, loss_diff)
    return loss.mean()


class NeuralQuotientEstimator:
    """GRU encoder + InfoNCE training for continuous quotient learning.

    Usage:
        estimator = NeuralQuotientEstimator(config)
        quotient = estimator.fit(trajectories, mmd_labels=mmd_quotient)
        embedding = estimator.encode(single_trajectory)
    """

    def __init__(self, config: NeuralEncoderConfig | None = None):
        if not HAS_TORCH:
            raise ImportError("torch is required for NeuralQuotientEstimator")
        self.config = config or NeuralEncoderConfig()
        self._encoder: Any = None
        self._input_dim: int = 0
        self.training_meta = TrainingMeta()
        self._torch_rng = None

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def fit(
        self,
        trajectories: list[ContinuousTrajectory] | ContinuousTrajectoryDataset,
        mmd_labels: CanonicalContinuousQuotient | None = None,
        history_max_length: int = 1,
        verbose: bool = False,
    ) -> CanonicalContinuousQuotient:
        """Train the encoder and extract quotient partition via k-means + silhouette.

        If mmd_labels is provided (a pre-computed ContinuousQuotientEstimator
        result), training uses those labels as supervision signal.
        Otherwise, same-trajectory-class labels are inferred from visitation.
        """
        import torch

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        if isinstance(trajectories, list):
            dataset = ContinuousTrajectoryDataset(trajectories=trajectories)
        else:
            dataset = trajectories

        # Build per-history sequence tensors.
        history_seqs, all_keys = self._build_history_seqs(dataset, history_max_length)
        if not history_seqs:
            return CanonicalContinuousQuotient(n_classes=0, partition={})

        # Use MMD partition labels if provided, else singletons.
        labels: dict[HistoryKey, int] = {}
        if mmd_labels is not None:
            for k in all_keys:
                labels[k] = mmd_labels.partition.get(k, -1)
        else:
            for i, k in enumerate(all_keys):
                labels[k] = i

        # Build model.
        sample_seq = next(iter(history_seqs.values()))
        self._input_dim = sample_seq.shape[1]
        cfg = self.config
        self._encoder = _GRUEncoder(
            self._input_dim, cfg.hidden_size, cfg.n_layers, cfg.embedding_dim
        )
        optimizer = torch.optim.Adam(self._encoder.parameters(), lr=cfg.lr)

        # Train.
        pair_ds = _PairDataset(history_seqs, labels)
        if len(pair_ds) == 0:
            return CanonicalContinuousQuotient(n_classes=1, partition={k: 0 for k in all_keys})

        loader = DataLoader(
            pair_ds, batch_size=min(cfg.batch_size, len(pair_ds)),
            shuffle=True, collate_fn=_pad_collate
        )
        self._encoder.train()
        for epoch in range(cfg.n_epochs):
            epoch_loss = 0.0
            for anc, pos, same in loader:
                z_a = self._encoder(anc)
                z_b = self._encoder(pos)
                loss = _infonce_loss(z_a, z_b, same, cfg.temperature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.training_meta.loss_history.append(epoch_loss / max(1, len(loader)))
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1}/{cfg.n_epochs}  loss={epoch_loss:.4f}")

        self.training_meta.n_epochs = cfg.n_epochs

        # Extract embeddings and cluster.
        embeddings = self._embed_all(history_seqs, all_keys)
        partition = self._cluster(embeddings, all_keys, labels)
        return partition

    def encode(self, trajectory: ContinuousTrajectory) -> np.ndarray:
        """Embed a single trajectory's full observation sequence."""
        if self._encoder is None:
            raise EncoderNotFittedError("Call .fit() before .encode()")
        import torch
        seq = self._traj_to_seq(trajectory)
        t = torch.tensor(seq[None], dtype=torch.float32)
        with torch.no_grad():
            return self._encoder(t).squeeze(0).numpy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _traj_to_seq(self, traj: ContinuousTrajectory) -> np.ndarray:
        """Convert trajectory to (T, d_O+d_A) array."""
        rows = []
        for i, o in enumerate(traj.observations):
            o_arr = np.asarray(o, dtype=np.float32)
            if i < len(traj.actions):
                a = traj.actions[i]
                if isinstance(a, (int, np.integer)):
                    a_arr = np.array([float(a)], dtype=np.float32)
                else:
                    a_arr = np.asarray(a, dtype=np.float32).ravel()
                rows.append(np.concatenate([o_arr, a_arr]))
            else:
                if i == 0 and len(traj.actions) == 0:
                    rows.append(o_arr)
                else:
                    rows.append(np.concatenate([o_arr, np.zeros(1, dtype=np.float32)]))
        return np.stack(rows, axis=0) if rows else np.zeros((1, 1), dtype=np.float32)

    def _build_history_seqs(
        self,
        dataset: ContinuousTrajectoryDataset,
        history_max_length: int,
    ) -> tuple[dict[HistoryKey, np.ndarray], list[HistoryKey]]:
        seqs: dict[HistoryKey, list[np.ndarray]] = {}
        # Infer action dimension from first trajectory with actions.
        _a_dim: int | None = None
        for traj in dataset.trajectories:
            if traj.actions:
                a0 = traj.actions[0]
                if isinstance(a0, (int, np.integer)):
                    _a_dim = 1
                else:
                    _a_dim = int(np.asarray(a0, dtype=np.float32).ravel().shape[0])
                break
        a_dim = _a_dim if _a_dim is not None else 1

        for traj in dataset.trajectories:
            obs, acts = traj.observations, traj.actions
            T = traj.length
            for start in range(1, T):
                hlen = min(history_max_length, start)
                h_obs = obs[start - hlen: start + 1]
                h_acts = acts[start - hlen: start]
                hkey = make_history_key(list(h_obs), list(h_acts))
                if hkey not in seqs:
                    # Build (L, d_O+d_A) array for this history.
                    # All rows must have identical width; pad missing action
                    # with zeros so the last observation step is uniform.
                    rows = []
                    for i, o in enumerate(h_obs):
                        o_arr = np.asarray(o, dtype=np.float32)
                        if i < len(h_acts):
                            a = h_acts[i]
                            if isinstance(a, (int, np.integer)):
                                a_arr = np.array([float(a)], dtype=np.float32)
                            else:
                                a_arr = np.asarray(a, dtype=np.float32).ravel()
                        else:
                            # Last observation in history has no following action;
                            # pad with zeros to keep uniform row width.
                            a_arr = np.zeros(a_dim, dtype=np.float32)
                        rows.append(np.concatenate([o_arr, a_arr]))
                    seqs[hkey] = np.stack(rows, axis=0) if rows else np.zeros((1, 1), np.float32)
        all_keys = list(seqs.keys())
        return seqs, all_keys

    def _embed_all(
        self,
        seqs: dict[HistoryKey, np.ndarray],
        keys: list[HistoryKey],
    ) -> np.ndarray:
        import torch
        self._encoder.eval()
        embeddings = []
        max_len = max(s.shape[0] for s in seqs.values())
        d = next(iter(seqs.values())).shape[1]
        with torch.no_grad():
            for k in keys:
                s = seqs[k]
                if s.shape[0] < max_len:
                    pad = np.zeros((max_len - s.shape[0], d), dtype=np.float32)
                    s = np.vstack([s, pad])
                t = torch.tensor(s[None], dtype=torch.float32)
                z = self._encoder(t).squeeze(0).numpy()
                embeddings.append(z)
        return np.stack(embeddings, axis=0)  # (N_hist, emb_dim)

    def _cluster(
        self,
        embeddings: np.ndarray,
        keys: list[HistoryKey],
        labels: dict[HistoryKey, int],
    ) -> CanonicalContinuousQuotient:
        """Auto-select k via silhouette score, then k-means cluster."""
        n = len(embeddings)
        if n < 2:
            partition = {k: 0 for k in keys}
            return CanonicalContinuousQuotient(
                n_classes=1, partition=partition,
                classes={0: ContinuousQuotientClass(0, keys)},
            )

        # Try k=2..min(n-1, k_max, n//2+1), pick k with best silhouette score.
        k_max = min(n - 1, self.config.k_max, n // 2 + 1)
        best_k, best_score, best_labels = 2, -1.0, None

        for k in range(2, k_max + 1):
            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=self.config.seed)
                cluster_ids = km.fit_predict(embeddings)
                if len(set(cluster_ids)) < 2:
                    continue
                score = silhouette_score(embeddings, cluster_ids)
                if score > best_score:
                    best_score, best_k, best_labels = score, k, cluster_ids
            except Exception:
                continue

        if best_labels is None:
            best_labels = np.zeros(n, dtype=int)
            best_k = 1

        # Rand index vs. MMD labels if available.
        true_labels = [labels.get(k, -1) for k in keys]
        valid = [(t, p) for t, p in zip(true_labels, best_labels) if t >= 0]
        if valid:
            t_arr, p_arr = zip(*valid)
            self.training_meta.val_rand_index = float(adjusted_rand_score(t_arr, p_arr))

        partition: dict[HistoryKey, int] = {k: int(best_labels[i]) for i, k in enumerate(keys)}
        n_classes = best_k
        classes = {
            cid: ContinuousQuotientClass(class_id=cid, member_keys=[
                keys[i] for i, x in enumerate(best_labels) if x == cid
            ])
            for cid in range(n_classes)
        }
        return CanonicalContinuousQuotient(
            n_classes=n_classes, partition=partition, classes=classes,
        )
