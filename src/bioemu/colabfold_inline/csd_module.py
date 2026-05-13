"""Coevolutionary Signal Decomposition (CoDeC) module — PyTorch projector.

The CSDModule sits between the two halves of AlphaFold's Evoformer stack
(see ``_vendor/alphafold/model/modules.py``).  AF2's first half runs in
JAX on GPU; at the boundary, ``jax.pure_callback`` ships the post-block-11
MSA representation to the host where **this** module runs.  The host
side is implemented in PyTorch because:

1. JAX's ``jax.jit`` traces forward + backward + AdamW into a single
   gigantic XLA program.  For 30-epoch training of a CNN on
   ``(8, 464, 464, c_z)`` inputs this compile takes hours.
2. The openfold2 reference implementation lives in PyTorch and works
   reliably; porting it preserves the exact architecture, optimiser, loss
   and hyperparameters.
3. PyTorch's eager mode dispatches each op immediately, so there is no
   compile wall — only the actual GPU compute.

Pipeline inside ``cluster_m_act_to_indices``:

    numpy m_act      (from jax.pure_callback)
      → torch tensors on cuda
      → fixed random a/b/w_out projections      (frozen)
      → 30-epoch AdamW training of the Projection CNN
        with cosine-similarity-preserving L1 loss
      → latent vectors (N_seq × latent_dim)
      → cosine k-means in numpy
      → fixed-shape ``(seq_idx, seq_mask)`` returned to JAX

The JAX side (``EmbeddingsAndEvoformer`` in ``_vendor/alphafold``) is
untouched — it gathers ``m_mid[seq_idx]`` and vmaps the second half over
the cluster axis exactly as before.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — chosen to track openfold2's ``OuterProductClustering``.
#   * c_hidden = 16  (openfold2 uses 32 but with the pretrained OPM linear_out;
#                     we use random fixed projections so a smaller c_hidden
#                     keeps the per-batch outer-product memory bounded.)
#   * c_z      = 128 (matches openfold2's Projection.in_dim).
#   * latent_dim = 128 (openfold2's Projection.out_dim).
# ---------------------------------------------------------------------------

_DEFAULT_C_HIDDEN = 16
_DEFAULT_C_Z = 128
_DEFAULT_LATENT = 128


def _pick_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Fixed (random) sequence-wise pair fingerprints
# ---------------------------------------------------------------------------


def _make_fixed_projections(
    rng: np.random.Generator,
    c_m: int,
    c_hidden: int,
    c_z: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three fixed random projection matrices (frozen, no grad)."""
    w_a = rng.standard_normal((c_m, c_hidden)).astype(np.float32) / np.sqrt(c_m)
    w_b = rng.standard_normal((c_m, c_hidden)).astype(np.float32) / np.sqrt(c_m)
    w_out = rng.standard_normal((c_hidden * c_hidden, c_z)).astype(np.float32) / np.sqrt(
        c_hidden * c_hidden
    )
    return (
        torch.from_numpy(w_a).to(device),
        torch.from_numpy(w_b).to(device),
        torch.from_numpy(w_out).to(device),
    )


def _opm(a: torch.Tensor, b: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
    """Sequence-wise outer product followed by a linear projection.

    Mirrors openfold2's ``OuterProductClustering._opm`` but with our fixed
    random ``w_out`` in place of the pretrained ``linear_out`` weights.

    a, b: ``[B, N_res, c_hidden]``
    Returns: ``[B, N_res, N_res, c_z]``
    """
    # (B, N, N, c_hidden, c_hidden)
    outer = torch.einsum("bic,bjd->bijcd", a, b)
    # (B, N, N, c_hidden ** 2)
    outer = outer.reshape(*outer.shape[:-2], -1)
    # (B, N, N, c_z)
    outer = outer @ w_out
    return outer


# ---------------------------------------------------------------------------
# 2. PyTorch CNN projector — direct port from openfold2's Projection / SEBlock
#    (openfold/model/outer_product_mean.py :: SEBlock, Projection)
# ---------------------------------------------------------------------------


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation block, NCHW (PyTorch standard)."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        red = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, red, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(red, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class _Projection(nn.Module):
    """Pair-image → latent CNN.

    Direct port of openfold2's ``Projection``: Conv 7×7/s2 + BN + GELU + SE,
    then two Conv 3×3/s2 + BN + GELU + SE blocks, then Dropout + Flatten,
    then a Linear → LN → GELU → Dropout → Linear → LN head.
    Input is permuted from NHWC (matching our ``_opm`` output) to NCHW for
    the convs.
    """

    def __init__(
        self,
        in_dim: int,
        in_size: int,
        out_dim: int,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(in_dim // 4),
            nn.GELU(),
            _SEBlock(in_dim // 4),
            nn.Conv2d(in_dim // 4, in_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_dim // 2),
            nn.GELU(),
            _SEBlock(in_dim // 2),
            nn.Conv2d(in_dim // 2, in_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            _SEBlock(in_dim),
            nn.Dropout(dropout),
            nn.Flatten(),
        )

        # Determine the encoder's flat output size by tracing a dummy input.
        with torch.no_grad():
            dummy = torch.zeros(1, in_dim, in_size, in_size)
            flat_feature_size = self.encoder(dummy).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(flat_feature_size, 256),
            nn.LayerNorm(256, eps=1e-8, elementwise_affine=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim, eps=1e-8, elementwise_affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: NHWC (B, H, W, C) → NCHW for Conv2d
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.encoder(x)
        x = self.fc_layers(x)
        return x


# ---------------------------------------------------------------------------
# 3. PyTorch training loop — port of openfold2's _train_projector
# ---------------------------------------------------------------------------


def _pairwise_cosine_upper(x: torch.Tensor) -> torch.Tensor:
    """Upper-triangle (k=1) of the normalized pairwise cosine similarity."""
    x_norm = F.normalize(x, p=2, dim=1)
    sim = x_norm @ x_norm.T
    b = x.shape[0]
    iu, ju = torch.triu_indices(b, b, offset=1, device=x.device)
    return sim[iu, ju]


def _train_projector_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    w_out: torch.Tensor,
    *,
    n_res: int,
    c_z: int,
    latent_dim: int = _DEFAULT_LATENT,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    dropout: float = 0.15,
    val_frac: float = 0.2,
    grad_clip: float = 1.0,
    rng_seed: int = 0,
    device: torch.device | None = None,
) -> _Projection:
    """Train the CNN projector on-the-fly (PyTorch eager mode).

    Mirrors openfold2's ``OuterProductClustering._train_projector``:
        - 30 epochs, batch_size=8.
        - AdamW(lr=5e-5, weight_decay=1e-4).
        - Gradient clipping at ``max_norm=1.0``.
        - L1 loss on pairwise cosine similarities (upper triangle).
        - 80/20 train/val split, drop-last validation batches.

    Returns the trained projector (in ``eval()`` mode).
    """
    device = device or a.device
    n_seq = a.shape[0]
    np_rng = np.random.default_rng(rng_seed)

    # 80/20 train/val split.
    perm = np_rng.permutation(n_seq)
    n_val = max(batch_size, int(round(val_frac * n_seq))) if n_seq >= 2 * batch_size else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if train_idx.size < 2:
        train_idx = np.arange(n_seq)
        val_idx = np.arange(n_seq)

    model = _Projection(in_dim=c_z, in_size=n_res, out_dim=latent_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _epoch_loss(loader_indices: np.ndarray, train: bool) -> float:
        if train:
            model.train()
        else:
            model.eval()
        losses: list[float] = []
        rng_local = np_rng if train else None
        order = rng_local.permutation(loader_indices) if train else loader_indices

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for start in range(0, order.size, batch_size):
                idx = order[start : start + batch_size]
                if not train and idx.size < batch_size:
                    # drop_last=True semantics on validation
                    continue
                if idx.size < 2:
                    continue
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
                a_b = a.index_select(0, idx_t)
                b_b = b.index_select(0, idx_t)

                batch_op = _opm(a_b, b_b, w_out)  # (B, N, N, c_z)

                # Target cosine similarities computed from the (frozen) outer
                # product flattened to a per-sequence vector.  No grad path
                # to w_a/w_b/w_out.
                with torch.no_grad():
                    flat = batch_op.reshape(batch_op.shape[0], -1)
                    flat = F.normalize(flat, p=2, dim=1)
                    cos_orig = _pairwise_cosine_upper(flat)

                if train:
                    optimizer.zero_grad(set_to_none=True)

                emb = model(batch_op)
                emb_n = F.normalize(emb, p=2, dim=1)
                cos_latent = _pairwise_cosine_upper(emb_n)

                loss = F.l1_loss(cos_latent, cos_orig)

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()

                losses.append(float(loss.detach()))

                # Free intermediates aggressively — per-batch tensors are
                # several hundred MB to a couple GB at typical AF2 shapes.
                del batch_op, flat, cos_orig, emb, emb_n, cos_latent, loss
        return float(np.mean(losses)) if losses else float("nan")

    for epoch in range(epochs):
        train_loss = _epoch_loss(train_idx, train=True)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_loss = _epoch_loss(val_idx, train=False) if val_idx.size else float("nan")
            logger.info(
                "CoDeC projector epoch %3d/%d - train L1=%.4f - val L1=%.4f",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
            )

    model.eval()
    return model


# ---------------------------------------------------------------------------
# 4. Inference: trained projector → per-sequence latents
# ---------------------------------------------------------------------------


def _compute_latents_torch(
    model: _Projection,
    a: torch.Tensor,
    b: torch.Tensor,
    w_out: torch.Tensor,
    batch_size: int = 8,
) -> np.ndarray:
    """Run the trained projector over every sequence and return L2-normalized
    latents as a numpy array."""
    model.eval()
    n_seq = a.shape[0]
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            end = min(start + batch_size, n_seq)
            batch_op = _opm(a[start:end], b[start:end], w_out)
            emb = model(batch_op)
            emb = F.normalize(emb, p=2, dim=1)
            out.append(emb.detach().cpu().numpy())
            del batch_op, emb
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# 5. Agglomerative cosine clustering (sklearn)
# ---------------------------------------------------------------------------


def _agglomerative_cosine(
    latents: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Fixed-n agglomerative clustering on L2-normalized latent vectors.

    Mirrors openfold2's ``OuterProductClustering._cluster_latent_vectors``
    choice of ``AgglomerativeClustering(n_clusters=..., linkage='average',
    metric='cosine')``.  We L2-normalize the latents first so the cosine
    metric is purely directional (the projector's final ``LayerNorm`` does
    this anyway, but normalizing again is cheap and defensive).

    Returns ``[N]`` int32 cluster assignments in ``range(n_clusters)``.
    """
    from sklearn.cluster import AgglomerativeClustering

    n = latents.shape[0]
    if n_clusters >= n:
        return np.arange(n, dtype=np.int32)

    x = latents.astype(np.float32)
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", metric="cosine")
    return agg.fit_predict(x).astype(np.int32)


# ---------------------------------------------------------------------------
# Public API — called from inside ``jax.pure_callback``
# ---------------------------------------------------------------------------


def cluster_m_act_to_indices(
    m_act: np.ndarray,
    *,
    n_clusters: int,
    max_size: int,
    c_hidden: int = _DEFAULT_C_HIDDEN,
    c_z: int = _DEFAULT_C_Z,
    latent_dim: int = _DEFAULT_LATENT,
    projector_epochs: int = 30,
    projector_batch_size: int = 8,
    projector_lr: float = 5e-5,
    projector_weight_decay: float = 1e-4,
    projector_dropout: float = 0.15,
    grad_clip: float = 1.0,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster the post-block-11 MSA and return fixed-shape gather indices.

    Host-side callback invoked from inside ``jax.pure_callback`` between the
    two halves of AF2's Evoformer.  Implemented entirely in PyTorch (no
    JAX in this function) to avoid the XLA compile-wall that the previous
    flax/optax version hit when tracing the full forward+backward+AdamW
    graph on a ``(8, 464, 464, c_z)`` outer-product tensor.

    Args:
        m_act: ``[N_seq, N_res, c_m]`` MSA representation **after** the first
            ``csd_block_idx + 1`` Evoformer blocks.  c_m is typically 256.
        n_clusters: Number of CoDeC clusters (= leading batch dim downstream).
        max_size: Maximum sequences per cluster *after padding*.  Mirrors
            openfold2's ``max_members = 128``.

    Returns:
        ``(seq_idx, seq_mask)`` with shapes ``[n_clusters, max_size]``:

        - ``seq_idx``: int32 indices into the original ``N_seq`` axis,
          gathered by ``m_act[seq_idx]`` downstream.
        - ``seq_mask``: float32 ``{0, 1}`` mask zeroing out padded slots.

        The query (sequence 0) is forced into every cluster at slot 0.
    """
    m_act_np = np.asarray(m_act, dtype=np.float32)
    n_seq, n_res, c_m = m_act_np.shape

    device = _pick_device()
    rng = np.random.default_rng(rng_seed)

    logger.info(
        "CoDeC: project + cluster on post-block-11 MSA "
        "(N_seq=%d, N_res=%d, c_m=%d, n_clusters=%d, max_size=%d, device=%s).",
        n_seq,
        n_res,
        c_m,
        n_clusters,
        max_size,
        device,
    )

    # Move m_act to the projector device once.
    m_act_t = torch.from_numpy(m_act_np).to(device, dtype=torch.float32)

    # Fixed random projections (no grad).  Linear projections of m_act produce
    # a, b each of shape (N_seq, N_res, c_hidden).  These are the inputs to
    # the per-sequence outer product.
    w_a, w_b, w_out = _make_fixed_projections(rng, c_m, c_hidden, c_z, device)
    with torch.no_grad():
        a = m_act_t @ w_a
        b = m_act_t @ w_b
    # Free m_act on GPU — a, b are the only thing the projector needs.
    del m_act_t

    effective_k = min(n_clusters, max(1, n_seq - 1))

    logger.info(
        "CoDeC: training projector (epochs=%d, batch_size=%d, lr=%g, wd=%g, "
        "c_hidden=%d, c_z=%d).",
        projector_epochs,
        projector_batch_size,
        projector_lr,
        projector_weight_decay,
        c_hidden,
        c_z,
    )
    model = _train_projector_torch(
        a,
        b,
        w_out,
        n_res=n_res,
        c_z=c_z,
        latent_dim=latent_dim,
        epochs=projector_epochs,
        batch_size=projector_batch_size,
        lr=projector_lr,
        weight_decay=projector_weight_decay,
        dropout=projector_dropout,
        grad_clip=grad_clip,
        rng_seed=rng_seed,
        device=device,
    )

    logger.info("CoDeC: computing latent vectors for all %d sequences.", n_seq)
    latents = _compute_latents_torch(model, a, b, w_out, batch_size=projector_batch_size)

    # Free projector + projections — k-means is numpy-only.
    del model, a, b, w_a, w_b, w_out
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Cluster the NON-query sequences only.
    # The query (index 0) lives at slot 0 of every cluster's MSA downstream
    # — it is not a member of any cluster.  Including it in the clustering
    # data lets agglomerative clustering carve out a singleton cluster
    # around the query (we previously saw cluster sizes like
    # ``[5223, 1, 3226, 1970]`` where the size-1 cluster was just the query),
    # which wrecks the affected cluster's MSA depth and collapses its
    # second-half evoformer to a one-sequence prediction.
    if n_seq <= 1:
        # No non-query sequences — every cluster is just the query.
        cluster_idxs_nonquery = np.zeros(0, dtype=np.int32)
    elif effective_k <= 1:
        cluster_idxs_nonquery = np.zeros(n_seq - 1, dtype=np.int32)
    else:
        logger.info(
            "CoDeC: running cosine agglomerative clustering "
            "(k=%d, linkage=average, on %d non-query sequences).",
            effective_k,
            n_seq - 1,
        )
        cluster_idxs_nonquery = _agglomerative_cosine(latents[1:], effective_k)
        # Contiguous-relabel in case any cluster came out empty.
        unique = np.unique(cluster_idxs_nonquery)
        remap = {old: new for new, old in enumerate(unique)}
        cluster_idxs_nonquery = np.array([remap[c] for c in cluster_idxs_nonquery], dtype=np.int32)

    actual_k = int(cluster_idxs_nonquery.max() + 1) if cluster_idxs_nonquery.size else 1
    counts = np.bincount(cluster_idxs_nonquery, minlength=n_clusters).tolist()
    logger.info(
        "CoDeC: non-query cluster sizes (pre-cap, query is added on top of each) = %s",
        counts,
    )

    # ---- Build fixed-shape (seq_idx, seq_mask) of shape [n_clusters, max_size] ----
    # Slot 0 of every cluster is the query (index 0) — ALWAYS, with mask=1.
    # Slots 1.. hold up to ``max_size - 1`` non-query members of that cluster
    # (random subsample if larger).  Empty cluster slots are filled with
    # index 0 but masked off so they don't contribute to attention.
    seq_idx = np.zeros((n_clusters, max_size), dtype=np.int32)
    seq_mask = np.zeros((n_clusters, max_size), dtype=np.float32)
    for c in range(n_clusters):
        # Query at slot 0, always present.
        seq_idx[c, 0] = 0
        seq_mask[c, 0] = 1.0

        if c >= actual_k or cluster_idxs_nonquery.size == 0:
            # No non-query members for this cluster — query-only MSA.
            continue

        # ``cluster_idxs_nonquery`` is indexed 0..N-2 over m_act[1:]; remap
        # back to original m_act indices by adding 1.
        members = np.where(cluster_idxs_nonquery == c)[0] + 1
        if members.size > max_size - 1:
            members = rng.choice(members, max_size - 1, replace=False)
        seq_idx[c, 1 : 1 + members.size] = members.astype(np.int32)
        seq_mask[c, 1 : 1 + members.size] = 1.0

    capped = [int(seq_mask[c].sum()) for c in range(n_clusters)]
    logger.info(
        "CoDeC: per-cluster MSA depth (query + capped members, max_size=%d) = %s",
        max_size,
        capped,
    )
    return seq_idx, seq_mask


# ---------------------------------------------------------------------------
# Legacy helper retained for the no-codec / sequential-cluster fallback path.
# ---------------------------------------------------------------------------


def build_cluster_a3m(
    a3m_string: str,
    cluster_idxs: np.ndarray,
    cluster_id: int,
) -> str:
    """Subset the original A3M to sequences belonging to ``cluster_id``.

    Only used by the legacy multi-pass code path; the in-graph single-pass
    CSDModule does its gathering on the m_mid tensor and never touches the
    A3M again after AF2's data pipeline has consumed it.
    """
    from alphafold.data.parsers import parse_a3m

    msa = parse_a3m(a3m_string)
    seen: set[str] = set()
    keep_indices: list[int] = []
    unique_idx = 0
    for i, seq in enumerate(msa.sequences):
        if seq in seen:
            continue
        seen.add(seq)
        if unique_idx == 0 or (
            unique_idx < len(cluster_idxs) and int(cluster_idxs[unique_idx]) == int(cluster_id)
        ):
            keep_indices.append(i)
        unique_idx += 1

    lines: list[str] = []
    for idx in keep_indices:
        desc = (
            msa.descriptions[idx]
            if idx < len(msa.descriptions) and msa.descriptions[idx]
            else f"seq_{idx}"
        )
        lines.append(f">{desc}")
        lines.append(msa.sequences[idx])
    return "\n".join(lines) + "\n"
