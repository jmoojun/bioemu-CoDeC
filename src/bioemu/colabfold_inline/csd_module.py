"""Coevolutionary Signal Decomposition (CoDeC) module — JAX/Flax implementation.

Faithful reimplementation of the openfold2 ``OuterProductClustering`` recipe
(Jun et al., 2026) in JAX/Flax:

1. Per-sequence pair fingerprints
   ----------------------------------
   For every MSA sequence ``s``, build a 2D pair "image"
   ``z_s ∈ R^(N_res × N_res × c_z)`` from a sequence-wise outer product
   ``a_s ⊗ b_s`` followed by a (fixed) linear projection — exactly as in
   ``OuterProductClustering._opm`` upstream of the projector training.

2. CNN projector trained on the fly
   ---------------------------------
   A small ConvNet with Squeeze-and-Excitation blocks
   (``Projection`` in ``openfold/model/outer_product_mean.py``):

       Conv 7x7/s2 → Norm → GELU → SE →
       Conv 3x3/s2 → Norm → GELU → SE →
       Conv 3x3/s2 → Norm → GELU → SE →
       Dropout → Flatten →
       Linear 256 → LN → GELU → Dropout →
       Linear out_dim → LN

   The projector is trained per inference call for **80 epochs** with
   ``AdamW(lr=5e-5, wd=1e-4)``, ``batch_size=8``, gradient clipping at
   ``max_norm=1.0``, and the cosine-similarity-preserving L1 loss

       L = mean_{i<j} | cos(MLP(z_i), MLP(z_j)) - cos(flat(z_i), flat(z_j)) |

   exactly as in openfold2's ``_train_projector``.  BatchNorm is replaced
   with LayerNorm so the recipe works at arbitrary batch size without
   running-stat state.

3. K-means cosine clustering
   --------------------------
   The final latent vectors are clustered with cosine k-means.

The resulting cluster ids feed ``build_cluster_a3m``, which subsets the
A3M so AlphaFold can be run once per cluster, yielding a multistate
``(single, pair)`` ensemble that BioEmu samples sequentially.
"""

from __future__ import annotations

import logging
from typing import Any

import flax.linen as fnn
import jax
import jax.numpy as jnp
import numpy as np
import optax

logger = logging.getLogger(__name__)

# Number of HHBLITS amino-acid types used by AlphaFold2 (includes gap + X + B + Z).
_AA_VOCAB = 22

# Default channel widths (chosen to mirror openfold2's c_hidden_opm=32 / c_z=128
# but trimmed so per-sequence pair tensors fit comfortably in GPU memory for
# typical proteins).  c_z=64 keeps the CNN expressive while halving memory vs
# the openfold2 default.
_DEFAULT_C_HIDDEN = 16
_DEFAULT_C_Z = 64
_DEFAULT_LATENT = 128


# ---------------------------------------------------------------------------
# 1. Per-sequence pair fingerprints
# ---------------------------------------------------------------------------


def _make_fixed_projections(
    rng: np.random.Generator, c_hidden: int, c_z: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allocate the three fixed projection matrices used to build ``z_s``."""
    w_a = rng.standard_normal((_AA_VOCAB, c_hidden)) / np.sqrt(_AA_VOCAB)
    w_b = rng.standard_normal((_AA_VOCAB, c_hidden)) / np.sqrt(_AA_VOCAB)
    w_out = rng.standard_normal((c_hidden * c_hidden, c_z)) / np.sqrt(c_hidden * c_hidden)
    return (
        jnp.asarray(w_a, dtype=jnp.float32),
        jnp.asarray(w_b, dtype=jnp.float32),
        jnp.asarray(w_out, dtype=jnp.float32),
    )


def _compute_ab(
    msa_int: np.ndarray, w_a: jnp.ndarray, w_b: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the sequence-wise linear projections a_s, b_s.

    Returns two ``[N_seq, N_res, c_hidden]`` arrays.
    """
    onehot = jax.nn.one_hot(jnp.asarray(msa_int), _AA_VOCAB, dtype=jnp.float32)
    return onehot @ w_a, onehot @ w_b


@jax.jit
def _batch_outer(a_batch: jnp.ndarray, b_batch: jnp.ndarray, w_out: jnp.ndarray) -> jnp.ndarray:
    """On-the-fly per-sequence outer product, batched.

    a_batch, b_batch: ``[B, N_res, c_hidden]``
    Returns:           ``[B, N_res, N_res, c_z]``

    Mirrors ``OuterProductMean._opm`` followed by ``linear_out``.
    """
    outer = jnp.einsum("bic,bjd->bijcd", a_batch, b_batch)
    outer = outer.reshape(*outer.shape[:-2], -1)
    return outer @ w_out


# ---------------------------------------------------------------------------
# 2. CNN projector (Flax) matching openfold2's Projection class
# ---------------------------------------------------------------------------


class _SEBlock(fnn.Module):
    """Squeeze-and-Excitation block (channels-last)."""

    features: int
    reduction: int = 16

    @fnn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, H, W, C]
        squeezed = jnp.mean(x, axis=(1, 2))  # [B, C]
        red = max(1, self.features // self.reduction)
        squeezed = fnn.Dense(red, use_bias=False)(squeezed)
        squeezed = fnn.relu(squeezed)
        squeezed = fnn.Dense(self.features, use_bias=False)(squeezed)
        squeezed = fnn.sigmoid(squeezed)
        return x * squeezed[:, None, None, :]


class _ProjectionCNN(fnn.Module):
    """Pair-image → latent vector CNN.

    Mirrors ``openfold/model/outer_product_mean.py::Projection`` but uses
    ``LayerNorm`` (channel-wise) in place of ``BatchNorm2d`` so the recipe
    is batch-size invariant and stateless.
    """

    in_dim: int
    out_dim: int = _DEFAULT_LATENT
    dropout: float = 0.15

    @fnn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # x: [B, H, W, C=in_dim] (NHWC).
        c1 = max(1, self.in_dim // 4)
        c2 = max(1, self.in_dim // 2)
        c3 = self.in_dim

        # Block 1: Conv 7x7 / stride 2
        x = fnn.Conv(c1, (7, 7), strides=(2, 2), padding="SAME")(x)
        x = fnn.LayerNorm(epsilon=1e-5)(x)
        x = fnn.gelu(x)
        x = _SEBlock(features=c1)(x)

        # Block 2: Conv 3x3 / stride 2
        x = fnn.Conv(c2, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = fnn.LayerNorm(epsilon=1e-5)(x)
        x = fnn.gelu(x)
        x = _SEBlock(features=c2)(x)

        # Block 3: Conv 3x3 / stride 2
        x = fnn.Conv(c3, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = fnn.LayerNorm(epsilon=1e-5)(x)
        x = fnn.gelu(x)
        x = _SEBlock(features=c3)(x)

        x = fnn.Dropout(rate=self.dropout, deterministic=not train)(x)
        x = x.reshape((x.shape[0], -1))

        # FC head — LayerNorm with use_scale/use_bias=False to match openfold2
        # (``elementwise_affine=False``).
        x = fnn.Dense(256)(x)
        x = fnn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-8)(x)
        x = fnn.gelu(x)
        x = fnn.Dropout(rate=self.dropout, deterministic=not train)(x)
        x = fnn.Dense(self.out_dim)(x)
        x = fnn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-8)(x)
        return x


# ---------------------------------------------------------------------------
# Training: pairwise cosine L1 loss, AdamW 80 epochs, grad clip 1.0
# ---------------------------------------------------------------------------


def _pairwise_cosine_upper(x: jnp.ndarray) -> jnp.ndarray:
    """Upper-triangle (k=1) of the pairwise cosine similarity matrix.

    Matches openfold2's ``compute_pairwise_cosine_similarities``.
    """
    x_norm = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = x_norm @ x_norm.T  # [B, B]
    b = x.shape[0]
    iu, ju = jnp.triu_indices(b, k=1)
    return sim[iu, ju]


def _train_projector_cnn(
    a: jnp.ndarray,
    b: jnp.ndarray,
    w_out: jnp.ndarray,
    *,
    n_res: int,
    c_z: int,
    latent_dim: int = _DEFAULT_LATENT,
    epochs: int = 80,
    batch_size: int = 8,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    dropout: float = 0.15,
    val_frac: float = 0.2,
    grad_clip: float = 1.0,
    rng_seed: int = 0,
) -> tuple[Any, _ProjectionCNN]:
    """Train the CNN projector on-the-fly, returning ``(params, model)``.

    Mirrors openfold2's ``OuterProductClustering._train_projector``:
    AdamW, 80 epochs, batch_size=8, L1 loss on pairwise cosine
    similarities (upper triangle), gradient clipping at ``max_norm=1.0``,
    80/20 train/val split.
    """
    n_seq = a.shape[0]
    np_rng = np.random.default_rng(rng_seed)

    # 80/20 train/val split.
    perm = np_rng.permutation(n_seq)
    n_val = max(batch_size, int(round(val_frac * n_seq))) if n_seq >= 2 * batch_size else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if train_idx.size < 2:
        # Pathologically tiny MSA — fall back to using all sequences for
        # both train and val so the projector still gets initialized.
        train_idx = np.arange(n_seq)
        val_idx = np.arange(n_seq)

    model = _ProjectionCNN(in_dim=c_z, out_dim=latent_dim, dropout=dropout)
    key = jax.random.PRNGKey(rng_seed)
    key, init_key, dropout_key = jax.random.split(key, 3)

    # Initialize using a dummy batch of size 1.
    dummy = jnp.zeros((1, n_res, n_res, c_z), dtype=jnp.float32)
    params = model.init({"params": init_key, "dropout": dropout_key}, dummy, train=True)

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(lr, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(params)

    def loss_fn(params, batch_op, dropout_key):
        # batch_op: [B, H, W, C]
        flat = batch_op.reshape(batch_op.shape[0], -1)
        cos_orig = _pairwise_cosine_upper(flat)

        latent = model.apply(params, batch_op, train=True, rngs={"dropout": dropout_key})
        cos_latent = _pairwise_cosine_upper(latent)
        # L1 loss in cosine-similarity space.
        return jnp.mean(jnp.abs(cos_latent - cos_orig))

    @jax.jit
    def train_step(params, opt_state, batch_a, batch_b, dropout_key):
        batch_op = _batch_outer(batch_a, batch_b, w_out)
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_op, dropout_key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params, batch_a, batch_b):
        batch_op = _batch_outer(batch_a, batch_b, w_out)
        flat = batch_op.reshape(batch_op.shape[0], -1)
        cos_orig = _pairwise_cosine_upper(flat)
        latent = model.apply(params, batch_op, train=False)
        cos_latent = _pairwise_cosine_upper(latent)
        return jnp.mean(jnp.abs(cos_latent - cos_orig))

    last_train = float("nan")
    last_val = float("nan")
    for epoch in range(epochs):
        epoch_perm = np_rng.permutation(train_idx)
        train_losses: list[float] = []
        for start in range(0, epoch_perm.size, batch_size):
            idx = epoch_perm[start : start + batch_size]
            if idx.size < 2:
                continue
            idx_j = jnp.asarray(idx)
            key, dk = jax.random.split(key)
            params, opt_state, loss = train_step(params, opt_state, a[idx_j], b[idx_j], dk)
            train_losses.append(float(loss))

        # Validation pass (drop_last=True semantics: only full batches).
        val_losses: list[float] = []
        for start in range(0, val_idx.size, batch_size):
            idx = val_idx[start : start + batch_size]
            if idx.size < batch_size:
                continue
            idx_j = jnp.asarray(idx)
            val_losses.append(float(eval_step(params, a[idx_j], b[idx_j])))

        last_train = float(np.mean(train_losses)) if train_losses else float("nan")
        last_val = float(np.mean(val_losses)) if val_losses else float("nan")
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            logger.info(
                "CoDeC projector epoch %3d/%d - train L1=%.4f - val L1=%.4f",
                epoch + 1,
                epochs,
                last_train,
                last_val,
            )

    return params, model


# ---------------------------------------------------------------------------
# Inference: latent vectors for every sequence
# ---------------------------------------------------------------------------


def _compute_latents(
    params: Any,
    model: _ProjectionCNN,
    a: jnp.ndarray,
    b: jnp.ndarray,
    w_out: jnp.ndarray,
    batch_size: int = 8,
) -> np.ndarray:
    """Run the trained projector over every sequence in (a, b).

    Returns ``[N_seq, latent_dim]`` L2-normalized numpy array.
    """
    n_seq = a.shape[0]

    @jax.jit
    def predict(batch_a, batch_b):
        batch_op = _batch_outer(batch_a, batch_b, w_out)
        latent = model.apply(params, batch_op, train=False)
        latent = latent / (jnp.linalg.norm(latent, axis=-1, keepdims=True) + 1e-8)
        return latent

    out: list[np.ndarray] = []
    for start in range(0, n_seq, batch_size):
        end = min(start + batch_size, n_seq)
        out.append(np.asarray(predict(a[start:end], b[start:end])))
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# 3. K-means clustering in cosine space (pure JAX)
# ---------------------------------------------------------------------------


def _kmeans_cosine(
    latents: np.ndarray, n_clusters: int, rng_seed: int = 0, n_iter: int = 30
) -> np.ndarray:
    """Cosine-similarity k-means.  Returns integer cluster assignments [N]."""
    n = latents.shape[0]
    if n_clusters >= n:
        return np.arange(n, dtype=np.int32)

    rng = np.random.default_rng(rng_seed)
    x = jnp.asarray(latents)
    x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    # k-means++-lite initialization (in cosine space).
    idxs = [int(rng.integers(0, n))]
    for _ in range(n_clusters - 1):
        centroids = x[jnp.asarray(idxs)]
        sims = x @ centroids.T
        max_sim = jnp.max(sims, axis=1)
        far = int(jnp.argmin(max_sim))
        idxs.append(far)
    centroids = x[jnp.asarray(idxs)]

    @jax.jit
    def step(centroids):
        sims = x @ centroids.T
        assign = jnp.argmax(sims, axis=1)
        new_centroids = []
        for k in range(n_clusters):
            mask = (assign == k).astype(jnp.float32)
            count = jnp.sum(mask) + 1e-8
            c = jnp.sum(x * mask[:, None], axis=0) / count
            c = c / (jnp.linalg.norm(c) + 1e-8)
            new_centroids.append(c)
        return jnp.stack(new_centroids), assign

    for _ in range(n_iter):
        new_centroids, assign = step(centroids)
        if jnp.allclose(new_centroids, centroids, atol=1e-5):
            centroids = new_centroids
            break
        centroids = new_centroids

    return np.asarray(assign, dtype=np.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_codec_clusters(
    msa_int: np.ndarray,
    n_clusters: int = 4,
    c_hidden: int = _DEFAULT_C_HIDDEN,
    c_z: int = _DEFAULT_C_Z,
    latent_dim: int = _DEFAULT_LATENT,
    projector_epochs: int = 80,
    projector_batch_size: int = 8,
    projector_lr: float = 5e-5,
    projector_weight_decay: float = 1e-4,
    projector_dropout: float = 0.15,
    grad_clip: float = 1.0,
    rng_seed: int = 0,
) -> np.ndarray:
    """Run CoDeC clustering on the integer MSA.

    Faithful re-implementation of openfold2's recipe:

        - Per-sequence outer-product pair fingerprints (with fixed random
          a/b projections and a fixed linear_out).
        - On-the-fly CNN projector with SE blocks.
        - AdamW(lr=5e-5, wd=1e-4), 80 epochs, batch_size=8,
          gradient clip max_norm=1.0, L1 cosine-similarity loss.
        - K-means cosine clustering of the resulting latent vectors.

    Args:
        msa_int: ``[N_seq, N_res]`` integer-encoded MSA (HHBLITS_AA_TO_ID).
            The query is at index 0.
        n_clusters: Number of clusters / coevolutionary states to produce.
        c_hidden: Channel dim of the a/b projections.
        c_z: Channel dim of the per-sequence pair fingerprint (CNN input).
        latent_dim: Dimensionality of the final latent vectors used for
            clustering.
        projector_epochs: Number of epochs to train the projector.
        projector_batch_size: Batch size for projector training.
        projector_lr: AdamW learning rate.
        projector_weight_decay: AdamW weight decay.
        projector_dropout: Dropout rate inside the CNN.
        grad_clip: Global-norm gradient clip threshold.
        rng_seed: Seed for projector init / random projections / k-means.

    Returns:
        ``cluster_idxs``: ``[N_seq]`` contiguous integer cluster assignments.
    """
    msa_int = np.asarray(msa_int, dtype=np.int32)
    n_seq, n_res = msa_int.shape
    if n_seq <= 1:
        return np.zeros(n_seq, dtype=np.int32)

    rng = np.random.default_rng(rng_seed)
    w_a, w_b, w_out = _make_fixed_projections(rng, c_hidden=c_hidden, c_z=c_z)

    logger.info(
        "CoDeC: building per-sequence pair fingerprints (N_seq=%d, N_res=%d, c_hidden=%d, c_z=%d).",
        n_seq,
        n_res,
        c_hidden,
        c_z,
    )
    a, b = _compute_ab(msa_int, w_a, w_b)  # [N_seq, N_res, c_hidden]

    effective_k = min(n_clusters, max(1, n_seq - 1))
    if effective_k <= 1:
        return np.zeros(n_seq, dtype=np.int32)

    logger.info(
        "CoDeC: training projector on-the-fly (epochs=%d, batch_size=%d, lr=%g, wd=%g).",
        projector_epochs,
        projector_batch_size,
        projector_lr,
        projector_weight_decay,
    )
    params, model = _train_projector_cnn(
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
    )

    logger.info("CoDeC: computing latent vectors for all %d sequences.", n_seq)
    latents = _compute_latents(params, model, a, b, w_out, batch_size=projector_batch_size)

    logger.info("CoDeC: running cosine k-means (k=%d).", effective_k)
    cluster_idxs = _kmeans_cosine(latents, effective_k, rng_seed=rng_seed)

    # Make cluster ids contiguous starting at 0.
    unique = np.unique(cluster_idxs)
    remap = {old: new for new, old in enumerate(unique)}
    cluster_idxs = np.array([remap[c] for c in cluster_idxs], dtype=np.int32)

    counts = np.bincount(cluster_idxs)
    logger.info("CoDeC: cluster sizes = %s", counts.tolist())
    return cluster_idxs


def build_cluster_a3m(
    a3m_string: str,
    cluster_idxs: np.ndarray,
    cluster_id: int,
) -> str:
    """Subset the original A3M to sequences belonging to ``cluster_id``.

    The query (sequence 0) is always retained at the top.  Sequences are
    selected by their *unique-sequence index* — the same indexing used by
    ``make_msa_features``, which deduplicates sequences as it parses the
    A3M.  We re-run the same dedup logic here so the cluster index aligns
    with the corresponding A3M entry.
    """
    from alphafold.data.parsers import parse_a3m  # local import (heavy)

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
