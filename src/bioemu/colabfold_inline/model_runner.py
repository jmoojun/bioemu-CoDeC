# Portions derived from ColabFold (MIT License)
# https://github.com/sokrypton/ColabFold
# Copyright (c) 2021 Sergey Ovchinnikov
#
# Portions derived from AlphaFold2 (Apache License 2.0)
# https://github.com/google-deepmind/alphafold
# Copyright 2021 DeepMind Technologies Limited
#
# Modified for BioEmu integration, 2026.
# Changes: Extracted model loading, configuration, weight downloading,
#   predict_structure, and pad_input from colabfold v1.5.4.
#   - Hardcoded to monomer alphafold2, model_3, num_recycle=0 (BioEmu defaults).
#   - Stripped relax, ranking, PDB output — only extracts representations.
#   - JAX/Haiku/AlphaFold are imported lazily (optional dependencies).
# See LICENSES/COLABFOLD-MIT and src/_vendor/alphafold/LICENSE for full license texts.
"""AlphaFold2 model runner for extracting Evoformer representations.

This module wraps the JAX-based AlphaFold2 forward pass to produce the single
and pair representations that BioEmu consumes.  JAX, Haiku, and the ``alphafold``
package are **optional dependencies** — they are only imported when this module
is actually used.

When CoDeC (Coevolutionary Decomposition and Clustering, Jun et al., 2026)
is enabled, the MSA is first clustered into ``n_clusters`` groups whose
sequences share similar residue-residue couplings.  AF2 is then run once
per cluster (with that cluster's MSA), and the resulting single/pair
representations are stacked along a leading batch dimension, giving a
multistate ensemble that BioEmu can sample over sequentially.

Typical usage::

    from bioemu.colabfold_inline.model_runner import get_embeddings
    single_repr, pair_repr = get_embeddings(sequence, a3m_string)
    # single_repr: [B, L, 384], pair_repr: [B, L, L, 128] where B = n_clusters
"""

from __future__ import annotations

import logging
import os
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import requests
from tqdm import tqdm

from bioemu.colabfold_inline.csd_module import build_cluster_a3m, compute_codec_clusters
from bioemu.colabfold_inline.features import FeatureDict, build_monomer_feature

logger = logging.getLogger(__name__)

# Default cache location for AF2 model weights.
DEFAULT_PARAMS_DIR = Path(os.path.expanduser("~/.cache/colabfold"))

# BioEmu always uses: alphafold2, model_3, num_recycle=0, no templates.
MODEL_TYPE = "alphafold2"
MODEL_NUMBER = 3
NUM_RECYCLE = 0
NUM_ENSEMBLE = 1

# Default number of CoDeC clusters → produces a multistate ensemble of this size.
DEFAULT_NUM_CLUSTERS = 4


# ---------------------------------------------------------------------------
# Weight downloading (from colabfold.download)
# ---------------------------------------------------------------------------


def download_alphafold_params(data_dir: Path = DEFAULT_PARAMS_DIR) -> Path:
    """Download AlphaFold2 model parameters if not already present.

    Returns the ``data_dir`` path (which contains a ``params/`` subdirectory).
    """
    params_dir = data_dir / "params"
    success_marker = params_dir / "download_finished.txt"

    if success_marker.is_file():
        logger.info(f"AlphaFold2 params already downloaded at {params_dir}")
        return data_dir

    params_dir.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar"
    logger.info(f"Downloading AlphaFold2 weights to {data_dir} ...")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    file_size = int(response.headers.get("Content-Length", 0))

    with tqdm.wrapattr(
        response.raw, "read", total=file_size, desc="Downloading AF2 weights"
    ) as raw:
        with tarfile.open(fileobj=raw, mode="r|") as tar:
            tar.extractall(path=params_dir)

    success_marker.touch()
    logger.info("Download complete.")
    return data_dir


# ---------------------------------------------------------------------------
# Model loading (from colabfold.alphafold.models)
# ---------------------------------------------------------------------------


def _count_a3m_sequences(a3m_string: str) -> int:
    """Count the number of sequences in an A3M string (``>``-prefixed headers)."""
    return sum(1 for line in a3m_string.splitlines() if line.startswith(">"))


def _load_model_and_params(
    data_dir: Path = DEFAULT_PARAMS_DIR,
    max_msa_clusters: int | None = None,
    max_extra_msa: int | None = None,
) -> tuple[Any, Any]:
    """Load model_3 runner and its parameters.

    Following the openfold2 CoDeC config (``openfold/config.py``), both
    ``max_msa_clusters`` and ``max_extra_msa`` are set to the *total*
    number of sequences in the input A3M so that AF2 does not subsample
    the MSA before the CSDModule sees it.  Pass ``max_msa_clusters`` and
    ``max_extra_msa`` to override.

    Returns ``(model_runner, params)`` — both are JAX/Haiku objects.
    """
    from alphafold.model import config, model, utils

    # Ensure weights are downloaded
    download_alphafold_params(data_dir)

    # Load parameters
    params_path = data_dir / "params" / f"params_model_{MODEL_NUMBER}.npz"
    raw_params = np.load(str(params_path), allow_pickle=False)
    params = utils.flat_params_to_haiku(raw_params, fuse=True, to_jnp=True)

    # Build model config — matches ColabFold defaults for BioEmu
    config_name = f"model_{MODEL_NUMBER}"
    model_config = config.model_config(config_name)
    model_config.data.common.num_recycle = NUM_RECYCLE
    model_config.model.num_recycle = NUM_RECYCLE
    model_config.data.eval.num_ensemble = NUM_ENSEMBLE

    # CoDeC config (openfold2): use the full MSA — no AF2-level subsampling.
    if max_msa_clusters is not None:
        model_config.data.eval.max_msa_clusters = int(max_msa_clusters)
    if max_extra_msa is not None:
        model_config.data.common.max_extra_msa = int(max_extra_msa)

    # Disable unused heads to save compute — we only need representations_evo
    model_config.model.heads.distogram.weight = 0.0
    model_config.model.heads.masked_msa.weight = 0.0
    model_config.model.heads.experimentally_resolved.weight = 0.0
    model_config.model.heads.structure_module.weight = 0.0
    model_config.model.heads.predicted_lddt.weight = 0.0
    model_config.model.heads.predicted_aligned_error.weight = 0.0
    # Enable fused projections
    evo = model_config.model.embeddings_and_evoformer.evoformer
    evo.triangle_multiplication_incoming.fuse_projection_weights = True
    evo.triangle_multiplication_outgoing.fuse_projection_weights = True

    logger.info(
        "AF2 config: max_msa_clusters=%d, max_extra_msa=%d",
        model_config.data.eval.max_msa_clusters,
        model_config.data.common.max_extra_msa,
    )

    model_runner = model.RunModel(model_config, params)
    return model_runner, params


# ---------------------------------------------------------------------------
# Input padding (from colabfold.batch.pad_input + colabfold.alphafold.msa)
# ---------------------------------------------------------------------------


def _make_fixed_size(
    feat: dict[str, Any],
    shape_schema: dict[str, list],
    msa_cluster_size: int,
    extra_msa_size: int,
    num_res: int,
    num_templates: int = 0,
) -> dict[str, Any]:
    """Pad feature arrays to fixed sizes for XLA compilation."""
    # Shape placeholder constants from alphafold.model.tf.shape_placeholders
    NUM_RES = "num residues placeholder"
    NUM_MSA_SEQ = "msa placeholder"
    NUM_EXTRA_SEQ = "extra msa placeholder"
    NUM_TEMPLATES = "num templates placeholder"

    pad_size_map = {
        NUM_RES: num_res,
        NUM_MSA_SEQ: msa_cluster_size,
        NUM_EXTRA_SEQ: extra_msa_size,
        NUM_TEMPLATES: num_templates,
    }
    for k, v in feat.items():
        if k == "extra_cluster_assignment":
            continue
        shape = list(v.shape)
        schema = shape_schema[k]
        assert len(shape) == len(schema), f"Rank mismatch for {k}: {shape} vs {schema}"
        pad_size = [pad_size_map.get(s2, None) or s1 for s1, s2 in zip(shape, schema)]
        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        if padding:
            feat[k] = np.pad(v, padding)
    return feat


def _pad_input(
    input_features: FeatureDict,
    model_runner: Any,
    pad_len: int,
) -> FeatureDict:
    """Pad processed features to ``pad_len`` residues."""
    model_config = model_runner.config
    eval_cfg = model_config.data.eval
    crop_feats = {k: [None] + v for k, v in dict(eval_cfg.feat).items()}

    max_msa_clusters = eval_cfg.max_msa_clusters
    max_extra_msa = model_config.data.common.max_extra_msa

    # model_3 does not use templates, so no adjustment needed
    return _make_fixed_size(
        input_features,
        crop_feats,
        msa_cluster_size=max_msa_clusters,
        extra_msa_size=max_extra_msa,
        num_res=pad_len,
        num_templates=4,
    )


# ---------------------------------------------------------------------------
# Forward pass — extract representations
# ---------------------------------------------------------------------------


def _run_model(
    feature_dict: FeatureDict,
    model_runner: Any,
    pad_len: int,
    random_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run the AF2 forward pass and return representations.

    Returns a dict with ``single`` and ``pair`` arrays.
    """
    seq_len = feature_dict["aatype"].shape[0]

    # Process features through the data pipeline
    input_features = model_runner.process_features(feature_dict, random_seed=random_seed)

    # Pad if needed
    if seq_len < pad_len:
        input_features = _pad_input(input_features, model_runner, pad_len)
        logger.info(f"Padded input to {pad_len} residues")

    # Run prediction with representations
    result, _recycles = model_runner.predict(
        input_features,
        random_seed=random_seed,
        return_representations=True,
    )

    # Extract Evoformer representations (before structure module, via patch)
    # representations_evo has the 384-dim single and 128-dim pair embeddings
    # that BioEmu uses, as opposed to the post-structure-module representations.
    evo_repr = result["representations_evo"]
    single_repr = np.array(evo_repr["single"][:seq_len])
    pair_repr = np.array(evo_repr["pair"][:seq_len, :seq_len])

    return {"single": single_repr, "pair": pair_repr}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _run_one_state(
    sequence: str,
    a3m_string: str,
    model_runner: Any,
    pad_len: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Build features for ``a3m_string`` and run a single AF2 forward pass."""
    feature_dict = build_monomer_feature(sequence, a3m_string)
    return _run_model(feature_dict, model_runner, pad_len, random_seed)


def get_embeddings(
    sequence: str,
    a3m_string: str,
    data_dir: str | Path = DEFAULT_PARAMS_DIR,
    random_seed: int = 0,
    n_clusters: int = DEFAULT_NUM_CLUSTERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the AF2 Evoformer and return single + pair representations.

    This is the main entry point for BioEmu embedding generation.  It builds
    features from the sequence and MSA, loads the AF2 model (downloading weights
    if necessary), runs the forward pass, and returns the Evoformer embeddings.

    With CoDeC enabled (``n_clusters > 1``), the MSA is first decomposed into
    ``n_clusters`` coevolutionary clusters; AF2 is run once per cluster and
    the resulting embeddings are stacked along a leading batch dimension.
    BioEmu then samples sequentially across this batch dimension.

    Args:
        sequence: Protein sequence (uppercase single-letter amino acids).
        a3m_string: MSA in A3M format (query sequence first).
        data_dir: Directory containing (or to download) AF2 model weights.
        random_seed: Random seed for the model.
        n_clusters: Number of CoDeC clusters / states to produce.  Set to ``1``
            to disable CoDeC and fall back to a single AF2 pass.

    Returns:
        ``(single_repr, pair_repr)`` where:
        - ``single_repr`` has shape ``(B, L, 384)``
        - ``pair_repr`` has shape ``(B, L, L, 128)``
        and ``B == n_clusters`` (or fewer if the MSA is too shallow for
        ``n_clusters`` distinct clusters).
    """
    data_dir = Path(data_dir)

    # Build features for the full MSA to size things and get the deduped int MSA
    # used as the basis for CoDeC clustering.
    feature_dict = build_monomer_feature(sequence, a3m_string)

    # Following the openfold2 CoDeC config: AF2 sees the full MSA (no
    # subsampling).  Count the number of A3M sequences and pass that as
    # max_msa_clusters / max_extra_msa.
    num_seqs = _count_a3m_sequences(a3m_string)
    logger.info("CoDeC: A3M contains %d sequences (used for max_msa_clusters).", num_seqs)
    model_runner, _params = _load_model_and_params(
        data_dir,
        max_msa_clusters=num_seqs,
        max_extra_msa=num_seqs,
    )

    # Compute padding length (round up to next multiple of 16)
    seq_len = len(sequence)
    pad_len = int(np.ceil(seq_len / 16) * 16)

    if n_clusters is None or n_clusters <= 1:
        # CoDeC disabled — single forward pass, but still return a batch dim
        # of size 1 so downstream code can assume a uniform contract.
        representations = _run_model(feature_dict, model_runner, pad_len, random_seed)
        single_repr = representations["single"][None, ...]
        pair_repr = representations["pair"][None, ...]
        logger.info(
            f"Embeddings computed (no CoDeC): single={single_repr.shape}, pair={pair_repr.shape}"
        )
        return single_repr, pair_repr

    # CoDeC: cluster MSA sequences by their coevolutionary fingerprint.
    msa_int = feature_dict["msa"]  # [N_seq, N_res], integer-encoded deduped MSA
    cluster_idxs = compute_codec_clusters(
        msa_int=msa_int,
        n_clusters=n_clusters,
        rng_seed=random_seed,
    )
    unique_clusters = np.unique(cluster_idxs)
    actual_k = int(unique_clusters.size)
    logger.info("CoDeC produced %d non-empty clusters (requested %d).", actual_k, n_clusters)

    single_batch = []
    pair_batch = []
    for cid in unique_clusters:
        cluster_a3m = build_cluster_a3m(a3m_string, cluster_idxs, int(cid))
        logger.info(
            "Running AF2 on cluster %d/%d (%d-line A3M).",
            int(cid),
            actual_k,
            cluster_a3m.count("\n"),
        )
        representations = _run_one_state(
            sequence,
            cluster_a3m,
            model_runner,
            pad_len,
            random_seed,
        )
        single_batch.append(representations["single"])
        pair_batch.append(representations["pair"])

    single_repr = np.stack(single_batch, axis=0)  # [B, L, 384]
    pair_repr = np.stack(pair_batch, axis=0)  # [B, L, L, 128]
    logger.info(f"CoDeC embeddings computed: single={single_repr.shape}, pair={pair_repr.shape}")
    return single_repr, pair_repr
