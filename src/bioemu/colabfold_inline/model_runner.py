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

# IMPORTANT: load JAX before the vendored AlphaFold (which transitively
# imports tensorflow inside its data pipeline).  Both JAX and TF ship their
# own copies of ``xla/xla_data.proto`` — whichever registers first wins, and
# the second one's duplicate is silently dropped.  But if TF registers
# first and JAX tries to do so during a *later* dlopen, protobuf's C++
# descriptor pool aborts the process with
#   F0000 descriptor.cc:2519] Check failed: GeneratedDatabase()->Add(...)
#   File already exists in database: xla/xla_data.proto
# The old code happened to load JAX first via csd_module (which used flax/
# optax/jax); after the PyTorch rewrite csd_module no longer imports JAX,
# so we anchor the load order explicitly here.
import jax  # noqa: F401  -- ordering guard, used inside alphafold below
import numpy as np
import requests
from tqdm import tqdm

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


def _split_evoformer_iteration_params(params: dict, n_first: int) -> dict:
    """Split the stacked ``evoformer_iteration`` params into ``_first``/``_second``.

    The AF2 checkpoint stores each Evoformer-block weight tensor with a leading
    axis of size ``evoformer_num_block`` (48 for monomer model_3).  The
    refactored ``EmbeddingsAndEvoformer`` (see ``_vendor/alphafold/model/
    modules.py``) replaces the single 48-block ``layer_stack`` with two
    named ``layer_stack``s — ``evoformer_iteration_first`` (size ``n_first``)
    and ``evoformer_iteration_second`` (size ``48 - n_first``) — so that the
    CSDModule can be inserted between them.  This function slices the
    pretrained stack along axis 0 and copies the slices into the two new
    haiku param namespaces.  The original ``evoformer_iteration/*`` keys are
    dropped to avoid surprising haiku's strict param-tree checks.
    """
    new_params = {}
    moved_keys = []
    for module_path, leaves in params.items():
        if "/evoformer_iteration" not in module_path:
            new_params[module_path] = leaves
            continue

        first_path = module_path.replace("/evoformer_iteration", "/evoformer_iteration_first", 1)
        second_path = module_path.replace("/evoformer_iteration", "/evoformer_iteration_second", 1)

        sliced_first = {}
        sliced_second = {}
        for leaf_name, value in leaves.items():
            if value.shape[0] != n_first + (value.shape[0] - n_first):
                # Should never happen — sanity check.
                raise ValueError(
                    f"Unexpected leading axis for {module_path}/{leaf_name}: " f"{value.shape}."
                )
            sliced_first[leaf_name] = value[:n_first]
            sliced_second[leaf_name] = value[n_first:]

        new_params[first_path] = sliced_first
        new_params[second_path] = sliced_second
        moved_keys.append(module_path)

    if moved_keys:
        logger.info(
            "CoDeC: split %d evoformer_iteration param modules into _first(%d) + _second(%d).",
            len(moved_keys),
            n_first,
            next(iter(new_params.values()))[next(iter(next(iter(new_params.values()))))].shape[0]
            if False
            else 0,
        )
    return new_params


def _load_model_and_params(
    data_dir: Path = DEFAULT_PARAMS_DIR,
    max_msa_clusters: int | None = None,
    max_extra_msa: int | None = None,
    csd_n_clusters: int | None = None,
    csd_max_size: int = 128,
    csd_block_idx: int = 11,
    csd_rng_seed: int = 0,
    csd_host_cluster_fn: Any = None,
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

    # Refactor pretrained evoformer params into the (first, second) split that
    # our patched Evoformer expects, so CSD can sit between them.  Even when
    # CSD is disabled, both stacks are present in the haiku model and need
    # their respective param slices.
    n_first_blocks = int(csd_block_idx) + 1
    params = _split_evoformer_iteration_params(params, n_first=n_first_blocks)

    # CoDeC config (openfold2): use the full MSA — no AF2-level subsampling.
    if max_msa_clusters is not None:
        model_config.data.eval.max_msa_clusters = int(max_msa_clusters)
    if max_extra_msa is not None:
        model_config.data.common.max_extra_msa = int(max_extra_msa)

    # Disable bfloat16 inside the Evoformer.  The vendored ColabFold AF2
    # config sets ``model.global_config.bfloat16=True`` for memory.  On the
    # jax-cuda12 plugin pinned by bioemu (0.4.35), XLA inserts bf16->f16
    # casts somewhere in the Evoformer graph (likely between bf16 internals
    # and f16 representation slots) that the plugin can't lower, yielding
    # ``LLVM ERROR: Unsupported rounding mode for conversion``.  Forcing
    # the whole stack to f32 sidesteps every bf16<->f16 boundary at the
    # cost of more memory.
    model_config.model.global_config.bfloat16 = False
    model_config.model.global_config.bfloat16_output = False

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

    # CoDeC sub-config consumed by the patched ``EmbeddingsAndEvoformer``.
    # When ``enabled=True`` the Evoformer splits after block ``block_idx``,
    # clusters the post-block-11 MSA via the host callback, gathers m by
    # cluster, and vmaps the remaining blocks over the cluster axis.
    import ml_collections

    eae = model_config.model.embeddings_and_evoformer
    eae.codec = ml_collections.ConfigDict()
    eae.codec.enabled = bool(csd_n_clusters is not None and csd_n_clusters > 1)
    eae.codec.n_clusters = int(csd_n_clusters) if csd_n_clusters else 1
    eae.codec.max_size = int(csd_max_size)
    eae.codec.block_idx = int(csd_block_idx)
    eae.codec.rng_seed = int(csd_rng_seed)
    eae.codec.host_cluster_fn = csd_host_cluster_fn

    logger.info(
        "AF2 config: max_msa_clusters=%d, max_extra_msa=%d | CSD enabled=%s n_clusters=%d max_size=%d block_idx=%d",
        model_config.data.eval.max_msa_clusters,
        model_config.data.common.max_extra_msa,
        eae.codec.enabled,
        eae.codec.n_clusters,
        eae.codec.max_size,
        eae.codec.block_idx,
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

    # Workaround: jax-cuda12 plugin 0.4.35 (the version bioemu pins via
    # pyproject) can't lower ``bf16 -> f16`` rounding casts that XLA inserts
    # when ColabFold's patched AlphaFold passes f16 recycling buffers
    # (``prev_msa_first_row``, ``prev_pair``, ``prev_pos``) into the
    # bf16 Evoformer.  Cast every f16 entry up to f32 here so the whole
    # apply_fn graph stays in f32 and the broken cast disappears.
    # Symptom without this: ``LLVM ERROR: Unsupported rounding mode for
    # conversion. Aborted (core dumped)`` at apply_fn JIT compile time.
    n_cast = 0
    for k, v in list(input_features.items()):
        if hasattr(v, "dtype") and v.dtype == np.float16:
            input_features[k] = v.astype(np.float32)
            n_cast += 1
    if n_cast:
        logger.info("Cast %d float16 features to float32 (XLA bf16->f16 workaround).", n_cast)

    # Run prediction with representations
    result, _recycles = model_runner.predict(
        input_features,
        random_seed=random_seed,
        return_representations=True,
    )

    # Extract Evoformer representations (before structure module, via patch).
    # representations_evo has the 384-dim single and 128-dim pair embeddings
    # that BioEmu uses, as opposed to the post-structure-module representations.
    #
    # With CSD enabled the patched ``EmbeddingsAndEvoformer`` returns these
    # with a leading cluster axis ([nC, L, 384] / [nC, L, L, 128]).  Without
    # CSD the shapes are the original [L, 384] / [L, L, 128]; we prepend a
    # singleton batch axis in that case for a uniform downstream contract.
    evo_repr = result["representations_evo"]
    single_arr = np.array(evo_repr["single"])
    pair_arr = np.array(evo_repr["pair"])
    if single_arr.ndim == 2:
        single_repr = single_arr[:seq_len][None, ...]  # [1, L, 384]
        pair_repr = pair_arr[:seq_len, :seq_len][None, ...]  # [1, L, L, 128]
    else:
        # CSD-batched
        single_repr = single_arr[:, :seq_len]  # [nC, L, 384]
        pair_repr = pair_arr[:, :seq_len, :seq_len]  # [nC, L, L, 128]

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
    max_size: int = 128,
    csd_block_idx: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the AF2 Evoformer with in-graph CoDeC and return batched embeddings.

    Single forward pass through AF2.  The patched ``EmbeddingsAndEvoformer``
    splits the 48 Evoformer blocks at ``csd_block_idx`` (default 11): the
    first ``csd_block_idx + 1`` blocks run with the full MSA, then a host
    callback (``cluster_m_act_to_indices``) trains the on-the-fly CNN
    projector on the post-block MSA representation, clusters the resulting
    latent vectors via cosine k-means, and returns fixed-shape gather
    indices.  The MSA is then reshaped to ``[n_clusters, max_size, L, c_m]``
    (padded slots zeroed via the mask), the pair representation is
    broadcast across the cluster axis, and the remaining 36 blocks run
    under ``jax.vmap`` — one cluster per batch element.  The single
    projection is applied per cluster, yielding the multistate ensemble
    that BioEmu samples sequentially over.

    Args:
        sequence: Protein sequence (uppercase single-letter amino acids).
        a3m_string: MSA in A3M format (query sequence first).
        data_dir: Directory containing (or to download) AF2 model weights.
        random_seed: Random seed for the model + CSD k-means + projector init.
        n_clusters: Number of CoDeC clusters / states to produce.  Set to ``1``
            to disable CoDeC (the second stack still runs but in single-state
            mode); the output then has a leading dim of 1.
        max_size: Maximum sequences per cluster after padding (openfold2 uses
            128 — capping memory of the batched second half).
        csd_block_idx: Index of the last Evoformer block in the first half
            (i.e. CSD happens between this block and the next).  openfold2
            uses 11 ⇒ first half of 12 blocks, second half of 36 blocks.

    Returns:
        ``(single_repr, pair_repr)`` where:
        - ``single_repr`` has shape ``(n_clusters, L, 384)``
        - ``pair_repr`` has shape ``(n_clusters, L, L, 128)``
    """
    data_dir = Path(data_dir)

    feature_dict = build_monomer_feature(sequence, a3m_string)

    # openfold2 CoDeC config: AF2 sees the full MSA — no AF2-level subsampling.
    num_seqs = _count_a3m_sequences(a3m_string)
    logger.info("CoDeC: A3M contains %d sequences (used for max_msa_clusters).", num_seqs)

    # Host-side cluster-and-index callback for jax.pure_callback inside the
    # Evoformer.  Bind ``n_clusters`` and ``max_size`` here so the output
    # shape is static at JIT trace time.
    import functools

    from bioemu.colabfold_inline.csd_module import cluster_m_act_to_indices

    host_cluster_fn = functools.partial(
        cluster_m_act_to_indices,
        n_clusters=int(n_clusters),
        max_size=int(max_size),
        rng_seed=int(random_seed),
    )

    model_runner, _params = _load_model_and_params(
        data_dir,
        max_msa_clusters=num_seqs,
        max_extra_msa=num_seqs,
        csd_n_clusters=int(n_clusters),
        csd_max_size=int(max_size),
        csd_block_idx=int(csd_block_idx),
        csd_rng_seed=int(random_seed),
        csd_host_cluster_fn=host_cluster_fn,
    )

    seq_len = len(sequence)
    pad_len = int(np.ceil(seq_len / 16) * 16)

    representations = _run_model(feature_dict, model_runner, pad_len, random_seed)
    single_repr = representations["single"]  # already [B, L, 384]
    pair_repr = representations["pair"]  # already [B, L, L, 128]
    logger.info(
        "CoDeC embeddings computed: single=%s, pair=%s",
        single_repr.shape,
        pair_repr.shape,
    )
    return single_repr, pair_repr
