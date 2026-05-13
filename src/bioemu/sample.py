# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script for sampling from a trained model."""

import logging
import time
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from bioemu.chemgraph import ChemGraph
from bioemu.convert_chemgraph import save_pdb_and_xtc
from bioemu.get_embeds import get_colabfold_embeds
from bioemu.model_utils import load_model, load_sdes, maybe_download_checkpoint
from bioemu.sde_lib import SDE
from bioemu.seq_io import check_protein_valid, parse_sequence, write_fasta
from bioemu.utils import format_npz_samples_filename, print_traceback_on_exception

logger = logging.getLogger(__name__)

DEFAULT_DENOISER_CONFIG_DIR = Path(__file__).parent / "config/denoiser/"
SupportedDenoisersLiteral = Literal["dpm", "heun"]
SUPPORTED_DENOISERS = list(typing.get_args(SupportedDenoisersLiteral))

# Mapping used in training of BioEmu-1.2 model.
_NODE_LABEL_MAPPING: dict[str, int] = {
    "A": 1,
    "R": 15,
    "N": 12,
    "D": 3,
    "C": 2,
    "Q": 14,
    "E": 4,
    "G": 6,
    "H": 7,
    "I": 8,
    "L": 10,
    "K": 9,
    "M": 11,
    "F": 5,
    "P": 13,
    "S": 16,
    "T": 17,
    "W": 19,
    "Y": 20,
    "V": 18,
    "U": 21,
    "O": 22,
    "X": 0,
    "B": 23,
    "Z": 25,
}


@print_traceback_on_exception
@torch.no_grad()
def main(
    sequence: str | Path,
    num_samples: int,
    output_dir: str | Path,
    batch_size_100: int = 10,
    model_name: Literal["bioemu-v1.0", "bioemu-v1.1", "bioemu-v1.2"] | None = "bioemu-v1.1",
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    denoiser_type: SupportedDenoisersLiteral | None = "dpm",
    denoiser_config: str | Path | dict | None = None,
    cache_embeds_dir: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    msa_host_url: str | None = None,
    filter_samples: bool = True,
    base_seed: int | None = None,
    n_clusters: int = 4,
) -> None:
    """
    Generate samples for a specified sequence, using a trained model.

    Args:
        sequence: Amino acid sequence for which to generate samples, or a path to a .fasta file, or a path to an .a3m file with MSAs.
            If it is not an a3m file, then colabfold will be used to generate an MSA and embedding.
        num_samples: Number of samples to generate. If `output_dir` already contains samples, this function will only generate additional samples necessary to reach the specified `num_samples`.
        output_dir: Directory to save the samples. Each batch of samples will initially be dumped as .npz files. Once all batches are sampled, they will be converted to .xtc and .pdb.
        batch_size_100: Batch size you'd use for a sequence of length 100. The batch size will be calculated from this, assuming
           that the memory requirement to compute each sample scales quadratically with the sequence length.
        model_name: Name of pretrained model to use. If this is set, you do not need to provide `ckpt_path` or `model_config_path`.
            The model will be retrieved from huggingface; the following models are currently available:
            - bioemu-v1.0: checkpoint used in the original preprint (https://www.biorxiv.org/content/10.1101/2024.12.05.626885v2)
            - bioemu-v1.1: checkpoint used for our paper (https://www.science.org/doi/10.1126/science.adv9817)
            - bioemu-v1.2: checkpoint trained with an extended set of MD simulations and experimental measurements of folding free energies.
        ckpt_path: Path to the model checkpoint. If this is set, `model_name` will be ignored.
        model_config_path: Path to the model config, defining score model architecture and the corruption process the model was trained with.
           Only required if `ckpt_path` is set.
        denoiser_type: Denoiser to use for sampling, if `denoiser_config` not specified. Comes in with default parameter configuration. Must be one of ['dpm', 'heun']
        denoiser_config: Path (str or :class:`os.PathLike`) to a denoiser config YAML, or a dict. For steered sampling,
            pass a steering config (e.g., config/steering/physical_steering.yaml) which includes
            the denoiser target, potentials, and steering parameters in one file.
        cache_embeds_dir: Directory to store MSA embeddings. If not set, this defaults to `COLABFOLD_DIR/embeds_cache`.
        cache_so3_dir: Directory to store SO3 precomputations. If not set, this defaults to `~/sampling_so3_cache`.
        msa_host_url: MSA server URL. If not set, this defaults to colabfold's remote server. If sequence is an a3m file, this is ignored.
        filter_samples: Filter out unphysical samples with e.g. long bond distances or steric clashes.
        base_seed: Base random seed for sampling. If set, each batch's seed will be set to base_seed + (num samples already generated).
        n_clusters: Number of CoDeC clusters / coevolutionary states.  AlphaFold2
            embeddings are computed once per cluster, and the requested
            ``batch_size`` samples are distributed (round-robin) across the
            resulting states.  Set to ``1`` to disable CoDeC.
    """

    if base_seed is None:
        # Use system time
        base_seed = time.time_ns()

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)  # Fail fast if output_dir is non-writeable

    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name=model_name, ckpt_path=ckpt_path, model_config_path=model_config_path
    )
    score_model = load_model(ckpt_path, model_config_path)

    sdes = load_sdes(model_config_path=model_config_path, cache_so3_dir=cache_so3_dir)

    # User may have provided an MSA file instead of a sequence. This will be used for embeddings.
    msa_file = sequence if str(sequence).endswith(".a3m") else None

    if msa_file is not None and msa_host_url is not None:
        logger.warning(f"msa_host_url is ignored because MSA file {msa_file} is provided.")

    # Parse FASTA or A3M file if sequence is a file path. Extract the actual sequence.
    sequence = parse_sequence(sequence)

    # Check input sequence is valid
    check_protein_valid(sequence)

    fasta_path = output_dir / "sequence.fasta"
    if fasta_path.is_file():
        if parse_sequence(fasta_path) != sequence:
            raise ValueError(
                f"{fasta_path} already exists, but contains a sequence different from {sequence}!"
            )
    else:
        # Save FASTA file in output_dir
        write_fasta([sequence], fasta_path)

    if denoiser_config is None:
        # load default config
        assert (
            denoiser_type in SUPPORTED_DENOISERS
        ), f"denoiser_type must be one of {SUPPORTED_DENOISERS}"
        denoiser_config = DEFAULT_DENOISER_CONFIG_DIR / f"{denoiser_type}.yaml"
        with open(denoiser_config) as f:
            denoiser_config = yaml.safe_load(f)
    elif isinstance(denoiser_config, str | Path):
        # path to denoiser config
        denoiser_config_path = Path(denoiser_config).expanduser().resolve()
        assert (
            denoiser_config_path.is_file()
        ), f"denoiser_config path '{denoiser_config_path}' does not exist or is not a file."
        with open(denoiser_config_path) as f:
            denoiser_config = yaml.safe_load(f)
    else:
        assert type(denoiser_config) in [
            dict,
            DictConfig,
        ], f"denoiser_config must be a path to a YAML file or a dict, but got {type(denoiser_config)}"

    denoiser = hydra.utils.instantiate(denoiser_config)

    logger.info(
        f"Sampling {num_samples} structures *per cluster* for sequence of length "
        f"{len(sequence)} residues..."
    )
    # Adjust batch size by sequence length (memory ~ L^2).  ``num_samples`` is
    # now interpreted as "samples per cluster", so cap by that.
    batch_size = int(batch_size_100 * (100 / len(sequence)) ** 2)
    batch_size = max(1, min(batch_size, num_samples))
    logger.info(f"Using batch size {batch_size} per cluster")

    # ---- Heavy step: AF2 + CSD runs *once* and returns one ChemGraph per
    # cluster.  PyTorch score model is parked on CPU while the JAX/AF2
    # forward is on GPU to avoid OOM on a single-GPU node.
    score_model_device = next(score_model.parameters()).device
    if score_model_device.type == "cuda":
        score_model.to("cpu")
        torch.cuda.empty_cache()

    context_chemgraphs = get_context_chemgraphs(
        sequence=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
        n_clusters=n_clusters,
    )

    if score_model_device.type == "cuda":
        import gc

        import jax

        jax.clear_caches()
        gc.collect()
        score_model.to(score_model_device)

    n_states_actual = len(context_chemgraphs)
    logger.info(
        "CoDeC produced %d coevolutionary states; sampling %d structures per state " "(total %d).",
        n_states_actual,
        num_samples,
        num_samples * n_states_actual,
    )

    # ---- Outer loop: each cluster's per-batch npz / topology / xtc files
    # are emitted directly into ``output_dir`` with a ``cluster_NN_`` prefix
    # in the filename — no subdirectories.  This keeps the layout flat
    # while still letting the 4+ ensembles coexist without overwriting
    # each other, and the final per-frame PDBs are named
    # ``cluster_NN_MM.pdb``. ----
    def _count_existing_for_cluster(state_idx: int) -> int:
        """Count samples whose npz files are tagged with this cluster.

        File pattern: ``cluster_<NN>_batch_<start>_<end>.npz``.  Sum
        ``end - start`` across all matching files.
        """
        prefix = f"cluster_{state_idx:02d}_batch_"
        total = 0
        for p in Path(output_dir).glob(f"{prefix}*.npz"):
            parts = p.stem.removeprefix(prefix).split("_")
            if len(parts) != 2:
                continue
            try:
                total += int(parts[1]) - int(parts[0])
            except ValueError:
                continue
        return total

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for state_idx, context_chemgraph in enumerate(context_chemgraphs):
        existing = _count_existing_for_cluster(state_idx)
        logger.info(
            "Cluster %02d: found %d previous samples in %s.",
            state_idx,
            existing,
            output_dir,
        )
        for start_idx in tqdm(
            range(existing, num_samples, batch_size),
            desc=f"Cluster {state_idx:02d} sampling",
        ):
            n = min(batch_size, num_samples - start_idx)
            npz_name = f"cluster_{state_idx:02d}_" + format_npz_samples_filename(start_idx, n)
            npz_path = output_dir / npz_name
            if npz_path.exists():
                raise ValueError(
                    f"Not sure why {npz_path} already exists when so far only "
                    f"{existing} samples have been generated for cluster {state_idx}."
                )
            # Seed: independent per (cluster, start_idx) so different clusters
            # explore independent stochastic trajectories.
            seed = base_seed + start_idx + state_idx * 1_000_000
            logger.info(
                "Cluster %02d, start_idx=%d, n=%d, seed=%d",
                state_idx,
                start_idx,
                n,
                seed,
            )
            batch = generate_batch_for_state(
                score_model=score_model,
                context_chemgraph=context_chemgraph,
                sdes=sdes,
                batch_size=n,
                seed=seed,
                denoiser=denoiser,
                device=device,
            )
            batch_np = {k: v.cpu().numpy() for k, v in batch.items()}
            np.savez(npz_path, **batch_np, sequence=sequence, state_idx=state_idx)

        # Per-cluster XTC + topology, also prefix-tagged.  Downstream
        # tools (run.sh's sidechain_relax + PDB conversion) read
        # ``cluster_NN_topology.pdb`` / ``cluster_NN_samples.xtc``.
        logger.info("Cluster %02d: converting samples to .pdb and .xtc...", state_idx)
        cluster_prefix = f"cluster_{state_idx:02d}_batch_"
        samples_files = sorted(output_dir.glob(f"{cluster_prefix}*.npz"))
        if not samples_files:
            logger.warning("Cluster %02d produced no samples (skipping).", state_idx)
            continue
        sequences = [np.load(f)["sequence"].item() for f in samples_files]
        if set(sequences) != {sequence}:
            raise ValueError(
                f"Expected all sequences to be {sequence}, but got {set(sequences)} "
                f"in cluster {state_idx}."
            )
        positions = torch.tensor(np.concatenate([np.load(f)["pos"] for f in samples_files]))
        node_orientations = torch.tensor(
            np.concatenate([np.load(f)["node_orientations"] for f in samples_files])
        )
        save_pdb_and_xtc(
            pos_nm=positions,
            node_orientations=node_orientations,
            topology_path=output_dir / f"cluster_{state_idx:02d}_topology.pdb",
            xtc_path=output_dir / f"cluster_{state_idx:02d}_samples.xtc",
            sequence=sequence,
            filter_samples=filter_samples,
        )

    logger.info(
        "Completed.  Per-cluster samples are in %s/cluster_NN_{topology.pdb,samples.xtc}.",
        output_dir,
    )


def get_context_chemgraphs(
    sequence: str,
    cache_embeds_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    n_clusters: int = 4,
) -> list[ChemGraph]:
    """Load CoDeC-batched embeddings and return one ChemGraph per state.

    The returned list has length equal to the number of CoDeC clusters that
    were actually produced (which may be less than ``n_clusters`` if the MSA
    was too shallow).  BioEmu sampling iterates over this list, generating
    a separate ensemble of structures for each cluster.
    """
    n = len(sequence)

    single_embeds_file, pair_embeds_file = get_colabfold_embeds(
        seq=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
        n_clusters=n_clusters,
    )
    single_embeds_batch = torch.from_numpy(np.load(single_embeds_file))  # [B, L, 384]
    pair_embeds_batch = torch.from_numpy(np.load(pair_embeds_file))  # [B, L, L, 128]

    # Backwards-compat: older caches were saved without a leading batch dim.
    if single_embeds_batch.ndim == 2:
        single_embeds_batch = single_embeds_batch.unsqueeze(0)
        pair_embeds_batch = pair_embeds_batch.unsqueeze(0)

    assert single_embeds_batch.ndim == 3, single_embeds_batch.shape
    assert pair_embeds_batch.ndim == 4, pair_embeds_batch.shape
    assert pair_embeds_batch.shape[1] == pair_embeds_batch.shape[2] == n
    assert single_embeds_batch.shape[1] == n

    n_states = single_embeds_batch.shape[0]
    _, _, _, n_pair_feats = pair_embeds_batch.shape

    edge_index = torch.cat(
        [
            torch.arange(n).repeat_interleave(n).view(1, n**2),
            torch.arange(n).repeat(n).view(1, n**2),
        ],
        dim=0,
    )
    pos = torch.full((n, 3), float("nan"))
    node_orientations = torch.full((n, 3, 3), float("nan"))
    node_labels = torch.LongTensor([_NODE_LABEL_MAPPING[aa] for aa in sequence])

    contexts: list[ChemGraph] = []
    for state_idx in range(n_states):
        single_embeds = single_embeds_batch[state_idx]
        pair_embeds = pair_embeds_batch[state_idx].reshape(n**2, n_pair_feats)
        contexts.append(
            ChemGraph(
                edge_index=edge_index.clone(),
                pos=pos.clone(),
                node_orientations=node_orientations.clone(),
                single_embeds=single_embeds,
                pair_embeds=pair_embeds,
                sequence=sequence,
                node_labels=node_labels.clone(),
            )
        )
    return contexts


def get_context_chemgraph(
    sequence: str,
    cache_embeds_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    n_clusters: int = 4,
    state_idx: int = 0,
) -> ChemGraph:
    """Return a single CoDeC state's ChemGraph, selected by ``state_idx``."""
    contexts = get_context_chemgraphs(
        sequence=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
        n_clusters=n_clusters,
    )
    return contexts[state_idx % len(contexts)]


def generate_batch_for_state(
    score_model: torch.nn.Module,
    context_chemgraph: ChemGraph,
    sdes: dict[str, SDE],
    batch_size: int,
    seed: int,
    denoiser: Callable,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Generate one batch of samples for a single CoDeC cluster.

    Unlike the legacy ``generate_batch``, this helper does **not** call AF2
    or re-derive the context — it consumes a single pre-built
    ``ChemGraph`` (one cluster's CoDeC state) and runs the score-based
    sampler ``batch_size`` times under that conditioning.  The caller is
    responsible for looping over clusters; this way AF2/CSD only runs
    once per inference even when many clusters and batches are requested.

    Args:
        score_model: Score model (already on the target device).
        context_chemgraph: A single cluster's CoDeC ``ChemGraph`` (single
            and pair embeddings already gathered for that state).
        sdes: SDEs defining the corruption process.
        batch_size: Number of samples to draw from this cluster's state.
        seed: PyTorch RNG seed for this batch.
        denoiser: Hydra-instantiated denoiser callable.
        device: Target device (defaults to ``cuda:0`` if available).

    Returns:
        ``{"pos": [B, L, 3], "node_orientations": [B, L, 3, 3]}``.
    """
    torch.manual_seed(seed)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    context_batch = Batch.from_data_list([context_chemgraph] * batch_size)
    result = denoiser(
        sdes=sdes,
        device=device,
        batch=context_batch,
        score_model=score_model,
    )

    # Steered denoisers (SMC) return (batch, log_weights); unsteered returns batch
    if isinstance(result, tuple):
        sampled_chemgraph_batch, _ = result
    else:
        sampled_chemgraph_batch = result
    assert isinstance(sampled_chemgraph_batch, Batch)
    sampled_chemgraphs = sampled_chemgraph_batch.to_data_list()
    pos = torch.stack([x.pos for x in sampled_chemgraphs]).to("cpu")
    node_orientations = torch.stack([x.node_orientations for x in sampled_chemgraphs]).to("cpu")
    return {"pos": pos, "node_orientations": node_orientations}


if __name__ == "__main__":
    import logging

    import fire

    logging.basicConfig(level=logging.DEBUG)

    fire.Fire(main)
