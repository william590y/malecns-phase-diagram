"""Public MaleCNS v1.0 ingestion; sparse arrays retain the full annotated CNS.

Weights are anatomical synapse counts, never conductances. Coordinates are soma
locations in micrometres. NT signs are an explicitly simplified model assumption.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.request import urlopen

import numpy as np

RELEASE = "male-cns:v1.0"
BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
SOURCE_FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
# SHA256 of the public v1.0 files retrieved 2026-09-09. Pinning detects source
# replacement as well as incomplete/corrupt downloads; these are not signatures.
SOURCE_SHA256 = {
    "annotations": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    "neurotransmitters": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    "weights": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
}
NT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}
NT_SIGN_ASSUMPTION = "acetylcholine +1, gaba/glutamate/histamine -1, other/unknown 0; ignores receptor-specific effects and neuromodulation"
NT_SIGN_REFERENCES = ["https://pubmed.ncbi.nlm.nih.gov/12196539/", "https://pmc.ncbi.nlm.nih.gov/articles/PMC6670387/"]
ARRAY_NAMES = (
    "body_ids", "pre", "post", "weight", "nt", "sign", "positions", "types",
    "classes", "superclasses", "subclasses", "sides", "instances", "nt_confidence",
)


@dataclass
class Connectome:
    body_ids: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray
    nt: np.ndarray
    sign: np.ndarray
    positions: np.ndarray
    types: np.ndarray
    classes: np.ndarray
    superclasses: np.ndarray
    subclasses: np.ndarray
    sides: np.ndarray
    instances: np.ndarray
    nt_confidence: np.ndarray
    metadata: dict

    @property
    def n(self) -> int:
        return len(self.body_ids)

    @property
    def n_edges(self) -> int:
        return len(self.pre)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name in ARRAY_NAMES:
            np.save(path / f"{name}.npy", getattr(self, name), allow_pickle=False)
        _write_metadata(path, self.metadata)
        return path

    @classmethod
    def load(cls, path: str | Path, mmap: bool = True) -> "Connectome":
        return load_connectome(path, mmap=mmap)


def _write_metadata(path: Path, metadata: dict) -> None:
    target = path / "metadata.json"
    tmp = path / "metadata.json.tmp"
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    tmp.replace(target)


def load_connectome(path: str | Path, mmap: bool = True) -> Connectome:
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("cache_format") != 1:
        raise ValueError("Unsupported or incomplete connectome cache")
    arrays = {
        name: np.load(path / f"{name}.npy", mmap_mode="r" if mmap else None, allow_pickle=False)
        for name in ARRAY_NAMES
    }
    result = Connectome(**arrays, metadata=metadata)
    if result.n != metadata["n_neurons"] or result.n_edges != metadata["n_edges"]:
        raise ValueError("Cache array lengths disagree with metadata")
    if result.post.shape != result.pre.shape or result.weight.shape != result.pre.shape:
        raise ValueError("Cache edge arrays have inconsistent lengths")
    if any(len(arrays[name]) != result.n for name in ARRAY_NAMES if name not in {"pre", "post", "weight"}):
        raise ValueError("Cache neuron arrays have inconsistent lengths")
    if result.positions.shape != (result.n, 3):
        raise ValueError("Cache positions must have shape (n_neurons, 3)")
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(raw_dir: str | Path = "data/raw") -> dict:
    """Fetch the three public files atomically, checking pinned SHA256 values."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for role, name in SOURCE_FILES.items():
        path = raw_dir / name
        headers = {}
        if not path.exists() or sha256_file(path) != SOURCE_SHA256[role]:
            tmp = path.with_suffix(".part")
            print(f"Downloading {name}", flush=True)
            try:
                with urlopen(BASE_URL + name, timeout=120) as response, tmp.open("wb") as output:
                    headers = dict(response.headers)
                    for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        output.write(chunk)
                if sha256_file(tmp) != SOURCE_SHA256[role]:
                    raise ValueError(f"SHA256 mismatch for {name}; refusing changed/corrupt source")
                tmp.replace(path)
            finally:
                if tmp.exists():
                    tmp.unlink()
        info = {
            "url": BASE_URL + name,
            "bytes": path.stat().st_size,
            "sha256": SOURCE_SHA256[role],
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        sidecar = path.with_suffix(".feather.provenance.json")
        if sidecar.exists():
            old = json.loads(sidecar.read_text())
            headers = headers or old.get("response_headers", {})
        info["response_headers"] = headers
        sidecar.write_text(json.dumps(info, indent=2) + "\n")
        provenance[role] = info
    return provenance


def _strings(frame, column: str) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), "", dtype="U1")
    return frame[column].fillna("").astype(str).to_numpy(dtype=str)


def _counts(values: np.ndarray) -> dict:
    names, counts = np.unique(values, return_counts=True)
    return {str(name): int(count) for name, count in zip(names, counts)}


def preprocess(
    raw_dir: str | Path = "data/raw",
    output: str | Path = "data/cache",
    *,
    verify_sources: bool = True,
) -> Connectome:
    """Stream the complete segment graph into the annotated-neuron induced graph.

    Include every annotation with a nonempty superclass OR status == Traced,
    excluding explicitly Glia/Unimportant rows. Retain isolates and every positive
    connection, including one-synapse edges and autapses. No top-k/minimum pruning.
    ``verify_sources=False`` is for test/custom fixtures and is recorded as such.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.feather as feather

    raw_dir, output = Path(raw_dir), Path(output)
    provenance = {}
    for role, name in SOURCE_FILES.items():
        path = raw_dir / name
        digest = sha256_file(path)
        if verify_sources and digest != SOURCE_SHA256[role]:
            raise ValueError(f"Source checksum mismatch: {path}")
        provenance[role] = {"url": BASE_URL + name, "sha256": digest, "bytes": path.stat().st_size}

    annotations = feather.read_table(raw_dir / SOURCE_FILES["annotations"]).to_pandas()
    required = {"bodyId", "superclass", "status"}
    if not required.issubset(annotations.columns):
        raise ValueError(f"Annotation schema lacks {required - set(annotations.columns)}")
    if annotations.bodyId.duplicated().any():
        raise ValueError("Duplicate annotation body IDs")
    classified = annotations.superclass.fillna("").ne("")
    included = (classified | annotations.status.eq("Traced")) & ~annotations.status.isin(["Glia", "Unimportant"])
    neurons = annotations.loc[included].sort_values("bodyId").reset_index(drop=True)
    if neurons.empty:
        raise ValueError("No annotated neurons selected")
    body_ids = neurons.bodyId.to_numpy(dtype=np.int64)
    index = pd.Index(body_ids)
    n = len(body_ids)
    positions = np.full((n, 3), np.nan, np.float32)
    if "somaLocation" in neurons:
        for i, location in enumerate(neurons.somaLocation):
            if location is not None and np.asarray(location).shape == (3,):
                positions[i] = np.asarray(location, dtype=np.float32) * 0.008

    nt_table = feather.read_table(
        raw_dir / SOURCE_FILES["neurotransmitters"],
        columns=["body", "consensus_nt", "predicted_nt_confidence"],
    ).to_pandas()
    if nt_table.body.duplicated().any():
        raise ValueError("Duplicate neurotransmitter body IDs")
    nt_table = nt_table.set_index("body").reindex(body_ids)
    nt = nt_table.consensus_nt.fillna("unknown").str.lower().to_numpy(dtype=str)
    nt[nt == ""] = "unknown"
    sign = np.asarray([NT_SIGN.get(value, 0.0) for value in nt], dtype=np.float32)
    confidence = nt_table.predicted_nt_confidence.to_numpy(dtype=np.float32)
    # For peripheral sensory neurons rootSide identifies the recorded receptor
    # entry side; otherwise use somaSide. Unresolved/M stay unresolved, not guessed.
    root_sides, soma_sides = _strings(neurons, "rootSide"), _strings(neurons, "somaSide")
    sides = np.where(np.isin(root_sides, ["L", "R"]), root_sides, soma_sides)

    output.mkdir(parents=True, exist_ok=True)
    # A missing metadata file marks an interrupted build as unusable.
    (output / "metadata.json").unlink(missing_ok=True)
    source_edges = retained_edges = source_synapses = retained_synapses = 0
    autapses = 0
    active_neurons = np.zeros(n, dtype=bool)
    source_reader = pa.ipc.open_file(str(raw_dir / SOURCE_FILES["weights"]))
    if not {"body_pre", "body_post", "weight"}.issubset(source_reader.schema.names):
        raise ValueError("Weight table must contain body_pre, body_post, weight")
    with tempfile.TemporaryDirectory(prefix=".preprocess-", dir=output) as tmp_name:
        tmp_dir = Path(tmp_name)
        with (tmp_dir / "pre.bin").open("wb") as pre_out, (tmp_dir / "post.bin").open("wb") as post_out, (tmp_dir / "weight.bin").open("wb") as weight_out:
            for batch_id in range(source_reader.num_record_batches):
                batch = source_reader.get_batch(batch_id)
                a = batch.column("body_pre").to_numpy(zero_copy_only=False)
                b = batch.column("body_post").to_numpy(zero_copy_only=False)
                weights = batch.column("weight").to_numpy(zero_copy_only=False)
                if np.any(~np.isfinite(weights)) or np.any(weights <= 0) or np.any(weights != np.floor(weights)):
                    raise ValueError("Source weights must be positive integral synapse counts")
                ai, bi = index.get_indexer(a), index.get_indexer(b)
                keep = (ai >= 0) & (bi >= 0)
                selected_a, selected_b, selected_w = ai[keep], bi[keep], weights[keep]
                if np.any(selected_w > 2**24):
                    raise ValueError("Source synapse count exceeds exact float32 integer range")
                pre_out.write(selected_a.astype(np.uint32).tobytes())
                post_out.write(selected_b.astype(np.uint32).tobytes())
                weight_out.write(selected_w.astype(np.float32).tobytes())
                active_neurons[selected_a] = True
                active_neurons[selected_b] = True
                autapses += int(np.count_nonzero(selected_a == selected_b))
                source_edges += len(a)
                retained_edges += len(selected_a)
                source_synapses += int(weights.sum())
                retained_synapses += int(selected_w.sum())
                if batch_id % 200 == 0:
                    print(f"Graph batch {batch_id + 1}/{source_reader.num_record_batches}: {retained_edges:,} neuron edges", flush=True)
        for name, dtype in [("pre", np.uint32), ("post", np.uint32), ("weight", np.float32)]:
            final = np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=dtype, shape=(retained_edges,))
            if retained_edges:
                raw = np.memmap(tmp_dir / f"{name}.bin", dtype=dtype, mode="r", shape=(retained_edges,))
                for start in range(0, retained_edges, 2_000_000):
                    final[start:start + 2_000_000] = raw[start:start + 2_000_000]
                del raw
            final.flush()
            del final

    arrays = {
        "body_ids": body_ids, "nt": nt, "sign": sign, "positions": positions,
        "types": _strings(neurons, "type"), "classes": _strings(neurons, "class"),
        "superclasses": _strings(neurons, "superclass"), "subclasses": _strings(neurons, "subclass"),
        "sides": sides, "instances": _strings(neurons, "instance"), "nt_confidence": confidence,
    }
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array, allow_pickle=False)
    metadata = {
        "cache_format": 1, "release": RELEASE if verify_sources else "custom-unverified",
        "source_release": RELEASE, "release_date": "2026-06-08", "sources_verified": verify_sources,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_page": "https://male-cns.janelia.org/download/", "license": "CC-BY-4.0",
        "sources": provenance, "n_neurons": n, "n_edges": retained_edges,
        "n_annotation_rows": len(annotations), "n_excluded_annotation_rows": int((~included).sum()),
        "n_unclassified_traced_retained": int((~classified & included).sum()),
        "n_isolated_neurons": int((~active_neurons).sum()), "n_autapses": autapses,
        "n_source_segment_edges": source_edges, "n_excluded_segment_edges": source_edges - retained_edges,
        "n_source_synapses": source_synapses, "n_retained_synapses": retained_synapses,
        "neuron_inclusion": "nonempty superclass OR status Traced; exclude status Glia/Unimportant; retain isolates",
        "edge_inclusion": "both endpoints retained; every positive synapse-count edge; retain autapses; no pruning",
        "weight_units": "anatomical synapse count", "position_units": "micrometre",
        "position_source": "somaLocation xyz, MaleCNS EM 8nm voxel coordinates multiplied by 0.008; missing NaN",
        "n_missing_positions": int(np.count_nonzero(~np.isfinite(positions).all(axis=1))),
        "nt_source": "official consensus_nt; nt_confidence is body predicted_nt_confidence, not confidence of consensus",
        "nt_sign_assumption": NT_SIGN_ASSUMPTION,
        "histamine_sign_references": NT_SIGN_REFERENCES,
        "nt_counts": _counts(nt), "nt_sign_counts": _counts(sign),
        "superclass_counts": _counts(arrays["superclasses"]), "class_counts": _counts(arrays["classes"]),
        "sides_source": "rootSide L/R when available, otherwise somaSide; unknown/M not assigned to a hemisphere",
        "excluded_annotation_status_counts": _counts(_strings(annotations.loc[~included], "status")),
        "scope_caveat": "annotated-neuron CNS induced graph, not every raw segmentation fragment; includes brain and VNC; anatomy does not specify neuronal dynamics",
    }
    _write_metadata(output, metadata)
    return load_connectome(output)


@dataclass
class IOSelection:
    sensory_left: np.ndarray
    sensory_right: np.ndarray
    motor_left: np.ndarray
    motor_right: np.ndarray
    metadata: dict

    def as_dict(self) -> dict:
        return {name: getattr(self, name).tolist() for name in ("sensory_left", "sensory_right", "motor_left", "motor_right")}


def select_io(
    connectome: Connectome,
    sensory_patterns: list[str] | None = None,
    motor_patterns: list[str] | None = None,
    *,
    sensory_body_ids: list[int] | None = None,
    motor_body_ids: list[int] | None = None,
) -> IOSelection:
    """Choose bilateral annotated sensory and descending populations.

    Defaults: mechanosensory_proprioceptive sensory neurons and all descending
    neurons. Patterns search type/instance (case insensitive). Explicit IDs
    override patterns/default selection; IDs and bilateral coverage are checked.
    No inferred anatomical links, random groups, or left/right coordinate guesses.
    """
    c = connectome

    def choose(patterns, ids, default):
        if ids is not None:
            unknown = sorted(set(ids) - set(c.body_ids.tolist()))
            if unknown:
                raise ValueError(f"I/O body IDs absent from graph: {unknown[:10]}")
            return np.isin(c.body_ids, ids)
        if patterns is not None:
            if not patterns:
                raise ValueError("I/O patterns cannot be empty")
            pattern = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
            return np.fromiter((bool(pattern.search(f"{t} {i}")) for t, i in zip(c.types, c.instances)), dtype=bool, count=c.n)
        return default

    sensory = choose(sensory_patterns, sensory_body_ids, c.classes == "mechanosensory_proprioceptive")
    motor = choose(motor_patterns, motor_body_ids, c.superclasses == "descending_neuron")
    arrays = {
        f"{role}_{side_name}": np.flatnonzero(mask & (c.sides == side)).astype(np.int32)
        for role, mask in [("sensory", sensory), ("motor", motor)]
        for side_name, side in [("left", "L"), ("right", "R")]
    }
    empty = [name for name, values in arrays.items() if not len(values)]
    if empty:
        raise ValueError(f"Empty anatomical I/O groups {empty}; provide bilateral annotated type patterns/body IDs")
    if np.any(sensory & motor):
        raise ValueError("Sensory and motor I/O populations overlap")
    metadata = {
        "sensory_default": "class mechanosensory_proprioceptive",
        "motor_default": "superclass descending_neuron",
        "sensory_patterns": sensory_patterns, "motor_patterns": motor_patterns,
        "sensory_body_ids_override": sensory_body_ids, "motor_body_ids_override": motor_body_ids,
        "n_sensory_side_unresolved": int(np.count_nonzero(sensory & ~np.isin(c.sides, ["L", "R"]))),
        "n_motor_side_unresolved": int(np.count_nonzero(motor & ~np.isin(c.sides, ["L", "R"]))),
        "counts": {name: len(values) for name, values in arrays.items()},
        "body_ids": {name: c.body_ids[values].tolist() for name, values in arrays.items()},
        "caveat": "population-level sensor encoding and descending rate decoding are engineering hypotheses; no identified joint or locomotor circuit mapping",
    }
    return IOSelection(**arrays, metadata=metadata)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download", help="Fetch and verify the public v1.0 Feather files (~1.1GB)")
    fetch.add_argument("--raw-dir", default="data/raw")
    prep = sub.add_parser("preprocess", help="Build the full annotated CNS sparse cache")
    prep.add_argument("--raw-dir", default="data/raw")
    prep.add_argument("--output", default="data/cache")
    inspect = sub.add_parser("inspect", help="Print provenance and I/O group counts")
    inspect.add_argument("path", nargs="?", default="data/cache")
    args = parser.parse_args(argv)
    if args.command == "download":
        print(json.dumps(download(args.raw_dir), indent=2))
    elif args.command == "preprocess":
        print(json.dumps(preprocess(args.raw_dir, args.output).metadata, indent=2))
    else:
        graph = load_connectome(args.path)
        print(json.dumps(graph.metadata, indent=2))
        print(json.dumps(select_io(graph).metadata["counts"], indent=2))


if __name__ == "__main__":
    main()
