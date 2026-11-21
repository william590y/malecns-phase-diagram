"""Meaningful ingestion checks using small public-schema Feather fixtures."""
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pytest

from malecns.data import NT_SIGN, SOURCE_FILES, load_connectome, preprocess, select_io


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    rows = [
        # Classified sensory neuron without Traced status is intentionally kept.
        (10, "vnc_sensory", None, "SNppL", "mechanosensory_proprioceptive", None, "L", [100, 200, 300]),
        (20, "vnc_sensory", "Traced", "SNppR", "mechanosensory_proprioceptive", None, "R", None),
        (30, "descending_neuron", "Traced", "DNa02_L", None, "L", None, [200, 300, 400]),
        (40, "descending_neuron", "Traced", "DNa02_R", None, "R", None, None),
        # Traced unclassified isolate must survive the edge filtering operation.
        (50, None, "Traced", None, None, None, None, None),
        (60, None, "Glia", None, None, None, None, None),
        (70, None, "Orphan", None, None, None, None, None),
    ]
    annotation = pd.DataFrame(rows, columns=["bodyId", "superclass", "status", "type", "class", "somaSide", "rootSide", "somaLocation"])
    annotation["instance"] = annotation["type"]
    feather.write_feather(annotation, root / SOURCE_FILES["annotations"])
    nt = pd.DataFrame({"body": [10, 20, 30, 40], "consensus_nt": ["acetylcholine", "gaba", "glutamate", "dopamine"], "predicted_nt_confidence": [0.9, 0.8, 0.7, 0.6]})
    feather.write_feather(nt, root / SOURCE_FILES["neurotransmitters"])
    edges = pd.DataFrame({"body_pre": [10, 20, 30, 40, 60, 70, 999], "body_post": [30, 40, 30, 10, 10, 20, 30], "weight": [1, 2, 3, 4, 5, 6, 7]})
    # Small record batches ensure the streaming path crosses batch boundaries.
    feather.write_feather(edges, root / SOURCE_FILES["weights"], chunksize=2)
    return root


def test_complete_annotated_graph_and_provenance(raw, tmp_path):
    graph = preprocess(raw, tmp_path / "cache", verify_sources=False)
    np.testing.assert_array_equal(graph.body_ids, [10, 20, 30, 40, 50])
    assert graph.n_edges == 4
    np.testing.assert_array_equal(graph.weight, [1, 2, 3, 4])
    np.testing.assert_array_equal(graph.pre, [0, 1, 2, 3])
    np.testing.assert_array_equal(graph.post, [2, 3, 2, 0])
    assert graph.metadata["n_isolated_neurons"] == 1
    assert graph.metadata["n_autapses"] == 1
    assert graph.metadata["n_source_segment_edges"] == 7
    assert graph.metadata["n_excluded_segment_edges"] == 3
    assert graph.metadata["n_excluded_annotation_rows"] == 2
    assert graph.metadata["n_retained_synapses"] == 10
    assert graph.metadata["n_unclassified_traced_retained"] == 1
    assert graph.metadata["release"] == "custom-unverified"
    assert graph.metadata["sources_verified"] is False
    assert len(graph.metadata["sources"]["weights"]["sha256"]) == 64
    assert graph.metadata["weight_units"] == "anatomical synapse count"


def test_coordinates_nt_and_io_are_anatomically_explicit(raw, tmp_path):
    graph = preprocess(raw, tmp_path / "cache", verify_sources=False)
    np.testing.assert_allclose(graph.positions[0], [0.8, 1.6, 2.4])
    assert np.isnan(graph.positions[1]).all()
    np.testing.assert_array_equal(graph.sign, [1, -1, -1, 0, 0])
    assert NT_SIGN["histamine"] == -1
    assert graph.nt[-1] == "unknown"
    selection = select_io(graph)
    np.testing.assert_array_equal(selection.sensory_left, [0])
    np.testing.assert_array_equal(selection.sensory_right, [1])
    np.testing.assert_array_equal(selection.motor_left, [2])
    np.testing.assert_array_equal(selection.motor_right, [3])
    overridden = select_io(graph, motor_patterns=["^DNa02"])
    np.testing.assert_array_equal(overridden.motor_right, selection.motor_right)
    with pytest.raises(ValueError, match="absent"):
        select_io(graph, motor_body_ids=[999])
    with pytest.raises(ValueError, match="Empty anatomical"):
        select_io(graph, motor_patterns=["not_a_neuron"])
    with pytest.raises(ValueError, match="overlap"):
        select_io(graph, motor_body_ids=[10, 20])


def test_pinned_sources_reject_custom_data(raw, tmp_path):
    with pytest.raises(ValueError, match="checksum"):
        preprocess(raw, tmp_path / "cache")


def test_bad_counts_rejected(raw, tmp_path):
    feather.write_feather(pd.DataFrame({"body_pre": [10], "body_post": [30], "weight": [-2]}), raw / SOURCE_FILES["weights"])
    with pytest.raises(ValueError, match="positive integral"):
        preprocess(raw, tmp_path / "cache", verify_sources=False)


def test_corrupt_cache_dimensions_rejected(raw, tmp_path):
    path = tmp_path / "cache"
    preprocess(raw, path, verify_sources=False)
    np.save(path / "positions.npy", np.zeros((5, 2)))
    with pytest.raises(ValueError, match="positions"):
        load_connectome(path)
