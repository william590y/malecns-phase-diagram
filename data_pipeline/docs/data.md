# MaleCNS data and anatomical scope

The cache uses **male-cns:v1.0**, released June 8, 2026, from the [official release notes](https://male-cns.janelia.org/release/). The [official download page](https://male-cns.janelia.org/download/) provides public Feather files for curated annotations, neuron neurotransmitter predictions, and the complete segment connectivity table. The upstream data license is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the MaleCNS consortium when distributing derived data or figures; this software's license does not replace the dataset license.

Rebuild from the project directory:

```bash
python -m malecns.data download --raw-dir data/raw
python -m malecns.data preprocess --raw-dir data/raw --output data/cache
python -m malecns.data inspect data/cache
```

Downloads need approximately 1.1 GB. The cache is approximately 399 MB. Preprocessing streams Arrow record batches and writes sparse arrays; it never creates an N×N connectivity matrix or loads the entire 151 million row source graph into a pandas DataFrame. Temporary edge arrays need approximately another 307 MB. Source SHA256 values are pinned in `data.py` and checked on both download and preprocessing. The pins identify the files retrieved on September 9, 2026; they are integrity checks, not publisher signatures. Source replacement causes an explicit checksum error. Raw download provenance includes URLs, response headers, sizes and hashes; derived `metadata.json` records hashes, selection rules, units and counts.

## Actual full-cache counts

The completed cache at `data/cache` contains:

| Quantity | Count |
| --- | ---: |
| Annotated neurons retained | 167,216 |
| Directed neuron-to-neuron connections | 25,587,572 |
| Anatomical synaptic contacts represented | 124,193,283 |
| Isolated annotated neurons retained | 548 |
| Autaptic connections retained | 102 |
| Original annotation rows | 211,577 |
| Original segment-to-segment edges | 151,856,684 |
| Original synaptic contacts | 311,833,243 |
| Excluded annotation rows | 44,361 |
| Edges involving excluded/unannotated segments | 126,269,112 |

This is the **complete annotated-neuron CNS induced graph under the stated inclusion rule**, including the brain and ventral nerve cord (VNC). A segmentation fragment is not automatically a neuron. We retain every row with a nonempty `superclass` or `status == Traced`, except explicitly `Glia` or `Unimportant`. This includes 516 traced but unclassified neurons. Excluded annotation statuses comprise 11,864 Glia, 10,751 Unimportant, 15,893 Orphan, 1,832 Assign, 551 Anchor and 3,470 without a status. Classified rows without Traced status are retained; they should not be interpreted as uniformly proofread neurons.

Both endpoints must be retained neurons for an edge to enter the model. Every such edge survives, including weight-one edges and autapses. There is no top-k pruning, minimum-synapse threshold, degree truncation or sampling in this cache. Isolated neurons survive even if they have no modeled synapses. The source has already applied its published `minconf-0.5` synapse confidence filter. We do not call this all raw segmented tissue, every biological neuron in the animal, or a physiological reconstruction.

Major superclasses are optic-lobe intrinsic (89,403), central-brain intrinsic (32,164), VNC intrinsic (13,161), visual projection (9,201), VNC sensory (6,370), optic-lobe sensory (6,098), central-brain sensory (4,868), ascending (1,846), descending (1,314) and VNC motor (708). All subclass counts are in the cache metadata.

## Arrays and units

`Connectome.load(path)` / `load_connectome(path)` load memory-mapped `.npy` arrays without Python pickle. Body IDs are int64 and sorted. `pre` and `post` are uint32 indices into `body_ids`, **not original body IDs**. `weight` is float32 anatomical contact count, represented exactly for these data, **not conductance or voltage**. The neural model is responsible for an explicitly documented conversion to physiological units.

`positions` uses annotated `somaLocation` x/y/z coordinates in the native MaleCNS EM frame, multiplying the 8 nm voxel units by 0.008 to obtain micrometres. Missing somata remain NaN (27,182 neurons). These are soma locations, not synapse centroids or arbitrary synthetic positions; they do not trace the shape of a neuron. `types`, `instances`, `classes`, `superclasses` and `subclasses` preserve the corresponding published annotation fields, with missing strings represented by `""`.

## Neurotransmitters and uncertainty

Each neuron is joined by `bodyId` ↔ neurotransmitter-table `body`, using the **official `consensus_nt`**. We do not silently replace it with the maximum-confidence individual or cell-type prediction. The table also contains body-level `predicted_nt`, its `predicted_nt_confidence`, cell-type predictions and their confidence, and `ground_truth`; these are not interchangeable.

`nt_confidence` preserves **body-level `predicted_nt_confidence`**, which may disagree with the assignment selected in `consensus_nt`. It must not be displayed as confidence in the consensus. The value is NaN if unavailable. No additional confidence cutoff is imposed. The raw file remains available for alternative confidence- or receptor-aware models.

| Consensus neurotransmitter | Neurons | Model output sign |
| --- | ---: | ---: |
| Acetylcholine | 103,746 | +1 |
| GABA | 22,073 | −1 |
| Glutamate | 29,303 | −1 |
| Histamine | 7,897 | −1 |
| Dopamine | 392 | 0 |
| Octopamine | 101 | 0 |
| Serotonin | 48 | 0 |
| Unclear | 3,142 | 0 |
| Unknown/missing | 514 | 0 |

This sign map is a simplified current-based model assumption. Real effects depend on postsynaptic receptors and cellular context; glutamate cannot be universally interpreted as inhibitory throughout the CNS and neuromuscular system. Histamine-mediated chloride currents at fly photoreceptor synapses are supported by [Gengs et al. (2002)](https://pubmed.ncbi.nlm.nih.gov/12196539/) and [Pantazis et al. (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6670387/); applying a uniform histamine sign throughout the network remains a simplification. In total, 4,197 neurons have modeled output sign zero, including neuromodulatory and uncertain assignments. Their anatomical edges remain stored, but their fast signed output does not affect the model. Neuromodulation, electrical synapses, receptor dynamics, individual cell physiology, plasticity and graded transmission are not recovered from this source graph. Simulating all neurons as spiking point neurons especially simplifies the fly visual system, where graded signaling is important.

## Anatomical sensor and motor selection

`select_io(connectome)` selects `class == mechanosensory_proprioceptive` for body feedback, and `superclass == descending_neuron` for a bilateral population-rate motor readout. Side is the annotated L/R `rootSide` where present, otherwise `somaSide`. Unknown and midline annotations are not guessed from coordinates.

The actual default groups are 735 sensory left, 718 sensory right, 656 descending left, and 648 descending right. One sensory and ten descending neurons have unresolved sides and remain in the network but outside bilateral I/O. Both sides must be nonempty, and sensory and motor groups cannot overlap. Selection records every chosen original body ID in the run provenance.

For specific hypotheses, `sensory_patterns`/`motor_patterns` search type and instance with regular expressions, or callers can provide `sensory_body_ids`/`motor_body_ids`. Unknown explicit IDs and empty groups fail clearly. For example, `motor_patterns=["^DNa02"]` finds body 523769 on the left and body 10360 on the right in this release. Selecting a named descending cell does not by itself validate a locomotor decoder.

The FlyGym body-feedback encoding and bilateral descending-rate decoding are **engineering hypotheses**. The source does not provide joint-by-joint muscles, sensory transduction functions, central-pattern-generator parameters or a proven mapping from those descending populations to forward speed/turning. FlyGym's locomotor controller supplies that missing motor transformation. The retained VNC motor neurons are simulated, but their individual synapses do not directly actuate reconstructed muscles. Movement produced by this system demonstrates a connectome-driven closed loop under model assumptions, not a validated digital organism.

## Verification

`tests/test_data.py` verifies streaming with multiple Arrow batches, exact original body-ID mapping, weight-one edges, autapse and isolate retention, glia/fragment exclusion, NT uncertainty handling, coordinate units, bilateral type-based selection, absent body-ID errors, overlapping I/O rejection, corrupt-cache dimensions and source-integrity rejection. Five test cases passed in the project environment. Full real preprocessing completed successfully with pinned checksums and the counts above; `data/preprocess.log` records its batch progress and final metadata.
