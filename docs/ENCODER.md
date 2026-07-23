# 3DMolMS encoder (`MolConv`)

The encoder maps a 3D molecular point cloud — up to `max_atom_num` atoms, each a
21-d vector whose first 3 dims are `xyz` and remaining 18 are one-hot atom
features — to a fixed embedding that a task head (MS/MS, RT, CCS) reads. A
correct encoder must be **permutation-invariant** (atom ordering is arbitrary)
and invariant to **rigid motions** of the conformer (its absolute pose in space
carries no chemistry).

There are two versions in the codebase: **v1** (`MolConv1`, legacy) and **v2**
(`MolConv2`, current/default). Select with `encoder_version: 1|2` in the model
config. The bare name **`MolConv`** is a soft link to `MolConv2`, so existing
code that imports `MolConv` transparently gets the current default (v2).

## Summary

| | **v1** (`MolConv1`) | **v2** (`MolConv2`) |
|---|---|---|
| First-layer Gram matrix | absolute `⟨xⱼ, xₗ⟩`, **uncentered** | relative `⟨xⱼ−xᵢ, xₗ−xᵢ⟩` on the **centered** frame |
| Padding atoms in kNN | included (sit at the origin) | **masked out** of real atoms' neighborhoods |
| Coordinate centering | none | on the real-atom centroid |
| Normalization | `BatchNorm2d` | `LayerNorm` |
| Rotation invariant | ✅ | ✅ |
| Reflection invariant | ✅ | ✅ |
| **Translation invariant** | ❌ | ✅ |
| Permutation invariant | ✅ | ✅ |
| **Symmetry group** | **O(3) + permutation** | **E(3) + permutation** |
| Params (RT model) | 11.7 M | 21.5 M |
| RT R² (from scratch) | 0.617 | 0.607 |
| CCS R² (from scratch) | 0.935 | **0.964** |
| MS/MS cosine | ≈ 0.749 † | **0.7493** |
| Status | legacy | **current / recommended** |

The invariance rows are measured, not claimed — see [Invariance
verification](#invariance-verification).

## v1 — `MolConv1` (legacy)

The first layer builds its Gram matrix from **absolute atom positions**
`⟨xⱼ, xₗ⟩` (via `graph_feat @ graph_featᵀ`) using the raw, uncentered
coordinates, and its k-nearest-neighbour graph includes the zero-padding atoms
(which all sit at the origin). Both choices make the output depend on **where the
molecule sits in space**:

- A rigid **translation** of the conformer changes the absolute positions and the
  real→padding distances, so the embedding changes. Measured shift: ~**11–18 %**
  of the output magnitude under a modest translation.
- Rotation, reflection and permutation *are* preserved (inner products and
  distances are orthogonally invariant; padding-at-origin is rotation-symmetric).

So v1 is **O(3) + permutation invariant but not translation-invariant** — it does
**not** satisfy SE(3)/E(3). Because conformers are stored pre-centered this
fragility is mostly latent in practice, but it is a real correctness gap (a shift
of the input silently changes the prediction). v1 is retained only for
reproducing legacy checkpoints.

## v2 — `MolConv2` (current)

v2 makes the geometry genuinely pose-independent:

1. **Relative-displacement Gram** `⟨xⱼ−xᵢ, xₗ−xᵢ⟩` — encodes the local angle
   `∠jik`, which is invariant to rotation and reflection.
2. **Center on the real-atom centroid** before the first-layer geometry. The
   centroid tracks any input translation, so the centered coordinates (and all
   distances) are identical under a shift → translation invariance becomes exact.
3. **Mask zero-padding atoms out of real atoms' kNN.** Once centered, padding
   sits at the centroid and would otherwise pollute central atoms' neighborhoods
   and leak a pose-dependent term through the atom-dimension normalization;
   masking keeps padding query rows inert.

The result is **full E(3) + permutation invariance** (rotation + reflection +
translation), verified to floating-point-exact precision in `float64`. v2 is the
production encoder for all released checkpoints.

## Performance (from scratch, lr 1e-4 for RT/CCS, paired)

| Task | v1 | v2 |
|---|---:|---:|
| **RT** (METLIN) R² | 0.617 | 0.607 |
| **CCS** (AllCCS) R² | 0.935 | **0.964** |
| **MS/MS** (qtof) cosine | ≈ 0.749 † | **0.7493** |

- **CCS:** v2 wins clearly (+0.03 R²) — v1's simpler architecture (BatchNorm, no
  `center_ff`, ~half the params) underperforms on the shape-driven CCS target.
- **RT:** essentially tied (within ~±0.02 run-to-run noise).
- **MS/MS:** insensitive to the encoder details; fragmentation is local.
- **Bottom line:** v2 matches or beats v1 on task accuracy **and** is the only one
  that is actually E(3)-invariant. The choice is correctness, at no accuracy cost.

> † v2 MS/MS = 0.7493 is the converged campaign qtof result. v1 MS/MS was not run
> to full convergence (~a day of compute for a metric that is encoder-insensitive);
> given MS/MS depends on local fragmentation rather than global geometry, v1 is
> expected to match. The encoder-dependent differences live in CCS and RT.

## Chirality: E(3) (default) vs SE(3)

Both v1 and v2 are **reflection-invariant** by default, so they give a molecule
and its mirror image (enantiomer) the **identical** embedding — they cannot tell
enantiomers apart. This is exactly **correct for the achiral 3DMolMS tasks**
(MS/MS, RT, CCS), where enantiomers share the target.

For a **chirality-dependent** task — e.g. **3DMolCSP** (chiral stationary-phase
separation), where enantiomers elute differently — set **`chirality: true`** on
v2. It appends a **signed-volume pseudoscalar** channel `dⱼ·(d₁×d₂)` in the first
layer, which is invariant under proper rotation and translation but **flips sign
under reflection**, so the encoder becomes **SE(3)** (reflection-*sensitive*).

| | `chirality: false` (default) | `chirality: true` |
|---|---|---|
| Symmetry group | **E(3)** | **SE(3)** |
| Rotation | invariant | invariant |
| Translation | invariant | invariant |
| **Reflection** | **invariant** (Δ = 0, exact) | **sensitive** (Δ ≈ 0.4) |
| Enantiomers (mirror images) | identical prediction | **can differ** |
| Correct for | achiral tasks: MS/MS, RT, CCS | chiral tasks: e.g. 3DMolCSP |

The reflection responses above are measured on random-weight layers (see
`test_v2_chirality_is_se3_reflection_sensitive`): `false` gives an *exactly zero*
reflection change, `true` gives a clearly non-zero one while keeping rotation and
translation invariant. An E(3) encoder is **provably unable** to model enantiomer
differences (it must output one value for both), so SE(3) is *required* for
chiral separation — not optional.

## Invariance verification

Both properties are checked with random-weight layers (invariance is
architectural, so no trained checkpoint is needed):

- `tests/test_encoder_invariance.py` — layer-level rotation/reflection/
  translation/permutation checks for v1, v2, and the `chirality` (SE(3)) mode.
- `tests/test_se3_invariance.py` — full-encoder SE(3) invariance in `float64`
  and `float32`, including the at-origin regression test for the padding mask.

```bash
pytest tests/test_encoder_invariance.py tests/test_se3_invariance.py -q
```

## Configuration

```yaml
model:
  encoder_version: 2   # 1 = MolConv1 (legacy O(3)); 2 = MolConv2 (E(3), default; also aliased as MolConv)
  chirality: false     # true -> SE(3), reflection-sensitive (chiral tasks, e.g. 3DMolCSP)
```

Both versions have identical `state_dict` key layout, so switching
`encoder_version` changes the forward computation, **not** the parameter shapes —
but a checkpoint must be evaluated with the same settings it was trained under.
