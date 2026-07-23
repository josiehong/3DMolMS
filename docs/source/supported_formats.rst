Supported formats
==================

3DMolMS reads molecules for **inference** from three file types — **CSV**, **MGF**,
and **PKL** — through ``MolNet.load_data``. **SDF** is supported for **training /
reference-set preparation** via the preprocessing scripts. Predictions are written
back as **MGF** (MS/MS) or **CSV** (RT / CCS).

Every input must satisfy the model's molecular limits:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Supported values
   * - Atom count
     - ≤ 300 (including hydrogens)
   * - Atom types
     - C, O, N, H, P, S, F, Cl, B, Br, I
   * - Precursor types
     - ``[M+H]+``, ``[M-H]-``, ``[M+H-H2O]+``, ``[M+Na]+``, ``[M+2H]2+``
   * - Collision energy
     - any number (e.g. ``40 V`` or ``40``)

Molecules that fall outside these limits — too many atoms, an unlisted element, an
unparseable SMILES, or an unlisted precursor type — are **silently skipped** during
loading.

Input formats
-------------

Which fields are required depends on the task:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Task
     - Requires
   * - MS/MS (``pred_msms``)
     - SMILES + precursor type + collision energy
   * - CCS (``pred_ccs``)
     - SMILES + precursor type
   * - RT (``pred_rt``) / features (``save_features``)
     - SMILES

CSV
~~~

A header row followed by one molecule per line. Column names are **case-sensitive**:

.. code-block:: text

   ID,SMILES,Precursor_Type,Collision_Energy
   demo_0,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,40 V

- ``ID`` — molecule identifier, used as the result title. **Required.**
- ``SMILES`` — the molecule structure. **Required.**
- ``Precursor_Type`` — adduct; one of the supported precursor types. Needed for
  MS/MS and CCS.
- ``Collision_Energy`` — e.g. ``40 V`` (the unit is optional). Needed for MS/MS.

Omit the columns a task does not use — ``ID,SMILES`` alone is enough for RT or
``save_features``. See ``examples/input_msms.csv``, ``examples/input_ccs.csv``, and
``examples/input_savefeat.csv``.

MGF
~~~

One ``BEGIN IONS`` … ``END IONS`` block per molecule. The parameters below are read
(keys are case-insensitive); peak lines are **optional** for prediction:

.. code-block:: text

   BEGIN IONS
   TITLE=demo_0
   SMILES=C/C(=C\CNc1nc[nH]c2ncnc1-2)CO
   PRECURSOR_TYPE=[M+H]+
   COLLISION_ENERGY=40 V
   END IONS

Only ``TITLE``, ``SMILES``, ``PRECURSOR_TYPE`` and ``COLLISION_ENERGY`` are used;
other fields (``PRECURSOR_MZ``, ``CHARGE``, peak lists, …) are ignored on input.
See ``examples/input_msms.mgf``.

SDF
~~~

Used for **preparing training or reference sets in bulk** (e.g. METLIN for RT, HMDB
for a reference library) via the preprocessing scripts (``scripts/preprocess.py``,
``scripts/hmdb2pkl.py``, ``scripts/refmet2pkl.py``). These read each SDF molecule
block and its properties (SMILES and task labels such as retention time) and emit a
PKL. SDF is **not** a direct ``MolNet.load_data`` inference input — convert it to a
PKL first.

PKL
~~~

The preprocessed, ready-to-run format: a pickled ``list`` of dicts. It is the
fastest input because the 3D conformation has already been computed (CSV and MGF
inputs are converted to this on load).

.. code-block:: python

   [
     {
       "title":  "demo_0",                  # str  — molecule id
       "smiles": "C/C(=C\\CNc1...)CO",      # str
       "mol":    np.ndarray,                 # [max_atom_num, 21] — 3D conformation
       "env":    np.ndarray,                 # collision-energy + precursor-type context
       "spec":   np.ndarray,                 # binned reference spectrum (training only)
     },
     ...
   ]

- ``mol`` — the 3D point cloud: dims 0–2 are the centered ``xyz`` coordinates,
  3–20 are per-atom attributes and the atom-type one-hot.
- ``env`` — the normalized collision energy plus the precursor-type one-hot
  (present for MS/MS and CCS).
- ``spec`` — the binned reference spectrum; needed only for training / evaluation,
  not for prediction.

Output formats
--------------

MS/MS — MGF
~~~~~~~~~~~

``pred_msms`` writes one ``BEGIN IONS`` block per molecule with the predicted
m/z–intensity peak list, alongside ``TITLE``, ``SMILES``, ``PRECURSOR_TYPE`` and
``COLLISION_ENERGY``:

.. code-block:: text

   BEGIN IONS
   TITLE=demo_0
   SMILES=C/C(=C\CNc1nc[nH]c2ncnc1-2)CO
   PRECURSOR_TYPE=[M+H]+
   COLLISION_ENERGY=39.98
   41.00000 39.8
   43.00000 172.5
   ...
   END IONS

RT / CCS — CSV
~~~~~~~~~~~~~~

``pred_rt`` and ``pred_ccs`` return a ``pandas.DataFrame`` (and write a CSV) with
one row per molecule; the prediction column is ``Pred RT`` or ``Pred CCS``:

.. code-block:: text

   ,ID,SMILES,Precursor Type,Pred CCS
   0,demo_0,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,154.62
