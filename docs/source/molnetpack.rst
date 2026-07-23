PyPI package: installation and inference
========================================

Using 3DMolMS through ``molnetpack`` requires minimal coding. This page covers installation and inference with the pre-trained models; to train your own models in Python, see :doc:`molnetpack_training`. If you prefer command-line scripts, see the :doc:`sourcecode` page.

Installing from PyPI
--------------------

3DMolMS is available on PyPI as the package ``molnetpack``. Install the latest version with ``pip``:

.. code-block:: bash

   pip install molnetpack

PyTorch must be installed separately. Check the `official PyTorch website <https://pytorch.org/get-started/locally/>`_ for the right version for your system, for example:

.. code-block:: bash

   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Using ``molnetpack`` for MS/MS prediction
-----------------------------------------

Sample inputs are at ``./examples/input_msms.csv`` and ``./examples/input_msms.mgf``. See :doc:`supported_formats` for the supported input/output formats; unsupported molecules are skipped automatically on load.

Instantiate a ``MolNet`` and load a CSV or MGF file with ``load_data``:

.. autofunction:: molnetpack.MolNet.load_data

Then predict the spectra with ``pred_msms``. Results are saved to the given path (MGF by default, or CSV if the filename ends in ``.csv``):

.. autofunction:: molnetpack.MolNet.pred_msms

For example:

.. code-block:: python

   import torch
   from molnetpack import MolNet, plot_msms

   device = torch.device("cpu")   # or torch.device(f"cuda:{gpu_index}") for GPU
   molnet_engine = MolNet(device, seed=42)

   molnet_engine.load_data(path_to_test_data='./examples/input_msms.csv')
   pred_spectra_df = molnet_engine.pred_msms(instrument='qtof')

Plot predicted MS/MS
--------------------

Visualize a predicted spectrum with ``plot_msms``:

.. autofunction:: molnetpack.plot_msms

For example:

.. code-block:: python

   # Plot the predicted MS/MS with its 3D molecular conformation
   plot_msms(pred_spectra_df, dir_to_img='./img/')

.. figure:: https://raw.githubusercontent.com/JosieHong/3DMolMS/main/img/demo_0.png
   :width: 600
   :align: center

Using ``molnetpack`` for properties prediction
----------------------------------------------

Instantiate ``MolNet`` first:

.. code-block:: python

   import torch
   from molnetpack import MolNet

   device = torch.device("cpu")   # or torch.device(f"cuda:{gpu_index}") for GPU
   molnet_engine = MolNet(device, seed=42)

Retention time (RT)
~~~~~~~~~~~~~~~~~~~~

Use ``pred_rt`` after instantiating ``MolNet``. The model is trained on METLIN-SMRT, so predictions are under the same experimental conditions as that dataset.

.. autofunction:: molnetpack.MolNet.pred_rt

For example:

.. code-block:: python

   molnet_engine.load_data(path_to_test_data='./examples/input_rt.csv')
   rt_df = molnet_engine.pred_rt()

Collision cross section (CCS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``pred_ccs`` after instantiating ``MolNet``:

.. autofunction:: molnetpack.MolNet.pred_ccs

For example:

.. code-block:: python

   molnet_engine.load_data(path_to_test_data='./examples/input_ccs.csv')
   ccs_df = molnet_engine.pred_ccs()

Molecular feature embedding
---------------------------

Use ``save_features`` to extract encoder embeddings for downstream tasks:

.. autofunction:: molnetpack.MolNet.save_features

For example:

.. code-block:: python

   molnet_engine.load_data(path_to_test_data='./examples/input_savefeat.csv')
   ids, features = molnet_engine.save_features()
   print('Titles:', ids)
   print('Features shape:', features.shape)
