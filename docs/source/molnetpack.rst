PyPI package installation and usage
====================================

Using 3DMolMS through ``molnetpack`` requires minimal coding and is easy to use. You can use it for both inference with pre-trained models and for training your own models entirely in Python. If you prefer command-line scripts, please refer to the :doc:`./sourcecode` page.

Installing from PyPI
--------------------

3DMolMS is available on PyPI as the package ``molnetpack``. You can install the latest version using ``pip``:

.. code-block:: bash

   pip install molnetpack

PyTorch must be installed separately. Check the `official PyTorch website <https://pytorch.org/get-started/locally/>`_ for the proper version for your system. For example:

.. code-block:: bash

   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Using ``molnetpack`` for MS/MS prediction
-----------------------------------------

The sample input files, a CSV and an MGF, are located at ``./examples/demo_input.csv`` and ``./examples/demo_input.mgf``, respectively. It's important to note that during the data loading phase, any input formats that are not supported will be automatically excluded. Below is a table outlining the types of input data that are supported:

.. list-table::
   :header-rows: 1

   * - Item
     - Supported input
   * - Atom number
     - <=300
   * - Atom types
     - 'C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I'
   * - Precursor types
     - '[M+H]+', '[M-H]-', '[M+H-H2O]+', '[M+Na]+', '[M+2H]2+'
   * - Collision energy
     - any number

To get started quickly, you can instantiate a MolNet and load a CSV or MGF file for MS/MS prediction using ``load_data`` function:

.. autofunction:: molnetpack.MolNet.load_data

Then predict the MS/MS spectra using ``pred_msms`` function. The predicted MS/MS spectra will be saved in the specified path. The default format is MGF, but you can also save it as a CSV file by specifying the file name with a ``.csv`` extension.

.. autofunction:: molnetpack.MolNet.pred_msms

For example: 

.. code-block:: python

   import torch
   from molnetpack import MolNet, plot_msms

   # Set the device to CPU for CPU-only usage:
   device = torch.device("cpu")

   # For GPU usage, set the device as follows (replace '0' with your desired GPU index):
   # gpu_index = 0
   # device = torch.device(f"cuda:{gpu_index}")

   # Instantiate a MolNet object
   molnet_engine = MolNet(device, seed=42) # The random seed can be any integer. 

   # Load input data (here we use a CSV file as an example)
   molnet_engine.load_data(path_to_test_data='./examples/input_msms.csv')
   
   # Predict MS/MS
   pred_spectra_df = molnet_engine.pred_msms(instrument='qtof')


Plot predicted MS/MS
--------------------

The predicted MS/MS spectra can be visualized using the ``plot_msms`` function: 

.. autofunction:: molnetpack.plot_msms

You may customize the plot by updating the source code directory, such as the size of the image and the color scheme. 

For example:

.. code-block:: python

   # Plot the predicted MS/MS with 3D molecular conformation
   plot_msms(pred_spectra_df, dir_to_img='./img/')

Below is an example of a predicted MS/MS spectrum plot.

.. figure:: https://raw.githubusercontent.com/JosieHong/3DMolMS/main/img/demo_0.png
   :width: 600
   :align: center

Using ``molnetpack`` for properties prediction
----------------------------------------------

Before doing any prediction, please intantiate ``MolNet``:

.. code-block:: python

   import torch
   from molnetpack import MolNet

   # Set the device to CPU for CPU-only usage:
   device = torch.device("cpu")

   # For GPU usage, set the device as follows (replace '0' with your desired GPU index):
   # gpu_index = 0
   # device = torch.device(f"cuda:{gpu_index}")

   # Instantiate a MolNet object
   molnet_engine = MolNet(device, seed=42) # The random seed can be any integer. 

RT prediction
~~~~~~~~~~~~~~

For RT prediction, please use ``pred_rt`` function after instantiating a MolNet object. Please note that since this model is trained on the METLIN-SMRT dataset, the predicted retention time is under the same experimental conditions as the METLIN-SMRT set.

.. autofunction:: molnetpack.MolNet.pred_rt

For example:

.. code-block:: python

   # Load input data
   molnet_engine.load_data(path_to_test_data='./examples/input_rt.csv')

   # Pred RT
   rt_df = molnet_engine.pred_rt()

CCS prediction
~~~~~~~~~~~~~~

For CCS prediction, please use ``pred_ccs`` function after instantiating a MolNet object. 

.. autofunction:: molnetpack.MolNet.pred_ccs

For example: 

.. code-block:: python

   # Load input data
   molnet_engine.load_data(path_to_test_data='./examples/input_ccs.csv')

   # Pred CCS
   ccs_df = molnet_engine.pred_ccs()

Molecular feature embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For saving the molecular embeddings, please use the following ``save_features`` function after instantiating a MolNet object. 

.. autofunction:: molnetpack.MolNet.save_features

For example: 

.. code-block:: python

   # Load input data
   molnet_engine.load_data(path_to_test_data='./examples/input_savefeat.csv')

   # Inference to get the features
   ids, features = molnet_engine.save_features()

   print('Titles:', ids)
   print('Features shape:', features.shape)

Training with ``molnetpack``
----------------------------

``MolNet`` supports training all task types (MS/MS, RT, CCS) directly from Python without using the command-line scripts.

.. autofunction:: molnetpack.MolNet.train

.. autofunction:: molnetpack.MolNet.evaluate

MS/MS model training
~~~~~~~~~~~~~~~~~~~~~

Fine-tune from a pretrained checkpoint:

.. code-block:: python

   import torch
   from molnetpack import MolNet

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   molnet_engine = MolNet(device, seed=42)

   best_cosine = molnet_engine.train(
       task='msms',
       train_data='./data/qtof_etkdgv3_train.pkl',
       valid_data='./data/qtof_etkdgv3_test.pkl',
       checkpoint_path='./check_point/molnet_qtof_etkdgv3_tl.pt',
       resume_path='./check_point/molnet_pre_etkdgv3.pt',
       transfer=True,
   )

   # The trained model is immediately ready for inference — no reload needed
   molnet_engine.load_data('./examples/input_msms.csv')
   pred_df = molnet_engine.pred_msms(instrument='qtof')

Evaluating predictions against ground truth:

.. code-block:: python

   results_df = molnet_engine.evaluate(
       test_pkl='./data/qtof_etkdgv3_test.pkl',
       pred_mgf='./result/pred_qtof_etkdgv3_test.mgf',
       result_path='./eval_qtof_etkdgv3_test.csv',
       plot_path='./eval_qtof_etkdgv3_test.png',
   )

Retention time model training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune from a pretrained MS/MS checkpoint using transfer learning:

.. code-block:: python

   import torch
   from molnetpack import MolNet

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   molnet_engine = MolNet(device, seed=42)

   best_mae = molnet_engine.train(
       task='rt',
       train_data='./data/metlin_etkdgv3_train.pkl',
       valid_data='./data/metlin_etkdgv3_test.pkl',
       checkpoint_path='./check_point/molnet_rt_etkdgv3_tl.pt',
       resume_path='./check_point/molnet_qtof_etkdgv3.pt',
       transfer=True,
       use_scaler=True,
   )

CCS model training
~~~~~~~~~~~~~~~~~~~

Fine-tune from a pretrained MS/MS checkpoint:

.. code-block:: python

   import torch
   from molnetpack import MolNet

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   molnet_engine = MolNet(device, seed=42)

   best_mae = molnet_engine.train(
       task='ccs',
       train_data='./data/allccs_etkdgv3_train.pkl',
       valid_data='./data/allccs_etkdgv3_test.pkl',
       checkpoint_path='./check_point/molnet_ccs_etkdgv3_tl.pt',
       resume_path='./check_point/molnet_qtof_etkdgv3.pt',
       transfer=True,
   )