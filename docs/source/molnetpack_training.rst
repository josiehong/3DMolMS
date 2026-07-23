PyPI package: training
======================

``MolNet`` can train all task types (MS/MS, RT, CCS) directly from Python, without the command-line scripts. For CLI-based training, see :doc:`usage/index`.

.. autofunction:: molnetpack.MolNet.train

.. autofunction:: molnetpack.MolNet.evaluate

MS/MS model training
--------------------

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

   # The trained model is ready for inference immediately — no reload needed
   molnet_engine.load_data('./examples/input_msms.csv')
   pred_df = molnet_engine.pred_msms(instrument='qtof')

Evaluate predictions against ground truth:

.. code-block:: python

   results_df = molnet_engine.evaluate(
       test_pkl='./data/qtof_etkdgv3_test.pkl',
       pred_mgf='./result/pred_qtof_etkdgv3_test.mgf',
       result_path='./eval_qtof_etkdgv3_test.csv',
       plot_path='./eval_qtof_etkdgv3_test.png',
   )

Retention time model training
-----------------------------

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
------------------

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
