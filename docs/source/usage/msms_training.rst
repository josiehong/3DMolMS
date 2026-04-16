Train your own model for MS/MS prediction
=========================================

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Obtain the Pretrained Model
---------------------------------------

Download the pretrained model (``molnet_pre_etkdgv3.pt.zip``) from `Releases <https://github.com/JosieHong/3DMolMS/releases>`_. You can also train the model from scratch. For details on pretraining the model on the `QM9 <https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904>`_ dataset, refer to :doc:`../advanced_usage/pretrain` page.

**Step 2**: Prepare the Datasets
--------------------------------

Download and organize the datasets into the ``./data/`` directory. The current version uses four datasets:

1. Agilent DPCL, provided by `Agilent Technologies <https://www.agilent.com/>`_.
2. `NIST20 <https://www.nist.gov/programs-projects/nist23-updates-nist-tandem-and-electron-ionization-spectral-libraries>`_, available under license for academic use.
3. `MoNA <https://mona.fiehnlab.ucdavis.edu/downloads>`_, publicly available.
4. Waters QTOF, our own experimental dataset.

The data directory structure should look like this:

.. code-block:: text

    |- data
      |- origin
        |- Agilent_Combined.sdf
        |- Agilent_Metlin.sdf
        |- hr_msms_nist.SDF
        |- MoNA-export-All_LC-MS-MS_QTOF.sdf
        |- MoNA-export-All_LC-MS-MS_Orbitrap.sdf
        |- waters_qtof.mgf

**Step 3**: Preprocess the Datasets
-----------------------------------

Run the following commands to preprocess the datasets. Specify the dataset with ``--dataset`` and select the instrument type as ``qtof``. Use ``--maxmin_pick`` to apply the MaxMin algorithm for selecting training molecules; otherwise, selection will be random. The dataset configurations are in ``./molnetpack/config/preprocess_etkdgv3.yml``.

.. code-block:: bash

    python scripts/preprocess.py --task msms \
    --dataset agilent nist mona waters gnps \
    --instrument_type qtof orbitrap \
    --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
    --mgf_dir ./data/mgf_debug/

**Step 4**: Train the Model
---------------------------

Use the following commands to train the model. Configuration settings for the model and training process are located in ``./molnetpack/config/molnet.yml``.

.. code-block:: bash

  # Train the model from pretrain:
  # Q-TOF:
  python scripts/train.py --task msms \
  --train_data ./data/qtof_etkdgv3_train.pkl \
  --test_data ./data/qtof_etkdgv3_test.pkl \
  --model_config_path ./molnetpack/config/molnet.yml \
  --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
  --checkpoint_path ./check_point/molnet_qtof_etkdgv3_tl.pt \
  --transfer \
  --resume_path ./check_point/molnet_pre_etkdgv3.pt
  # Orbitrap can be done in a similar way.

  # Train the model from scratch
  # Q-TOF:
  python scripts/train.py --task msms \
  --train_data ./data/qtof_etkdgv3_train.pkl \
  --test_data ./data/qtof_etkdgv3_test.pkl \
  --model_config_path ./molnetpack/config/molnet.yml \
  --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
  --checkpoint_path ./check_point/molnet_qtof_etkdgv3.pt \
  --ex_model_path ./check_point/molnet_qtof_etkdgv3_jit.pt --device 0

  # Orbitrap:
  python scripts/train.py --task msms \
  --train_data ./data/orbitrap_etkdgv3_train.pkl \
  --test_data ./data/orbitrap_etkdgv3_test.pkl \
  --model_config_path ./molnetpack/config/molnet.yml \
  --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
  --checkpoint_path ./check_point/molnet_orbitrap_etkdgv3.pt \
  --ex_model_path ./check_point/molnet_orbitrap_etkdgv3_jit.pt --device 0

**Step 5**: Evaluation
----------------------

Let's evaluate the model trained above!

.. code-block:: bash

  # Predict the spectra:
  # Q-TOF:
  python scripts/predict.py --task msms \
  --test_data ./data/qtof_etkdgv3_test.pkl \
  --model_config_path ./molnetpack/config/molnet.yml \
  --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
  --resume_path ./check_point/molnet_qtof_etkdgv3.pt \
  --result_path ./result/pred_qtof_etkdgv3_test.mgf
  # Orbitrap:
  python scripts/predict.py --task msms \
  --test_data ./data/orbitrap_etkdgv3_test.pkl \
  --model_config_path ./molnetpack/config/molnet.yml \
  --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
  --resume_path ./check_point/molnet_orbitrap_etkdgv3.pt \
  --result_path ./result/pred_orbitrap_etkdgv3_test.mgf

  # Evaluate the cosine similarity between experimental spectra and predicted spectra:
  # Q-TOF:
  python scripts/eval.py ./data/qtof_etkdgv3_test.pkl ./result/pred_qtof_etkdgv3_test.mgf \
  --result_path ./eval_qtof_etkdgv3_test.csv --plot_path ./eval_qtof_etkdgv3_test.png
  # Orbitrap:
  python scripts/eval.py ./data/orbitrap_etkdgv3_test.pkl ./result/pred_orbitrap_etkdgv3_test.mgf \
  --result_path ./eval_orbitrap_etkdgv3_test.csv --plot_path ./eval_orbitrap_etkdgv3_test.png
