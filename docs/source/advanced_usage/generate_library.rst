Generating a reference library for molecular identification
===========================================================

3DMolMS can be used to generate a reference library of small molecule MS/MS spectra, which can then be used for small molecule identification through MS/MS searching.

Using molecules from HMDB
-------------------------

Setup
~~~~~

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Data preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the HMDB molecules dataset from `HMDB Downloads <https://hmdb.ca/downloads>`_. The expected data directory structure is:

.. code-block:: text

   |- data
     |- hmdb
       |- structures.sdf

**Step 2**: Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the following commands to preprocess the datasets. The dataset configuration is stored in ``./molnetpack/config/preprocess_etkdgv3.yml``.

Using ETKDGv3:

.. code-block:: bash

   python scripts/hmdb2pkl.py --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml

Or using original conformation:

.. code-block:: bash

   python scripts/hmdb2pkl.py --data_config_path ./molnetpack/config/preprocess_hmdb.yml

**Step 3**: MS/MS generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the following commands to generate MS/MS spectra. The model configuration is stored in ``./molnetpack/config/molnet.yml``. Remember to modify the commands if you're using the original conformations from HMDB.

.. code-block:: bash

   for i in {0..21}; do
     python scripts/predict.py --task msms \
     --test_data ./data/hmdb/hmdb_etkdgv3_$i.pkl \
     --model_config_path ./molnetpack/config/molnet.yml \
     --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
     --resume_path ./check_point/molnet_qtof_etkdgv3.pt \
     --result_path ./data/hmdb/molnet_hmdb_etkdgv3_$i.mgf
   done

Using molecules from RefMet
---------------------------

Setup
~~~~~

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Data preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the RefMet molecules dataset from `RefMet Browse <https://www.metabolomicsworkbench.org/databases/refmet/browse.php>`_. The expected data directory structure is:

.. code-block:: text

   |- data
     |- refmet
       |- refmet.csv

**Step 2**: Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the following commands to preprocess the datasets. The dataset configuration is stored in ``./molnetpack/config/preprocess_etkdgv3.yml``.

.. code-block:: bash

   python scripts/refmet2pkl.py --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml

**Step 3**: MS/MS generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the following commands to generate MS/MS spectra. The model configuration is stored in ``./molnetpack/config/molnet.yml``.

.. code-block:: bash

   python scripts/predict.py --task msms \
   --test_data ./data/refmet/refmet_etkdgv3.pkl \
   --model_config_path ./molnetpack/config/molnet.yml \
   --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
   --resume_path ./check_point/molnet_qtof_etkdgv3.pt \
   --result_path ./data/refmet/molnet_refmet_etkdgv3.mgf
