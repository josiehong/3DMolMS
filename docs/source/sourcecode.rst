Source code setup
=================

Installing from source code
---------------------------

3DMolMS can also be installed through source code:

.. code-block:: bash

   git clone https://github.com/JosieHong/3DMolMS.git
   cd 3DMolMS
   pip install .

PyTorch must be installed separately. Check the `official PyTorch website <https://pytorch.org/get-started/locally/>`_ for the proper version for your system. For example:

.. code-block:: bash

   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Update package locally
----------------------

Every time you update the code, you can run the following command to update the package:

.. code-block:: bash

   cd 3DMolMS
   pip install .

Then you can use the package for various tasks as shown in the :doc:`./usage/index` and :doc:`./advanced_usage/index` page.

Requirements
------------

3DMolMS has the following dependencies:

* Python 3.8+
* PyTorch
* RDKit
* NumPy
* Pandas
* matplotlib
* PyYAML
* Other dependencies listed in ``pyproject.toml``

Most dependencies will be automatically installed when using pip, but PyTorch should be installed separately as mentioned above to ensure compatibility with your system's CUDA version.