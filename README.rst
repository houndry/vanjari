.. image:: https://github.com/bloodhound-devs/vanjari/blob/main/docs/images/vanjari-banner.jpg?raw=true
    
=============
Vanjari
=============

.. start-badges

|black badge| |torchapp badge|

.. .. |testing badge| image:: https://github.com/bloodhound-devs/vanjari/actions/workflows/testing.yml/badge.svg
..     :target: https://github.com/bloodhound-devs/vanjari/actions

.. .. |docs badge| image:: https://github.com/bloodhound-devs/vanjari/actions/workflows/docs.yml/badge.svg
..     :target: https://bloodhound-devs.github.io/bloodhound
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |torchapp badge| image:: https://img.shields.io/badge/MLOpps-torchapp-B1230A.svg
    :target: https://rbturnbull.github.io/torchapp/
    
.. end-badges

.. start-quickstart

Virus Assessment using Neural networks for Just-in-time Analysis and Rapid Identification

Vanjari is a tool for the classification of viruses using neural networks. It uses the ICTV taxonomy to hierarchically classify viruses.


Installation
------------

Vanjari can be installed using pip:

.. code-block:: bash

    pip install git+https://github.com/bloodhound-devs/vanjari.git


Usage
---------

The main Vanjari app can be used as follows:

For more information see:

.. code-block:: bash

    vanjari --input virus.fa --output-csv virus-predictions.csv

This outputs a CSV file with the predictions for each virus in the input file.

This will build embeddings for all sequences in the input file and then classify them using the neural network.

To save the embeddings to disk as a Numpy memmap array, provide a location for the file and the index listing:

.. code-block:: bash

    vanjari --input virus.fa --output-csv virus-predictions.csv --memmap-array-path virus-embeddings.npy --memmap-index virus-index.txt

If the memmap and index exist already, then they will be used without recomputing the embeddings.

The input file can also be a directory of FASTA files:

Faster Inference
----------------

To use a Vanjari model without computing the embeddings, use the `vanjari-fast` command which uses a simplier convolutional neural network:

.. code-block:: bash

    vanjari-fast --input virus.fa --output-csv virus-predictions.csv

Advanced Usage
--------------

More documentation is coming with advanced usage. For now, please see the help:

.. code-block:: bash

    vanjari --help
    vanjari-tools --help
    vanjari-fast --help
    vanjari-fast-tools --help


ICTV Challenge
--------------

This project is submitted as part of the 2024 `ICTV Computational Virus Taxonomy Challenge <https://ictv-vbeg.github.io/ICTV-TaxonomyChallenge/>`_.

The results are in ``./results``:

- `results/vanjari-0.1.csv <https://github.com/bloodhound-devs/vanjari/blob/main/results/vanjari-0.1.csv>`_: The results for the main Vanjari model.
- ``results/vanjari-fast-0.1.csv``: The results for the fast Vanjari model.

To reproduce the results, use the following command to download the dataset:

.. code-block:: bash

    wget "https://github.com/ICTV-VBEG/ICTV-TaxonomyChallenge/raw/refs/heads/main/dataset/dataset_challenge.tar.gz?download=" -O ictv-challenge.tar.gz
    tar zxvf ictv-challenge.tar.gz

This will create a directory called ``dataset_challenge`` with the sequences. Now run the following commands to classify the sequences using the two Vanjari models:

.. code-block:: bash

    vanjari --input dataset_challenge/ --output-csv ictv-challenge-vanjari.csv --memmap-array-path ictv-challenge.npy --memmap-index ictv-challenge.txt
    vanjari-fast --input dataset_challenge/ --output-csv ictv-challenge-vanjari-fast.csv

.. end-quickstart


Credits
-------

.. start-credits

This package was created by:

- Robert Turnbull (University of Melbourne)
- George Spyro Bouras (University of Adelaide)
- Wytamma Wirth (University of Melbourne)
- Torsten Seemann (University of Melbourne)


.. end-credits
