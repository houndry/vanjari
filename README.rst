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

.. note::

    The first time that Vanjari is run, it will download the model weights. These are large files and may take some time to download.

This outputs a CSV file with the predictions for each virus in the input file.

This will build embeddings for all sequences in the input file and then classify them using the neural network.

To save the embeddings to disk as a Numpy memmap array, provide a location for the file and the index listing:

.. code-block:: bash

    vanjari --input virus.fa --output-csv virus-predictions.csv --memmap-array-path virus-embeddings.npy --memmap-index virus-index.txt

If the memmap and index exist already, then they will be used without recomputing the embeddings.

The input file can also be a directory of FASTA files.

Thresholds
----------

You can set a threshold for the predictions using the ``--prediction-threshold`` option. This must be a value between 0 and 1. 
If the probability of a classification at any rank in the taxonomy is below this threshold, the classification will be set to "NA".

The threshold can be increased later using the ``vanjari-tools increase-threshold`` command:

Faster Inference
----------------

To use a Vanjari model without computing the embeddings, use the ``vanjari-fast`` command which uses a simplier convolutional neural network:

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

.. end-quickstart


ICTV Challenge
--------------

.. start-ictv

This project is submitted as part of the 2024 `ICTV Computational Virus Taxonomy Challenge <https://ictv-vbeg.github.io/ICTV-TaxonomyChallenge/>`_.

The results are in ``./results``:

- `results/vanjari-0.1.csv <https://github.com/bloodhound-devs/vanjari/blob/main/results/vanjari-0.1.csv>`_: The results for the main Vanjari model.
- `results/vanjari-fast-0.1.csv <https://github.com/bloodhound-devs/vanjari/blob/main/results/vanjari-fast-0.1.csv>`_: The results for the fast Vanjari model.
- `results/vanjari-ensemble-0.1.csv <https://github.com/bloodhound-devs/vanjari/blob/main/results/vanjari-ensemble-0.1.csv>`_: The results for the fast Vanjari model.

There are also versions of the results with a threshold of 0.5:

- `results/vanjari-0.1-threshold0.5.csv <https://github.com/bloodhound-devs/vanjari/blob/main/results/vanjari-0.1-threshold0.5.csv>`_
- `results/vanjari-fast-0.1-threshold0.5.csv <https://github.com/bloodhound-devs/vanjari/blob/main/results/vanjari-fast-0.1-threshold0.5.csv>`_
- `results/vanjari-ensemble-0.1-threshold0.5.csv <https://github.com/bloodhound-devs/vanjari/blob/main/results/vanjari-ensemble-0.1-threshold0.5.csv>`_

To reproduce the results, use the following command to download the dataset:

.. code-block:: bash

    wget "https://github.com/ICTV-VBEG/ICTV-TaxonomyChallenge/raw/refs/heads/main/dataset/dataset_challenge.tar.gz?download=" -O ictv-challenge.tar.gz
    tar zxvf ictv-challenge.tar.gz

This will create a directory called ``dataset_challenge`` with the sequences. Now run the following commands to classify the sequences using the two Vanjari models:

.. code-block:: bash

    # Generage results for single models
    vanjari --input dataset_challenge/ --output-csv ictv-challenge/vanjari-0.1.csv --memmap-array-path ictv-challenge/embeddings.npy --memmap-index ictv-challenge/embeddings.txt
    vanjari-fast --input dataset_challenge/ --output-csv ictv-challenge/vanjari-fast-0.1.csv

    # Generate results for ensemble
    vanjari-tools ensemble-csvs --input ictv-challenge/vanjari-0.1.csv --input ictv-challenge/vanjari-fast-0.1.csv --output ictv-challenge/vanjari-ensemble-0.1.csv

    # Set the threshold for the all results to 0.5
    vanjari-tools increase-threshold --input ictv-challenge/vanjari-0.1.csv --output ictv-challenge/vanjari-0.1-threshold0.5.csv --threshold 0.5
    vanjari-tools increase-threshold --input ictv-challenge/vanjari-fast-0.1.csv --output ictv-challenge/vanjari-fast-0.1-threshold0.5.csv --threshold 0.5
    vanjari-tools increase-threshold --input ictv-challenge/vanjari-ensemble-0.1.csv --output ictv-challenge/vanjari-ensemble-0.1-threshold0.5.csv --threshold 0.5

.. end-ictv


Credits
-------

.. start-credits

This package was created by members of the University of Melbourne and the University of Adelaide. Citation details to come.


.. end-credits
