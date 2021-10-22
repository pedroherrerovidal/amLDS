# Across-animal odor decoding byprobabilistic manifold alignment (NeurIPS 2021)

This repository is the official implementation of aligned mixture of latent dynamical systems (amLDS) published at NeurIPS 2021.

amLDS is a probabilistic method to align neural responses and efficiently decode stimuli across animals. It learns independent mappings of different recordings into a shared latent manifold, where stimulus-evoked dynamics are similar across animals but distint across stimuli allowing for accurate stimulus decoding. 

./funs/figs.png

A full description of the method can be found in the [preprint](https://www.biorxiv.org/content/10.1101/2021.06.06.447279v1).

## Requirements

* numpy == 1.16.2
* matplotlib == 3.0.3
* seaborn == 0.9.0
* sklearn == 0.20.3
* scipy == 1.2.1
* python == 3.7.3+

## Usage

To get started, run the example notebook ['amLDS_example'](amLDS_example.ipynb). This notebook contains an example on the use of amLDS on synthetic data. It shows how to perform parameter learning, inference and stimulus decoding; as well as latent dimensionality estimation.

To explore other properties and capabilities of amLDS check the ['amLDS_mixturesConcentration'](amLDS_mixturesConcentration.ipynb) notebook or run the performance script as python3 ['amLDS_Performance_DataDemands_ModelComparison.py'](amLDS_mixturesConcentration.ipynb).

## Copyrights and license
This code has been released under the GNU AGPLv3 license. The code in this repository can be used noncommercially, if so cite as:

'''bibtex
@article {Herrero-Vidal2021.06.06.447279,
	author = {Herrero-Vidal, Pedro and Rinberg, Dmitry and Savin, Cristina},
	title = {Across-animal odor decoding by probabilistic manifold alignment},
	elocation-id = {2021.06.06.447279},
	year = {2021},
	doi = {10.1101/2021.06.06.447279},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/06/08/2021.06.06.447279},
	eprint = {https://www.biorxiv.org/content/early/2021/06/08/2021.06.06.447279.full.pdf},
  journal = {bioRxiv}}
  '''
