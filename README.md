# PST_host_prediction

This is code used for the host prediction proof-of-concept analysis associated with the manuscript:

Protein Set Transformer: A protein-based genome language model to power high diversity viromics.  
Cody Martin, Anthony Gitter, and Karthik Anantharaman.
*bioRxiv*, 2024, doi: [10.1101/2024.07.26.605391](https://doi.org/10.1101/2024.07.26.605391).

The parent repository for the PST can be found here: [https://github.com/AnantharamanLab/protein_set_transformer](https://github.com/AnantharamanLab/protein_set_transformer)

## Installation

**DISCLAIMER**: This is not a standalone host prediction tool, so we have not created an installable package.

If you wish to make use of our code or reproduce our results, you can use these workflows to setup a Pytorch-Geometric environment with other dependencies:

### Without GPUs

```bash
mamba create -n pyg -c pytorch -c pyg -c conda-forge 'python<3.12' 'pytorch>=2.0' cpuonly 'pyg>=2.3.1' torch_scatter 'pydantic>=2,<2.1' 'numpy<2.0' lightning torchmetrics
```

### With GPUs

```bash
mamba create -n pyg -c pytorch -c nvidia -c pyg -c conda-forge 'python<3.12' 'pytorch>=2.0' pytorch-cuda=11.8 'pyg>=2.3.1' torch_scatter 'pydantic>=2,<2.1' 'numpy<2.0' lightning torchmetrics
```

This environment will enable you to try the code in this repository.

## Manuscript

For the manuscript, we ran the following command:

```bash
python main.py -i KNOWLEDGE_GRAPH.pt --dummy
```

which enumerates several hyperparameters to train different models using the input `KNOWLEDGE_GRAPH.pt`. These knowledge graphs can be found at [data/knowledge_graphs](data/knowledge_graphs/).

### Data availability

All data can be found in the PST manuscript DRYAD repository: [https://doi.org/10.5061/dryad.d7wm37q8w](https://doi.org/10.5061/dryad.d7wm37q8w). Additionally, **Supplementary Table 8** contains the metadata for the train and test viruses, along with the associated genomes we used as their hosts.
