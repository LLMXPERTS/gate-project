# MatryoshkaEmbeddings

This repository contains code for training and evaluating Matryoshka Embeddings for multilingual text representations as described in our paper "*Matryoshka Representation Learning for Cross-lingual Transfer*".

## Overview

This project implements Matryoshka Representation Learning, which enables efficient nested representations at multiple dimensionalities. We train transformer models on the SNLI + MultiNLI (AllNLI) dataset using MatryoshkaLoss with MultipleNegativesRankingLoss to produce embeddings at various dimensions [768, 512, 256, 128, 64].

Our approach shows strong performance on cross-lingual semantic textual similarity tasks, especially for Arabic language.

## Repository Structure

```
MatryoshkaEmbeddings/
├── training/
│   ├── matryoshka_nli.py          # Training with Matryoshka loss on NLI datasets
│   ├── hybrid_training.py         # Multi-dataset hybrid training approach
│   └── utils/
│       ├── __init__.py
│       ├── data_loading.py        # Dataset loading utilities
│       └── model_utils.py         # Model initialization helpers
├── evaluation/
│   ├── evaluate_mteb.py           # MTEB evaluation script
│   ├── evaluate_sts.py            # STS benchmark evaluation
│   └── utils/
│       ├── __init__.py
│       └── evaluation_utils.py    # Evaluation utilities
├── configs/
│   ├── training_config.json       # Training hyperparameters
│   └── eval_config.json           # Evaluation settings
├── scripts/
│   ├── run_training.sh            # Training script
│   └── run_evaluation.sh          # Evaluation script
├── requirements.txt               # Dependencies
├── environment.yml                # Conda environment
├── LICENSE                        # License information
└── README.md                      # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MatryoshkaEmbeddings.git
cd MatryoshkaEmbeddings

# Option 1: Using pip
pip install -r requirements.txt

# Option 2: Using conda
conda env create -f environment.yml
conda activate matryoshka
```

## Training

We provide scripts for training models with Matryoshka embedding architecture:

```bash
# Run training with default parameters
python training/matryoshka_nli.py 

# Or specify a custom pre-trained model
python training/matryoshka_nli.py distilroberta-base
```

For the hybrid training approach:

```bash
python training/hybrid_training.py
```

## Evaluation

Evaluate models using the MTEB benchmark:

```bash
python evaluation/evaluate_mteb.py --model_name your-model-name
```

Or evaluate on specific STS tasks:

```bash
python evaluation/evaluate_sts.py --model_name your-model-name --task STS17
```

## Pre-trained Models

Our models are available on the Hugging Face Hub:

- [Omartificial-Intelligence-Space/GATE-AraBert-v0](https://huggingface.co/Omartificial-Intelligence-Space/GATE-AraBert-v0)

## Datasets

We use the following datasets for training and evaluation:

- [sentence-transformers/all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli)
- [sentence-transformers/stsb](https://huggingface.co/datasets/sentence-transformers/stsb)
- MTEB Arabic evaluation datasets (STS17, STS22, STS22.v2)

## Citation

If you use this code or our pre-trained models, please cite our paper:

```
@inproceedings{matryoshka-embeddings-2025,
  title={Matryoshka Representation Learning for Cross-lingual Transfer},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the Conference Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We thank the Sentence Transformers team for their excellent library, and the creators of the MTEB benchmark for providing standardized evaluation protocols.
