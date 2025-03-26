# Getting Started

See the README for general setup instructions.

## SIC Classification Library Utilities

This repository provides utility functions used by the **SIC Classification Library** for classifying the Standard Industrial Code, used by the **Survey Assist API** hosted in **Google Cloud Platform (GCP)**.

### Features

- SIC Embedding and Vector Store

### Installation

To use this code in another repository using ssh:

```bash
poetry add git+ssh://git@/ONSdigital/sic-classification-utils.git@v0.1.0
```

or https:

```bash
poetry add git+https://github.com/ONSdigital/sic-classification-utils.git@v.0.1.0
```

### Usage

Example code that uses the embeddings is in [sic_embedding_example.py](https://github.com/ONSdigital/sic-classification-utils/blob/main/src/industrial_classification_utils/embed/sic_embedding_example.py)
