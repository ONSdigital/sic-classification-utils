# SIC Classification Utils

Standard Industrial Classification (SIC) Utilities, initially developed for Survey Assist API and complements the SIC Classification Library code.

## Overview

SIC classification utilities used in the classification of industry.  This repository contains core code used by the SIC Classification Library.

## Features

- Embeddings.  Functionality for embedding SIC hierarchy data, managing vector stores,
and performing similarity searches
- Data Access. Functions to load CSV data files related to SIC.

## Prerequisites

Ensure you have the following installed on your local machine:

- [ ] Python 3.12 (Recommended: use `pyenv` to manage versions)
- [ ] `poetry` (for dependency management)
- [ ] Colima (if running locally with containers)
- [ ] Terraform (for infrastructure management)
- [ ] Google Cloud SDK (`gcloud`) with appropriate permissions

### Local Development Setup

The Makefile defines a set of commonly used commands and workflows.  Where possible use the files defined in the Makefile.

#### Clone the repository

```bash
git clone https://github.com/ONSdigital/sic-classification-utils.git
cd sic-classification-utils
```

#### Install Dependencies

```bash
poetry install
```

#### Add Git Hooks

Git hooks can be used to check code before commit. To install run:

```bash
pre-commit install
```

### Run Locally

There is example source for using the SIC Embedding functionality in [sic_embedding_example.py](src/industrial_classification_utils/embed/sic_embedding_example.py) to run:

```bash
poetry run python src/industrial_classification_utils/embed/sic_embedding_example.py
```

This will output semantic search of the files in [src/industrial_classification_utils/data/sic_index](src/industrial_classification_utils/data/sic_index) based on the query "school teacher primary education"

### Structure

[docs](docs) - documentation as code using mkdocs

[scripts](scripts) - location of any supporting scripts (e.g data cleansing etc)

[src/industrial_classification_utils/data](src/industrial_classification_utils/data) - example data and SIC classification data used for embeddings

[src/industrial_classification_utils/embed](src/industrial_classification_utils/embed) - ChromaDB vector store and embedding code, includes an example use of the store.

[src/industrial_classification_utils/models](src/industrial_classification_utils/models) - common data structures that need to be shared

[src/industrial_classification_utils/utils](src/industrial_classification_utils/utils) - common utility functions such as xls file read for embeddings.

[tests](tests) - PyTest unit testing for code base, aim is for 80% coverage.

### GCP Setup

Placeholder

### Code Quality

Code quality and static analysis will be enforced using isort, black, ruff, mypy and pylint. Security checking will be enhanced by running bandit.

To check the code quality, but only report any errors without auto-fix run:

```bash
make check-python-nofix
```

To check the code quality and automatically fix errors where possible run:

```bash
make check-python
```

### Documentation

Documentation is available in the docs folder and can be viewed using mkdocs

```bash
make run-docs
```

### Testing

Pytest is used for testing alongside pytest-cov for coverage testing.  [/tests/conftest.py](/tests/conftest.py) defines config used by the tests.

Unit testing for embedding functions is added to the [/tests/test_embedding.py](./tests/test_embedding.py)
Unit testing for utility functions is added to the [/tests/test_sic_data_access.py](./tests/test_sic_data_access.py)

```bash
make embed-tests
```

```bash
make utils-tests
```

All tests can be run using

```bash
make all-tests
```

### Environment Variables

Placeholder
