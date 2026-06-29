# Statistical Rethinking — Python examples

My Python implementations of the examples from [Statistical Rethinking (2nd edition)](https://xcelab.net/rm/) by Richard McElreath. The book uses R and the [`rethinking` package](https://github.com/rmcelreath/rethinking) (source code and example data also available there); I'm working through it using PyMC and ArviZ primarily instead.

Rethinking/overthinking examples are not included. DAG plotting is also not included.

This is a work in progress. I'll keep adding examples as I work my way through the book.

## A note on versions

The examples use PyMC 6.0 and ArviZ 1.2.0. Because of breaking changes at the ArviZ 1.0.0 release, the examples do not work with pre-1.0.0 versions.

I tried to keep the list of dependencies to a minimum, but that turned out to be harder than expected — McElreath uses a wide range of functionality across the book, and no single Python package covers all of it. The current set of packages is about as lean as I could get it.

## Installation

### pip

Create and activate a virtual environment, then install:

```bash
pip install .
```

This installs all required dependencies listed in `pyproject.toml`.

### conda

```bash
conda env create -f environment.yml
conda activate rethinking
```

## Structure

Each file corresponds to a section of the book:

| File | Content |
|---|---|
| `Chapter3.py` | Chapter 3 — Sampling the Imaginary |
| `Chapter4_1.py` | Chapter 4.1 — Why normal distributions are normal |
| `Chapter4_3.py` | Chapter 4.3 — Gaussian model of height |
| `Chapter4_4.py` | Chapter 4.4 — Linear prediction |
| `Chapter4_5.py` | Chapter 4.5 — Curves from lines |
| `Chapter5_1.py` | Chapter 5.1 — Spurious association |
| `Chapter5_2.py` | Chapter 5.2 — Masked relationships |
| `Chapter5_3.py` | Chapter 5.3 — Categorical variables |
| `Chapter6_1.py` | Chapter 6.1 — Multicollinearity |
| `Chapter6_2.py` | Chapter 6.2 — Post-treatment bias |
| `Chapter6_3.py` | Chapter 6.3 — Collider bias |

## Requirements

- Python ≥ 3.12
- See `pyproject.toml` or `environment.yml` for the full dependency list
