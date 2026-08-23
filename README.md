# Statistical Rethinking — Python examples

My Python implementations of the examples from [Statistical Rethinking (2nd edition)](https://xcelab.net/rm/) by Richard McElreath. The book uses R and the [`rethinking` package](https://github.com/rmcelreath/rethinking) (source code and example data also available there); I'm working through it using PyMC and ArviZ primarily instead.

Rethinking/overthinking examples are not included. DAG plotting is also not included.

This is a work in progress. I'll keep adding examples as I work my way through the book.

## A note on versions

The examples use PyMC 6.2 and ArviZ 1.3.0. Because of breaking changes at the ArviZ 1.0.0 release, the examples do not work with pre-1.0.0 versions.

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
| `Chapter7.py` | Chapter 7 — Ulysses' Compass |
| `Chapter8_12.py` | Chapter 8.1 — Building an interaction and Chapter 8.2 — Symmetry of interactions |
| `Chapter8_3.py` | Chapter 8.3 — Continuous interactions |
| `Chapter9_1.py` | Chapter 9.1 — Good King Markov and his island kingdom |
| `Chapter9_2.py` | Chapter 9.2 — Metropolis algorithms |
| `Chapter9_4.py` | Chapter 9.4 — Easy HMC: ulam |
| `Chapter9_5.py` | Chapter 9.5 — Care and feeding of your Markov chain |


## Requirements

- Python ≥ 3.12
- See `pyproject.toml` or `environment.yml` for the full dependency list
