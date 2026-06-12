# Statistical Rethinking — Python examples

My Python implementations of the examples from [Statistical Rethinking (2nd edition)](https://xcelab.net/rm/) by Richard McElreath. The book uses R and the [`rethinking` package](https://github.com/rmcelreath/rethinking) (source code and example data also available there); I'm working through it using PyMC and ArviZ primarily instead.

This is a work in progress. I'll keep adding examples as I work my way through the book.

## A note on versions

I started writing these examples before PyMC 6.0 was released, so the dependencies are pinned to PyMC 5.x and ArviZ < 1.0. I'll update to the newer versions at a later stage.

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
| `Chapter4_3.py` | Chapter 4.3 — Gaussian model of height |
| `Chapter4_4.py` | Chapter 4.4 — Linear prediction |
| `Chapter4_5_1.py` | Chapter 4.5.1 — Curves from lines - Polynomial regression |
| `Chapter4_5_2.py` | Chapter 4.5.2 — Curves from lines - Splines |
| `Chapter5_1.py` | Chapter 5.1 — Spurious association |
| `Chapter5_2.py` | Chapter 5.2 — Masked relationships |
| `Chapter5_3.py` | Chapter 5.3 — Categorical variables |
| `Chapter6_1.py` | Chapter 6.1 — Multicollinearity |
| `Chapter6_2.py` | Chapter 6.2 — Post-treatment bias |
| `Chapter6_3.py` | Chapter 6.3 — Collider bias |

## Requirements

- Python ≥ 3.10
- See `pyproject.toml` or `environment.yml` for the full dependency list
