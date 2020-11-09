# Modeling Dynamics of Euler's Three Body Problem

[![DOI](https://zenodo.org/badge/294831403.svg)](https://zenodo.org/badge/latestdoi/294831403)

## How to run

### Prereqs

1. Clone the repo
2. Install Python 3 and optionally Jupyter Notebook (`pip install notebook` or
`conda install -C conda-force notebook`).
3. Install `matplotlib`, `numpy`, and (if using Jupyter Notebook) `ipywidgets`
with `conda` or `pip`.

### Jupyter Notebook

Note: unforunately, GitHub does not have built in support for `ipywidgets`, so
you will need to run it locally.

1. If you have jupyter notebook installed, run `jupyter notebook main.ipynb`.
Alternatively, use an IDE to run `main.ipynb` (for example, in Visual Studio
Code, if you have the Python extension installed, a Jupyter Notebook kernel will
automatically be started when you open the file).
2. Run all the cells in order
3. Change the inputs of the interact widgets (they are in the order of Euler's
Method, Runge-Kutta, and Forest & Neri)
4. Click "Run Interact"

Each method is given its own graph. The color scale indicates the time dimension
of the particle (lighter is earlier, darker is later).

![Jupyter Example](images/Jupyter%20Ex.png)

### Standard Python

1. Run `python3 main.py`
2. Select your inputs and initial conditions

All three numerical methods will be displayed on the same graph

![Standard Example](images/Overlay%20400%200.005.png)

## General Notes

* Higher `δt` results in more accurate graphs but requires more iterations (and
thus is slower)
* Euler's Method is the least accurate by far, but also the fastest
