# Computational Methods
This repository contains essential python tutorials and functions for numerical techniques such as root finding, solving ODEs and optimization that are most commonly used by undergraduate students in UBC CHBE. The material was originally developed for teaching CHBE 230 (Computational Methods) by **Arman Seyed-Ahmadi**. Additional examples and notebooks were later added by **Danie Benetton** as a part of CHBE Python Project.

# Folders

- `./`: The root folder contains the main Jupyter notebooks where you can find the functions, as well as several detailed examples for each function that are prepared to help understanding how the functions are used.
- `./Python Crash Course`: You can find a quick introduction to python here. The tutorial is provided both as Jupyter notebooks and PDF slides.
- `./images`: This folder stores the image files used in the Jupyer notebooks.
- `./package`: The modules and functions are stored as a python package here. This package is hosted on [PyPi.org](https://pypi.org/project/ubc-chbe-computational-methods/) so that anyone can install and use the functions without the need for copying functions from the notebooks.

## Package installation
In case you want to have the modules and functions globally available on your system, such that you can use them anywhere else, you can install the package by running the following command:
```bash
pip install ubc_chbe_computational_methods
```
This makes the functions available anywhere on your system. After installing the package, you can import the package, and the included modules and functions simply by running
```python
import ubc_chbe_computational_methods as ubc_chbe
ubc_chbe.ode_solve.RK4_ode_sys([args])
```
with the required arguments.