# Computational Methods
This repository contains essential python tutorials and functions for numerical techniques such as root finding, solving ODEs and optimization that are most commonly used by undergraduate students in UBC CHBE. The material was originally developed for teaching CHBE 230 (Computational Methods) by **Arman Seyed-Ahmadi**. Additional examples and notebooks were later added by **Danie Benetton** as a part of CHBE Python Project.

The Jupyter notebooks contain both the functions and several examples that help understanding how the functions are used. The folder `./package` contains the modules and functions only, which can be used anywhere if installed, as shown in the following.

## Package installation
You can install the package by running the following command:
```bash
pip install ubc_chbe_computational_methods
```
This makes the functions available anywhere on your system. After installing the package, you can import the package, and the included modules and functions simply by running
```python
import ubc_chbe_computational_methods as ubc_chbe
ubc_chbe.ode_solve.RK4_ode_sys([args])
```
with the required arguments.
