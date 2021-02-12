import setuptools

setuptools.setup(name='ubc_chbe_computational_methods',
      version='0.1.3',
      description='Essential python functions for numerical techniques',
      long_description='This package contains all functions developed for teaching the course CHBE 230 (Computational Methods). Computational techniques including root-finding, linear and nonlinear systems, curve-fitting, differentiation, integration, and ODE solution methods are implemented. For more information and examples for each method, please refer to the github page.',
      long_description_content_type='text/markdown',
      url='https://github.com/OpenChemE/Computational_Methods',
      author='Arman Seyed-Ahmadi',
      author_email='arman.awn@gmail.com',
      license='GPLv3',
      packages=setuptools.find_packages(),
      install_requires=['matplotlib', 'numpy', 'pandas'],
      classifiers=['Programming Language :: Python :: 3'],
      zip_safe=False)
