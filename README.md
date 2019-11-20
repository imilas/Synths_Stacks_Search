
### Notes
- Use python3.6 and Make sure python3-dev tools are installed. If using apt:
    ~~~~
    sudo apt-get install python3-dev
   ~~~~~
- notebooks are tested in jupyterlab (extension for jupyter-notebook)
- need gcc4.9+ (tested on gcc7) to install pippi
- need nodejs5+ for some jupyterlab widgets
- Since gcc and nodejs are required for parts of this project, a miniconda environment is ideal https://docs.conda.io/en/latest/miniconda.html
- check nodejs --version and gcc --version, then update your conda env with:
        conda install -c conda-forge nodejs
        conda install -c creditx gcc-7
        
### installation
- Install pippi and other requirements:
  ~~~~
  sh install.sh
  ~~~~
- Once installed, check random_gen.ipynb for a start
