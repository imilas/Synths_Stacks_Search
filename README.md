
### Notes
- use python3.6 and Make sure python3-dev tools are installed. If using apt:
    ~~~~
    sudo apt-get install python3-dev
   ~~~~~
- notebooks are best run in jupyterlab
- need gcc4.9+ (tested on gcc7) to install pippi
- need nodejs5+ for some jupyterlab widgets
- since gcc and nodejs are required for parts of this project, a miniconda environment is ideal https://docs.conda.io/en/latest/miniconda.html
- check nodejs --version and gcc --version, then (if needed) update your conda env with:
    ~~~~
    conda install -c conda-forge nodejs
    conda install -c creditx gcc-7
    ~~~~  
- install the following widgets for jupyter-lab: jupyterlab-plotly and plotlywidget v4.9.0
    ~~~~
    jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.9.0
    jupyter labextension install jupyterlab-plotly@4.9.0
    ~~~~
### Installation
- install pippi and other requirements:
    ~~~~
    sh install.sh
    ~~~~
- once installed, check random_gen.ipynb for a start
