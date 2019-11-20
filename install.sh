#!/bin/bash
pippi_source=https://github.com/luvsound/pippi.git
git clone $pippi_source && cd pippi
pip install -r requirements.txt
sudo apt-get install python3-dev
make install
cd ..
pip install -r requirements.txt
