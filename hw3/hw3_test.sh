#!/bin/bash
wget -O mymodel.h5 'https://www.dropbox.com/s/48npnsb6va2snep/model-00022-0.64176.h5?dl=1'
python3 test.py $1 $2