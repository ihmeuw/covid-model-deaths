#!/bin/bash
export PATH="/ihme/covid-19/miniconda3/bin:$PATH"
dt=`date '+%Y-%m-%d_%H-%M-%S'`
echo "Creating environment covid-deaths-$dt"
conda create -y --name=covid-deaths-$dt python=3.6
source /ihme/covid-19/miniconda3/bin/activate covid-deaths-$dt
#which pip
