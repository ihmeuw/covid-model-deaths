#!/bin/sh
#$ -S /bin/bash

#export PATH="/ihme/code/rmbarber/miniconda3/bin:$PATH"
#source /ihme/code/evidence_score/miniconda3/bin/activate mr_brt_refactor_env

# need to do this dynamically...
/ihme/code/covid-19/deaths/conda/miniconda3/envs/covid_dev_2020_04_12/bin/python "$@"