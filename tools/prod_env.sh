#!/bin/sh
#$ -S /bin/bash

#export PATH="/ihme/code/rmbarber/miniconda3/bin:$PATH"
#source /ihme/code/evidence_score/miniconda3/bin/activate mr_brt_refactor_env

/ihme/code/evidence_score/miniconda3/envs/mr_brt_refactor_env/bin/python "$@"