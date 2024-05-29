#!/bin/bash

# ------------------------------------------------
# forces a commit and push before running anything
#
# for instance:
#
#  > ./scripts/gitandrun.py python train.py 
# ------------------------------------------------

git add configs
git commit -am "experiment iteration"
git push

$*
