#!/bin/bash
source /home/spack/spack/share/spack/setup-env.sh
spack load cuda

bash compile.sh

bash run.sh $1

