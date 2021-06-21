#!/bin/bash
source /home/spack/spack/share/spack/setup-env.sh
spack load cuda

if [ "$1" = "" ]; then
	echo Usage: ./run.sh testcase
	exit 1
fi

srun cuda-memcheck ~/PA4_build/test/unit_tests --dataset $1 --datadir ~/PA4/data/ --len 256

