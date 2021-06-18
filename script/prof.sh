source /home/spack/spack/share/spack/setup-env.sh
spack load cuda
srun nvprof -o ../prof/${1}.nvprof ~/PA4_build/test/unit_tests --dataset $1 --datadir ~/PA4/data/ --len 32
