source /home/spack/spack/share/spack/setup-env.sh
spack load cuda
srun nvprof --kernels "spmm_kernel_merge" --analysis-metrics -f --export-profile ../prof/${1}-${2}.nvprof ~/PA4_build/test/unit_tests --dataset $1 --datadir ~/PA4/data/ --len $2
