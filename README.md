[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/2CuuzvUr)
# assignment-7


module load nvhpc
module load craype-accel-nvidia80
nvc++ -mp=gpu -gpu=cc80 -Ofast cg.cpp -o cg_idk -Minfo=accel,mp
srun -p gpu --gres=gpu:1 --ntasks=1 --time=00:05:00 --mem=40G --reservation=p_es_itkpp_204 ./cg_idk


nvc++ laplace2d.cpp -mp -Ofast -o laplace_cpu


nvc++ cg.cpp -mp -Ofast -o cg_cpu
srun -p gpu --gres=gpu:1 --ntasks=1 --time=00:05:00 --mem=40G --reservation=p_es_itkpp_204 ./cg_cpu