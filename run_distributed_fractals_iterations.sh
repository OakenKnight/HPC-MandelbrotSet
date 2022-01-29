#!/bin/bash
make distributed-fractal
echo -e "\e[01;32mGenerating distributed memory fractal algorithm files bw...\e[00m"

mpiexec ./distributed-fractal 1024 1024 10 0

mpiexec ./distributed-fractal 1024 1024 20 0

mpiexec ./distributed-fractal 1024 1024 50 0

mpiexec ./distributed-fractal 1024 1024 100 0

mpiexec ./distributed-fractal 1024 1024 500 0 

mpiexec ./distributed-fractal 1024 1024 1000 0

mpiexec ./distributed-fractal 1024 1024 5000 0

mpiexec ./distributed-fractal 1024 1024 10000 0

mpiexec ./distributed-fractal 1024 1024 25000 0


echo -e "\e[01;32mGenerating distributed memory fractal algorithm files rgb...\e[00m"

mpiexec ./distributed-fractal 1024 1024 10 1

mpiexec ./distributed-fractal 1024 1024 20 1

mpiexec ./distributed-fractal 1024 1024 50 1

mpiexec ./distributed-fractal 1024 1024 100 1

mpiexec ./distributed-fractal 1024 1024 500 1 

mpiexec ./distributed-fractal 1024 1024 1000 1

mpiexec ./distributed-fractal 1024 1024 5000 1

mpiexec ./distributed-fractal 1024 1024 10000 1

mpiexec ./distributed-fractal 1024 1024 25000 1







