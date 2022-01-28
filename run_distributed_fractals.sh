#!/bin/bash
make distributed-fractal

mpiexec ./distributed-fractal 128 128 5000

mpiexec ./distributed-fractal 256 256 5000

mpiexec ./distributed-fractal 512 512 5000

mpiexec ./distributed-fractal 1024 1024 5000



