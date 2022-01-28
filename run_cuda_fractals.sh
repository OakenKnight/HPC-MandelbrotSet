#!/bin/bash
make cuda

./gpu-fractal 128 128 5000

./gpu-fractal 256 256 5000

./gpu-fractal 512 512 5000

./gpu-fractal 1024 1024 5000



