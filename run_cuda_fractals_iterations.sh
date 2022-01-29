#!/bin/bash
make cuda

echo -e "\e[01;32mGenerating gpu fractal algorithm files BW...\e[00m"

./gpu-fractal 1024 1024 10 0

./gpu-fractal 1024 1024 20 0

./gpu-fractal 1024 1024 50 0

./gpu-fractal 1024 1024 100 0

./gpu-fractal 1024 1024 500 0

./gpu-fractal 1024 1024 1000 0

./gpu-fractal 1024 1024 5000 0

./gpu-fractal 1024 1024 10000 0

./gpu-fractal 1024 1024 25000 0

echo -e "\e[01;32mGenerating gpu fractal algorithm files rgb...\e[00m"

./gpu-fractal 475 475 5 1

./gpu-fractal 475 475 10 1

./gpu-fractal 475 475 20 1

./gpu-fractal 475 475 50 1

./gpu-fractal 475 475 100 1

./gpu-fractal 475 475 500 1

./gpu-fractal 475 475 1000 1

./gpu-fractal 475 475 5000 1

./gpu-fractal 475 475 10000 1

./gpu-fractal 475 475 25000 1


