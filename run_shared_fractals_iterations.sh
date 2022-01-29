#!/bin/bash
make shared-fractal

./shared-fractal 1024 1024 10 0

./shared-fractal 1024 1024 20 0

./shared-fractal 1024 1024 50 0

./shared-fractal 1024 1024 100 0

./shared-fractal 1024 1024 500 0

./shared-fractal 1024 1024 1000 0

./shared-fractal 1024 1024 5000 0

./shared-fractal 1024 1024 10000 0

./shared-fractal 1024 1024 25000 0
