#!/bin/bash
make shared-fractal

./shared-fractal 128 128 5000 0

./shared-fractal 256 256 5000 0

./shared-fractal 512 512 5000 0

./shared-fractal 1024 1024 5000 0



