all: fractal

seq-fractal: code/seq-fractal.c
	@echo "\e[01;32mGenerating sequential fractal algorithm files...\e[00m"
	gcc code/seq-fractal.c -g -Wall --std=c99 -lX11 -lm -o seq-fractal

shared-fractal: code/shared-fractal.c
	@echo "\e[01;32mGenerating shared memory fractal algorithm files...\e[00m"
	gcc code/shared-fractal.c -g -Wall -fopenmp --std=c99 -lX11 -lm -o shared-fractal

distributed-fractal: code/distributed-fractal.c
	@echo "\e[01;32mGenerating distributed memory fractal algorithm files...\e[00m"
	mpicc code/distributed-fractal.c --std=c99 -fopenmp  -lX11 -lm -o distributed-fractal

run_distrib: ./distributed-fractal
	mpiexec ./distributed-fractal

cuda:	code/cuda-fractal.cu
	@echo "\e[01;32mGenerating gpu fractal algorithm files...\e[00m"
	nvcc -o gpu-fractal code/cuda-fractal.cu 

clean:
	@echo "\e[01;32mDeleting temporary files...\e[00m"
	@rm -f $(COMPILER_CLEAN)
	@rm -f seq-fractal
	@rm -f fractal-images/fractal.jpg
	@rm -f shared-fractal
	@rm -f fractal-images/shared-fractal.jpg
	@rm -f distributed-fractal
	@rm -f fractal-images/distrib-fractal.jpg
	@rm -f gpu-fractal
