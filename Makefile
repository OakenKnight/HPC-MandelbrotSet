all: fractal

seq-fractal: code/seq-fractal.c
	@echo "\e[01;32mGenerating sequential fractal algorithm files...\e[00m"
	gcc code/seq-fractal.c -g -Wall --std=c99 -lX11 -lm -o seq-fractal
shared-fractal: code/shared-fractal.c
	@echo "\e[01;32mGenerating shared memory fractal algorithm files...\e[00m"
	gcc code/shared-fractal.c -g -Wall -fopenmp --std=c99 -lX11 -lm -o shared-fractal
clean:
	@echo "\e[01;32mDeleting temporary files...\e[00m"
	@rm -f $(COMPILER_CLEAN)
	@rm -f seq-fractal
	@rm -f fractal.jpg
	@rm -f shared-fractal
	@rm -f shared-fractal.jpg
