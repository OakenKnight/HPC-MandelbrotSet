#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <complex.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static int compute_point( double x, double y, int max )
{
	double complex z = 0;
	double complex alpha = x + I*y;

	int iter = 0;

	while( cabs(z)<4 && iter < max ) {
		z = cpow(z,2) + alpha;
		iter++;
	}

	return iter;
}

void compute_image( double xmin, double xmax, double ymin, double ymax, int maxiter, int width, int height)
{
	int i,j;
    char buffer[width*height];

	for(i=0;i<height;i++) {
		for(j=0;j<width;j++) {

			double x = xmin + i*(xmax-xmin)/width;
			double y = ymin + j*(ymax-ymin)/height;

			int iter = compute_point(x,y,maxiter);

			int gray = 255 * iter / maxiter;
            
            buffer[j*height+i] = (char) gray;
		}

	}
     stbi_write_jpg("fractal-images/fractal.jpg", width, height, 1, buffer, 200);

}

int main( int argc, char *argv[] )
{

	double xmin=-1.5;
	double xmax= 0.5;
	double ymin=-1.0;
	double ymax= 1.0;

    int width = 1000;
    int height = 1000;
	
	int maxiter=1000;

	printf("Timer started\n");

	clock_t begin = clock();
	compute_image(xmin,xmax,ymin,ymax,maxiter, width, height);
	clock_t end = clock();

	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("time took for execution of sequential algorithm: %f\n", time_spent);

	return 0;
}