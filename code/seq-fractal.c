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

/*
Compute the number of iterations at point x, y
in the complex space, up to a maximum of maxiter.
Return the number of iterations at that point.

This example computes the Mandelbrot fractal:
z = z^2 + alpha

Where z is initially zero, and alpha is the location x + iy
in the complex plane.  Note that we are using the "complex"
numeric type in C, which has the special functions cabs()
and cpow() to compute the absolute values and powers of
complex values.
*/

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

/*
Compute an entire image, writing each point to the given bitmap.
Scale the image to the range (xmin-xmax,ymin-ymax).
*/
void compute_image( double xmin, double xmax, double ymin, double ymax, int maxiter, int width, int height)
{
	int i,j;

	// For every pixel i,j, in the image...
    char buffer[width*height];



	for(i=0;i<height;i++) {
		for(j=0;j<width;j++) {

			// Scale from pixels i,j to coordinates x,y
			double x = xmin + i*(xmax-xmin)/width;

			double y = ymin + j*(ymax-ymin)/height;

			// Compute the iterations at x,y
			int iter = compute_point(x,y,maxiter);

			// Convert a iteration number to an RGB color.
			int gray = 255 * iter / maxiter;
            
            buffer[j*height+i] = (char) gray;
		}

	}
     stbi_write_jpg("fractal-images/fractal.jpg", width, height, 1, buffer, 200);

}

int main( int argc, char *argv[] )
{
	// The initial boundaries of the fractal image in x,y space.

	double xmin=-1.5;
	double xmax= 0.5;
	double ymin=-1.0;
	double ymax= 1.0;

    int width = 1000;
    int height = 1000;

	// Maximum number of iterations to compute.
	// Higher values take longer but have more detail.
	int maxiter=1000;

	// Showing the configuration, just in case you want to recreate it.
	printf("Coordinates: %lf %lf %lf %lf\n",xmin,xmax,ymin,ymax);
	printf("Timer started\n");

	clock_t begin = clock();
	compute_image(xmin,xmax,ymin,ymax,maxiter, width, height);
	clock_t end = clock();

	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("time took for execution of sequential algorithm: %f\n", time_spent);

	return 0;
}