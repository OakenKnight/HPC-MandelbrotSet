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
#include <omp.h>
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
double complex calculateZ(double complex z, int lvl, double complex alpha){
	z = cpow(z,lvl) + alpha;
	return z;
}

static int compute_point( double x, double y, int max )
{
	double complex z = 0;
	double complex alpha = x + I*y;

	int iter = 0;

	
		// int i = 1;
		// int max_num_threads = omp_get_max_threads();

		// #pragma omp parallel for num_threads(max_num_threads)
		// for(i=1; i>0;i++){
		// 	if(cabs(z)<4 && iter < max){
        //         z = calculateZ(z, 2, alpha);
        //         iter++;

		// 	}else{
		// 		exit;
		// 	}
		// }
	// {

    

	
// double start; 
// 	double end; 
// 	start = omp_get_wtime(); 
	while( cabs(z)<4 && iter < max ) {
		z = cpow(z,2) + alpha;
		iter++;
	}
		// {
		// 	#pragma omp parallel
		// 	{
		// 		#pragma omp single
		// 		while (cabs(z)<4 && iter < max) {
		// 			{
		// 				#pragma omp task 
		// 				z=calculateZ(z, 2, alpha);

		// 			}
		// 			#pragma omp atomic
		// 			iter++;
		// 		}
		// 	}
		// }
        

// 	end = omp_get_wtime(); 
// 	printf("Vreme za while: %f\n", start-end);

// volatile int flag=0;

// #pragma omp parallel for shared(flag)
// for(iter=0; iter<max; iter++)
// {    
//     if(flag){
// 		continue;
// 	}
// 	z = cpow(z,2) + alpha;
//     if(cabs(z)>=4)
//     {
//           flag=1;
//     }
// }

	// #pragma omp parallel for
	// for(iter = 0; iter<max;iter++){
	// 	if(cabs(z)<4)
	// 		z=calculateZ(z, 2, alpha);
	// }

// now i is the first index for which \n{a[i]} is zero.
// We replace the while loop by a for loop that examines all locations:
// result = -1;
// 	#pragma omp parallel for  lastprivate(result)
// 	for (i=0; i<imax; i++) {
// 		if (a[i]!=0 && result<0) 
// 		result = i;
// 	}


	return iter;
}

/*
Compute an entire image, writing each point to the given bitmap.
Scale the image to the range (xmin-xmax,ymin-ymax).
*/

void compute_image( double xmin, double xmax, double ymin, double ymax, int maxiter, int width, int height, int threads)
{
	int i,j;

	// For every pixel i,j, in the image...
    char buffer[width*height];
	// int max_num_threads = omp_get_max_threads();
	#pragma omp parallel num_threads(threads)
	#pragma omp for collapse(2)
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
     stbi_write_jpg("fractal-images/shared-fractal.jpg", width, height, 1, buffer, 200);

}

int main( int argc, char *argv[] )
{
	// The initial boundaries of the fractal image in x,y space.
	printf("adfasda");

	double xmin=-1.5;
	double xmax= 0.5;
	double ymin=-1.0;
	double ymax= 1.0;

    int width = 1000;
    int height = 1000;
	int threadct = omp_get_max_threads();
	int maxiter=5000;
	printf("adfasda");

	if (argc > 1)
    	threadct = atoi(argv[1]);
	if(argc>2){
		maxiter = atoi(argv[2]);
	}
	// Maximum number of iterations to compute.
	// Higher values take longer but have more detail.

	// Showing the configuration, just in case you want to recreate it.
	printf("Coordinates: %lf %lf %lf %lf\n",xmin,xmax,ymin,ymax);
	printf("Timer started\n");
	printf("adfasda2");

	// clock_t begin = clock();
	// clock_t end = clock();
	
	// double time_spent = omp_get_wtime();
	double start; 
	double end; 
	start = omp_get_wtime(); 
		printf("adfasda3");

	compute_image(xmin,xmax,ymin,ymax,maxiter, width, height,threadct);

	end = omp_get_wtime(); 
	printf("Work took %f seconds\n", end - start);
	// printf("time took for execution of parallel algorithm with shared memory: %f\n", time_spent);
	// testOMP();
	return 0;
}