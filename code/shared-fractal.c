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

double complex calculateZ(double complex z, int lvl, double complex alpha){
	z = cpow(z,lvl) + alpha;
	return z;
}

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


void compute_image( double xmin, double xmax, double ymin, double ymax, int maxiter, int width, int height, int threads)
{
	int i,j;

    char buffer[width*height];

	#pragma omp parallel num_threads(threads)
	#pragma omp for collapse(2)
	for(i=0;i<height;i++) {
		for(j=0;j<width;j++) {

			double x = xmin + i*(xmax-xmin)/width;

			double y = ymin + j*(ymax-ymin)/height;

			int iter = compute_point(x,y,maxiter);

			int gray = 255 * iter / maxiter;
            
            buffer[j*height+i] = (char) gray;
		}

	}
     stbi_write_jpg("fractal-images/shared-fractal.jpg", width, height, 1, buffer, 200);

}


void compute_image_opt( double xmin, double xmax, double ymin, double ymax, int maxiter, int width, int height, char* result, int threads){
    int i, iter;
	double xstep = (xmax-xmin) / (width-1);
	double ystep = (ymax-ymin) / (height-1);

    #pragma omp parallel shared(result, maxiter) private(i,iter) num_threads(threads)
    #pragma omp for schedule(runtime)
    for (i = 0; i < width*height; i++) {

		double x = xmin + (i%width)*xstep;
		double y = ymin + (i/height)*ystep;
		
		iter = compute_point(x,y,maxiter);
        
		int gray = 255 * iter / maxiter;

        result[i] = (char)gray;
    }
	
	stbi_write_jpg("fractal-images/shared-fractal.jpg", width, height, 1, result, 200);

}

void compute_image_rgb( double xmin, double xmax, double ymin, double ymax, int maxiter, int width, int height, char* result, int threads, int num_chanels, int rgb){
    int i, iter;
	char name[60];

	double xstep = (xmax-xmin) / (width-1);
	double ystep = (ymax-ymin) / (height-1);

    #pragma omp parallel shared(result, maxiter) private(i,iter) num_threads(threads)
    #pragma omp for schedule(runtime)
    for (i = 0; i < width*height; i++) {

		double x = xmin + (i%width)*xstep;
		double y = ymin + (i/height)*ystep;
		
		iter = compute_point(x,y,maxiter);
        
		if(rgb==1){
        	result[num_chanels*i ] = (unsigned char) ((int)(iter* sin(iter/maxiter))%255);
        	result[num_chanels*i +1] = (unsigned char)(iter%256);
        	result[num_chanels*i +2] = (unsigned char) ((iter*iter)%255);
		}
		else{
			int gray = 255 * iter / maxiter;
			result[i] = (unsigned char) gray;
		}

    }
	if(rgb==1){
		sprintf(name,"fractal-images/shared/rgb_img%dx%d_%d.jpg", width, height, maxiter);
	}else{	
		sprintf(name,"fractal-images/shared/bw_img%dx%d_%d.jpg", width, height, maxiter);
	}
	stbi_write_jpg(name, width, height, num_chanels, result, 200);

}

int main( int argc, char *argv[] )
{
	FILE *out_file = fopen("code/data/shared.txt", "a");
    if (out_file == NULL) {   
		printf("Error! Could not open file\n"); 
        exit(-1);
    } 
	double xmin=-1.5;
	double xmax= 0.5;
	double ymin=-1.0;
	double ymax= 1.0;

    int width = 1200;
    int height = 1200;
	int threadct = omp_get_max_threads();
	int maxiter=10;
	int rgb = 1;
	if (argc > 1)
    	height = atoi(argv[1]);
	if(argc>2){
		width = atoi(argv[2]);
	}
	if(argc>3){
		maxiter = atoi(argv[3]);
	}

	if(argc>4){
		rgb = atoi(argv[4]);
	}


	printf("Timer started\n");

	double start; 
	double end; 
	int num_chanels;
	if(rgb==1){
		num_chanels=3;
	}
	else{
		num_chanels=1;
	}

	unsigned char buffer[width*height*num_chanels];

	start = omp_get_wtime(); 
	compute_image_rgb(xmin,xmax,ymin,ymax,maxiter, width, height,buffer, threadct, num_chanels, rgb);
	// compute_image(xmin,xmax,ymin,ymax,maxiter, width, height,threadct);
	end = omp_get_wtime(); 
	double time_spent = end - start;
	fprintf(out_file, "%d,%d,%d,%f,%d \n", width, height, maxiter, time_spent,rgb); // write to file 

	printf("Time took for shared memory parallel algorithm with parameters(%dx%d,%d) %f seconds\n",  width, height, maxiter,time_spent);

	return 0;
}