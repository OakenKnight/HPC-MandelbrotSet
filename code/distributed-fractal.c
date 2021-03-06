#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <complex.h>
#include <time.h>
#include <string.h>
#include <omp.h>
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

void compute_image( double xmin, double xmax, double ymin, double ymax, int maxiter, int width, int height, int start, int end, char* result, int rank, int size, int rgb){
	printf("Working with rank %d and size %d\n\n", rank, size);
    
	int i, iter;
	double xstep = (xmax-xmin) / (width-1);
	double ystep = (ymax-ymin) / (height-1);

//    #pragma omp parallel shared(result, maxiter, start, end) private(i,iter)
//	  #pragma omp for
    #pragma omp parallel for shared(result, maxiter, start, end) private(i,iter)
    for (i = start; i < end; i++) {

		double x = xmin + (i%width)*xstep;
		double y = ymin + (i/height)*ystep;
		
		iter = compute_point(x,y,maxiter);
        
		int gray = 255 * iter / maxiter;

		if(rgb==1){
			result[3*(i-start)] = (char) ((int)(iter* sin(iter/maxiter))%255);
			result[3*(i-start)+1] = (char)(iter%256);
			result[3*(i-start)+2] = (char) ((iter*iter)%255);
		}else{
			result[i-start] = (char) gray;
		}
        
    }

}

void run(int maxiter, int current_processor, int processors_amount, int width, int height, double xmin, double xmax, double ymin, double ymax, FILE *file, int num_chanels, int rgb) 
{
	char* result = (char *) malloc(width*height*num_chanels);
    int part_width =  (width*height) / processors_amount;
    int start = current_processor * part_width;
    char* partial_result = (char *) malloc(part_width*num_chanels); 
	char name[60];

	clock_t begin;
	clock_t end;


    if (current_processor == 0 ) {
        result = (char *) malloc(width*height*num_chanels); 
		printf("Timer started\n");

		begin = clock();
    }

    MPI_Scatter(result, part_width*num_chanels, MPI_CHAR, partial_result, part_width*num_chanels, MPI_CHAR, 0, MPI_COMM_WORLD);
 	
	printf("Distributing work on node %d out of %d\n", current_processor, processors_amount);

	compute_image(xmin,xmax,ymin,ymax,maxiter,width, height, start, start+part_width, partial_result, current_processor, processors_amount, rgb);
   	
	MPI_Gather(partial_result, part_width*num_chanels, MPI_CHAR, result, part_width*num_chanels, MPI_CHAR, 0, MPI_COMM_WORLD);

	  if (current_processor == 0) {
		end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

		printf("time took for execution of distributed algorithm with parameters(%dx%d,%d): %f\n", width,height,maxiter,time_spent);
		
		fprintf(file, "%d,%d,%d,%f,%d \n", width, height, maxiter, time_spent,rgb);

		if(rgb==1){
			sprintf(name,"fractal-images/distributed/rgb_img%dx%d_%d.jpg", width, height, maxiter);
		}else{	
			sprintf(name,"fractal-images/distributed/bw_img%dx%d_%d.jpg", width, height, maxiter);
		}
		stbi_write_jpg(name, width, height, num_chanels, result, 300);
		
        free(result);
    }

    free(partial_result);

}

int  main( int argc, char **argv ){


	FILE *out_file = fopen("code/data/distributed.txt", "a");
    if (out_file == NULL) {   
		printf("Error! Could not open file\n"); 
        exit(-1);
    } 

	double xmin=-1.5;
	double xmax= 0.5;
	double ymin=-1.0;
	double ymax= 1.0;

    int width = 120;
    int height = 120;
	int maxiter=5000;

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
	int num_chanels;
	if(rgb==1){
		num_chanels=3;
	}
	else{
		num_chanels=1;
	}
    int size, rank;
	
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	run(maxiter, rank, size, width, height, xmin, xmax, ymin, ymax, out_file,num_chanels, rgb);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
	return 0;
}	
