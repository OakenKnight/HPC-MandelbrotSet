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

struct RGB {
    unsigned char R;
    unsigned char G;
    unsigned char B;
};


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

void compute_image( double xmin, double xmax, double ymin, double ymax, int maxiter, int width, int height, int num_chanels, int rgb)
{
	int i,j;
    unsigned char buffer[width*height*num_chanels];
	char name[60];

	for(i=0;i<height;i++) {
		for(j=0;j<width;j++) {

			double x = xmin + i*(xmax-xmin)/width;
			double y = ymin + j*(ymax-ymin)/height;

			int iter = compute_point(x,y,maxiter);

            if(rgb==1){
				buffer[3*(j*height+i)] = (unsigned char) ((int)(iter* sin(iter/maxiter))%255);
				buffer[3*(j*height+i)+1] = (unsigned char) ((iter*iter)%255);
				buffer[3*(j*height+i)+2] = (unsigned char)(iter%256);
			}else{
				int gray = 255 * iter / maxiter; 
            	buffer[j*height+i] = (char) gray;
			}
          
		}

	}
	if(rgb==1){
		sprintf(name,"fractal-images/seq/rgb_img%dx%d_%d.jpg", width, height, maxiter);
	}else{	
		sprintf(name,"fractal-images/seq/bw_img%dx%d_%d.jpg", width, height, maxiter);
	}
    stbi_write_jpg(name, width, height, num_chanels, buffer,  100 );
	
}
int main( int argc, char *argv[] )
{
	FILE *out_file = fopen("code/data/seq.txt", "a");
    if (out_file == NULL) {   
		printf("Error! Could not open file\n"); 
        exit(-1);
    } 
         
	double xmin=-1.5;
	double xmax= 0.5;
	double ymin=-1.0;
	double ymax= 1.0;

    int width = 500;
    int height = 500;
	
	int maxiter=1000;
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


	printf("Timer started\n");

	clock_t begin = clock();
	compute_image(xmin,xmax,ymin,ymax,maxiter, width, height, num_chanels, rgb);
	clock_t end = clock();

	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    
	fprintf(out_file, "%d,%d,%d,%f,%d\n", width,height,maxiter,time_spent,rgb); // write to file 

	printf("time took for execution of sequential algorithm with parameters(%dx%d,%d): %f\n", width,height,maxiter,time_spent);

	return 0;
}