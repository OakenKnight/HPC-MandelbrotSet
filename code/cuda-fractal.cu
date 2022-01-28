
  
#include <stdio.h>
#include <unistd.h>
#include <err.h>
#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


void checkErr(cudaError_t err, char* msg)
{
    if (err != cudaSuccess){
        fprintf(stderr, "%s (error code %d: '%s'", msg, err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__device__ int compute_point( double x, double y, int max )
{
	double zr = 0;
    double zi = 0;
    double zrsqr = 0;
    double zisqr = 0;

    int iter;

    for (iter = 0; iter < max; iter++){
		zi = zr * zi;
		zi += zi;
		zi += y;
		zr = (zrsqr - zisqr) + x;
		zrsqr = zr * zr;
		zisqr = zi * zi;
		
		if (zrsqr + zisqr >= 4.0) break;
    }
	
    return iter;
}

__global__ void compute_image_kernel(double xmin, double xmax, double ymin, double ymax, int maxiter, int width, int height, char* result) 
{
    int pix_per_thread = width * height / (gridDim.x * blockDim.x);
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = pix_per_thread * tId;
    int iter;
	double xstep = (xmax-xmin) / (width-1);
	double ystep = (ymax-ymin) / (height-1);

    for (int i = offset; i < offset + pix_per_thread; i++){
        
        int iw = i%width;
        int ih = i/height;
        
        double x = xmin + iw*xstep;
		double y = ymin + ih*ystep;

        iter = compute_point(x, y, maxiter);
  		int gray = 255 * iter / maxiter;

        result[ih * width + iw]  = (char)gray;    
        
    
    }

    if (gridDim.x * blockDim.x * pix_per_thread < width * height && tId < (width * height) - (blockDim.x * gridDim.x)){
        int i = blockDim.x * gridDim.x * pix_per_thread + tId;
        
        int iw = i%width;
        int ih = i/height;
        
        double x = xmin + iw*xstep;
		double y = ymin + ih*ystep;

        iter = compute_point(x, y, maxiter);
  		int gray = 255 * iter / maxiter;

        result[ih * width + iw]  = (char)gray;    
    }
    
    
}

static void run(double xmin, double xmax, double ymin, double ymax, int width, int height, int max_iter, char* result)
{
    dim3 numBlocks(width,height);
    cudaError_t err = cudaSuccess;
    compute_image_kernel<<<width, height>>>(xmin, xmax, ymin, ymax, max_iter, width, height, result);
    checkErr(err, "Failed to run Kernel");
    void *data = malloc(height * width * sizeof(char));
    err = cudaMemcpy(data, result, width * height * sizeof(char), cudaMemcpyDeviceToHost);
    checkErr(err, "Failed to copy result back");

    stbi_write_jpg("fractal-images/cuda-fractal.jpg", width, height, 1, data, 200);

}

int main(int argc, char** argv){
    
	FILE *out_file = fopen("code/data/cuda.txt", "a");
    if (out_file == NULL) {   
		printf("Error! Could not open file\n"); 
        exit(-1);
    } 
    cudaError_t err = cudaSuccess;

    int width = 300;
    int height = 300;
    int max_iter = 25000;
        
    double xmin=-1.5;
	double xmax= 0.5;
	double ymin=-1.0;
	double ymax= 1.0;


	if (argc > 1)
    	height = atoi(argv[1]);
	if(argc>2){
		width = atoi(argv[2]);
	}
	if(argc>3){
		max_iter = atoi(argv[3]);
	}
	clock_t begin = clock();

    char *result = NULL;
    err = cudaMalloc(&result, width*height*sizeof(char));
    checkErr(err, "Failed to allocate result memory on gpu");

    run(xmin, xmax, ymin, ymax, width, height, max_iter, result);
	clock_t end = clock();
double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    
	fprintf(out_file, "%f & ", time_spent); // write to file 

	printf("time took for execution of sequential algorithm: %f\n", time_spent);

    cudaFree(result);
	return 0;
}