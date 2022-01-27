void compute_image( double xmin, double xmax, double ymin, 
double ymax, int maxiter, int width, int height){
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
  stbi_write_jpg("fractal-images/fractal.jpg", width, height, 
  1, buffer, 200);
}
