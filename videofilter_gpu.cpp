#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"
#define STRING_BUFFER_LEN 1024
using namespace std;
using namespace cv;
#define SHOW

void print_matrix(unsigned N, float * mat) {
	for(unsigned k = 0; k < N*N; ++k) {
    	printf("%2f ", *(mat+k));
		if ((k+1)%N == 0) {
			printf("\n");
		}
	}
}


void print_clbuild_errors(cl_program program,cl_device_id device)
	{
		cout<<"Program Build failed\n";
		size_t length;
		char buffer[2048];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		cout<<"--- Build log ---\n "<<buffer<<endl;
		exit(1);
	}


unsigned char ** read_file(const char *name) {
  size_t size;
  unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  FILE* fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s",name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  unsigned char **outputstr=(unsigned char **)malloc(sizeof(unsigned char *));
  *outputstr= (unsigned char *)malloc(size);
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s",name);
    exit(-1);
  }

  if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  printf("file size %d\n",size);
  printf("-------------------------------------------\n");
  snprintf((char *)*outputstr,size,"%s\n",*output);
  printf("%s\n",*outputstr);
  printf("-------------------------------------------\n");
  return outputstr;
}


void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}


void checkError(int status, const char *msg) {
	if(status!=CL_SUCCESS)	
		printf("%s\n",msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

int main()
{
     char char_buffer[STRING_BUFFER_LEN];
     cl_platform_id platform;
     cl_device_id device;
     cl_context context;
     cl_context_properties context_properties[] =
     { 
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
     };
     cl_command_queue queue;
     cl_program program;
     cl_kernel kernel;



//--------------------------------------------------------------------

// capture the video data 
VideoCapture camera("./bourne.mp4");
    if(!camera.isOpened())  // check if we succeeded
        return -1;

    const string NAME = "./output.avi";   // Form the new name with container
    int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    Size S = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
	//Size S =Size(1280,720);
	cout << "SIZE:" << S << endl;
	
    VideoWriter outputVideo;                                        // Open the output
        outputVideo.open(NAME, ex, 25, S, true);

    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << NAME << endl;
        return -1;
    }
	time_t start,end;
	double diff,tot;
	tot=0;
	int count=0;
	const char *windowName = "filter";   // Name shown in the GUI window.
    #ifdef SHOW
    namedWindow(windowName); // Resizable window, might not work on Windows.
	#endif

while (true) {
        Mat cameraFrame,displayframe;
        count=count+1;
        if(count > 100) break;
        camera >> cameraFrame;
        time (&start);
        Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);
        Mat grayframe,edge_x,edge_y,edge;
        // create a mat with float number
        //Mat floatframe = Mat(cameraFrame.size(), CV_32FC1);

		// cvtColor : turn cameraFrame to grayframe, defined by CV_BGR2GRAY
        cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);

		int h = grayframe.size.p[0];
        int w = grayframe.size.p[1];

		// create a matrix
        float *image = (float *) malloc(sizeof(float)*h*w);
        for (int i=0; i < h*w; ++i) {
            *(image + i) = (float) grayframe.data[i];
        }
        printf("\n the first 5 element is \n");

//        for (int i =0; i< 5; ++i) {
//            printf("%f  ",image[i]);
//        }

		// create a gaussian kernel
		int k_row = 3;
		int k_col = 3; 
		//float g_kernel[9] = {1,2,1, 2, 4, 2, 1,2,1};
		float g_kernel[9] = {-1, 0, 1,-1, 0, 1, -1, 0, 1};
		for (int i = 0; i<9; ++i) {
			g_kernel[i] = g_kernel[i];
		}

		//printf("the kernel is :\n");
		//print_matrix(3, g_kernel);

//float *input_a=(float *) malloc(sizeof(float)*N*N);
//float *input_b=(float *) malloc(sizeof(float)*N*N);

float *input_a = image;
float *input_b = g_kernel;

float *output = (float *) malloc(sizeof(float)*h*w);  // for the results returned from GPU
int *dimension = (int *) malloc(sizeof(int)*2); // pass the height, weight to the GPU
cl_mem dimension_buf; 
cl_mem input_a_buf; // num_devices elements
cl_mem input_b_buf; // num_devices elements
cl_mem output_buf; // num_devices elements
int status;

// we will pass the pointer dimension to our buff
dimension[0] = h;
dimension[1] = w;

// for the GPU calculation
   time (&start);
     clGetPlatformIDs(1, &platform, NULL);
     clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

     context_properties[1] = (cl_context_properties)platform;
     clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
     context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
     queue = clCreateCommandQueue(context, device, 0, NULL);

     unsigned char **opencl_program=read_file("videofilter_gpu.cl");
     program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
     if (program == NULL)
	{
         printf("Program creation failed\n");
         return 1;
	}	
     int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	 if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
     kernel = clCreateKernel(program, "videofilter_gpu", &status);
     checkError(status, "Failed to create kernel");

 // Input buffers. of type cl_mem
	dimension_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
       2*sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for dimension");

    input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
       h*w*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        k_row*k_col*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        h*w*sizeof(float), NULL, &status); // change to one dimension
    checkError(status, "Failed to create buffer for output");


    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[3];
	cl_event kernel_event,finish_event;
	status = clEnqueueWriteBuffer(queue, dimension_buf, CL_FALSE,
        0, 2*sizeof(int), dimension, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer dimension");	

    status = clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE,
        0, h*w*sizeof(float), input_a, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue, input_b_buf, CL_FALSE,
        0, k_row*k_col*sizeof(float), input_b, 0, NULL, &write_event[2]);
    checkError(status, "Failed to transfer input B");

    // Set kernel arguments.
//    unsigned argi = 0;

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dimension_buf);
    checkError(status, "Failed to set argument 0");

    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3");

	// use w*h work items
    const size_t global_work_size = w*h;

    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        &global_work_size, NULL, 3, write_event, &kernel_event);  // add 
    checkError(status, "Failed to launch kernel");

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
        0, w*h*sizeof(float), output, 1, &kernel_event, &finish_event);

   time (&end);
   diff = difftime (end,start);
   tot += diff;
//   printf ("count = %d, GPU took %.8lf seconds to run.\n", count, diff);

	// Verify results GPU
	for(int j = 0; j < 10; ++j) {
		printf("%f ", output[j]);
	}
	
	Mat newframe = Mat(h, w, CV_32FC1);
	memcpy(newframe.data, output, h*w);

	newframe.convertTo(grayframe, CV_8U);    
	
	// edge detection
	Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT );
    Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT );
    addWeighted( edge_x, 0.5, edge_y, 0.5, 0, edge );

	// TODO: change grayframe to edge
	cvtColor(grayframe, displayframe, CV_GRAY2BGR);
	outputVideo << displayframe;	

	#ifdef SHOW
//        imshow(windowName, displayframe);
    #endif

    // Release local events.
clReleaseEvent(write_event[0]);
clReleaseEvent(write_event[1]);
clReleaseKernel(kernel);
clReleaseCommandQueue(queue);
clReleaseMemObject(input_a_buf);
clReleaseMemObject(input_b_buf);
clReleaseMemObject(output_buf);
clReleaseProgram(program);
clReleaseContext(context);


//--------------------------------------------------------------------
     clFinish(queue);

} // end while
	
	outputVideo.release();
	camera.release();
	printf ("In total, GPU took %.8lf seconds to run.\n", tot );
	
     return 0;
}

