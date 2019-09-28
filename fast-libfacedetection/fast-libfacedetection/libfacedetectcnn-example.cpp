#include <stdio.h>
#include <windows.h>
#include <opencv2/opencv.hpp>
#include "facedetectcnn.h"

#include<iostream>
using namespace std;

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000
using namespace cv;

int main()
{
	//load an image and convert it to gray (single-channel)
	Mat image = imread("C:\\Users\\huiling\\Desktop\\class.jpg"); 
	if(image.empty())
	{
		fprintf(stderr, "Can not load the image file\n");
		return -1;
	}

	int * pResults = NULL; 

    unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    if(!pBuffer)
    {
        fprintf(stderr, "Can not alloc buffer.\n");
        return -1;
    }

	LARGE_INTEGER nFreq;
	LARGE_INTEGER t1;
	LARGE_INTEGER t2;

	double dt;

	QueryPerformanceFrequency(&nFreq);

	QueryPerformanceCounter(&t1);

	/* ÈËÁ³¼ì²âÖ÷º¯Êý */
	pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), (int)(image.cols), (int)(image.rows), (int)image.step);
	
	QueryPerformanceCounter(&t2);
	dt = (t2.QuadPart - t1.QuadPart) / (double)nFreq.QuadPart;

	cout << "LastTIME = " << dt * 1000000 << endl;

    printf("%d faces detected.\n", (pResults ? *pResults : 0));
	Mat result_cnn = image.clone();

	for(int i = 0; i < (pResults ? *pResults : 0); i++)
	{
        short * p = ((short*)(pResults+1)) + 142 * i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int confidence = p[4];
		int angle = p[5];

		printf("face_rect=[%d, %d, %d, %d], confidence=%d, angle=%d\n", x, y, w, h, confidence, angle);
		rectangle(result_cnn, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
	}
	imshow("result_cnn", result_cnn);

	waitKey();

    free(pBuffer);

	return 0;
}
