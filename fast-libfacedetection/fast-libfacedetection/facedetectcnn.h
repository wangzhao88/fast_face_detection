#ifndef __FACEDETECNN_H__
#define __FACEDETECNN_H__

#if defined(_ENABLE_AVX2)
#include <immintrin.h>
#endif

#if defined(_ENABLE_NEON)
#include "arm_neon.h"
#define _ENABLE_INT8_CONV
#endif

#if defined(_ENABLE_AVX2)
#define _MALLOC_ALIGN 256
#else
#define _MALLOC_ALIGN 128
#endif

#if defined(_ENABLE_AVX2)&& defined(_ENABLE_NEON)
#error Cannot enable the two if SSE2 AVX and NEON at the same time.
#endif

#if defined(_OPENMP)
#include<omp.h>
#endif

#include<string.h>
#include<stdio.h>
#include<math.h>
#include<float.h>
#include<malloc.h>
#include<stdbool.h>
#include"MyVector.h"

void* myAlloc(size_t size);
void myFree_(void* ptr);
#define myFree(ptr) (myFree_(*(ptr)), *(ptr) = 0);

#define FILTERS_MAX_NUM		1000

#ifndef MIN
#  define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

typedef struct NormalizedBBox_
{
	float xmin;
	float ymin;
	float xmax;
	float ymax;
}NormalizedBBox;

typedef struct BoxScore_
{
	float score;
	NormalizedBBox box;
}BoxScore;

typedef struct FaceRect_
{
	float score;
	int x;
	int y;
	int w;
	int h;
}FaceRect;

typedef struct FaceResult_
{
	FaceRect faces[256];
	int num;
}FaceResult;

typedef struct CDataBlob_
{
	float * data_float;
	signed char * data_int8;
	int width;
	int height;
	int channels;
	int floatChannelStepInByte;
	int int8ChannelStepInByte;
	float int8float_scale;
	bool int8_data_valid;
}CDataBlob;

typedef struct Filters_
{
	CDataBlob *filters;
	int tail;
	int pad;
	int stride;
	float scale;
}Filters;

void setNULL(CDataBlob *data);
bool create(CDataBlob *data, int width, int height, int channel);
void CDataBlobInit1(CDataBlob *data);
void CDataBlobInit2(CDataBlob *data, int width, int height, int channel);
void CDataBlobDeinit(CDataBlob *data);
bool setInt8DataFromCaffeFormat(CDataBlob *data, signed char * pData, int dataWidth, int dataHeight, int dataChannels);
bool setFloatDataFromCaffeFormat(CDataBlob *data, float * pData, int dataWidth, int dataHeight, int dataChannels);
bool setDataFromImage(CDataBlob *data, const unsigned char *imgData, int imgWidth, int imgHeight, int imgChannels, int imgWidthStep, int *pChannelMean);
bool setDataFrom3x3S2P1to1x1S1P0FromImage(CDataBlob *data, const unsigned char *imgData, int imgWidth, int imgHeight, int imgChannels, int imgWidthStep, int *pChannelMean);
float getElementFloat(CDataBlob *data, int x, int y, int channel);
int getElementint8(CDataBlob *data, int x, int y, int channel);

bool convolution(CDataBlob *inputData, const Filters* filters, CDataBlob *outputData);
bool maxpooling2x2S2(const CDataBlob *inputData, CDataBlob *outputData);
bool concat4(const CDataBlob *inputData1, const CDataBlob *inputData2, const CDataBlob *inputData3, const CDataBlob *inputData4, CDataBlob *outputData);
bool scale(CDataBlob *dataBlob, float scale);
bool relu(const CDataBlob *inputOutputData);
bool priorbox(const CDataBlob *featureData, const CDataBlob *imageData, int num_sizes, float *pWinSizes, CDataBlob *outputData);
bool normalize(CDataBlob *inputOutputData, float *scale);
bool blob2vector(const CDataBlob * inputData, CDataBlob * outputData, bool isFloat);
bool detection_output(const CDataBlob *priorbox, const CDataBlob *loc, const CDataBlob *conf, float overlap_threshold, float confinence_threshold, int top_k, int keep_top_k, CDataBlob *outputData);
bool softmax1vector2class(const CDataBlob *inputOutputData);
cvector objectdetect_cnn(unsigned char *rgbImageData, int width, int height, int step);
int *facedetect_cnn(unsigned char *result_buffer, unsigned char *rgb_image_data, int width, int height, int step);
int TPQuickIntoSortedArrayPairI(int *piArray1, int *piArray2, unsigned char ucOrder, int iLeft, int iRight);

void CDataBlobInit1hlhlhlhl(CDataBlob *data);
#endif
