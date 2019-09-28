#include "facedetectcnn.h"
#include <cmath>

#if defined( __WIN__) || defined(_WINDOWS)
#define SSE_256ELEMENT(vec, idx) vec.m256_f32[(idx)]
#else
#define SSE_256ELEMENT(vec, idx) vec[(idx)]
#endif

#if !defined(_ENABLE_OPENMP_SIMD) && ((defined(_OPENMP) && (_OPENMP >= 201307L)))
#  define _ENABLE_OPENMP_SIMD
#elif defined(__cilk)
#  define _ENABLE_CILKPLUS
#endif

void CDataBlobInit1(CDataBlob *data)
{
	data->data_float = 0;
	data->data_int8 = 0;
	data->width = 0;
	data->height = 0;
	data->channels = 0;
	data->floatChannelStepInByte = 0;
	data->int8ChannelStepInByte = 0;
	data->int8float_scale = 1.0f;
	data->int8_data_valid = false;
}

void setNULL(CDataBlob *data)
{
	if (NULL == data)
	{
		printf("error\n");
		return;
	}

	if (data->data_float)
	{
		myFree(&(data->data_float));
	}

	if (data->data_int8)
	{
		myFree(&(data->data_int8));
	}

	data->width = data->height = data->channels = data->floatChannelStepInByte = data->int8ChannelStepInByte = 0;

	data->int8float_scale = 1.0f;
	data->int8_data_valid = false;
}

bool create(CDataBlob *data, int width, int height, int channel)
{
	setNULL(data);

	data->width = width;
	data->height = height;
	data->channels = channel;

	int remBytes = (sizeof(float)* channel) % (_MALLOC_ALIGN / 8);

	if (0 == remBytes)
	{
		(data)->floatChannelStepInByte = channel * sizeof(float);
	}
	else
	{
		data->floatChannelStepInByte = (channel * sizeof(float)) + (_MALLOC_ALIGN / 8) - remBytes;
	}

	data->data_float = (float*)myAlloc(width * height * data->floatChannelStepInByte);

	remBytes = (sizeof(char)* channel) % (_MALLOC_ALIGN / 8);

	if (0 == remBytes)
	{
		data->int8ChannelStepInByte = channel * sizeof(char);
	}
	else
	{
		data->int8ChannelStepInByte = (channel * sizeof(char)) + (_MALLOC_ALIGN / 8) - remBytes;
	}
	data->data_int8 = (signed char*)myAlloc(width * height *data->int8ChannelStepInByte);

	if (NULL == data->data_float)
	{
		printf("Cannot alloc memory for float data blob:%d * %d * %d", width, height, channel);
		return false;
	}

	if (NULL == data->data_int8)
	{
		printf("Cannot alloc memory for uint8 data blob:%d * %d * %d", width, height, channel);
		return false;
	}

	int r = 0, c = 0, ch = 0;

	for (r = 0; r < data->height; r++)
	{
		for (c = 0; c < data->width; c++)
		{
			int pixel_end = data->floatChannelStepInByte / sizeof(float);
			float *pF = (float*)(data->data_float + (r * data->width + c) * data->floatChannelStepInByte / sizeof(float));
			for (ch = data->channels; ch < pixel_end; ch++)
			{
				pF[ch] = 0;
			}

			pixel_end = data->int8ChannelStepInByte / sizeof(char);
			char *pI = (char*)(data->data_int8 + (r * data->width + c) * data->int8ChannelStepInByte / sizeof(char));
			for (ch = data->channels; ch < pixel_end; ch++)
			{
				pI[ch] = 0;
			}
		}
	}

	return true;
}

void CDataBlobInit2(CDataBlob *data, int width, int height, int channel)
{
	data->data_float = 0;
	data->data_int8 = 0;
	 
	create(data, width, height, channel);
}

void CDataBlobDeinit(CDataBlob *data)
{
	setNULL(data);
}

bool setInt8DataFromCaffeFormat(CDataBlob *data, signed char * pData, int dataWidth, int dataHeight, int dataChannels)
{
	if (NULL == pData)
	{
		return false;
	}

	if (dataWidth != dataWidth ||
		dataHeight != dataHeight ||
		dataChannels != data->channels)
	{
		printf("The dim of the data can not match that of the Blob.\n");
		return false;
	}

	int row = 0, col = 0, ch = 0;
	for (row = 0; row < dataHeight; row++)
	{
		for (col = 0; col < dataWidth; col++)
		{
			signed char *p = (data->data_int8 + (dataWidth * row + col) * data->int8ChannelStepInByte / sizeof(char));
			for (ch = 0; ch < dataChannels; ch++)
			{
				p[ch] = pData[ch * dataHeight * dataWidth + row * dataWidth + col];
			}
		}
	}
	
	return true;
}

bool setFloatDataFromCaffeFormat(CDataBlob *data, float *pData, int dataWidth, int dataHeight, int dataChannels)
{
	if (NULL == pData)
	{
		printf("The input image data is null.\n");
		return false;
	}

	if (dataWidth != data->width ||
		dataHeight != data->height ||
		dataChannels != data->channels)
	{
		printf("The dim of the data can not match that of the Blob.\n");
		return false;
	}

	int row, col, ch;
	for (row = 0; row < dataHeight; ++row)
	{
		for (col = 0; col < dataWidth; ++col)
		{
			float *p = (data->data_float + (dataWidth * row + col) * data->floatChannelStepInByte / sizeof(float));
			for (ch = 0; ch < dataChannels; ++ch)
			{
				p[ch] = pData[ch * dataHeight * dataWidth + row * dataWidth + col];
			}
		}
	}

	return true;
}

bool setDataFromImage(CDataBlob *data, const unsigned char *imgData, int imgWidth, int imgHeight, int imgChannels, int imgWidthStep, int *pChannelMean)
{
	if (NULL == imgData)
	{
		printf("The input image data is null.\n");
		return false;
	}

	if (NULL == pChannelMean)
	{
		printf("The mean values is null.\n");
		return false;
	}

	create(data, imgWidth, imgHeight, imgChannels);

	int r = 0, c = 0, ch = 0;
	for (r = 0; r < imgHeight; r++)
	{
		for (c = 0; c < imgWidth; c++)
		{
			const unsigned char * pImgData = imgData + imgWidthStep * r + imgChannels * c;
			float * pBlobData = data->data_float + (data->width * r + c) * data->floatChannelStepInByte / sizeof(float);

			for (ch = 0; ch < imgChannels; ch++)
			{
				pBlobData[ch] = (float)(pImgData[ch] - pChannelMean[ch]);
			}
		}
	}

	return true;
}

bool setDataFrom3x3S2P1to1x1S1P0FromImage(CDataBlob *data, const unsigned char *imgData, int imgWidth, int imgHeight, int imgChannels, int imgWidthStep, int *pChannelMean)
{
	if (NULL == imgData)
	{
		printf("The input image data is null");
		return false;
	}

	if (NULL == pChannelMean)
	{
		printf("The mean values is null. \n");
		return false;
	}

	if (imgChannels != 3)
	{
		printf("The input image must be a 3-channel RGB image. \n");
		return false;
	}

	create(data, (imgWidth + 1) / 2, (imgHeight + 1) / 2, 27);

	memset(data->data_float, 0, data->width * data->height * data->floatChannelStepInByte);

#if defined(_OPENMP)
#pragma omp parallel for
#endif

	float *pData = NULL;
	int r = 0, c = 0, fy = 0, fx = 0, output_channel_offset = 0;
	const unsigned char * pImgData = NULL;

	for (r = 0; r < data->height; ++r)
	{
		for (c = 0; c < data->width; ++c)
		{
			pData = data->data_float + (r * data->width + c) * data->floatChannelStepInByte / sizeof(float);

			for (fy = -1; fy <= 1; fy++)
			{
				int srcy = r * 2 + fy;

				if (srcy < 0 || srcy >= imgHeight)
				{
					continue;
				}

				for (fx = -1; fx <= 1; fx++)
				{
					int srcx = c * 2 + fx;

					if (srcx < 0 || srcx >= imgWidth)
					{
						continue;
					}

					pImgData = imgData + imgWidthStep * srcy + imgChannels * srcx;

					output_channel_offset = ((fy + 1) * 3 + fx + 1) * 3;

					pData[output_channel_offset] = (float)(pImgData[0] - pChannelMean[0]);
					pData[output_channel_offset + 1] = (float)(pImgData[1] - pChannelMean[1]);
					pData[output_channel_offset + 2] = (float)(pImgData[2] - pChannelMean[2]);
				}
			}
		}
	}
	return true;
}

float getElementFloat(CDataBlob *data, int x, int y, int channel)
{
	if (data->data_float)
	{
		if (x >= 0 && x < data->width &&
			y >= 0 && y < data->height &&
			channel >= 0 && channel < data->channels)
		{
			float * p = (float*)(data->data_float + (y * data->width + x) * data->floatChannelStepInByte / sizeof(float));
			return p[channel];
		}
	}

	return 0.f;
}

int getElementint8(CDataBlob *data, int x, int y, int channel)
{
	if (data->data_int8 && data->int8_data_valid)
	{
		if (x >= 0 && x < data->width &&
			y >= 0 && y < data->height &&
			channel >= 0 && channel < data->channels)
		{
			signed char *p = data->data_int8 + (y * data->width + x) * data->int8ChannelStepInByte / sizeof(int);
			return p[channel];
		}
	}

	return 0;
}

void *myAlloc(size_t size)
{
	char *ptr, *ptr0;
	ptr0 = (char*)malloc((size_t)(size + _MALLOC_ALIGN * ((size >= 4096) + 1) + sizeof(char*)));

	if (NULL == ptr0)
	{
		return 0;
	}

	ptr = (char*)(((size_t)(ptr0 + sizeof(char*)+1) + _MALLOC_ALIGN - 1) & ~(size_t)(_MALLOC_ALIGN - 1));
	*(char**)(ptr - sizeof(char*)) = ptr0;

	return ptr;
}

void myFree_(void *ptr)
{
	if (ptr)
	{
		if (((size_t)ptr & (_MALLOC_ALIGN - 1)) != 0)
		{
			return;
		}
		free(*((char**)ptr - 1));
	}
}

inline float dotProductFloatChGeneral(float *p1, float *p2, int num, int lengthInBytes)
{
#if defined(_ENABLE_NEON) && !defined(_ENABLE_INT8_CONV)
	float sum = 0.0f;
	float32x4_t a, b;
	float32x4_t result_vec;

	result_vec = vdupq_n_f32(0);

	for (int i = 0; i < num; i += 4)
	{
		a = vld1q_f32(p1 + i);
		b = vld1q_f32(p2 + i);
		result_vec = vmlaq_f32(result_vec, a, b);
	}

	sum += vget_lane_f32(result_vec, 0);
	sum += vget_lane_f32(result_vec, 1);
	sum += vget_lane_f32(result_vec, 2);
	sum += vget_lane_f32(result_vec, 3);

	return sum;
#elif defined(_ENABLE_AVX2) && !defined(_ENABLE_INT8_CONV)
	float sum = 0;
	int end = lengthInBytes / sizeof(float);

	__m256 sumvec = _mm256_setzero_ps();
	__m256 avec, bvec;
	for (int i = 0; i < end; i += 8)
	{
		avec = _mm256_load_ps(p1 + i);
		bvec = _mm256_load_ps(p2 + i);

		sumvec = _mm256_fmadd_ps((avec, bvec, sumvec);
	}
	sumvec = _mm256_hadd_ps(sumvec, sumvec);
	sumvec = _mm256_hadd_ps(sumvec, sumvec);
	sum += SSE_256ELEMENT(sumvec, 0);
	sum += SSE_256ELEMENT(sumvec, 4);

	return sum;
#else
	float sum = 0;

#if defined(_ENABLE_OPENMP_SIMD)
#pragma omp simd reduction(+:sum)
#endif
	int i = 0;
	for (i = 0; i < num; i++)
	{
		sum += (p1[i] * p2[i]);
	}
	return sum;
#endif
}

inline int dotProductInt8ChGeneral(signed char * p1, signed char * p2, int num, int lengthInBytes)
{
#if defined(_ENABLE_NEON) && defined(_ENABLE_INT8_CONV)

	int sum = 0;
	int8x8x2_t a, b;
	int16x8_t result_vec;
	int32x4_t d;

	result_vec = vdupq_n_s16(0);
	for (int i = 0; i < num; i += 16)
	{
		a = vld2_s8(p1 + i);
		b = vld2_s8(p2 + i);
		result_vec = vmlal_s8(result_vec, a.val[0], b.val[0]);
		result_vec = vmlal_s8(result_vec, a.val[1], b.val[1]);
	}
	d = vpaddlq_s16(result_vec);
	sum += vgetq_lane_s32(d, 0);
	sum += vgetq_lane_s32(d, 1);
	sum += vgetq_lane_s32(d, 2);
	sum += vgetq_lane_s32(d, 3);

	return sum;
#elif defined(_ENABLE_AVX2) && defined(_ENABLE_INT8_CONV)
	int sum = 0;
	int i = 0;

	short sumarray[16];

	__m256i temp_sum;
	__m128i ac, bc;
	__m256i as, bs;
	for (; i < num; i += 16)
	{
		ac = _mm_load_si128((__m128i*)(p1 + i));
		bc = _mm_load_si128((__m128i*)(p2 + i));
		as = _mm256_cvtepi8_epi16(ac);
		bs = _mm256_cvtepi8_epi16(bc);
		temp_sum = _mm256_mullo_epi16(as, bs);
		temp_sum = _mm256_hadd_epi16(temp_sum, temp_sum);
		temp_sum = _mm256_hadd_epi16(temp_sum, temp_sum);
		_mm256_store_si256((__m256i*)sumarray, temp_sum);
		sum += ((int)(sumarray[0] + (int)(sumarray[1]) + (int)(sumarray[8]) + (int)(sumarray[9]));
	}
	return sum;
#else

	int sum = 0;
#if defined(_ENABLE_OPENMP_SIMD)
#pragma omp simd reduction(+:sum)
#endif
	int i;
	for (i = 0; i < num; i++)
	{
		sum += ((int)(p1[i]) * (int)p2[i]);
	}
	return sum;
#endif
}

bool convolutionFloat1x1P0S1(const CDataBlob *inputData, const Filters *filters, CDataBlob *outputData)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	int row, col, ch;
	for (row = 0; row < outputData->height; row++)
	{
		for (col = 0; col < outputData->width; col++)
		{
			float * pOut = (outputData->data_float + (row * outputData->width + col) * outputData->floatChannelStepInByte / sizeof(float));
			float * pIn = (inputData->data_float + (row * inputData->width + col) * inputData->floatChannelStepInByte / sizeof(float));
			for (ch = 0; ch < outputData->channels; ch++)
			{
				float * pF = (float*)(filters->filters[ch].data_float);
				pOut[ch] = dotProductFloatChGeneral(pIn, pF, inputData->channels, inputData->floatChannelStepInByte);
			}
		}
	}
	return true;
}

bool convolutionInt81x1P0S1(const CDataBlob *inputData, const Filters * filters, CDataBlob * outputData)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	int row, col, ch;
	for (row = 0; row < outputData->height; row++)
	{
		for (col = 0; col < outputData->width; col++)
		{
			float * pOut = (outputData->data_float + (row * outputData->width + col) * outputData->floatChannelStepInByte / sizeof(float));
			signed char * pIn = (inputData->data_int8 + (row * inputData->width + col) * inputData->int8ChannelStepInByte / sizeof(char));
			for (ch = 0; ch < outputData->channels; ch++)
			{
				signed char * pF = (filters->filters[ch].data_int8);
				pOut[ch] = (float)dotProductInt8ChGeneral(pIn, pF, inputData->channels, inputData->int8ChannelStepInByte);
			}
		}
	}
	return true;
}

bool convolutionFloat3x3P1ChGeneral(const CDataBlob *inputData, const Filters * filters, CDataBlob * outputData)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	int row, col, ch;
	for (row = 0; row < outputData->height; ++row)
	{
		int elementStepInFloat = inputData->floatChannelStepInByte / sizeof(float);
		int stride = filters->stride;
		int src_centery = row *stride;
		for (col = 0; col < outputData->width; ++col)
		{
			int srcx_start = col * stride - 1;
			int srcx_end = srcx_start + 3;
			srcx_start = MAX(0, srcx_start);
			srcx_end = MIN(srcx_end, inputData->width);
			int num_pixels = srcx_end - srcx_start;
			int num_pixels_infloat = (srcx_end - srcx_start) * elementStepInFloat;

			for (ch = 0; ch < outputData->channels; ++ch)
			{
				int srcy = src_centery - 1;

				float *pIn = (inputData->data_float + (srcy * inputData->width + srcx_start) * elementStepInFloat);
				float *pF = (filters->filters[ch].data_float) + (srcx_start - col * stride + 1) * elementStepInFloat;
				float *pOut = (outputData->data_float + (row*outputData->width + col) * outputData->floatChannelStepInByte / sizeof(float));
				pOut[ch] = 0;

				{
					if (srcy >= 0)
					{
						pOut[ch] += dotProductFloatChGeneral(pIn, pF, 
															 num_pixels_infloat, 
															 num_pixels_infloat * sizeof(float));
					}
				}
				{
					++srcy;
					{
						pIn += (inputData->width * elementStepInFloat);
						pOut[ch] += dotProductFloatChGeneral(pIn, pF + (3 * elementStepInFloat),
															 num_pixels_infloat,
															 num_pixels_infloat * sizeof(float));
					}
				}
				{
					++srcy;
					if (srcy < inputData->height)
					{
						pIn += (inputData->width * elementStepInFloat);
						pOut[ch] += dotProductFloatChGeneral(pIn, pF + (6 * elementStepInFloat),
							num_pixels_infloat,
							num_pixels_infloat * sizeof(float));
					}
				}
			}
		}
	}
	return true;
}

bool convolutionInt83x3P1ChGeneral(const CDataBlob *inputData, const Filters* filters, CDataBlob *outputData)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	int row, col, ch;
	for (row = 0; row < outputData->height; row++)
	{
		int elementStep = inputData->int8ChannelStepInByte;
		int stride = filters->stride;
		int src_centery = row * stride;
		for (col = 0; col < outputData->width; col++)
		{
			int srcx_start = col * stride - 1;
			int srcx_end = srcx_start + 3;
			srcx_start = MAX(0, srcx_start);
			srcx_end = MIN(srcx_end, inputData->width);
			int num_pixels_inbytes = (srcx_end - srcx_start) * elementStep;

			for (ch = 0; ch < outputData->channels; ch++)
			{
				int srcy = src_centery - 1;

				signed char * pIn = (inputData->data_int8 + (srcy * inputData->width + srcx_start) * elementStep);
				signed char * pF = (filters->filters[ch].data_int8) + ((srcx_start - col * stride + 1)) * elementStep;
				float * pOut = (outputData->data_float + (row * outputData->width + col) * outputData->floatChannelStepInByte / sizeof(float));
				pOut[ch] = 0;

				{
					if (srcy >= 0)
					{
						pOut[ch] += dotProductInt8ChGeneral(pIn, pF,
							num_pixels_inbytes,
							num_pixels_inbytes);
					}
				}
				{
					srcy++;
					{
						pIn += (inputData->width * elementStep);
						pOut[ch] += dotProductInt8ChGeneral(pIn, pF + (3 * elementStep),
							num_pixels_inbytes,
							num_pixels_inbytes);
					}
				}
				{
					srcy++;
					if (srcy < inputData->height)
					{
						pIn += (inputData->width * elementStep);
						pOut[ch] = dotProductInt8ChGeneral(pIn, pF + (6 * elementStep),
							num_pixels_inbytes,
							num_pixels_inbytes);
					}
				}
			}
		}
	}
	return true;
}

bool convertFloat2Int8(CDataBlob *dataBlob)
{
	if (NULL == dataBlob->data_float || NULL == dataBlob->data_int8)
	{
		return false;
	}

	float maxval = -FLT_MAX;
#if defined(_ENABLE_NEON)
	float32x4_t maxvalvec = vdupq_n_f32(-FLT_MAX);
	float32x4_t scalevec;
#elif defined(_ENABLE_AVX2)
	__m256 scalevec;
#endif

	float scale = 1.f;

	if (dataBlob->int8_data_valid)
	{
		return true;
	}

	int row, col, ch;
	for (row = 0; row < dataBlob->height; row++)
	{
		for (col = 0; col < dataBlob->width; col++)
		{
			float *pF = (dataBlob->data_float + (row*dataBlob->width + col) * dataBlob->floatChannelStepInByte / sizeof(float));
		
#if defined(_ENABLE_NEON)
			for (ch = 0; ch < dataBlob->channels; ch += 4)
			{
				float32x4_t a;
				a = vld1q_f32(pF + ch);
				a = vabsq_f32(a);
				maxvalvec = vmaxq_f32(maxvalvec, a);
			}
#else

#if defined(_ENABLE_OPNMP_SIMD)
#pragma omp simd reduction(max:maxval)
#endif
			for (ch = 0; ch < dataBlob->channels; ch++)
			{
				float tmp;

				tmp = pF[ch];
				tmp = tmp * ((tmp > 0) * 2 - 1);
				maxval = MAX(maxval, tmp);
			}
#endif
		}
	}
#if defined(_ENABLE_NEON)
	{
		float tmp;
		tmp = vgetq_lane_f32(maxvalvec, 0);
		maxval = MAX(maxval, tmp);
		tmp = vgetq_lane_f32(maxvalvec, 1);
		maxval = MAX(maxval, tmp);
		tmp = vgetq_lane_f32(maxvalvec, 2);
		maxval = MAX(maxval, tmp);
		tmp = vgetq_lane_f32(maxvalvec, 3);
		maxval = MAX(maxval, tmp);
	}
#endif
	scale = 127.f / (maxval + FLT_EPSILON);

#if defined(_ENABLE_NEON)
	scalevec = vdupq_n_f32(scale);
#elif defined(_ENABLE_AVX2)
	scalevec = _mm256_set1_ps(scale);
#endif

#if defined(_OPENMP)
#pragma omp parallel for
#endif

	for (row = 0; row < dataBlob->height; row++)
	{
		for (col = 0; col < dataBlob->width; col++)
		{
			float * pF = (dataBlob->data_float + (row * dataBlob->width + col) * dataBlob->floatChannelStepInByte / sizeof(float));
			signed char * pI = (dataBlob->data_int8 + (row * dataBlob->width + col) * dataBlob->int8ChannelStepInByte / sizeof(char));
		
#if defined(_ENABLE_NEON)
			for (ch = 0; ch < dataBlob->channels; ch += 4)
			{
				float tmp;
				float32x4_t a = vld1q_f32(pF + ch);
				float32x4_t resultvec = vmulq_f32(a, scalevec);

				tmp = vgetq_lane_f32(resultvec, 0);
				pI[ch] = (signed char)(tmp + ((tmp>0) - 0.5f));
				tmp = vgetq_lane_f32(resultvec, 1);
				pI[ch + 1] = (signed char)(tmp + ((tmp > 0) - 0.5f));
				tmp = vgetq_lane_f32(resultvec, 2);
				pI[ch + 2] = (signed char)(tmp + ((tmp > 0) - 0.5f));
				tmp = vgetq_lane_f32(resultvec, 3);
				pI[ch + 3] = (signed char)(tmp + ((tmp > 0) - 0.5f));
			}
#else
#if defined(_ENABLE_OPENMP_SIMD)
#pragma omp simd
#endif
			for (ch = 0; ch < dataBlob->channels; ch++)
			{
				float tmp;
				tmp = pF[ch];
				pI[ch] = (signed char)(tmp * scale + ((tmp > 0) - 0.5f));
			}
#endif
		}
	}
	dataBlob->int8float_scale = scale;
	dataBlob->int8_data_valid = true;

	return true;
}

bool convolution(CDataBlob *inputData, const Filters* filters, CDataBlob *outputData)
{
	if (NULL == inputData->data_float || NULL == inputData->data_int8)
	{
		printf("inputData is null\n");
		return false;
	}

	if (0 == filters->tail)
	{
		printf("filters_tail is null\n");
		return false;
	}

	int filterW = filters->filters[0].width;
	int filterH = filters->filters[0].height;
	int filterC = filters->filters[0].channels;
	int filterS = filters->stride;
	int filterP = filters->pad;

	int outputW = 0;
	int outputH = 0;
	int outputC = (int)(filters->tail);

	int i;
	for (i = 1; i < outputC; ++i)
	{
		if ((filterW != filters->filters[i].width) ||
			(filterH != filters->filters[i].height) ||
			(filterC != filters->filters[i].channels))
		{
			printf("filters must be the same size\n");
			return false;
		}
	}

	if (filterC != inputData->channels)
	{
		printf("the number of channels of filters must be the same with the input\n");
		return false;
	}

	if (1 == filterW && 1 == filterH)
	{
		if (filterS != 1)
		{
			printf("only stride = 1 is supported for 1x1 filters.\n");
			return false;
		}

		if (filterP != 0)
		{
			printf("only stride = 0 is supported for 1x1 filters.\n");
			return false;
		}

		outputW = inputData->width;
		outputH = inputData->height;
	}
	else if (3 == filterW && 3 == filterH)
	{
		if (1 == filterS && 1 == filterP)
		{
			outputW = inputData->width;
			outputH = inputData->height;
		}
		else if (2 == filterS && 1 == filterP)
		{
			outputW = (inputData->width + 1) / 2;
			outputH = (inputData->height + 1) / 2;
		}
		else
		{
			printf("unsupported filter stride\n\n");
			return false;
		}
	}
	else
	{
		printf("unsupported filter size\n");
		return false;
	}

	if (outputW < 1 || outputH < 1)
	{
		printf("the size of the output is not correct\n");
		return false;
	}

	create(outputData, outputW, outputH, outputC);

#if defined(_ENABLE_INT8_CONV)
	convertFloat2Int8(inputData);
#endif

	if (1 == filterW && 1 == filterH)
	{
#if defined(_ENABLE_INT8_CONV)
		convolutionInt81x1P0S1(inputData, filters, outputData);
#else
		convolutionFloat1x1P0S1(inputData, filters, outputData);
#endif
	}
	else if (3 == filterW && 3 == filterH)
	{
#if defined(_ENABLE_INT8_CONV)
		convolutionInt83x3P1ChGeneral(inputData, filters, outputData);
#else
		convolutionFloat3x3P1ChGeneral(inputData, filters, outputData);
#endif
	}

#if defined(_ENABLE_INT8_CONV)
	scale(outputData, 1.0f / (inputData->int8float_scale * filters->scale));
#endif

	return true;
}

bool maxpooling2x2S2(const CDataBlob *inputData, CDataBlob *outputData)
{
	if (NULL == inputData->data_float)
	{
		printf("maxpooling2x2S2:input data is null\n");
		return false;
	}

	int outputW = (int)(ceil((inputData->width - 3.0f) / 2)) + 1;
	int outputH = (int)(ceil((inputData->height - 3.0f) / 2)) + 1;
	int outputC = inputData->channels;

	if (outputW < 1 || outputH < 1)
	{
		printf("maxpooling2x2S2:the size of the output is not correct\n");
		return false;
	}

	int elementStep = inputData->floatChannelStepInByte / sizeof(float);
	int lineElementStep = inputData->width * elementStep;

	create(outputData, outputW, outputH, outputC);

	int row, col, ch, fy, fx, el;
	for (row = 0; row < outputData->height; row++)
	{
		for (col = 0; col < outputData->width; col++)
		{
			int inputMatOffsetsInElement[4];
			int elementCount = 0;

			int hstart = row * 2;
			int wstart = col * 2;
			int hend = MIN(hstart + 2, inputData->height);
			int wend = MIN(wstart + 2, inputData->width);

			for (fy = hstart; fy < hend; fy++)
			{
				for (fx = wstart; fx < wend; fx++)
				{
					inputMatOffsetsInElement[elementCount++] = (fy * inputData->width + fx) * inputData->floatChannelStepInByte / sizeof(float);
				}
			}

			float * pOut = outputData->data_float + (row * outputData->width + col) * outputData->floatChannelStepInByte / sizeof(float);
			float * pIn = inputData->data_float;

#if defined(_ENABLE_NEON)
			for (ch = 0; ch < outputData->channels; ch += 4)
			{
				float32x4_t a;
				float32x4_t maxval = vld1q_f32(pIn + ch + inputMatOffsetsInElement[0]);
				for (int el = 1; el < elementCount; el++)
				{
					a = vld1q_f32(pIn + ch + inputMatOffsetsInElement[el]);
					maxval = vmaxq_f32(maxval, a);
				}
				vst1q_f32(pOut + ch, maxval);
			}
#elif defined(_ENABLE_AVX2)
			for (int ch = 0; ch < outputData->channels; ch += 8)
			{
				__m256 a;
				__m256 maxval = _mm256_load_ps(pIn + ch + inputMatOffsetsInElement[0]);
				for (int el = 1; el < elementCount; el++)
				{
					a = _mm256_load_ps(pIn + ch + inputMatOffsetsInElement[el]);
					maxval = _mm256_max_ps(maxval, a);
				}
				_mm256_store_ps(pOut + ch, maxval);
			}
#else
			for (ch = 0; ch < outputData->channels; ++ch)
			{
				float maxval = pIn[ch + inputMatOffsetsInElement[0]];
#if defined(_ENABLE_OPENMP_SIMD)
#pragma omp simd reduction(max:maxval)
#endif
				for (el = 1; el < elementCount; ++el)
				{
					maxval = MAX(maxval, pIn[ch + inputMatOffsetsInElement[el]]);
				}
				pOut[ch] = maxval;
			}
#endif
		}
	}

	return true;
}

bool concat4(const CDataBlob *inputData1, const CDataBlob *inputData2, const CDataBlob *inputData3, const CDataBlob *inputData4, CDataBlob *outputData)
{
	if ((NULL == inputData1->data_float) ||
		(NULL == inputData2->data_float) ||
		(NULL == inputData3->data_float) ||
		(NULL == inputData4->data_float))
	{
		return false;
	}

	if ((inputData1->width != inputData2->width) ||
		(inputData1->height != inputData2->height) ||
		(inputData1->width != inputData3->width) ||
		(inputData1->height != inputData3->height) ||
		(inputData1->width != inputData4->width) ||
		(inputData1->height != inputData4->height))
	{
			return false;
	}

	int outputW = inputData1->width;
	int outputH = inputData1->height;
	int outputC = inputData1->channels + inputData2->channels + inputData3->channels + inputData4->channels;

	if (outputW < 1 || outputH < 1 || outputC < 1)
	{
		return false;
	}

	create(outputData, outputW, outputH, outputC);

	int row, col, ch;
	for (row = 0; row < outputData->height; ++row)
	{
		for (col = 0; col < outputData->width; ++col)
		{
			float * pOut = (outputData->data_float + (row * outputData->width + col) * outputData->floatChannelStepInByte / sizeof(float));
			float * pIn1 = (inputData1->data_float + (row * inputData1->width + col) * inputData1->floatChannelStepInByte / sizeof(float));
			float * pIn2 = (inputData2->data_float + (row * inputData1->width + col) * inputData2->floatChannelStepInByte / sizeof(float));
			float * pIn3 = (inputData3->data_float + (row * inputData1->width + col) * inputData3->floatChannelStepInByte / sizeof(float));
			float * pIn4 = (inputData4->data_float + (row * inputData1->width + col) * inputData4->floatChannelStepInByte / sizeof(float));

			memcpy(pOut, pIn1, sizeof(float)* inputData1->channels);
			memcpy(pOut + inputData1->channels, pIn2, sizeof(float) * inputData2->channels);
			memcpy(pOut + inputData1->channels + inputData2->channels, pIn3, sizeof(float)* inputData3->channels);
			memcpy(pOut + inputData1->channels + inputData2->channels + inputData3->channels, pIn4, sizeof(float)* inputData4->channels);
		}
	}

	return true;
}

bool scale(CDataBlob *dataBlob, float scale)
{
	if (NULL == dataBlob->data_float || NULL == dataBlob->data_int8)
	{
		return false;
	}

	int row, col, ch;
	for (row = 0; row < dataBlob->height; row++)
	{
		for (col = 0; col < dataBlob->width; col++)
		{
			float *pF = (dataBlob->data_float + (row * dataBlob->width + col) * dataBlob->floatChannelStepInByte / sizeof(float));
#if defined(_ENBALE_NEON)
			float32x4_t a, bscale;
			float32x4_t result_vec;

			bscale = vdupq_n_f32(scale);
			for (int ch = 0; ch < dataBlob->channels; ch += 4)
			{
				a = vld1q_f32(pF + ch);
				result_vec = vmulq_f32(a, bscale);
				vst1q_f32(pF + ch, result_vec);
			}
#elif defined(_ENABLE_AVX2)
			__m256 a, bsacle;

			bscale = _mm256_set1_ps(scale);
			for (int ch = 0; ch < dataBlob->channels; ch += 8)
			{
				a = _mm256_load_ps(pF + ch);
				a = _mm256_mul_ps(a, bsacle);
				_mm256_store_ps(pF = ch, a);
			}
#else
#if defined(_ENABLE_OPENMP_SIMD)
#pragma omp simd
#endif
			for (ch = 0; ch < dataBlob->channels; ch++)
			{
				pF[ch] *= scale;
			}
#endif
		}
	}

	return true;
}

bool relu(const CDataBlob * inputOutputData)
{
	if (NULL == inputOutputData->data_float)
	{
		printf("relu: the input data is null\n");
		return false;
	}

	float *pData = NULL;
	int row, col, ch;
	for (row = 0; row < inputOutputData->height; row++)
	{
		for (col = 0; col < inputOutputData->width; col++)
		{
			pData = (float*)(inputOutputData->data_float + (row * inputOutputData->width + col) * inputOutputData->floatChannelStepInByte / sizeof(float));

#if defined(_ENABLE_NEON)
			float 32x4_t a, bzeros;
			float 32x4_t result_vec;

			bzeros = vdupq_n_f32(0);
			for (int ch = 0; ch < inputOutputData->channels; ch += 4)
			{
				a = vld1q_f32(pData + ch);
				result_vec = vmaxq_f32(a, bzeros);
				vst1q_f32(pData + ch, result_vec);
			}
#elif defined(_ENABLE_AVX2)
			__m256 a, bzeros;

			bzeros = _mm256_setzero_ps();
			for (int ch = 0; ch < inputOutputData->channels; ch += 8)
			{
				a = _mm256_load_ps(pData + ch);
				a = _mm256_max_ps(a, bzeros);
				_mm256_store_ps(pData + ch, a);
			}

#else
#if defined(_ENABLE_OPENMP_SIMD)
#pragma omp simd
#endif
			for (int ch = 0; ch < inputOutputData->channels; ++ch)
			{
				pData[ch] = MAX(pData[ch], 0);
				if (pData[ch] != 0)
				{
					//printf("pData[ch] = %f\n", pData[ch]);
				}
			}
#endif
		}
	}

	return true;
}

bool priorbox(const CDataBlob * featureData, const CDataBlob* imageData, int num_sizes, float * pWinSizes, CDataBlob * outputData)
{
	if ((NULL == featureData->data_float) ||
		NULL == imageData->data_float ||
		NULL == pWinSizes)
	{
		return false;
	}

	int feature_width = featureData->width;
	int feature_height = featureData->height;
	int image_width = imageData->width * 2;
	int image_height = imageData->height * 2;

	float step_w = (float)(image_width) / feature_width;
	float step_h = (float)(image_height) / feature_height;

	float * output_data = outputData->data_float;

	create(outputData, feature_width, feature_height, num_sizes * 4);

	int h, w, s;
	for (h = 0; h < feature_height; h++)
	{
		for (w = 0; w < feature_width; w++)
		{
			float *pOut = (float*)(outputData->data_float + (h * outputData->width + w) * outputData->floatChannelStepInByte / sizeof(float));
			int idx = 0;

			for (s = 0; s < num_sizes; s++)
			{
				float min_size_ = pWinSizes[s];
				float box_width, box_height;
				box_width = box_height = min_size_;

				float center_x = w * step_w + step_w / 2.0f;
				float center_y = h * step_h + step_h / 2.0f;

				pOut[idx++] = (center_x - box_width / 2.f) / image_width;
				pOut[idx++] = (center_y - box_height / 2.f) / image_height;
				pOut[idx++] = (center_x + box_width / 2.f) / image_width;
				pOut[idx++] = (center_y + box_height / 2.f) / image_height;
			}
		}
	}

	return true;
}

bool normalize(CDataBlob * inputOutputData, float *pScale)
{
	if ((NULL == inputOutputData->data_float) || NULL == pScale)
	{
		return false;
	}

	int row, col, ch;
	for (row = 0; row < inputOutputData->height; row++)
	{
		for (col = 0; col < inputOutputData->width; col++)
		{
			float * pData = (float*)(inputOutputData->data_float + (row * inputOutputData->width + col) * inputOutputData->floatChannelStepInByte / sizeof(float));
			float sum = FLT_EPSILON;
			float s = 0;
#if defined(_ENABLE_NEON)
			float32x4_t a, b, scale;
			float32x4_t result_vec;
			for (ch = 0; ch < inputOutputData->channels; ch += 4)
			{
				a = vld1q_f32(pData + ch);
				result_vec = vmulq_f32(a, a);
				sum += vgetq_lane_f32(result_vec, 0);
				sum += vgetq_lane_f32(result_vec, 1);
				sum += vgetq_lane_f32(result_vec, 2);
				sum += vgetq_lane_f32(result_vec, 3);
			}

			s = 1.0f / sqrt(sum);
			cscale = vdupq_n_f32(s);

			for (ch = 0; ch < inputOutputData->channels; ch += 4)
			{
				a = vld1q_f32(pData + ch);
				b = vld1q_f32(pScale + ch);

				result_vec = vmulq_f32(a, b);
				result_vec = vmulq_f32(result_vec, cscale);
				vst1q_f32(pData + ch, result_vec);
			}
#elif defined(_ENABLE_AVX2)
			__m256 a, b, cscale;
			__m256 result_vec;
			for (ch = 0; ch < inputOutputData->channels; ch += 8)
			{
				a = _mm256_load_ps(pData + ch);
				a = _mm256_mul_ps(a, a);
				a = _mm256_hadd_ps(a, a);
				a = _mm256_hadd_ps(a, a);
				sum += SSE_256ELEMENT(a, 0);
				sum += SSE_256ELEMENT(a, 4);
			}

			s = 1.0f / sqrt(sum);
			cscale = _mm256_set1_ps((s);

			for (ch = 0; ch < inputOutputData->channels; ch += 8)
			{
				a = _mm256_load_ps(pData + ch);
				b = _mm256_load_ps(pScale + ch);

				result_vec = _mm256_mul_ps(a, b);
				result_vec = _mm256_mul_ps(result_vec, cscale);
				_mm256_store_ps(pData + ch, result_vec);
			}
#else

#if defined(_ENABLE_OPENMP_SIMD)
#pragma omp simd reduction(+:sum)
#endif
			for (ch = 0; ch < inputOutputData->channels; ch++)
			{
				sum += (pData[ch] * pData[ch]);
			}

			s = 1.0f / sqrt(sum);
#if defined(_ENABLE_OPENMP_SIMD)
#pragma omp simd
#endif
			for (ch = 0; ch < inputOutputData->channels; ch++)
			{
				pData[ch] = pData[ch] * pScale[ch] * s;
			}
#endif
		}
	}

	return true;
}

bool softmax1vector2class(const CDataBlob *inputOutputData)
{
	if (NULL == inputOutputData->data_float)
	{
		return false;
	}

	if (inputOutputData->width != 1 || inputOutputData->height != 1)
	{
		return false;
	}

	int num = inputOutputData->channels;
	float * pData = (inputOutputData->data_float);

#if defined(_OPENMP)
#pragma  omp prarllel for
#endif
	int i;
	for (i = 0; i < num; i += 2)
	{
		float v1 = pData[i];
		float v2 = pData[i + 1];
		float vm = MAX(v1, v2);
		v1 -= vm;
		v2 -= vm;
		v1 = expf(v1);
		v2 = expf(v2);
		vm = v1 + v2;
		pData[i] = v1 / vm;
		pData[i + 1] = v2 / vm;
	}

	return true;
}

bool blob2vector(const CDataBlob * inputData, CDataBlob * outputData, bool isFloat)
{
	if (NULL == inputData->data_float)
	{
		return false;
	}

	create(outputData, 1, 1, inputData->width * inputData->height * inputData->channels);

	int row, col, ch;
	if (isFloat)
	{
		int bytesOfAChannel = inputData->channels * sizeof(float);
		float * pOut = outputData->data_float;
		for (row = 0; row < inputData->height; row++)
		{
			for (col = 0; col < inputData->width; col++)
			{
				float * pIn = (inputData->data_float + (row * inputData->width + col) * inputData->floatChannelStepInByte / sizeof(float));
				memcpy(pOut, pIn, bytesOfAChannel);
				pOut += inputData->channels;
			}
		}
	}
	else
	{
		int bytesOfAChannel = inputData->channels * sizeof(char);
		signed char * pOut = outputData->data_int8;
		for (row = 0; row < inputData->height; row++)
		{
			for (col = 0; col < inputData->width; col++)
			{
				signed char * pIn = (inputData->data_int8 + (row * inputData->width + col) * inputData->int8ChannelStepInByte / sizeof(char));
			}
		}
	}

	return true;
}

void IntersectBBox(const NormalizedBBox *bbox1, const NormalizedBBox *bbox2, NormalizedBBox *intersect_bbox)
{
	if (bbox2->xmin > bbox1->xmax || bbox2->xmax < bbox1->xmin ||
		bbox2->ymin > bbox1->ymax || bbox2->ymax < bbox1->ymin)
	{
		intersect_bbox->xmin = 0;
		intersect_bbox->ymin = 0;
		intersect_bbox->xmax = 0;
		intersect_bbox->ymax = 0;
	}
	else
	{
		intersect_bbox->xmin = (MAX(bbox1->xmin, bbox2->xmin));
		intersect_bbox->ymin = (MAX(bbox1->ymin, bbox2->ymin));
		intersect_bbox->xmax = (MIN(bbox1->xmax, bbox2->xmax));
		intersect_bbox->ymax = (MIN(bbox1->ymax, bbox2->ymax));
	}
}

float JaccardOverlap(const NormalizedBBox *bbox1, const NormalizedBBox *bbox2)
{
	NormalizedBBox intersect_bbox;
	IntersectBBox(bbox1, bbox2, &intersect_bbox);
	float intersect_width, intersect_height;
	intersect_width = intersect_bbox.xmax - intersect_bbox.xmin;
	intersect_height = intersect_bbox.ymax - intersect_bbox.ymin;

	if (intersect_width > 0 && intersect_height > 0)
	{
		float intersect_size = intersect_width * intersect_height;
		float bsize1 = (bbox1->xmax - bbox1->xmin) * (bbox1->ymax - bbox1->ymin);
		float bsize2 = (bbox2->xmax - bbox2->xmin) * (bbox2->ymax - bbox2->ymin);
		return intersect_size / (bsize1 + bsize2 - intersect_size);
	}
	else
	{
		return 0.f;
	}
}

int TPQuickIntoSortedArrayPairIFloat(float *piArray1, int *piArray2, unsigned char ucOrder, int iLeft, int iRight)
{
	int i = iLeft;
	int j = iRight;
	float iTemp1 = piArray1[iLeft];
	float iTemp2 = piArray2[iLeft];

	if (i >= j)
	{
		return 0;
	}

	if (0 == ucOrder)
	{
		while (i < j)
		{
			while (i < j && piArray1[j] >= iTemp1)
			{
				j--;
			}
			if (i < j)
			{
				piArray1[i] = piArray1[j];
				piArray2[i] = piArray2[j];
				++i;
			}
			while (i < j && piArray1[i] <= iTemp1)
			{
				++i;
			}
			if (i < j)
			{
				piArray1[j] = piArray1[i];
				piArray2[j] = piArray2[i];
				--j;
			}
		}
	}
	else if (1 == ucOrder)
	{
		while (i < j)
		{
			while (i < j && piArray1[j] <= iTemp1)
			{
				--j;
			}
			if (i < j)
			{
				piArray1[i] = piArray1[j];
				piArray2[i] = piArray2[j];
				++i;
			}
			while (i < j && piArray1[i] >= iTemp1)
			{
				++i;
			}
			if (i < j)
			{
				piArray1[j] = piArray1[i];
				piArray2[j] = piArray2[i];
				--j;
			}
		}
	}
	else
	{
		return -1;
	}
	piArray1[i] = iTemp1;
	piArray2[i] = iTemp2;
	TPQuickIntoSortedArrayPairIFloat(piArray1, piArray2, ucOrder, iLeft, i - 1);
	TPQuickIntoSortedArrayPairIFloat(piArray1, piArray2, ucOrder, i + 1, iRight);
}

int TPQuickIntoSortedArrayPairI(int *piArray1, int *piArray2, unsigned char ucOrder, int iLeft, int iRight)
{
	int i = iLeft;
	int j = iRight;
	int iTemp1 = piArray1[iLeft];
	int iTemp2 = piArray2[iLeft];

	if (i >= j)
	{
		return 0;
	}

	if (0 == ucOrder)
	{
		while (i < j)
		{
			while (i < j && piArray1[j] >= iTemp1)
			{
				--j;
			}

			if (i < j)
			{
				piArray1[i] = piArray1[j];
				piArray2[i] = piArray2[j];
				++i;
			}
			while (i < j && piArray1[i] < iTemp1)
			{
				++i;
			}

			if (i < j)
			{
				piArray1[j] = piArray1[i];
				piArray2[j] = piArray2[i];
			}
		}
	}
	else if (1 == ucOrder)
	{
		while (i < j)
		{
			while (i < j && piArray1[j] <= iTemp1)
			{
				j--;
			}

			if (i < j)
			{
				piArray1[i] = piArray1[j];
				piArray2[i] = piArray2[j];
				++i;
			}

			while (i < j && piArray1[i] >= iTemp1)
			{
				++i;
			}
			if (i < j)
			{
				piArray1[j] = piArray1[i];
				piArray2[j] = piArray2[i];
				--j;
			}
		}
	}
	else
	{
		return -1;
	}
	piArray1[i] = iTemp1;
	piArray2[i] = iTemp2;
	TPQuickIntoSortedArrayPairI(piArray1, piArray2, ucOrder, iLeft, i - 1);
	TPQuickIntoSortedArrayPairI(piArray1, piArray2, ucOrder, i + 1, iRight);

	return 0;
}

bool detection_output(const CDataBlob * priorbox, const CDataBlob * loc, const CDataBlob * conf, float overlap_threshold, float confidence_threshold, int top_k, int keep_top_k, CDataBlob * outputData)
{
	if (NULL == priorbox->data_float || NULL == loc->data_float || NULL == conf->data_float)
	{
		return 0;
	}

	if (priorbox->channels != loc->channels || loc->channels != conf->channels * 2)
	{
		return 0;
	}

	float prior_variance[4] = { 0.1f, 0.1f, 0.2f, 0.2f };
	float * pPriorBox = priorbox->data_float;
	float * pLoc = loc->data_float;
	float * pConf = conf->data_float;

	cvector scores = cvector_create(sizeof(float));
	cvector boxes = cvector_create(sizeof(NormalizedBBox));
	cvector finalScores = cvector_create(sizeof(float));
	cvector finalBoxes = cvector_create(sizeof(NormalizedBBox));

	int i;
	for (i = 1; i < conf->channels; i += 2)
	{
		if (pConf[i] > confidence_threshold)
		{
			float fx1 = pPriorBox[i * 2 - 2];
			float fy1 = pPriorBox[i * 2 - 1];
			float fx2 = pPriorBox[i * 2];
			float fy2 = pPriorBox[i * 2 + 1];

			float locx1 = pLoc[i * 2 - 2];
			float locy1 = pLoc[i * 2 - 1];
			float locx2 = pLoc[i * 2];
			float locy2 = pLoc[i * 2 + 1];

			float prior_width = fx2 - fx1;
			float prior_height = fy2 - fy1;
			float prior_center_x = (fx1 + fx2) / 2;
			float prior_center_y = (fy1 + fy2) / 2;

			float box_centerx = prior_variance[0] * locx1 * prior_width + prior_center_x;
			float box_centery = prior_variance[0] * locy1 * prior_height + prior_center_y;
			float box_width = expf(prior_variance[2] * locx2) * prior_width;
			float box_height = expf(prior_variance[3] * locy2) * prior_height;

			fx1 = box_centerx - box_width / 2.f;
			fy1 = box_centery - box_height / 2.f;
			fx2 = box_centerx + box_width / 2.f;
			fy2 = box_centery + box_height / 2.f;

			fx1 = MAX(0, fx1);
			fy1 = MAX(0, fy1);
			fx2 = MIN(1.f, fx2);
			fy2 = MIN(1.f, fy2);

			NormalizedBBox bb;
			bb.xmin = fx1;
			bb.ymin = fy1;
			bb.xmax = fx2;
			bb.ymax = fy2;

			cvector_pushback(scores, (void *)&pConf[i]);
			cvector_pushback(boxes, (void *)&bb);
		}
	}

	int *piIndex = (int*)malloc(sizeof(int)* cvector_length(scores));
	float *arrayScores = (float*)malloc(sizeof(float)* cvector_length(scores));

	for (i = 0; i < cvector_length(scores); ++i)
	{
		piIndex[i] = i;
		cvector_val_at(scores, i, &arrayScores[i]);
	}

	TPQuickIntoSortedArrayPairIFloat(arrayScores, piIndex, 1, 0, (int)cvector_length(scores) - 1);
	cvector tempScores = cvector_create(sizeof(float));
	cvector tempBoxes = cvector_create(sizeof(NormalizedBBox));

	NormalizedBBox bbtemp = { 0 };
	float scoretemp = 0.0;
	for (i = 0; i < cvector_length(scores); ++i)
	{
		cvector_val_at(boxes, piIndex[i], &bbtemp);
		cvector_val_at(scores, piIndex[i], &scoretemp);
		cvector_pushback(tempScores, (void *)&scoretemp);
		cvector_pushback(tempBoxes, (void *)&bbtemp);
	}


	if (top_k > -1 && top_k < cvector_length(tempScores))
	{
		i = cvector_length(tempScores) - 1;
		for (i; i >= top_k; --i)
		{
			cvector_rm_at(tempScores, i);
			cvector_rm_at(tempBoxes, i);
		}
	}

	int k;
	while (cvector_length(tempScores) != 0)
	{
		const NormalizedBBox bb1 = *(NormalizedBBox *)cvector_begin(tempBoxes);
		bool keep = true;
		for (k = 0; k < cvector_length(finalScores); ++k)
		{
			if (keep)
			{
				const NormalizedBBox bb2;
				cvector_val_at(finalBoxes, k, (void *)&bb2);
				float overlap = JaccardOverlap(&bb1, &bb2);
				keep = (overlap <= overlap_threshold);
			}
			else
			{
				break;
			}
		}

		if (keep)
		{
			cvector_pushback(finalBoxes, cvector_begin(tempBoxes));
			cvector_pushback(finalScores, cvector_begin(tempScores));
		}
		cvector_rm_at(tempBoxes, 0);
		cvector_rm_at(tempScores, 0);
	}

	if (keep_top_k > -1 && keep_top_k < cvector_length(finalScores))
	{
		int i = cvector_length(finalScores) - 1;
		for (i; i >= keep_top_k; i--)
		{
			cvector_rm_at(finalScores, i);
			cvector_rm_at(finalBoxes, i);
		}
	}

	int num_faces = cvector_length(finalScores);
	if (0 == num_faces)
	{
		setNULL(outputData);
	}
	else
	{
		int fi;
		create(outputData, num_faces, 1, 5);
		for (fi = 0; fi < num_faces; ++fi)
		{
			float * pOut = (outputData->data_float + fi * outputData->floatChannelStepInByte / sizeof(float));
			cvector_val_at(finalScores, fi, &pOut[0]);
			NormalizedBBox bb = { 0 };
			cvector_val_at(finalBoxes, fi, &bb);
			pOut[1] = bb.xmin;
			pOut[2] = bb.ymin;
			pOut[3] = bb.xmax;
			pOut[4] = bb.ymax;
		}
	}

	free(piIndex);
	free(arrayScores);
	piIndex = 0;
	arrayScores = 0;

	cvector_destroy(scores);
	cvector_destroy(boxes);
	cvector_destroy(finalScores);
	cvector_destroy(finalBoxes);

	return true;
}