
/*
 * use the fft peaks to calculate the salience of the frequency from 70 to 1400
 *
 */

#ifndef _SALIENCEFUNCTION_H_
#define _SALIENCEFUNCTION_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>

/*
 * calculate the salience using the peaks of fft. Invoked on CPU, executed on GPU
 *
 * Input
 * --------------------------------------
 * peaks_gpu: the peaks value of every frame,length HALF_PEAK_NUM, but only a few
 *            value is useful.
 * index_gpu: the index of the peaks, length HALF_PEAK_NUM, but only a few value
 *            is useful. peaks and index must be used together.
 * peaks_num_gpu: the number of peaks of every frame. 
 * saliencebins_gpu_p: the result salience of every frame. transfer a NULL pointer
 *                     is ok, the space is mallocated inside the function.
 * N: fft length
 * sampleRate: sampleRate of the wav file. N and sampleRate is used to calculate 
 *             the frequency of the peaks.
 * framenumber: how many frames 
 */
void Saliencefunc(float *peaks_gpu,int *index_gpu,int *peaks_num_gpu,float** filteredpeaks_gpu_p,int ** filteredindex_gpu_p,int **filteredpeaks_num_gpu_p,float**leftpeaks_gpu_p,int ** leftindex_gpu_p,int **leftpeaks_num_gpu_p,int N,int sampleRate,int framenumber);


/*
 * spilt the salience peaks to two parts, one is bigger than the max*ratio,the other is smaller than that.
 *
 * Input
 * ------------------------------------
 * max_gpu: the max of every array
 * peaks_gpu: salience peaks of every array
 * index_gpu: the salience peaks index of every array
 * peaks_num_gpu: the salience peaks number of every array.
 * filteredpeaks_gpu: the salience peaks smaller than the max*ratio of every array
 * filteredindex_gpu: the salience peaks index smaller than the max*ratio of every array
 * filteredpeaks_num_gpu: the salience peaks number smaller than the max*ratio of every array
 * leftpeaks_gpu: the salience peaks bigger than the max*ratio of every array
 * leftindex_gpu: the salience peaks index bigger than the max*ratio of every array
 * leftpeaks_num_gpu: the salience peaks number bigger than the max*ratio of every array
 * framenumber: how many arraier
 * ratio: spilt the salience peaks to two parts, bigger or smaller than max*ratio
 */
__global__ static void spilt(float *max_gpu,float *peaks_gpu,int *index_gpu,int *peaks_num_gpu,
                             float *filteredpeaks_gpu, int *filteredindex_gpu,int *filteredpeaks_num_gpu,
                             float *leftpeaks_gpu, int *leftindex_gpu,int *leftpeaks_num_gpu,
                             int framenumber);


__global__ static void spilt2(float *max_gpu,float *peaks_gpu,int *index_gpu,int *peaks_num_gpu,
                             float *filteredpeaks_gpu, int *filteredindex_gpu,int *filteredpeaks_num_gpu,
                             float *leftpeaks_gpu, int *leftindex_gpu,int *leftpeaks_num_gpu,
                             int framenumber);


__global__ static void multipy(float* data_gpu,int length,float ratio);

/*
 * calculate the salience using the peaks of fft. Invoked on GPU, execute on GPU
 *
 * Input
 * ----------------------------------
 * all the same as the above function except the saliencebins_gpu is not a NULL
 * pointer.
 */
__global__ static void saliencefunc(float *peaks_gpu,int *index_gpu,int *peaks_num_gpu,float *saliencebins_gpu,int N,int sampleRate,int framenumber);


/*
 * calculate the salience of one bin.
 *
 * Input
 * ----------------------------------
 * peaks_gpu: the peaks of one frame, mallocated in shared memory
 * index_gpu: the index of the peaks, mallocated in shared memory
 * frame: calculate the salience of which frame
 * peaks_num: how many peaks in this frame
 * N/sampleRate: the same as above
 * b: calculate the salience of which bin in the frame.
 */
__device__ static float salience(float *peaks_gpu,int *index_gpu,int frame,int peaks_num,int N,int sampleRate,int b);


/*
 * calculate the weight to frequency when specify the bin and harmonic
 * 
 * Input
 * --------------------------------
 * bin: the bin number
 * harmonic: the harmonic number
 * frequency: frequency of fft
 */
__device__ static float weight(int bin,int harmonic,float frequency);


/*
 * change the unit of the frequency from Hz to cent scale
 *
 * Input
 * -----------------------------------
 * frequency: the frequency of fft
 */
__device__ static int bin(float frequency);


#endif

