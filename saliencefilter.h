/*
 * this file filters the salience peaks through the salience mean and std in all
 * frames the second time.
 *
 */

#ifndef _SALIENCEFILTER_H_
#define _SALIENCEFILTER_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>


/*
 * filter the salience peaks using salience mean and std in all frames the second time.
 *
 * Input
 * ------------------------------------
 * filteredpeaks_gpu: the filtered salience peaks the first time
 * filteredindex_gpu: the filtered salience peaks index the first time
 * filteredpeaks_num_gpu: the filtered salience peaks number in every frame the first time
 *
 * leftpeaks_gpu: the left salience peaks the first time
 * leftindex_gpu: the left salience peaks index the first time
 * leftpeaks_num_gpu: the left salience peaks number in every frame the first time
 *
 * leftpeaks_gpu_p: a pointer points to the left salience peaks filtered the second time.
 * leftpeaks_num_gpu_p: a pointer points to the left salience peaks number filtered the second time.
 *
 * framenumber: how many frames in all
 */
void saliencefilter(float *filteredpeaks_gpu, int *filteredindex_gpu,int *filteredpeaks_num_gpu,float *leftpeaks_gpu, int *leftindex_gpu,int *leftpeaks_num_gpu,int framenumber);


/*
 * calculate the square sum of all frames.
 *
 * Input
 * ----------------------------------
 * leftpeaks_gpu: the data calculated the square sum.
 * square_gpu: the square sum of all frames.
 * nx: the length of every frame
 * batch: how many frames in all
 * salience_mean: square the data-salience_mean
 */
void Saliencesquare(float *leftpeaks_gpu,float** square_gpu,int nx,int batch,float salience_mean);


/*
 * calculate the square sum of all frames in GPU
 *
 * Input
 * ------------------------------
 * the same as the above function
 *
 */
__global__ static void saliencesquare(float *leftpeaks_gpu,float* leftpeaks_square_gpu,int nx,int batch,float salience_mean);


/*
 * filter the salience peaks the second time
 *
 * Input
 * -------------------------------------
 * threshold: salience peaks in left is dropped if the peaks are below the threshold
 * the other is the same as the first function
 */
__global__ static void filter(float *filteredpeaks_gpu, int *filteredindex_gpu,int *filteredpeaks_num_gpu,float *leftpeaks_gpu, int *leftindex_gpu,int *leftpeaks_num_gpu,int framenumber,float threshold);


/*
 * clear the zero values scattered among non-zero values in peaks array.
 *
 * Input
 * ------------------------------------
 * leftpeaks_gpu: the peaks values with zero value scattered among non-zero value
 * leftindex_gpu: the peaks index following the peaks
 * framenumber: how many frame in all
 */
__global__ static void shift(float *leftpeaks_gpu, int *leftindex_gpu,int framenumber);
#endif
