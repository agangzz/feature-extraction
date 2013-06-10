/*
 * statistics function executed on GPU
 */

#ifndef _STATISTIC_H__
#define _STATISTIC_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cufft.h>
#include <cutil_inline.h>


/*
 * using redunction method to sum float array in GPU. Revoked on CPU, executed 
 * on CPU
 * 
 * Input
 * -------------
 * data_gpu: data_array, type is float
 * s_gpu: sum result in GPU. sometimes, the result will be used in GPU not CPU
 *        s_gpu is not initialized,so is just a NULL pointer. It will be initialize
 *        inside the function
 * length: length of data array
 *
 * Output:
 * ------------------------
 * return the sum result of the array in CPU memory.
 */
float Sum(float *data_gpu,float** s_gpu,int length);


/*
 * using redunction method to sum int array in GPU. Revoked on CPU, executed 
 * on CPU
 * 
 * Input
 * -------------
 * data_gpu: data_array, type is int
 * s_gpu: sum result in GPU. sometimes, the result will be used in GPU not CPU
 *        s_gpu is not initialized,so is just a NULL pointer. It will be initialize
 *        inside the function
 * length: length of data array
 *
 * Output:
 * ------------------------
 * return the sum result of the array in CPU memory.
 */

int Sum(int *data_gpu,int **s_gpu,int length);


/*
 * sum of the float data in multi-arraies.
 *
 * Input
 * ----------------------
 * data_gpu: float data arraies
 * s_gpu: sum pointer points to the sum array in GPU memory.
 * nx: the length of every array
 * batch: the number of arraies
 *
 * Output
 * -------------------------
 * the sum array in CPU memory
 *
 */
float* Sum(float* data_gpu,float **s_gpu,int nx,int batch);


/*
 * sum the float data array in GPU. Revoked on CPU, executed on GPU
 *
 * Input:
 * ----------------------
 * block_gpu: used to sum the value of every block
 * the others are the same as the above function Sum
 *
 */
template <class T>
__global__ static void sum(T* data_gpu,T* block_gpu,int length);


template <class T>
__global__ static void sumblock(T *block_gpu,T *sum_gpu,int length);


/*
 * sum the float data in multi-arraies in GPU memory.
 *
 * Input
 * ------------------------
 * the same as the above three function
 */
__global__ static void sum(float* data_gpu,float* sum_gpu,int nx,int batch);


/*
 * get the max of float array with elements bigger than zero.
 * Revoked on CPU, executed on GPU
 *
 * Input
 * ------------------------------------
 * data_gpu: data_array, type is float
 * m_gpu: max of the array in GPU. sometimes, the result will be used in GPU not CPU
 * length: length of data array
 *
 * Output:
 * ------------------------
 * return the max of the array in CPU memory.
 */
float Max(float *data_gpu,float **m_gpu,int length);


/*
 * get the maxes of the arraies with elements bigger than zero.
 *
 * Input
 * ------------------------------------
 * data_gpu/m_gpu: the same as the above function
 * nx: the length of every array, named from cufft
 * batch:number of arraies of size of nx, named from cufft
 *
 * Output
 * --------------------------------------
 * the maxes arraies in CPU memory
 */
float* Max(float *data_gpu,float **m_gpu,int nx,int batch);


/*
 * get the maxes of the arraies with elements bigger than zero.
 * The above function processes one array every time in GPU, and use for loop to calculate the max of every array.
 * Max2 processes all the arraies in GPU, one block deals with one array. So it avoids the invoke from host so many times.
 * The speed is about 100 times faster than the above Max function.
 *
 * Input
 * ------------------------------------
 * data_gpu/m_gpu: the same as the above function
 * nx: the length of every array, named from cufft
 * batch:number of arraies of size of nx, named from cufft
 *
 * Output
 * --------------------------------------
 * the maxes arraies in CPU memory
 */
float * Max2(float *data_gpu,float **m_gpu,int nx,int batch);


/*
 * get the max float value of array on GPU
 * Revoked and executed on GPU
 *
 * Input
 * -----------------------------
 * block_gpu: used to get the max value of every block
 * the others are the same as Max
 */
__global__ static void max(float* data_gpu,float* block_gpu,float *max_gpu,int length);


/*
 * get the max float value of array on GPU
 * Revoked and executed on GPU
 *
 * Input
 * -----------------------------
 * the same as Max2
 */
__global__ static void max(float* data_gpu,float *max_gpu,int nx,int batch);


/*
 * Revoke the cublas library to calcuals the sum of an array
 *
 * Input
 * -----------------------------------
 * the same as the function Sum
 *
 * Output
 * ----------------------------------
 * the same as the function Sum
 */
float Sum_cublas(float *data_gpu,float **s_gpu,int length);


#endif
