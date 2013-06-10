/*
 * This file defines a host function and a global function which will find peaks of an array. 
 * The data is a 1-D array, size is length * framenumber.
 *
 *
 * Detail:
 * The function will find the peaks in every frame which has length length. But
 * we won't find the peaks of length length. We will find the peaks in minlength
 * among every frame.
 *
 * ______________________________________________________________________________________
 * |________length 1________|________length n_________|______length framenumber_________|
 * |                |       |                 |       |                         |       |
 * |    minlength   |       |      minlength  |       |       minlength         |       |
 * |________________|_______|_________________|_______| ________________________|_______|
 * |________________________|_________________________|_________________________________|
 */

#ifndef _FINDPEAKS_H_
#define _FINDPEAKS_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>

/*
 * This host function will find the peaks in array data_gpu. The detail is on top.
 * It is a wrap function of function findpeaks below.
 *
 * Input
 * ----------------------------------
 * data_gpu: the source data, find the peaks of these data
 * peaks_gpu_p: a pointer points to the device memory which has the peaks of every 
 *              frame. Just thansfer a NULL pointer, the space will be mallocd in 
 *              this function.
 * peaks_gpu_num_p: a pointer points to the device memory which has the peaks number
 *              of every frame.Just thansfer a NULL pointer, the space will be mallo *              cd in this function.
 * framenumber: how many frames in data_gpu
 * length: the length of every frame
 * minlength: find the peaks of every frame which has a part length minlength
 */
void Findpeaks(float *data_gpu,float **peaks_gpu_p,int framenumber,int length,int minlength);


/*
 * This host function will find the peaks bigger than max*ratio in array data_gpu. 
 * I don't use the method above to achieve the function, I just use atomic operation
 * to achieve it as the length is small for find peaks of salience.
 * It is a wrap function of function findpeaks3 below.
 *
 * Input
 * ----------------------------------
 * data_gpu: the source data, find the peaks of these data
 * max_gpu: the max value of every array
 * peaks_gpu_p: a pointer points to the device memory which has the peaks of every 
 *              frame. Just transfer a NULL pointer, the space will be mallocd in 
 *              this function.
 * index_gpu_p: a pointer points to the device memory which has the peaks index of
 *              every frame. Just transfer a NULL pointer, the space will be mallocd
 *              in this function
 * peaks_gpu_num_p: a pointer points to the device memory which has the peaks number
 *              of every frame.Just thansfer a NULL pointer, the space will be mallo *              cd in this function.
 * framenumber: how many frames in data_gpu
 * length: the length of every frame
 * minpeakratio: the peaks must be bigger than the ratio mutilplied by max
 */
void Findpeaks(float* data_gpu,float *max_gpu,float **peaks_gpu_p,int **index_gpu_p,int **peaks_num_gpu_p,int framenumber,int length);


/*
 * This global function will find the peaks in array data_gpu. The detail in on top.
 *
 * Input
 * -----------------------------------
 * The same as the function Findpeaks
 *
 */
__global__ static void findpeaks(float *data_gpu,float* peaks_gpu,int framenumber,int length,int minlength);


/*
 * achieve the same function as the above function, but use different schedule 
 * method
 *
 * Input
 * -------------------------------
 * The same as the function Findpeaks
 */
__global__ static void findpeaks2(float *data_gpu,float* peaks_gpu,int framenumber,int length,int minlength);


/*
 * find the peaks bigger than the max*ratio in every array. used by Findpeaks in
 * line 70.
 *
 * Input
 * ------------------------------------
 * the same as the function Findpeaks in line 70
 *
 */
__global__ static void findpeaks(float *data_gpu,float* max_gpu,float* peaks_gpu,int *index_gpu,int *peaks_num_gpu,int framenumber,int length);


/*
 * Find the values bigger than max*filterratio in arraies. Executed on CPU
 *
 * Input
 * --------------------------------------
 * data_gpu: arraies found bigger values. composed of framenumber*length
 * max_gpu: the max value of every array, size is length
 * bigger_gpu_p: a pointer points to bigger_gpu in GPU mallocated in this function,  *               so just transfer a NULL pointer. can be used outside the function
 *               It has size of PEAK_NUM*framenumber
 * index_gpu_p: Function and size are the same as the above. store the index of the 
 *              bigger value. The execution of GPU makes the index out of order.
 * bigger_num_gpu_p: Function is the same as the above. size is framenumber. stores
 *                   the count of bigger value of every array.
 * framenumber: how many frames in data_gpu
 * length: the length of every frame in data_gpu
 * filterratio: the big value must be bigger than max*filterratio.
 */
void Biggerfilter(float *data_gpu,float *max_gpu,float **bigger_gpu_p,int **index_gpu_p,int **bigger_num_gpu_p,int framenumber,int length,float filterratio);


/*
 * Find the values bigger than max*filterratio in arraies. Executed on GPU
 *
 * Input
 * ---------------------------------
 * data_gpu: the same as above
 * max_gpu: the same as above
 * bigger_gpu: stores the results of bigger value of every frame. the space is mallo *             cated in the above function.
 * index_gpu: stores the results of indexes of bigger value of every frame. the spac *            e is mallocated in the above function.
 * bigger_num_gpu: stores the results of the count of bigger value of every frame.
 *                 the space is mallocated in the above function.
 * framenumber: the same as above
 * length: the same as above
 * filterratio: the same as above
 */
__global__ static void biggerfilter(float *data_gpu,float *max_gpu,float *bigger_gpu,int *index_gpu,int *bigger_num_gpu,int framenumber,int length,float filterratio);


/*
 * After filter using biggerfilter, the peaks number may be still large, so filter a * gain
 *
 * Input
 * -------------------------------------
 * data_gpu: data arraies after biggerfilter.
 * prev_index_gpu: the index calculated in biggerfilter.
 * max_gpu: the same as above
 * bigger_gpu: the same as above, the difference is the size of every frame reduced  *             to half of theabove
 * index_gpu: the same as_above, the difference is the size of every frame reduced   *            to half 
 * bigger_num_gpu: the same as above
 * framenumber: the same as above
 * filterratio: the same as above
 */
__global__ static void morefilter(float *data_gpu,int* prev_index_gpu,float *max_gpu,float *bigger_gpu,int *index_gpu,int *bigger_num_gpu,int framenumber,float filterratio);


__global__ static void morefilter2(float *data_gpu,int* prev_index_gpu,float *max_gpu,float *bigger_gpu,int *index_gpu,int *bigger_num_gpu,int framenumber,float filterratio);
#endif
