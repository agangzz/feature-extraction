/*
 * This head file has the function about calculating fft in GPU
 */

#ifndef _FFT_H_
#define _FFT_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cufft.h>
#include <cutil_inline.h>

/*
 * use cufft to fft the wav data with zero padding. 
 * Revoked on CPU, mainly executed on GPU
 *
 * Input
 * --------------------------------
 * originalwavdata_gpu: wav data get from wav file
 * wavdata_zp_gpu: rearranged wav data with zero padding
 * fft: save the fft result in GPU
 * hannwin_gpu: data of hann window, used to adjust the wav frame
 * ws_gpu: sum of hann window data in GPU
 * framenumber: the wav file has how many frames in all
 * framelength: the length of fft of every frame
 * hoplength: hop length of original wav data forward
 */
void fft_zp(float *originalwavdata_gpu,float* wavdata_zp_gpu,cufftComplex *fft_gpu,float * fft_result_gpu,float* hannwin_gpu,float *ws_gpu,int framenumber,int framelength,int hoplength);

//use cufft to fft the wav data without zero padding
void fft_nzp(float *originalwavdata,float* wavdata_zp);

/*
 * rearrange the wav data so as to the data are more easy to fft
 * Revoked on CPU, executed on GPU
 *
 * Input
 * -------------------------------
 * same as function fft_zp
 */
__global__ static void rearrange_data(float *originalwavdata_gpu,float* wavdata_zp_gpu,float* hannwin_gpu,int framenumber,int framelength,int hoplength);


/*
 * calculate the modulus of complex array. The implementation details use the sqrt t * to calculate the modulus. named abs is from matlab.
 *
 * Input
 * --------------------------------
 * fft_gpu: fft result, every element is a complex value
 * fft_result_gpu: modulus of every complex, the ultimate result
 * framenumber: how many frame with length (N/2+1) 
 */
__global__ static void fft_abs(cufftComplex* fft_gpu,float *fft_result_gpu,float *ws_gpu,int framenumber);

#endif
