/*
 *this head file spectral transform the wav data, and get the peaks of every frame
 */

#ifndef _SPECTRALTRANSFORM_H_
#define _SPECTRALTRANSFORM_H_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>


/*
 * spectral transform the wav data, and get the peaks of every frame
 * 
 * Input
 * ----------------------------------------
 * wavdata: original wav data extracted from wav file
 * datalength: the length of wavdata
 * sampleRate: sample rate of the wav file
 * N: the length of fft
 * framenumber: the frame number for fft of wavdata
 * framelength: the length of every frame
 * hoplength: the stride of frame
 */
void spectraltransform(float * wavdata,int datalength,unsigned long sampleRate,int N,int framenumber,int framelength,int hoplength,float **bigger_gpu_p,int **index_gpu_p,int **bigger_num_gpu_p);



#endif
