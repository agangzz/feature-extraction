/*
 * this head file calculate the hann window in GPU
 */

#ifndef _HANN_H_
#define _HANN_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


#include <cutil_inline.h>

/*get hann window in GPU
 *
 * Input
 * ----------------------------
 * length: the length of hann window
 * 
 * Return
 * ----------------------------
 * all the value of hann window in GPU
 */
float *Hann(int length);


/*calculate the hann window in GPU
 *
 * Input
 * ----------------------------
 * data: calculated value will be stored in the memory
 * length: hann window length
 */
__global__ static void  hanning(float *data,int length);

#endif
