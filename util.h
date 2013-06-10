/*
 * this file has all the useful functions which have some general use
 * !!!!all the functions are executed on CPU!!!!
 */

#ifndef _UTIL_H_
#define _UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/*
 * get the diff time in ms 
 *
 * Input
 * -----------------
 * finish: the finish time of the programming point.
 * start: the start time of the programming point.
 *
 * Return:
 * --------------------
 * the function returns the time between finish and start.
 */
float difftime_ms(struct timeval finish,struct timeval start);


/*
 * sum of the float data with length
 *
 * Input:
 * -----------------
 * data: float data in CPU memory
 * length: the length of the data
 *
 * Return:
 * --------------------------------
 * sum of the data with length
 */
float sum(float *data,int length);


/*
 * calculate the hann window with length
 *
 * Input
 * -----------------------------
 * length: the length of hann window
 *
 * Output
 * ------------------------------
 * value of hann window with length
 */
float * hann(int length);

#endif
