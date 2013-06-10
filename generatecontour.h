
/*
 * generate the contours using the pitch in the left and filtered parts 
 *
 */

#ifndef _GENERATECONTOUR_H_
#define _GENERATECONTOUR_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include "contour.h"

/*
 * generate the contours using the pitch in the left and filtered parts
 *
 * Input
 * ------------------------------------------
 * filteredpeaks: the peaks of every frame in filtered parts
 * filteredindex: the peak index of every frame in filtered parts
 * filteredpeaks_num: the peak number of every frame in filtered parts
 * leftpeaks: the peaks of every frame in left parts
 * leftindex: the peak index of every frame in left parts
 * leftpeaks_num: the peak number of every frame in left parts
 * framenumber: the number of frames
 * contourgap: the max number of the contour allowed to visit the filtered parts
 *
 * Output
 * -----------------------------------------
 * all the contours
 */
contours* generateContours(float *filteredpeaks,int *filteredindex,int *filteredpeaks_num,float *leftpeaks,int *leftindex,int *leftpeaks_num,int framenumber,int contourgap);

#endif
