#ifndef _MELODYCONTOUR_H_
#define _MELODYCONTOUR_H_


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/*
 * process the wav file, and get the melody contour
 *
 * Input
 * ------------------------------------------
 * wavfile:wav clip
 * kind: the kind of wav file:1 for polyphonic music, 2 for monophonic
 */
void melodycontour(char* wavfile,int kind);

#endif
