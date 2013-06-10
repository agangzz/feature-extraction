
/*
 * this file delete the contour using the contour characteristics.
 * Include three procedures: voice detection, octave detection and outlier deletion.
 */
#ifndef _FILTERCONTOUR_H_
#define _FILTERCONTOUR_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include "contour.h"

void deleteusingvoiced(contours *cons);

void deleteusingpitchstdandlength(contours *cons,double std_threshold,int length_threshold);


/*
 * delete contours using salience mean and salience std
 *
 * Input
 * ------------------------------------------
 * cons: the set of all the contours
 * threshold: equals salience mean minus (v*salience_std)
 */
void deleteusingsaliencemean(contours* cons,double threshold);


void deleteusingpitchmean(contours *cons,double lthreshold);


/*
 * delete contours using length
 *
 * Input
 * ------------------------------------------
 * cons: the set of all the contours
 * length: the length threshold of the contour
 */
void deleteusinglength(contours* cons,int length);


/*
 * delete contours using melody pitch mean
 *
 * Input
 * ------------------------------------------
 * cons: the set of all the contours
 * melodypitchmean: the mean pitch of all the contours
 * start: the start position of the mean pitch contour
 * threshold: the threshold between the pitch mean of the contour and the melody pitch mean
 */
void deleteusingmelodypitchmean(contours* cons,float *melodypitchmean,int start,double threshold);


/*
 * delete contours found octave error using melody pitch mean
 *
 * Input
 * ------------------------------------------
 * cons: the set of all the contours
 * melodypitchmean: the mean pitch of all the contours
 * mstart: the start position of the mean pitch contour
 */
void deleteoctave(contours *cons,float *melodypitchmean,int mstart,int label);


/*
 * delete contours found octave error using length
 *
 * Input
 * ------------------------------------------
 * cons: the set of all the contours
 */
void deleteoctaveusinglength(contours *cons);


/*
 * delete pitch outliers using melody pitch mean
 *
 * Input
 * ------------------------------------------
 * cons: the set of all the contours
 * melodypitchmean: the mean pitch of all the contours
 * start: the start position of the mean pitch contour
 */
void deletepitchoutlier(contours* cons,float *melodypitchmean,int start,int label);


/*
 * delete the pitches at given frame
 * 
 * Input
 * ------------------------------------------
 * cons: the set of all the contours
 * frame: the number of given frame
 */
contour* deletepitchesatframe(contours *cons,int frame);


/*
 * generate the ultimate melody contour through delete the pitch which is longer with melody pitch mean
 *
 * Input
 * ------------------------------------------
 * cons: the set of all the contours
 * framenumber: how many frames in all. also the length of the contour
 */
int* ultimatemelodycontour(contours *cons,int framenumber);


int* ultimatemelodycontour(contours *cons,float *melodypitchmean,int start,int framenumber);

#endif
