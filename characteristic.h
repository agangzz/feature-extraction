
/*
 * thie file calculates the characteristics of all the contours
 */
#ifndef _FEATURES_H_
#define _FEATURES_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#include "contour.h"


/*
 * cal the pitch mean of every the contours
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 */
void Pitchmeans(contours *cons);


/*
 * cal the pitch std of every the contours
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 */
void Pitchstds(contours *cons);


/*
 * cal the salience sum of every the contours
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 */
void Saliencesums(contours *cons);


/*
 * cal the salience mean of every the contours
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 */
void Saliencemeans(contours *cons);


/*
 * cal the salience std of every the contours
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 */
void Saliencestds(contours *cons);


/*
 * cal the salience mean of all the contours, only one result
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 */
double Contoursaliencemean(contours *cons);


/*
 * cal the salience std of all the contours, only one result
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 * mean: the contour salience mean
 */
double Contoursaliencestd(contours *cons,double mean);

/*
 * cal the pitch mean of all the contours, only one result
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 */
double Contourpitchmean(contours *cons);

/*
 * cal the pitch std of all the contours, only one result
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 * mean: the contour pitch mean
 */
double Contourpitchstd(contours *cons,double mean);


/*
 * cal the pitch mean of one contour
 *
 * Input:
 * ------------------------------------------
 * con: the contour
 */
void pitchmean(contour *con);


/*
 * cal the pitch std of one contour
 *
 * Input:
 * ------------------------------------------
 * con: the contour
 */
void pitchstd(contour *con);


/*
 * cal the salience sum of one contour
 *
 * Input:
 * ------------------------------------------
 * con: the contour
 */
void saliencesum(contour *con);


/*
 * cal the salience mean of one contour
 *
 * Input:
 * ------------------------------------------
 * con: the contour
 * flag: means if the salience sum is calculated. 1 cal, 0 not cal
 */
void saliencemean(contour *con,int flag);


/*
 * cal the salience std of one contour
 *
 * Input:
 * ------------------------------------------
 * con: the contour
 */
void saliencestd(contour *con);


#endif
