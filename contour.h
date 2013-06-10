
/*
 * the struct of the contours and the operation functions of the contours
 *
 */

#ifndef _CONTOUR_H_
#define _CONTOUR_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cufft.h>
#include <cutil_inline.h>

typedef struct _pitch
{
    double salience;
    int binindex;

    struct _pitch *next;
}pitch;

typedef struct _contour
{
    int size;

    int start;
    int end;

    int voiced;

    struct _pitch *head;
    struct _pitch *tail;

    struct _contour *next;//it is used to generate the contours below

    double salience_sum;
    double salience_mean;
    double salience_std;

    double pitch_mean;
    double pitch_std;

    double vibrato;
}contour;

//used to save the other attribute of the contour except the pitch linkedlist
typedef struct _contour_attribute
{
    int size;

    int start;
    int end;

    int voiced;
}contour_attribute;

typedef struct _contours
{
    int length;

    int maxcontoursize;

    struct _contour *first;
    struct _contour *last;
}contours;


/*
 * append one pitch at the end of the contour
 *
 * Input
 * -------------------------------------------
 * con: the contour  added the pitch
 * pit: the added pitch
 */
void insertaftertail(contour* con,pitch* pit);


/*
 * delete head from one contour
 *
 * Input
 * ------------------------------------------
 * con: the contour will be deleted the head
 *
 * Output
 * ------------------------------------------
 * return the deleted pitch head
 */
pitch* deletehead(contour *con);


/*
 * add one contour to the contours set
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 * con: the contour will be added to the contours
 *
 */
void addcontour(contours *cons,contour *con);


/*
 * get the melody range of all the contours
 *
 * Input
 * ------------------------------------------
 * cons: the set of the contours
 * start: the pointer points to the melody start
 * end: the pointer pionts to the melody end
 */
void melodyrange(contours *cons,int *start,int *end);


/*
 * get the salience and the binindex of the contour at the given frame
 *
 * Input
 * ------------------------------------------
 * con: given contour
 * frame: specified frame
 * salience: the pointer points to the pitch salience
 * binindex: the pointer points to the pitch binindex
 *
 * Output
 * ------------------------------------------
 * return the frame is whether right
 */
int getframe(contour* con,int frame,double *salience,int *binindex);


/*
 * get all the pitches at given frame
 *
 * Input
 * ------------------------------------------
 * cons: the set of the contours
 * frame: specified frame
 *
 * Output
 * ------------------------------------------
 * return the pitch contour at the given frame
 */
contour* pitchesatframe(contours* cons,int frame);


/*
 * get the weighted pitch of the contour using weighted mean method of salience
 *
 * Input
 * ------------------------------------------
 * con: the given contour
 *
 * Output
 * ------------------------------------------
 * return the weighted pitch
 */
double weightedpitch(contour *con);


/*
 * generate the mean contour based on all the contours
 *
 * Input
 * ------------------------------------------
 * cons: the set of all the contours
 * hop: half length of the brute force filter
 * start: the pointer points to the melody start
 * end: the pointer points to the melody end
 *
 * Output
 * ------------------------------------------
 * return the generate mean contour
 */
float *pitchmeancontour(contours* cons,int hop,int *start,int *end);


__global__ static void smoothfilter(float *data,int length,int hop);


__global__ static void smoothfilter2(float *data,int length,int hop);


/*
 * print the contours
 *
 * Input
 * ------------------------------------------
 * cons: the set of the contours
 */
void printcontours(contours *cons);


/* 
 * print one contour
 *
 * Input
 * ------------------------------------------
 * con: the contour will be printed
 */
void printcontour(contour *con);


/*
 * release the space of one contour
 * 
 * Input
 * ------------------------------------------
 * con: the contour will be released the space
 */
void freecontour(contour *con);


/*
 * get the length of all the contours
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 *
 * Output
 * ------------------------------------------
 * the length of all the contours
 */
int contourslen(contours *cons);


/*
 * change the contour size to array format
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 *
 * Output
 * ------------------------------------------
 * the size array
 */
int *sizetoarray(contours *cons);


/*
 * cal the acculated size of sizes
 * 
 * Input
 * ------------------------------------------
 * sizes: the size array
 * length: the length of the size array
 *
 * Output
 * ------------------------------------------
 * the acculated sum array
 */
int *acculatedsize(int *sizes,int length);
/*
 * change the contour pitches to array format
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 * sizes: the contours size in array format
 * lenall: the length of all the contours
 *
 * Output
 * ------------------------------------------
 * the pitches array
 */
pitch *pitchtoarray(contours *cons,int *sizes,int lenall);


/*
 * change the contour attribute except the pitches to arraies
 *
 * Input
 * ------------------------------------------
 * cons: the set of contours
 *
 * Output
 * ------------------------------------------
 * the attibute arraies
 */
contour_attribute* contourtoattributearray(contours* cons);


/*
 * add the contour attribute from arraies to the contour struct
 *
 * Input
 * ------------------------------------------
 * cons: the set of the contours
 * salience_sum: the salience sum of the contours
 * salience_mean: the salience mean of the contours
 * salience_std: the salience std of the contours
 * pitch_mean: the pitch mean of the contours
 * pitch_std: the pitch std of the contours
 */
void attrtocontours(contours *cons,double *salience_sum,double *salience_mean,double *salience_std,double *pitch_mean,double* pitch_std,double *vibrato);


double contoursvibratomean(contours *cons);


double contoursvibratostd(contours *cons,double vibratomean);


/*
 * decide if two pitches are different in one octave
 * 
 * Input
 * ------------------------------------------
 * p1: pitch one
 * p2: pitch two
 *
 * Output
 * ------------------------------------------
 * return 1 if they are different in one octave, or return 0
 */
int octave(double p1,double p2);


/*
 * decide the distance of two pitches is longer than one octave
 * 
 * Input
 * ------------------------------------------
 * p1: pitch one
 * p2: pitch two
 *
 * Output
 * ------------------------------------------
 * return 1 if the distance is longer than one octave, or return 0
 */
int pitchoutlier(double p1,double p2);

#endif

