#include "util.h"

float difftime_ms(struct timeval finish,struct timeval start)
{
    return (float)(finish.tv_sec-start.tv_sec)*1000+(float)(finish.tv_usec-start.tv_usec)*1.0/1000;
}

float sum(float *data,int length)
{
    float s=0;
    for(int i=0;i<length;i++)
    {
        s+=data[i];
    }
    return s;
}

float * hann(int length)
{
    float * data;
    data=(float*)malloc(sizeof(float)*length);
    for(int i=0;i<length;i++)
    {
        data[i]=(float)0.5*(1-cos(6.283186*i/(length-1)));
    }
    return data;
}


