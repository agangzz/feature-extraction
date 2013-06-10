#include "hann.h"

#define BLOCK_NUM 64
#define THREAD_NUM 512

float *Hann(int length)
{
    float *data_gpu;
    cutilSafeCall(cudaMalloc((void **)&data_gpu,sizeof(float)*length));
    hanning<<<BLOCK_NUM,THREAD_NUM,0>>>(data_gpu,length);
    cutilCheckMsg("hann calculate failed\n");
    
    return data_gpu;
}

__global__ static void  hanning(float *data_gpu,int length)
{
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    for(int i=bid*THREAD_NUM+tid;i<length;i+=BLOCK_NUM*THREAD_NUM)
    {
        data_gpu[i]=(float)(1-cosf(6.283183*i/(length-1)))/2;
    } 
}
