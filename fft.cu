#include "fft.h"
#include "util.h"
#include <sys/time.h>
#include <unistd.h>

#define N 8192
#define BLOCK_NUM 60
#define THREAD_NUM 512

void fft_zp(float *originalwavdata_gpu,float* wavdata_zp_gpu,cufftComplex *fft_gpu,float * fft_result_gpu,float* hannwin_gpu,float * ws_gpu,int framenumber,int framelength,int hoplength)
{
    struct timeval start,finish;
    cufftHandle plan;


    //change the data more easy to use in fft
    gettimeofday(&start,NULL);
    rearrange_data<<<BLOCK_NUM,THREAD_NUM,sizeof(float)*framelength>>>(originalwavdata_gpu, wavdata_zp_gpu,hannwin_gpu,framenumber,framelength,hoplength);
    cudaThreadSynchronize();
    gettimeofday(&finish,NULL);
    printf("time of rearrange_data is %f\n",difftime_ms(finish,start));

    //cufft
    gettimeofday(&start,NULL);
    cufftSafeCall(cufftPlan1d(&plan,8192,CUFFT_R2C,framenumber));
    cufftSafeCall(cufftExecR2C(plan,(cufftReal *)wavdata_zp_gpu,(cufftComplex*)fft_gpu));
    cudaThreadSynchronize();
    gettimeofday(&finish,NULL);
    printf("time of cufft is %f\n",difftime_ms(finish,start));

    
    //abs(f)
    gettimeofday(&start,NULL);
    fft_abs<<<BLOCK_NUM,THREAD_NUM,0>>>(fft_gpu,fft_result_gpu,ws_gpu,framenumber);
    cudaThreadSynchronize();
    gettimeofday(&finish,NULL);
    printf("time of fft abs is %f\n",difftime_ms(finish,start));


    cutilSafeCall(cudaFree(originalwavdata_gpu));
    cutilSafeCall(cudaFree(wavdata_zp_gpu));
    cutilSafeCall(cudaFree(fft_gpu));
    cutilSafeCall(cudaFree(hannwin_gpu));
    cutilSafeCall(cudaFree(ws_gpu));
    cufftDestroy(plan);
    cudaThreadSynchronize();
}

__global__ static void rearrange_data(float *originalwavdata_gpu,float* wavdata_zp_gpu,float* hannwin_gpu,int framenumber,int framelength,int hoplength)
{
    int tid=threadIdx.x;
    int bid=blockIdx.x;

    extern __shared__ float hannwin[];

    for(int i=tid;i<framelength;i+=THREAD_NUM)
    {
        hannwin[i]=hannwin_gpu[i];
    }

    __syncthreads();

    for(int i=bid;i<framenumber;i+=BLOCK_NUM)
    {
        for(int j=tid;j<framelength;j+=THREAD_NUM)
        {
            wavdata_zp_gpu[j+i*N]=originalwavdata_gpu[j+i*hoplength]*hannwin[j];
        }
    }
}


__global__ static void fft_abs(cufftComplex* fft_gpu,float *fft_result_gpu,float *ws_gpu,int framenumber)
{
    const int tid=threadIdx.x;
    const int bid=blockIdx.x;

    float ws=(*ws_gpu)*2;

    for(int i=bid;i<framenumber;i+=BLOCK_NUM)
    {
        for(int j=tid;j<(N/2+1);j+=THREAD_NUM)
        {
            float x=fft_gpu[j+i*(N/2+1)].x;
            float y=fft_gpu[j+i*(N/2+1)].y;
            fft_result_gpu[j+i*(N/2+1)]=sqrtf(x*x+y*y)/ws;
        }
    }
}


void fft_nzp(float *originalwavdata_gpu,float* wavdata_zp_gpu)
{

}
