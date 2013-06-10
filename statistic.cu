#include "statistic.h"
#include "util.h"

#ifndef CUBLAS_V2
#include <cublas.h>
#else
#include <cublas_v2.h>
#endif

#define BLOCK_NUM 60
#define THREAD_NUM 512

float Sum(float *data_gpu,float **s_gpu,int length)
{
    float *s;
    float *block_gpu;
    float *sum_gpu;
    s=(float*)malloc(sizeof(float));
    cutilSafeCall(cudaMalloc((void**)&sum_gpu,sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&block_gpu,sizeof(float)*BLOCK_NUM));
    sum<float><<<BLOCK_NUM,THREAD_NUM,THREAD_NUM*sizeof(float)>>>(data_gpu,block_gpu,length);
    sumblock<<<1,1,0>>>(block_gpu,sum_gpu,BLOCK_NUM);
    cutilCheckMsg("gpu sum failed\n");
    cutilSafeCall(cudaMemcpy(s,sum_gpu,sizeof(float),cudaMemcpyDeviceToHost));

    if(s_gpu!=NULL)
        *s_gpu=sum_gpu; 
    return *s;
}

int Sum(int *data_gpu,int **s_gpu,int length)
{
    int *s;
    int *block_gpu;
    int *sum_gpu;
    s=(int*)malloc(sizeof(int));
    cutilSafeCall(cudaMalloc((void**)&sum_gpu,sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&block_gpu,sizeof(int)*BLOCK_NUM));
    sum<int><<<BLOCK_NUM,THREAD_NUM,THREAD_NUM*sizeof(int)>>>(data_gpu,block_gpu,length);
    sumblock<<<1,1,0>>>(block_gpu,sum_gpu,BLOCK_NUM);
    cutilCheckMsg("gpu sum failed\n");
    cutilSafeCall(cudaMemcpy(s,sum_gpu,sizeof(int),cudaMemcpyDeviceToHost));
    if(s_gpu!=NULL)
        *s_gpu=sum_gpu; 
    return *s;
}

float* Sum(float* data_gpu,float **s_gpu,int nx,int batch)
{
    float *s;
    float *sum_gpu;
    s=(float*)malloc(sizeof(float)*batch);
    cutilSafeCall(cudaMalloc((void**)&sum_gpu,sizeof(float)*batch));
    sum<<<BLOCK_NUM,THREAD_NUM,THREAD_NUM*sizeof(float)>>>(data_gpu,sum_gpu,nx,batch);
    
    cutilCheckMsg("gpu sum failed\n");

    cutilSafeCall(cudaMemcpy(s,sum_gpu,sizeof(float)*batch,cudaMemcpyDeviceToHost));

    if(s_gpu!=NULL)
        *s_gpu=sum_gpu;
    return s;

}

template <class T>
__global__ static void sum(T* data_gpu,T* block_gpu,int length)
{
    extern __shared__ T blocksum[];
    int offset;

    const int tid=threadIdx.x;
    const int bid=blockIdx.x;
    blocksum[tid]=0;
    for(int i=bid*THREAD_NUM+tid;i<length;i+=BLOCK_NUM*THREAD_NUM)
    {
        blocksum[tid]+=data_gpu[i];
    } 

    __syncthreads();
    offset=THREAD_NUM/2;
    while(offset>0)
    {
        if(tid<offset)
        {
            blocksum[tid]+=blocksum[tid+offset];
        }
        offset>>=1;
        __syncthreads();//we should synchronize the threads to make sure all the output is calculated
    }

    if(tid==0)
    {
        block_gpu[bid]=blocksum[0];
    }
}


template <class T>
__global__ static void sumblock(T *block_gpu,T *sum_gpu,int length)
{
    T sum=0;

    for(int i=0;i<length;i++)
    {
        sum+=block_gpu[i];    
    }

    *sum_gpu=sum;
}


__global__ static void sum(float* data_gpu,float* sum_gpu,int nx,int batch)
{
    extern __shared__ float blocksum[];
    int offset;

    const int tid=threadIdx.x;
    const int bid=blockIdx.x;

    for(int i=bid;i<batch;i+=BLOCK_NUM)
    {
        blocksum[tid]=0; 
        for(int j=i*nx+tid;j<i*nx+nx;j+=THREAD_NUM)
        {
            blocksum[tid]+=data_gpu[j];
        }
        __syncthreads();
        offset=THREAD_NUM/2;
        while(offset>0)
        {
            if(tid<offset)
            {
                blocksum[tid]+=blocksum[tid+offset];
            }

            offset>>=1;
            __syncthreads();//we should synchronize the threads to make sure all the output is calculated
        }

        if(tid==0)
        {
            sum_gpu[i]=blocksum[0];
        }
    }
}



float Max(float *data_gpu,float **m_gpu,int length)
{
    float *m;
    float *block_gpu;
    float *max_gpu;
    m=(float*)malloc(sizeof(float));
    cutilSafeCall(cudaMalloc((void**)&max_gpu,sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&block_gpu,sizeof(float)*BLOCK_NUM));
    max<<<BLOCK_NUM,THREAD_NUM,THREAD_NUM*sizeof(float)>>>(data_gpu,block_gpu,max_gpu,length);
    cutilCheckMsg("GPU max failed\n");
    cutilSafeCall(cudaMemcpy(m,max_gpu,sizeof(float),cudaMemcpyDeviceToHost));

    *m_gpu=max_gpu; 
    return *m;
}

//low performance method
float*  Max(float *data_gpu,float **m_gpu,int nx,int batch)
{
    float *m;
    float *block_gpu;
    float *max_gpu;
    float *data;
    m=(float*)malloc(sizeof(float)*batch);
    cutilSafeCall(cudaMalloc((void**)&max_gpu,sizeof(float)*batch));
    cutilSafeCall(cudaMalloc((void**)&block_gpu,sizeof(float)*BLOCK_NUM));
    data=data_gpu;
    for(int i=0;i<batch;i++)
    {
        max<<<BLOCK_NUM,THREAD_NUM,THREAD_NUM*sizeof(float)>>>(data,block_gpu,&max_gpu[i],nx);
        cutilCheckMsg("GPU max failed\n");
        data+=nx;
    }
    cutilSafeCall(cudaMemcpy(m,max_gpu,sizeof(float)*batch,cudaMemcpyDeviceToHost));

    *m_gpu=max_gpu; 
    return m;
}

//high performance method, use this not above
float * Max2(float *data_gpu,float **m_gpu,int nx,int batch)
{
    struct timeval start,finish;

    float *m;
    float *max_gpu;
    m=(float*)malloc(sizeof(float)*batch);
    cutilSafeCall(cudaMalloc((void**)&max_gpu,sizeof(float)*batch));

    gettimeofday(&start,NULL);
    max<<<BLOCK_NUM,THREAD_NUM,THREAD_NUM*sizeof(float)>>>(data_gpu,max_gpu,nx,batch);
    cudaThreadSynchronize();
    gettimeofday(&finish,NULL);
    printf("time of peaks max  is %f\n",difftime_ms(finish,start));
    cutilSafeCall(cudaMemcpy(m,max_gpu,sizeof(float)*batch,cudaMemcpyDeviceToHost));
    
    *m_gpu=max_gpu; 
    return m;

}

//calculate the max of one array
__global__ static void max(float* data_gpu,float* block_gpu,float *max_gpu,int length)
{
    extern __shared__ float blockmax[];
    int offset;

    const int tid=threadIdx.x;
    const int bid=blockIdx.x;
    blockmax[tid]=0;//the smallest value of the array must be bigger than zero
    for(int i=bid*THREAD_NUM+tid;i<length;i+=BLOCK_NUM*THREAD_NUM)
    {
        if(blockmax[tid]<data_gpu[i])
            blockmax[tid]=data_gpu[i];
    } 

    __syncthreads();
    offset=THREAD_NUM/2;
    while(offset>0)
    {
        if(tid<offset&&blockmax[tid]<blockmax[tid+offset])
        {
            blockmax[tid]=blockmax[tid+offset];
        }
        offset>>=1;
        __syncthreads();//we should synchronize the threads to make sure all the output is calculated
    }

    if(tid==0)
    {
        block_gpu[bid]=blockmax[0];
    }

    if(bid==0&&tid==0)
    {
        *max_gpu=block_gpu[0];
        for(int i=1;i<BLOCK_NUM;i++)
        {
            if(*max_gpu<block_gpu[i])
                *max_gpu=block_gpu[i];
        }
    }
}

//calculate the max of batch arraies
__global__ static void max(float* data_gpu,float *max_gpu,int nx,int batch)
{
    extern __shared__ float blockmax[];
    int offset;

    const int tid=threadIdx.x;
    const int bid=blockIdx.x;

    for(int i=bid;i<batch;i+=BLOCK_NUM)
    {
        blockmax[tid]=0;//the smallest value of the array must be bigger than zero 
        for(int j=i*nx+tid;j<i*nx+nx;j+=THREAD_NUM)
        {
            if(blockmax[tid]<data_gpu[j])
                blockmax[tid]=data_gpu[j];
        }
        __syncthreads();
        offset=THREAD_NUM/2;
        while(offset>0)
        {
            if(tid<offset&&blockmax[tid]<blockmax[tid+offset])
            {
                blockmax[tid]=blockmax[tid+offset];
            }
            offset>>=1;
            __syncthreads();//we should synchronize the threads to make sure all the output is calculated
        }

        if(tid==0)
        {
            max_gpu[i]=blockmax[0];
        }
    }
}


float Sum_cublas(float *data_gpu,float **s_gpu,int length)
{
    float s;
    float *sum_gpu;
#ifndef CUBLAS_V2

    cublasStatus status=cublasInit();   
    cublasGetError();


    s=cublasSasum(length,data_gpu,1);
    cublasGetError();

    cutilSafeCall(cudaMalloc((void**)&sum_gpu,sizeof(float)));
    cutilSafeCall(cudaMemcpy(sum_gpu,&s,sizeof(float),cudaMemcpyHostToDevice));

    *s_gpu=sum_gpu; 
#else
    cublasHandle_t handle;
    cublasStatus_t status;

    status=cublasCreate(&handle);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr,"cublas create handle failed!\n");
        return EXIT_FAILURE;
    }
    cutilSafeCall(cudaMalloc((void**)&sum_gpu,sizeof(float)));

    status=cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr,"cublas set pointer mode failed!\n");
        return EXIT_FAILURE;
    }

    status=cublasSasum(handle,length,data_gpu,1,sum_gpu);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr,"cublas calculate sum failed!\n");
        return EXIT_FAILURE;
    }

    cutilSafeCall(cudaMemcpy(&s,sum_gpu,sizeof(float),cudaMemcpyDeviceToHost));

    cublasDestroy(handle);
#endif
    return s;
}
