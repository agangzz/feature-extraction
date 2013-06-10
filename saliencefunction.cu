#include "saliencefunction.h"
#include "statistic.h"
#include "findpeaks.h"
#include "util.h"

#define BLOCK_NUM 60
#define THREAD_NUM 512
#define BIN_NUM 480
#define PEAK_NUM 64
#define HALF_PEAK_NUM 32
#define QTR_PEAK_NUM 16
#define NH 8
#define ALPHA 0.8f


void Saliencefunc(float *peaks_gpu,int *index_gpu,int *peaks_num_gpu,float** filteredpeaks_gpu_p,int ** filteredindex_gpu_p,int **filteredpeaks_num_gpu_p,float**leftpeaks_gpu_p,int ** leftindex_gpu_p,int **leftpeaks_num_gpu_p,int N,int sampleRate,int framenumber)
{
    struct timeval start,finish;

    float tfirst=0.25f;
    float tplus=0.9f;

    float* saliencepeaks_gpu;

    float *saliencebins_gpu;
    float *saliencemax_gpu;

    float *saliencemaxpeaks_gpu;
    int *salienceindex_gpu;
    int *saliencepeaks_num_gpu;

    float *filteredpeaks_gpu;
    int *filteredindex_gpu;
    int *filteredpeaks_num_gpu;

    float *leftpeaks_gpu;
    int *leftindex_gpu;
    int *leftpeaks_num_gpu;


    cudaThreadSynchronize();

    gettimeofday(&start,NULL);
    cutilSafeCall(cudaMalloc((void **)&saliencebins_gpu,sizeof(float)*BIN_NUM*framenumber));


    saliencefunc<<<BLOCK_NUM,THREAD_NUM,0>>>(peaks_gpu,index_gpu,peaks_num_gpu,saliencebins_gpu,N,sampleRate,framenumber);
    cutilCheckMsg("saliencefunc failed\n");
    cudaThreadSynchronize();
    gettimeofday(&finish,NULL);
    printf("time of salience func is %f\n",difftime_ms(finish,start));

    cutilSafeCall(cudaFree(peaks_num_gpu));
    cutilSafeCall(cudaFree(peaks_gpu));
    cutilSafeCall(cudaFree(index_gpu));


    //gettimeofday(&start,NULL);
    //first find the peaks of the salience, every peaks frame has the same length with salience frame.
    //none peaks are zero. the space of salience is released.
    //doing this first is that the max may not be the peaks sometimes
    Findpeaks(saliencebins_gpu,&saliencepeaks_gpu,framenumber,BIN_NUM,BIN_NUM);

    //get the salience max of every frame peaks
    Max2(saliencepeaks_gpu,&saliencemax_gpu,BIN_NUM,framenumber);

    multipy<<<BLOCK_NUM,THREAD_NUM,0>>>(saliencemax_gpu,framenumber,tfirst);

    //get the peaks in every frame using salience max and tfirst,save HALF_PEAK_NUM peaks at most
    //so that the peaks number in  leached part is small enough.
    Findpeaks(saliencepeaks_gpu,saliencemax_gpu,&saliencemaxpeaks_gpu,&salienceindex_gpu,&saliencepeaks_num_gpu,framenumber,BIN_NUM);


    cutilSafeCall(cudaMalloc((void **)&filteredpeaks_gpu,sizeof(float)*HALF_PEAK_NUM*framenumber));
    cutilSafeCall(cudaMemset(filteredpeaks_gpu,0,sizeof(float)*HALF_PEAK_NUM*framenumber));
    cutilSafeCall(cudaMalloc((void **)&filteredindex_gpu,sizeof(int)*HALF_PEAK_NUM*framenumber));
    cutilSafeCall(cudaMemset(filteredindex_gpu,0,sizeof(int)*HALF_PEAK_NUM*framenumber));
    cutilSafeCall(cudaMalloc((void **)&filteredpeaks_num_gpu,sizeof(int)*framenumber));

    cutilSafeCall(cudaMalloc((void **)&leftpeaks_gpu,sizeof(float)*QTR_PEAK_NUM*framenumber));
    cutilSafeCall(cudaMemset(leftpeaks_gpu,0,sizeof(float)*QTR_PEAK_NUM*framenumber));
    cutilSafeCall(cudaMalloc((void **)&leftindex_gpu,sizeof(int)*QTR_PEAK_NUM*framenumber));
    cutilSafeCall(cudaMemset(leftindex_gpu,0,sizeof(int)*QTR_PEAK_NUM*framenumber));
    cutilSafeCall(cudaMalloc((void **)&leftpeaks_num_gpu,sizeof(int)*framenumber));

    multipy<<<BLOCK_NUM,THREAD_NUM,0>>>(saliencemax_gpu,framenumber,tplus/tfirst);

    //spilt the salince peaks to two parts, one bigger than max*tplus, the other is smaller.
    gettimeofday(&start,NULL);
    spilt2<<<BLOCK_NUM,THREAD_NUM,0>>>(saliencemax_gpu,saliencemaxpeaks_gpu,salienceindex_gpu,saliencepeaks_num_gpu,
                             filteredpeaks_gpu,filteredindex_gpu,filteredpeaks_num_gpu,
                             leftpeaks_gpu,leftindex_gpu,leftpeaks_num_gpu,
                             framenumber);
    cudaThreadSynchronize();
    gettimeofday(&finish,NULL);
    printf("time of spilt is %f\n",difftime_ms(finish,start));

    *filteredpeaks_gpu_p=filteredpeaks_gpu;
    *filteredindex_gpu_p=filteredindex_gpu;
    *filteredpeaks_num_gpu_p=filteredpeaks_num_gpu;
    *leftpeaks_gpu_p=leftpeaks_gpu;
    *leftindex_gpu_p=leftindex_gpu;
    *leftpeaks_num_gpu_p=leftpeaks_num_gpu;


    cutilSafeCall(cudaFree(saliencemax_gpu));
    cutilSafeCall(cudaFree(saliencemaxpeaks_gpu));
    cutilSafeCall(cudaFree(salienceindex_gpu));
    cutilSafeCall(cudaFree(saliencepeaks_num_gpu));//printed outside this function

    cudaThreadSynchronize();
}


__global__ static void spilt(float *max_gpu,float *peaks_gpu,int *index_gpu,int *peaks_num_gpu,
                             float *filteredpeaks_gpu, int *filteredindex_gpu,int *filteredpeaks_num_gpu,
                             float *leftpeaks_gpu, int *leftindex_gpu,int *leftpeaks_num_gpu,
                             int framenumber)
{
    __shared__ int filtered_num;
    __shared__ int left_num;
    float threshold;

    int tid=threadIdx.x;
    int bid=blockIdx.x;

    if(tid==0)
    {
        filtered_num=-1;
        left_num=-1;
    }


    for(int i=bid;i<framenumber;i+=BLOCK_NUM)
    {   
        threshold=max_gpu[i];
        if(tid<HALF_PEAK_NUM)
        {
            float data=peaks_gpu[i*HALF_PEAK_NUM+tid];
            
            if(data>threshold)
            {
                int old=atomicAdd(&left_num,1); 
                old++;
                
                leftpeaks_gpu[i*QTR_PEAK_NUM+old]=data;
                leftindex_gpu[i*QTR_PEAK_NUM+old]=index_gpu[i*HALF_PEAK_NUM+tid];
            }
            else if(data>0)
            {
                int old=atomicAdd(&filtered_num,1); 
                old++;
                filteredpeaks_gpu[i*HALF_PEAK_NUM+old]=data;
                filteredindex_gpu[i*HALF_PEAK_NUM+old]=index_gpu[i*HALF_PEAK_NUM+tid];
            }
        }

        __syncthreads();

        if(tid==0)
        {
            filteredpeaks_num_gpu[i]=filtered_num+1;
            filtered_num=-1;
        }

        if(tid==1)
        {
            leftpeaks_num_gpu[i]=left_num+1;
            left_num=-1;
        }
    }
}
__global__ static void spilt2(float *max_gpu,float *peaks_gpu,int *index_gpu,int *peaks_num_gpu,
                             float *filteredpeaks_gpu, int *filteredindex_gpu,int *filteredpeaks_num_gpu,
                             float *leftpeaks_gpu, int *leftindex_gpu,int *leftpeaks_num_gpu,
                             int framenumber)
{
    int frameperblock=THREAD_NUM/HALF_PEAK_NUM;

    __shared__ int filtered_num[THREAD_NUM/HALF_PEAK_NUM];
    __shared__ int left_num[THREAD_NUM/HALF_PEAK_NUM];
    __shared__ float threshold[THREAD_NUM/HALF_PEAK_NUM];

    int tid=threadIdx.x;
    int bid=blockIdx.x;

    if(tid<frameperblock)
    {
        filtered_num[tid]=-1;
        left_num[tid]=-1;
    }

    int innernumber=tid/HALF_PEAK_NUM;
    int innertid=tid%HALF_PEAK_NUM;

    for(int i=bid*frameperblock;i<framenumber;i+=BLOCK_NUM*frameperblock)
    {   
        if(i+innernumber<framenumber)//there is the possibility the index exceeds the frame number
        {
            threshold[innernumber]=max_gpu[i+innernumber];

            float data=peaks_gpu[(i+innernumber)*HALF_PEAK_NUM+innertid];

            if(data>threshold[innernumber])
            {
                int old=atomicAdd(&left_num[innernumber],1); 
                old++;

                leftpeaks_gpu[(i+innernumber)*QTR_PEAK_NUM+old]=data;
                leftindex_gpu[(i+innernumber)*QTR_PEAK_NUM+old]=index_gpu[(i+innernumber)*HALF_PEAK_NUM+innertid];
            }
            else if(data>0)
            {
                int old=atomicAdd(&filtered_num[innernumber],1); 
                old++;
                filteredpeaks_gpu[(i+innernumber)*HALF_PEAK_NUM+old]=data;
                filteredindex_gpu[(i+innernumber)*HALF_PEAK_NUM+old]=index_gpu[(i+innernumber)*HALF_PEAK_NUM+innertid];
            }

            __syncthreads();

            if(innertid==0)
            {
                filteredpeaks_num_gpu[i+innernumber]=filtered_num[innernumber]+1;
                filtered_num[innernumber]=-1;
            }

            if(innertid==1)
            {
                leftpeaks_num_gpu[i+innernumber]=left_num[innernumber]+1;
                left_num[innernumber]=-1;
            }
        }
    }
}


__global__ static void multipy(float* data_gpu,int length,float ratio)
{
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    
    for(int i=bid*THREAD_NUM+tid;i<length;i+=BLOCK_NUM*THREAD_NUM)
    {
        data_gpu[i]=data_gpu[i]*ratio;
    }
}

//!!!use so many registers, need optimization!!!
__global__ static void saliencefunc(float *peaks_gpu,int *index_gpu,int *peaks_num_gpu,float *saliencebins_gpu,int N,int sampleRate,int framenumber)
{
    __shared__ float peaks[HALF_PEAK_NUM];
    __shared__ int index[HALF_PEAK_NUM];

    int tid=threadIdx.x;
    int bid=blockIdx.x;

    for(int i=bid;i<framenumber;i+=BLOCK_NUM)
    {
        if(tid<HALF_PEAK_NUM)
        {
            peaks[tid]=peaks_gpu[HALF_PEAK_NUM*i+tid];//!!!use i, not bid!!!
            index[tid]=index_gpu[HALF_PEAK_NUM*i+tid];
        }
        __syncthreads();

        for(int j=tid;j<BIN_NUM;j+=THREAD_NUM)//one thread cal one bins!!! can take out the for circulation
        {
            saliencebins_gpu[i*BIN_NUM+j]=salience(peaks,index,i,peaks_num_gpu[i],N,sampleRate,j+1);//???exists bank conflict???
        }
        __syncthreads();//we must synchronize all the threads in the same block or the shared memory 
                        // may be overlapped
    }
}

__device__ inline static float salience(float *peaks,int *index,int frame,int peaks_num,int N,int sampleRate,int b)
{
    float ss=0;
    float rate=sampleRate*1.0f/N;
    for(int p=0;p<peaks_num;p++)
    {
#pragma unroll 8
        for(int h=1;h<=NH;h++)
        {
            ss+=weight(b,h,index[p]*rate)*peaks[p];
        }
    } 

    return ss;
}

__device__ inline static float weight(int b,int harmonic,float frequency)
{
    float delta=fabsf(bin(frequency/harmonic)-b)/10;

    if(delta<=1)
        return cosf(delta*1.5708f)*cosf(delta*1.5708f)*powf(ALPHA,harmonic-1);
    else
        return 0;
}


__device__ inline static int bin(float frequency)
{
    return (int)floorf(120*log2f(frequency/90)+1);
}
