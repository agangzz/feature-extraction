#include "findpeaks.h"
#include "util.h"

#define BLOCK_NUM 60
#define THREAD_NUM 512
#define PEAK_NUM 64
#define HALF_PEAK_NUM 32
#define QTR_PEAK_NUM 16

void Findpeaks(float *data_gpu,float **peaks_gpu_p,int framenumber,int length,int minlength)
{
    struct timeval start,finish;

    float *peaks_gpu;

    cutilSafeCall(cudaMalloc((void **)&peaks_gpu,sizeof(float)*minlength*framenumber));
    cutilSafeCall(cudaMemset(peaks_gpu,0,sizeof(float)*minlength*framenumber));


    gettimeofday(&start,NULL);
    findpeaks2<<<BLOCK_NUM,THREAD_NUM,0>>>(data_gpu,peaks_gpu,framenumber,length,minlength);
    cudaThreadSynchronize();

    gettimeofday(&finish,NULL);
    printf("time of find peaks is %f\n",difftime_ms(finish,start));

    *peaks_gpu_p=peaks_gpu;

    cutilSafeCall(cudaFree(data_gpu));
}


//find peaks with peaks value bigger than minpeakheight,used in salience function
void Findpeaks(float* data_gpu,float *max_gpu,float **peaks_gpu_p,int **index_gpu_p,int **peaks_num_gpu_p,int framenumber,int length)
{
    float *peaks_gpu;
    int *index_gpu;
    int *peaks_num_gpu;

    cutilSafeCall(cudaMalloc((void **)&peaks_gpu,sizeof(float)*HALF_PEAK_NUM*framenumber));
    cutilSafeCall(cudaMemset(peaks_gpu,0,sizeof(float)*HALF_PEAK_NUM*framenumber));

    cutilSafeCall(cudaMalloc((void **)&index_gpu,sizeof(int)*HALF_PEAK_NUM*framenumber));
    cutilSafeCall(cudaMemset(index_gpu,0,sizeof(int)*HALF_PEAK_NUM*framenumber));

    cutilSafeCall(cudaMalloc((void **)&peaks_num_gpu,sizeof(int)*framenumber));
    cutilSafeCall(cudaMemset(peaks_num_gpu,0,sizeof(int)*framenumber));

    findpeaks<<<BLOCK_NUM,THREAD_NUM,0>>>(data_gpu,max_gpu,peaks_gpu,index_gpu,peaks_num_gpu,framenumber,length);
    cutilCheckMsg("find peaks failed\n");


    *peaks_gpu_p=peaks_gpu;
    *index_gpu_p=index_gpu;
    *peaks_num_gpu_p=peaks_num_gpu;

    cutilSafeCall(cudaFree(data_gpu));

}

//schedule method one
__global__ static void findpeaks(float *data_gpu,float* peaks_gpu,int framenumber,int length,int minlength)
{
    int tid=threadIdx.x;
    int bid=blockIdx.x;

    for(int i=0;i<framenumber;i++)
    {
        for(int j=bid*THREAD_NUM+tid+1;j<minlength-1;j+=BLOCK_NUM*THREAD_NUM)
        {
            float data=data_gpu[i*length+j];
            if(data>data_gpu[i*length+j-1]&&data>data_gpu[i*length+j+1])
            {
                peaks_gpu[i*minlength+j]=data;
            }
        }
    }
}

//schedule method two
__global__ static void findpeaks2(float *data_gpu,float* peaks_gpu,int framenumber,int length,int minlength)
{
    int tid=threadIdx.x;
    int bid=blockIdx.x;

    for(int i=bid;i<framenumber;i+=BLOCK_NUM)
    {
        for(int j=tid+1;j<minlength-1;j+=THREAD_NUM)
        {
            float data=data_gpu[i*length+j];
            if(data>data_gpu[i*length+j-1]&&data>data_gpu[i*length+j+1])
            {
                peaks_gpu[i*minlength+j]=data;
            }
        }
    }
}


__global__ static void findpeaks(float *data_gpu,float* max_gpu,float* peaks_gpu,int *index_gpu,int *peaks_num_gpu,int framenumber,int length)
{
    __shared__ int peaks_num;//a variable in shared memory as a counter
    float threshold;

    peaks_num=-1;

    int tid=threadIdx.x;
    int bid=blockIdx.x;

    for(int i=bid;i<framenumber;i+=BLOCK_NUM)
    {
        threshold=max_gpu[i];
        for(int j=tid+1;j<length-1;j+=THREAD_NUM)
        {
            float data=data_gpu[i*length+j];
            if(data>threshold&&data>data_gpu[i*length+j-1]&&data>data_gpu[i*length+j+1])
            {
                int old=atomicAdd(&peaks_num,1); 
                if(old>=HALF_PEAK_NUM-1)
                {
                    atomicAdd(&peaks_num,-1);
                    break;
                }
                else
                {
                    old++;
                    peaks_gpu[i*HALF_PEAK_NUM+old]=data;
                    index_gpu[i*HALF_PEAK_NUM+old]=j+1;
                }
            }
        }
        __syncthreads();

        if(tid==0)
        {
            peaks_num_gpu[i]=peaks_num+1;
            peaks_num=-1;
        }
    }
}


void Biggerfilter(float *data_gpu,float *max_gpu,float **bigger_gpu_p,int **index_gpu_p,int **bigger_num_gpu_p,int framenumber,int length,float filterratio)
{
    struct timeval start,finish;

    float *bigger_gpu;
    int *index_gpu;
    int *bigger_num_gpu;

    cutilSafeCall(cudaMalloc((void **)&bigger_gpu,sizeof(float)*PEAK_NUM*framenumber));
    cutilSafeCall(cudaMemset(bigger_gpu,0,sizeof(float)*PEAK_NUM*framenumber));

    cutilSafeCall(cudaMalloc((void **)&index_gpu,sizeof(int)*PEAK_NUM*framenumber));
    cutilSafeCall(cudaMemset(index_gpu,0,sizeof(int)*PEAK_NUM*framenumber));

    cutilSafeCall(cudaMalloc((void **)&bigger_num_gpu,sizeof(int)*framenumber));
    cutilSafeCall(cudaMemset(bigger_num_gpu,0,sizeof(int)*framenumber));

    //junior filter
    gettimeofday(&start,NULL);
    biggerfilter<<<BLOCK_NUM,THREAD_NUM,0>>>(data_gpu,max_gpu,bigger_gpu,index_gpu,bigger_num_gpu,framenumber,length,filterratio);
    cudaThreadSynchronize();
    gettimeofday(&finish,NULL);
    printf("time of  filter junior is %f\n",difftime_ms(finish,start));


    float *last_bigger_gpu;
    int *last_index_gpu;

    cutilSafeCall(cudaMalloc((void **)&last_bigger_gpu,sizeof(float)*HALF_PEAK_NUM*framenumber));
    cutilSafeCall(cudaMemset(last_bigger_gpu,0,sizeof(float)*HALF_PEAK_NUM*framenumber));

    cutilSafeCall(cudaMalloc((void **)&last_index_gpu,sizeof(int)*HALF_PEAK_NUM*framenumber));
    cutilSafeCall(cudaMemset(last_index_gpu,0,sizeof(int)*HALF_PEAK_NUM*framenumber));

    //senior filter
    gettimeofday(&start,NULL);
    morefilter<<<BLOCK_NUM,THREAD_NUM,0>>>(bigger_gpu,index_gpu,max_gpu,last_bigger_gpu,last_index_gpu,bigger_num_gpu,framenumber,filterratio);
    cudaThreadSynchronize();
    gettimeofday(&finish,NULL);
    printf("time of  filter senior is %f\n",difftime_ms(finish,start));

    *bigger_gpu_p=last_bigger_gpu;
    *index_gpu_p=last_index_gpu;
    *bigger_num_gpu_p=bigger_num_gpu;

    cutilSafeCall(cudaFree(data_gpu));
    cutilSafeCall(cudaFree(max_gpu));
    cutilSafeCall(cudaFree(bigger_gpu));
    cutilSafeCall(cudaFree(index_gpu));
}

__global__ static void biggerfilter(float *data_gpu,float *max_gpu,float *bigger_gpu,int *index_gpu,int *bigger_num_gpu,int framenumber,int length,float filterratio)
{
    __shared__ int bigger_num;//a variable in shared memory as a counter
    bigger_num=-1;

    int tid=threadIdx.x;
    int bid=blockIdx.x;

    for(int i=bid;i<framenumber;i+=BLOCK_NUM)//one block handles one frame
    {
        for(int j=tid;j<length;j+=THREAD_NUM)
        {
            float data=data_gpu[i*length+j];
            if(max_gpu[i]*filterratio<data)
            {
                int old=atomicAdd(&bigger_num,1);/*skillful statement, first we atomic add the counter, 
                                                  * so the counter will have the right count. And the 
                                                  * function will return the original value, we can use
                                                  * original value to access the right memory. From this
                                                  * way, we make the bottleneck limited to this statement.
                                                  * because of the fast access of shared memory, it's really
                                                  * a good way. ;)
                                                  */
                if(old>=PEAK_NUM-1)//we only save PEAK_NUM values bigger than max*filterratio.
                {
                    atomicAdd(&bigger_num,-1);
                    break;
                }
                else
                {
                    old++;
                    bigger_gpu[i*PEAK_NUM+old]=data;
                    index_gpu[i*PEAK_NUM+old]=j;
                }
            }
        }
        __syncthreads();

        if(tid==0)
        {
            bigger_num_gpu[i]=bigger_num+1;//because it starts from -1, we need add 1 at last.
            bigger_num=-1;
        }
    }
}


__global__ static void morefilter(float *data_gpu,int* prev_index_gpu,float *max_gpu,float *bigger_gpu,int *index_gpu,int *bigger_num_gpu,int framenumber,float filterratio)
{
    int frameperblock=THREAD_NUM/PEAK_NUM;

    __shared__ int bigger_num[THREAD_NUM/PEAK_NUM];
    __shared__ int number[THREAD_NUM/PEAK_NUM];

    int tid=threadIdx.x;
    int bid=blockIdx.x;

    if(tid<frameperblock)
    {
        bigger_num[tid]=-1;
    }

    int innernumber=tid/PEAK_NUM;
    int innertid=tid%PEAK_NUM;


    for(int i=bid*frameperblock;i<framenumber;i+=BLOCK_NUM*frameperblock)
    {
        if(i+innernumber<framenumber)
        {
            number[innernumber]=bigger_num_gpu[i+innernumber];

            if(number[innernumber]>16)
            {
                float data=data_gpu[(i+innernumber)*PEAK_NUM+innertid];
                if(max_gpu[(i+innernumber)]*filterratio*number[innernumber]/16<data)
                {
                    int old=atomicAdd(&bigger_num[innernumber],1);
                    if(old>=HALF_PEAK_NUM-1)
                    {
                        atomicAdd(&bigger_num[innernumber],-1);
                        break;
                    }
                    else
                    {
                        old++;
                        bigger_gpu[(i+innernumber)*HALF_PEAK_NUM+old]=data;
                        index_gpu[(i+innernumber)*HALF_PEAK_NUM+old]=prev_index_gpu[(i+innernumber)*PEAK_NUM+innertid];
                    }
                }
                __syncthreads();

                if(innertid==0)
                {
                    bigger_num_gpu[i+innernumber]=bigger_num[innernumber]+1;
                    bigger_num[innernumber]=-1;
                }
            }
            else
            {
                if(innertid<HALF_PEAK_NUM)
                {
                    bigger_gpu[(i+innernumber)*HALF_PEAK_NUM+innertid]=data_gpu[(i+innernumber)*PEAK_NUM+innertid];
                    index_gpu[(i+innernumber)*HALF_PEAK_NUM+innertid]=prev_index_gpu[(i+innernumber)*PEAK_NUM+innertid];
                }
            }
        }
    }
}

__global__ static void morefilter2(float *data_gpu,int* prev_index_gpu,float *max_gpu,float *bigger_gpu,int *index_gpu,int *bigger_num_gpu,int framenumber,float filterratio)
{
    __shared__ int bigger_num;//a variable in shared memory as a counter
    __shared__ int number;
    bigger_num=-1;

    int tid=threadIdx.x;
    int bid=blockIdx.x;

    for(int i=bid;i<framenumber;i+=BLOCK_NUM)//one block handles one frame
    {
        number=bigger_num_gpu[i];
        if(number>16)
        {
            for(int j=tid;j<PEAK_NUM;j+=THREAD_NUM)
            {
                float data=data_gpu[i*PEAK_NUM+j];
                if(max_gpu[i]*filterratio*number/16<data)
                {
                    int old=atomicAdd(&bigger_num,1);
                    if(old>=HALF_PEAK_NUM-1)
                    {
                        atomicAdd(&bigger_num,-1);
                        break;
                    }
                    else
                    {
                        old++;
                        bigger_gpu[i*HALF_PEAK_NUM+old]=data;
                        index_gpu[i*HALF_PEAK_NUM+old]=prev_index_gpu[i*PEAK_NUM+j];
                    }
                }
            }
            __syncthreads();

            if(tid==0)
            {
                bigger_num_gpu[i]=bigger_num+1;//because it starts from -1, we need add 1 at last.
                bigger_num=-1;
            }
        }
        else
        {
            if(tid<HALF_PEAK_NUM)
            {
                bigger_gpu[i*HALF_PEAK_NUM+tid]=data_gpu[i*PEAK_NUM+tid];
                index_gpu[i*HALF_PEAK_NUM+tid]=prev_index_gpu[i*PEAK_NUM+tid];
            }

        }
    }
}

