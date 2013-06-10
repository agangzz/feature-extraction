#include "saliencefilter.h"
#include "statistic.h"

#define BLOCK_NUM 60
#define THREAD_NUM 512
#define HALF_PEAK_NUM 32
#define QTR_PEAK_NUM 16

void saliencefilter(float *filteredpeaks_gpu,int *filteredindex_gpu,int *filteredpeaks_num_gpu,float *leftpeaks_gpu,int *leftindex_gpu,int *leftpeaks_num_gpu,int framenumber)
{
    float* leftpeaks_sum_gpu;
    float* leftpeaks_square_gpu;
    float salience_sum;
    float salience_square_sum;
    int salience_length;
    float salience_mean;
    float salience_std;
    float tdelta=0.9f;


    //get the salience sum in all frames of left part
    Sum(leftpeaks_gpu,&leftpeaks_sum_gpu,QTR_PEAK_NUM,framenumber);
    salience_sum=Sum(leftpeaks_sum_gpu,NULL,framenumber);
    salience_length=Sum(leftpeaks_num_gpu,NULL,framenumber);
    salience_mean=salience_sum/salience_length;

    //get the square sum of all frames of left parts
    Saliencesquare(leftpeaks_gpu,&leftpeaks_square_gpu,QTR_PEAK_NUM,framenumber,salience_mean);
    salience_square_sum=Sum(leftpeaks_square_gpu,NULL,framenumber);
    salience_std=sqrt(salience_square_sum/salience_length);
 
    printf("salience mean is %f,std is %f\n",salience_mean,salience_std);


    //filter the left parts
    filter<<<BLOCK_NUM,THREAD_NUM,0>>>(filteredpeaks_gpu,filteredindex_gpu,filteredpeaks_num_gpu,leftpeaks_gpu,leftindex_gpu,leftpeaks_num_gpu,framenumber,salience_mean-tdelta*salience_std);

    //move forward the data to remove zero value in front end.
    shift<<<BLOCK_NUM,THREAD_NUM,0>>>(leftpeaks_gpu,leftindex_gpu,framenumber);

}



void Saliencesquare(float *leftpeaks_gpu,float** square_gpu,int nx,int batch,float salience_mean)
{
    float *leftpeaks_square_gpu;
    cutilSafeCall(cudaMalloc((void**)&leftpeaks_square_gpu,sizeof(float)*batch));

    saliencesquare<<<BLOCK_NUM,THREAD_NUM,THREAD_NUM*sizeof(float)>>>(leftpeaks_gpu,leftpeaks_square_gpu,nx,batch,salience_mean);

    *square_gpu=leftpeaks_square_gpu;
}



__global__ static void saliencesquare(float *leftpeaks_gpu,float* leftpeaks_square_gpu,int nx,int batch,float salience_mean)
{
    extern __shared__ float blocksquare[];
    int offset;

    const int tid=threadIdx.x;
    const int bid=blockIdx.x;

    for(int i=bid;i<batch;i+=BLOCK_NUM)
    {
        blocksquare[tid]=0; 
        for(int j=i*nx+tid;j<i*nx+nx;j+=THREAD_NUM)
        {
            if(leftpeaks_gpu[j]>0)
                blocksquare[tid]+=(leftpeaks_gpu[j]-salience_mean)*(leftpeaks_gpu[j]-salience_mean);
        }
        __syncthreads();
        offset=THREAD_NUM/2;
        while(offset>0)
        {
            if(tid<offset)
            {
                blocksquare[tid]+=blocksquare[tid+offset];
            }

            offset>>=1;
            __syncthreads();//we should synchronize the threads to make sure all the output is calculated
        }

        if(tid==0)
        {
            leftpeaks_square_gpu[i]=blocksquare[0];
        }
    }
}

//filter the peaks in left part using salience mean and std to filtered part.
__global__ static void filter(float *filteredpeaks_gpu, int *filteredindex_gpu,int *filteredpeaks_num_gpu,float *leftpeaks_gpu, int *leftindex_gpu,int *leftpeaks_num_gpu,int framenumber,float threshold)
{
    __shared__ int filtered_num;
    __shared__ int left_num;


    const int tid=threadIdx.x;
    const int bid=blockIdx.x;

    for(int i=bid;i<framenumber;i+=BLOCK_NUM)
    {
        if(tid==0)
        {
            filtered_num=filteredpeaks_num_gpu[i];
            left_num=leftpeaks_num_gpu[i];
            __threadfence_block();/*there is very little possiblilty that the other threads
                                    can execute more faster than thread 0 and more earlier
                                    to visit the variable in shared memory in the following
                                    statement:atomicAdd(&left_num,-1); because of the assignment
                                    value to peaks and index in one warp.
                                        But in different warps, warp 0 is excuting assignment value
                                    to left_num, the other may be executing assignment value
                                    to peaks and index, and finish assignment earlier than warp
                                    0. so the other warp may execute the atomic statement before
                                    the assignment to left_num in warp 0.
                                        But here, because of the short length of QTR_PEAK_NUM, the 
                                    other warps don't execute the following judge statement. It's
                                    ok to delete this statement safely.
                                    */
        }
        if(tid<QTR_PEAK_NUM)
        {
            float peaks=leftpeaks_gpu[i*QTR_PEAK_NUM+tid];
            int index=leftindex_gpu[i*QTR_PEAK_NUM+tid];

            if(peaks<threshold&&peaks>0)
            {
                //left_num--;
                atomicAdd(&left_num,-1);
                //atomicAdd(&leftpeaks_num_gpu[i],-1);
                leftpeaks_gpu[i*QTR_PEAK_NUM+tid]=0;
                leftindex_gpu[i*QTR_PEAK_NUM+tid]=0;

                int old=atomicAdd(&filtered_num,1);

                filteredpeaks_gpu[i*HALF_PEAK_NUM+old]=peaks;
                filteredindex_gpu[i*HALF_PEAK_NUM+old]=index;
            }

        }
        __syncthreads();


        if(tid==0)
        {
            filteredpeaks_num_gpu[i]=filtered_num;
        }

        if(tid==1)
        {
            leftpeaks_num_gpu[i]=left_num;

        }
        /*there is no need to synchronize all the threads in one block.
          the threads in one warp are shcheduled simultaneously, and executed
          the same time.
            when thread 1 writes the shared memory back to global memory, the 
          other threads in one warp may start to processing the other frames.
          but they will not modify the shared memory and wait for thread 0
          to modify it because thread 0 is also writing the shared memory 
          back to global memory.
            the code doesn't show they will wait for thread 0 to modify, but
          the thread fence will.
            thread 2-15: satify the if(tid<QTR_PEAK_NUM) statement, will modify
          peaks and index. only two or three of these threads will satify the
          inner if statement, and will modify the left_num in shared memory.
          but they will not sub left_num unless thread 0 changed left_num
          first using thread fence.
            thread 16-511: do null operation, and wairt the __syncthread();
         */
    }
}

__global__ static void shift(float *leftpeaks_gpu, int *leftindex_gpu,int framenumber)
{
    __shared__ int left_num;//a variable in shared memory as a counter
    left_num=-1;

    const int tid=threadIdx.x;
    const int bid=blockIdx.x;

    for(int i=bid;i<framenumber;i+=BLOCK_NUM)
    {
        if(tid<QTR_PEAK_NUM)
        {
            float data=leftpeaks_gpu[i*QTR_PEAK_NUM+tid];
            int index=leftindex_gpu[i*QTR_PEAK_NUM+tid];
            if(data>0)
            {
                leftpeaks_gpu[i*QTR_PEAK_NUM+tid]=0;
                leftindex_gpu[i*QTR_PEAK_NUM+tid]=0;

                __threadfence_block();/*there is no need to add fence here because the short length
                                        of QTR_PEAK_NUM makes the number doing real work to half warp.
                                            if QTR_PEAK_NUM is long enough that every thread in one
                                        block does real work, the fence should be added.
                                            because that the other warp may be executed faster than
                                        warp 0. so there exists some warp which zero-clear the global
                                        memory and assign the new value to the global array in the
                                        front end. but at this time, warp 0 still doesn't execute 
                                        the clear operation. When warp 0 executes the clear operation
                                        afterwards, it will overload the new value assigned by the
                                        other warp. so the other warp should wait for the warp 0 to
                                        clear the global memory or there will be wrong.
                                            we can also use the __syncthreads(); operation, but it
                                        will make all the threads in one block stop in this point.
                                        This will drop the speed.
                                        */
                int old=atomicAdd(&left_num,1);
                old++;
                leftpeaks_gpu[i*QTR_PEAK_NUM+old]=data;
                leftindex_gpu[i*QTR_PEAK_NUM+old]=index;

            }
        }
        __syncthreads();
        if(tid==0)
        {
            left_num=-1;
        }
    }

}
