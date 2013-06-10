#include "spectraltransform.h"
#include "fft.h"
#include "hann.h"
#include "statistic.h"
#include "findpeaks.h"
#include "util.h"

#define PEAK_NUM 64
#define HALF_PEAK_NUM 32
#define FRAME_NUM 10

void spectraltransform(float * wavdata,int datalength,unsigned long sampleRate,int N,int framenumber,int framelength,int hoplength,float **bigger_gpu_p,int **index_gpu_p,int **bigger_num_gpu_p)
{
    //time variable
    struct timeval start,finish;

    //GPU fft data
    float *originalwavdata_gpu;
    float *wavdata_zp_gpu;
    cufftComplex * fft_gpu;
    float *fft_result_gpu;


    //GPU hann data
    float *hannwin_gpu;
    float *ws_gpu;


    int edgelength;
    int minlength;

    //GPU peaks data
    float *peaks_gpu;
    float* max_gpu;

    //GPU bigger data
    float *bigger_gpu;
    int *index_gpu;
    int *bigger_num_gpu;



    //calculate hann window on GPU
    gettimeofday(&start,NULL);
    hannwin_gpu=Hann(framelength);
    Sum(hannwin_gpu,&ws_gpu,framelength);
    gettimeofday(&finish,NULL);
    printf("time of hann is %f\n\n",difftime_ms(finish,start));
    printf("data size is %d,sample rate is %ld\n",datalength,sampleRate);
    printf("frame length is: %d,hop length is %d,frame number is %d\n",framelength,hoplength,framenumber);


    ///////////////////////////////////////////////////////////////////////
    //doing real work on GPU
    //////////////////////////////////////////////////////////////////////
    gettimeofday(&start,NULL);
    cutilSafeCall(cudaMalloc((void **)&originalwavdata_gpu,sizeof(float)*datalength));
    cutilSafeCall(cudaMalloc((void **)&wavdata_zp_gpu,sizeof(float)*N*framenumber));
    cutilSafeCall(cudaMalloc((void **)&fft_gpu,sizeof(cufftComplex)*(N/2+1)*framenumber));
    cutilSafeCall(cudaMemcpy(originalwavdata_gpu,wavdata,datalength*sizeof(float),cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemset(wavdata_zp_gpu,0,sizeof(float)*N*framenumber));
    cutilSafeCall(cudaMalloc((void **)&fft_result_gpu,sizeof(float)*(N/2+1)*framenumber));

    //fft the wav data with zero padding 
    fft_zp(originalwavdata_gpu,wavdata_zp_gpu,fft_gpu,fft_result_gpu,hannwin_gpu,ws_gpu,framenumber,framelength,hoplength);
    gettimeofday(&finish,NULL);
    printf("time of fft is %f\n",difftime_ms(finish,start));

    edgelength=(int)ceil(5000*N*1.0/sampleRate);
    minlength=(int)fmin((float)(N/2+1),(float)edgelength);

    gettimeofday(&start,NULL);
    //find the peaks of the fft result
    Findpeaks(fft_result_gpu,&peaks_gpu,framenumber,(N/2+1),minlength);
    //gettimeofday(&finish,NULL);
    //printf("time of find peaks is %f\n",difftime_ms(finish,start));

    //gettimeofday(&start,NULL);
    float *m=Max2(peaks_gpu,&max_gpu,minlength,framenumber);
    cudaThreadSynchronize();
    //gettimeofday(&finish,NULL);
    //printf("time of peaks max  is %f\n",difftime_ms(finish,start));

    //gettimeofday(&start,NULL);
    Biggerfilter(peaks_gpu,max_gpu,&bigger_gpu,&index_gpu,&bigger_num_gpu,framenumber,minlength,0.2);
    cudaThreadSynchronize();
    gettimeofday(&finish,NULL);
    printf("time of spectral peaks filter  is %f\n",difftime_ms(finish,start));

    *bigger_gpu_p=bigger_gpu;
    *index_gpu_p=index_gpu;
    *bigger_num_gpu_p=bigger_num_gpu;
}




