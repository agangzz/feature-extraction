#include "ReadWav.h"
#include "spectraltransform.h"
#include "saliencefunction.h"
#include "saliencefilter.h"
#include "generatecontour.h"
#include "contour.h"
#include "characteristic.h"
#include "filtercontour.h"
#include "util.h"

#include <cutil_inline.h>


#define N 8192
#define BLOCK_NUM 60
#define THREAD_NUM 512
#define BIN_NUM 480
#define HALF_PEAK_NUM 32
#define QTR_PEAK_NUM 16
#define FRAME_NUM 29


void melodycontour(char* wavfile,int kind)
{
    struct timeval start,finish,first,end;
    float frametime=46.44f;
    float hoptime=10.0f;
    float v=0.2f;
   
    //wav data
    int framelength;//length of every frame
    int hoplength;
    int framenumber;//how many FFT frames in all,one frame will get a pitch
    int datalength;//=sampleRate*wavDuration. length of all frames(has stride)
    unsigned long sampleRate;

    //spectral transform result in GPU
    float *bigger_gpu;
    int *index_gpu;
    int *bigger_num_gpu;

    float* filteredpeaks_gpu;
    int* filteredindex_gpu;
    int* filteredpeaks_num_gpu;

    float* leftpeaks_gpu;
    int* leftindex_gpu;
    int* leftpeaks_num_gpu;

    float* filteredpeaks;
    int* filteredindex;
    int* filteredpeaks_num;

    float* leftpeaks;
    int* leftindex;
    int* leftpeaks_num;

    //FILE *salience_fp;

    //read wav file to get data
    CReadWav wavFile;

    gettimeofday(&first,NULL);

    gettimeofday(&start,NULL);
    wavFile.readData(wavfile);
    sampleRate=wavFile.pWav.nSamplesPerSec;
    datalength=wavFile.framelen;
    gettimeofday(&finish,NULL);
    printf("time of read wav is %f\n\n",difftime_ms(finish,start));

    //calculate variables about frame data
    framelength=(int)floor(sampleRate*frametime/1000);
    hoplength=(int)floor(sampleRate*hoptime/1000);
    framenumber=(datalength-framelength)/hoplength;

    gettimeofday(&start,NULL);
    spectraltransform(wavFile.data,datalength,sampleRate,N,framenumber,framelength,hoplength,&bigger_gpu,&index_gpu,&bigger_num_gpu);
    gettimeofday(&finish,NULL);
    printf("time of spectral transform is %f\n\n",difftime_ms(finish,start));

    gettimeofday(&start,NULL);
    Saliencefunc(bigger_gpu,index_gpu,bigger_num_gpu,&filteredpeaks_gpu,&filteredindex_gpu,&filteredpeaks_num_gpu,&leftpeaks_gpu,&leftindex_gpu,&leftpeaks_num_gpu,N,sampleRate,framenumber);
    gettimeofday(&finish,NULL);
    printf("time of salience is %f\n\n",difftime_ms(finish,start));

    gettimeofday(&start,NULL);
    saliencefilter(filteredpeaks_gpu,filteredindex_gpu,filteredpeaks_num_gpu,leftpeaks_gpu,leftindex_gpu,leftpeaks_num_gpu,framenumber);
    gettimeofday(&finish,NULL);
    printf("time of salience filter is %f\n\n",difftime_ms(finish,start));

    filteredpeaks=(float*)malloc(sizeof(float)*HALF_PEAK_NUM*framenumber);
    filteredindex=(int*)malloc(sizeof(int)*HALF_PEAK_NUM*framenumber);
    filteredpeaks_num=(int*)malloc(sizeof(int)*framenumber);

    leftpeaks=(float*)malloc(sizeof(float)*QTR_PEAK_NUM*framenumber);
    leftindex=(int*)malloc(sizeof(int)*QTR_PEAK_NUM*framenumber);
    leftpeaks_num=(int*)malloc(sizeof(int)*framenumber);

    cutilSafeCall(cudaMemcpy(filteredpeaks,filteredpeaks_gpu,sizeof(float)*HALF_PEAK_NUM*framenumber,cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(filteredindex,filteredindex_gpu,sizeof(int)*HALF_PEAK_NUM*framenumber,cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(filteredpeaks_num,filteredpeaks_num_gpu,sizeof(int)*framenumber,cudaMemcpyDeviceToHost));

    cutilSafeCall(cudaMemcpy(leftpeaks,leftpeaks_gpu,sizeof(float)*QTR_PEAK_NUM*framenumber,cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(leftindex,leftindex_gpu,sizeof(int)*QTR_PEAK_NUM*framenumber,cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(leftpeaks_num,leftpeaks_num_gpu,sizeof(int)*framenumber,cudaMemcpyDeviceToHost));

    cutilSafeCall(cudaFree(filteredpeaks_gpu));
    cutilSafeCall(cudaFree(filteredindex_gpu));
    cutilSafeCall(cudaFree(filteredpeaks_num_gpu));
    cutilSafeCall(cudaFree(leftpeaks_gpu));
    cutilSafeCall(cudaFree(leftindex_gpu));
    cutilSafeCall(cudaFree(leftpeaks_num_gpu));



    /*
    salience_fp=fopen("./output/filteredsalience.txt","wb");
    if(salience_fp!=NULL)
    {
        for(int i=0;i<framenumber;i++)
        {
            fprintf(salience_fp,"frame %d left %d peaks,filtered is %d\n",i,leftpeaks_num[i],filteredpeaks_num[i]);
            for(int j=i*QTR_PEAK_NUM;j<i*QTR_PEAK_NUM+QTR_PEAK_NUM;j++)
            {
                fprintf(salience_fp,"%f(%d) %f(%d)\n",leftpeaks[j],leftindex[j],filteredpeaks[j+i*QTR_PEAK_NUM],filteredindex[j+i*QTR_PEAK_NUM]);
            }
        }
    }
    fclose(salience_fp);
    */


    int contourgap=(int)(50/hoptime)+1;
    contours *cons;

    gettimeofday(&start,NULL);
    cons=generateContours(filteredpeaks,filteredindex,filteredpeaks_num,leftpeaks,leftindex,leftpeaks_num,framenumber,contourgap);
    printf("have %d contours in all\n",cons->length);

    free(filteredpeaks);
    free(filteredindex);
    free(filteredpeaks_num);
    free(leftpeaks);
    free(leftindex);
    free(leftpeaks_num);

    deleteusinglength(cons,6);
    //printf("after delete using length, have %d contours in all\n",cons->length);

    gettimeofday(&finish,NULL);
    //printf("time of generate contours is %f\n",difftime_ms(finish,start));


    float contours_salience_mean;
    float contours_salience_std;

    float contours_pitch_mean;
    float contours_pitch_std;

    gettimeofday(&start,NULL);
    Pitchmeans(cons);
    Pitchstds(cons);
    Saliencesums(cons);
    Saliencemeans(cons);
    Saliencestds(cons);
    gettimeofday(&finish,NULL);
    printf("time of characteristic is %f\n",difftime_ms(finish,start));

    //printcontours(cons);


    gettimeofday(&start,NULL);
    if(kind==1)
    {
        deleteusingpitchstdandlength(cons,10,20);
        printf("after delete using pitch std and length, have %d contours in all\n",cons->length);
    }

    contours_salience_mean=Contoursaliencemean(cons);
    contours_salience_std=Contoursaliencestd(cons,contours_salience_mean);
    //printf("salience mean is %f, salience std is %f,salience threshold is %f\n",contours_salience_mean,contours_salience_std,contours_salience_mean-v*(contours_salience_std));

    deleteusingsaliencemean(cons,contours_salience_mean-v*(contours_salience_std));
    printf("after delete using salience mean, have %d contours in all\n",cons->length);

    if(kind==1)
    {
        contours_pitch_mean=Contourpitchmean(cons);
        contours_pitch_std=Contourpitchstd(cons,contours_pitch_mean);
        printf("contours pitch mean is %f,contours pitch std is %f,threshold is %f\n",contours_pitch_mean,contours_pitch_std,contours_pitch_mean-1.2f*contours_pitch_std);

        deleteusingpitchmean(cons,contours_pitch_mean-1.2f*contours_pitch_std);
        printf("after delete using pitch mean, have %d contours in all\n",cons->length);
    }

    int s,e;
    float *melodypitchmean;
    for(int i=0;i<3;i++)
    {
        melodypitchmean=pitchmeancontour(cons,250,&s,&e);
        deletepitchoutlier(cons,melodypitchmean,s,0);
        free(melodypitchmean);
    }
    printf("after delete pitch outlier, have %d contours in all\n",cons->length);


    melodypitchmean=pitchmeancontour(cons,250,&s,&e);
    deleteoctave(cons,melodypitchmean,s,0);
    free(melodypitchmean);
    printf("after delete octave, have %d contours in all\n",cons->length);


    melodypitchmean=pitchmeancontour(cons,250,&s,&e);
    int *ultimatemelodypitch;
    //ultimatemelodypitch=ultimatemelodycontour(cons,framenumber);
    ultimatemelodypitch=ultimatemelodycontour(cons,melodypitchmean,s,framenumber);
    free(melodypitchmean);

    int i=0;
    
    //continuation naturally
    while(i<framenumber)
    {
        if(ultimatemelodypitch[i]==0)
        {
            int j=i+1;
            while(j<framenumber&&ultimatemelodypitch[j]==0)
            {
                j++;
            }
            if(j-i<6&&i>0)
            {
                for(int k=i;k<=j-1;k++)
                {
                    ultimatemelodypitch[k]=ultimatemelodypitch[i-1];
                }
            }
            i=j+1;
        }
        else
        {
            i++;
        }
    }

    gettimeofday(&finish,NULL);
    printf("time of voice detection is %f\n",difftime_ms(finish,start));

    gettimeofday(&start,NULL);
    char *wav=strrchr(wavfile,(int)'.');
    strncpy(wav,".txt",4);

    FILE *fp;

    fp=fopen(wavfile,"wb");
    if(fp!=NULL)
    {
        fprintf(fp,"%d\r\n",framenumber);
        for(int i=0;i<framenumber;i++)
        {
            fprintf(fp,"%d\r\n",ultimatemelodypitch[i]);
        }

        fclose(fp);
    }
    else
    {
        fprintf(stderr,"open file failed!\n");
    }

    free(ultimatemelodypitch);

    gettimeofday(&finish,NULL);
    printf("time of write to file is %f\n",difftime_ms(finish,start));
    gettimeofday(&end,NULL);
    printf("time  is %f\n",difftime_ms(end,first));

}

