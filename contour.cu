
#include "contour.h"
#include "util.h"


#define BLOCK_NUM 240
#define THREAD_NUM 128

#define Free(p) do {                                            \
         printf("%s:%d:%s:free(0x%lx)\n",__FILE__,__LINE__,     \
         __func__,(unsigned long)p);                            \
        free(p);                                                \
}while(0)


void insertaftertail(contour* con,pitch* pit)
{
  if(con->size==0)
  {
      con->head=pit;
      con->tail=pit;
  }  
  else
  {
      con->tail->next=pit;
      con->tail=pit;
  }
  con->size=con->size+1;
}

pitch* deletehead(contour *con)
{
    pitch *temp=0;

    if(con->head!=NULL)
    {
        temp=con->head;
        con->head=con->head->next;
        temp->next=0;

        con->size=con->size-1;
        con->start=con->start+1;
    }

    return temp;
}

void addcontour(contours *cons,contour *con)
{
    if(cons->length==0)
    {
        cons->first=con;
        cons->last=con;
    }
    else
    {
        cons->last->next=con;
        cons->last=con;
    }
    cons->length=cons->length+1;
}

void melodyrange(contours *cons,int *start,int *end)
{
    contour *temp;
    int s,e;
    
    temp=cons->first;
    s=temp->start;
    e=temp->end;
    while(temp!=NULL)
    {
        if(e<temp->end)
            e=temp->end;
        temp=temp->next;
    }

    *start=s;
    *end=e;
}

int getframe(contour* con,int frame,double *salience,int *binindex)
{
    pitch* temp;
    int pos;



    if(frame<con->start||frame>con->end)
    {
        return 0;
    }

    temp=con->head;
    pos=con->start;
    while(temp!=NULL)
    {
        if(pos==frame)
        {
            *salience=temp->salience;
            *binindex=temp->binindex;
            break;
        }
        pos++;
        temp=temp->next;
    }

    return 1;
}


contour *pitchesatframe(contours* cons,int frame)
{
    contour *con=(contour *)malloc(sizeof(contour));
    contour *temp;

    con->size=0;
    con->next=0;
    con->head=0;

    //printf("find the pitches at frame %d\n",frame);

    temp=cons->first;
    while(temp!=NULL)
    {
        if(temp->start<=frame&&temp->voiced==1)
        {
            if(temp->end>=frame)
            {
                double salience;
                int binindex;
                pitch *pit;

                getframe(temp,frame,&salience,&binindex);
                pit=(pitch*)malloc(sizeof(pitch));
                pit->next=0;
                pit->salience=salience;
                pit->binindex=binindex;

                insertaftertail(con,pit);
                              
            }
        }
        else
        {
            break;
        }
        temp=temp->next;
    }
    //printf("contour length is %d\n",con->size);

    return con;
}

double weightedpitch(contour *con)
{
    pitch *temp;
    double sum=0;
    double wp=0;

    temp=con->head;
    while(temp!=NULL)
    {
        sum+=temp->salience;
        temp=temp->next;
    }

    temp=con->head;
    while(temp!=NULL)
    {
        wp+=temp->binindex*10*(temp->salience/sum);
        temp=temp->next;
    }


    return wp;
}

float *pitchmeancontour(contours* cons,int hop,int *s,int *e)
{
    struct timeval starts,finish;
    int start,end;
    float *melodypitchmean,wp;
    contour *framepitches;


    melodyrange(cons,&start,&end);
    //printf("melody start %d,end %d\n",start,end);

    melodypitchmean=(float*)malloc(sizeof(float)*(end-start+1));

    for(int i=0;i<end-start+1;i++)
    {
        framepitches=pitchesatframe(cons,i+start);
        //printf("size is %d\n",framepitches->size);
        wp=(float)weightedpitch(framepitches);
        //printf("wp is %f\n",wp);
        if(wp==0)
        {
            melodypitchmean[i]=melodypitchmean[i-1];
        }
        else
        {
            melodypitchmean[i]=wp;
        }
        freecontour(framepitches);
    }

    gettimeofday(&starts,NULL);

    /*
    for(int i=0;i<end-start+1;i++)
    {   
        double s=0;
        int k=0;

        for(int j=-hop;j<=hop;j++)
        {
            if(i+j>=0&&i+j<end-start+1)
            {
                s+=melodypitchmean[i+j];
                k++;
            }
        }
        melodypitchmean[i]=s/k;
    }

    gettimeofday(&finish,NULL);
    printf("time of melody pitch mean is %f\n",difftime_ms(finish,starts));
    */
    
    float *melodypitchmean_gpu;

    cutilSafeCall(cudaMalloc((void **)&melodypitchmean_gpu,sizeof(float)*(end-start+1)));
    cutilSafeCall(cudaMemcpy(melodypitchmean_gpu,melodypitchmean,sizeof(float)*(end-start+1),cudaMemcpyHostToDevice));
    smoothfilter<<<BLOCK_NUM,THREAD_NUM,sizeof(float)*THREAD_NUM>>>(melodypitchmean_gpu,end-start+1,hop);
    cutilSafeCall(cudaMemcpy(melodypitchmean,melodypitchmean_gpu,sizeof(float)*(end-start+1),cudaMemcpyDeviceToHost));
    gettimeofday(&starts,NULL);
    cutilSafeCall(cudaFree(melodypitchmean_gpu));
    gettimeofday(&finish,NULL);

    //gettimeofday(&finish,NULL);
    printf("time of melody pitch mean is %f\n",difftime_ms(finish,starts));

    /*
    for(int i=0;i<10;i++)
    {
        printf("%d: %f\n",i,melodypitchmean[i]);
    }
    */

    *s=start;
    *e=end;

    return melodypitchmean;
}


__global__ static void smoothfilter(float *data,int length,int hop)
{
    extern __shared__ float s[];
    int k;
    int offset;

    int tid=threadIdx.x;
    int bid=blockIdx.x;

    for(int i=bid;i<length;i+=BLOCK_NUM)
    {
        s[tid]=0;
        k=0;

        if(tid==0)
        {
            if(i<hop-1)
            {
                k+=(i+1);
            }
            else
            {
                k+=hop;
            }

            if(i+hop<length)
            {
                k+=hop;
            }
            else
            {
                k+=(length-i);
            }
        }
        for(int j=tid-hop;j<=hop;j+=THREAD_NUM)
        {
            if(i+j>=0&&i+j<length)
            {
               s[tid]+=data[i+j];
            }
        }
        __syncthreads();
        offset=THREAD_NUM/2;
        while(offset>0)
        {
            if(tid<offset)
            {
                s[tid]+=s[tid+offset];
            }

            offset>>=1;
            __syncthreads();
        }

        if(tid==0)
        {
            data[i]=s[0]/k;
        }
    }
}


__global__ static void smoothfilter2(float *data,int length,int hop)
{
    float s;
    int k;
    int kk;

    int bid=blockIdx.x;

    for(int i=bid;i<length;i+=BLOCK_NUM)
    {
        s=0;
        k=0;

        if(i<hop-1)
        {
            k+=(i+1);
        }
        else
        {
            k+=hop;
        }

        if(i+hop<length)
        {
            k+=hop;
        }
        else
        {
            k+=(length-i);
        }

        for(int j=-hop;j<=hop;j++)
        {
            if(i+j>=0&&i+j<length)
            {
               s+=data[i+j];
            }
        }
        data[i]=s/k;
    }
}


void printcontours(contours *cons)
{
    contour *temp;
    int i;

    temp=cons->first;
    i=1;
    while(temp!=NULL)
    {
        printf("contour %d,size is %d,start %d,end %d,pitch mean is %lf,pitch std is %lf,salience mean is %lf,salience std is %lf,salience sum is %lf\n",i++,temp->size,temp->start,temp->end,temp->pitch_mean,temp->pitch_std,temp->salience_mean,temp->salience_std,temp->salience_sum);
        printcontour(temp);
        printf("\n");
        temp=temp->next;
    }
}

void printcontour(contour *con)
{
    pitch *temp;

    temp=con->head;
    for(int i=0;i<con->size-1;i++)
    {
        printf("%d(%lf)->",temp->binindex,temp->salience);
        temp=temp->next;
    }
    printf("%d(%lf)",temp->binindex,temp->salience);
}


void freecontour(contour *con)
{
    pitch *current;

    //printf("delete contour,start %d,end %d,size %d,salience mean is %f,pitch std is %f\n",con->start,con->end,con->size,con->salience_mean,con->pitch_std);

    //printcontour(con);
    //printf("\n");
    while(con->head!=NULL)
    {
        current=con->head;
        //printf("delete addr is %x,bin index is %d\n",current,current->binindex);
        con->head=con->head->next;
        free(current);
        //printf("delete success, addr is %x\n",current);
    }
    //printf("ok\n");
    free(con);
    //printf("ok\n");
}

int contourslen(contours *cons)
{
    contour *temp;
    int len=0;
    temp=cons->first;

    while(temp!=NULL)
    {
        len+=temp->size;
        temp=temp->next;
    }
    return len;
}

int *sizetoarray(contours *cons)
{
    int *sizes;
    contour *temp;

    sizes=(int *)malloc(sizeof(int)*cons->length);
    temp=cons->first;

    for(int i=0;i<cons->length;i++)
    {
        sizes[i]=temp->size;
        temp=temp->next;
    }

    return sizes;
}

int *acculatedsize(int *sizes,int length)
{
    int *accsize;
    int s=0;
    accsize=(int *)malloc(sizeof(int)*length);

    for(int i=0;i<length;i++)
    {
        accsize[i]=s;
        s+=sizes[i];
    }

    return accsize;
}

pitch *pitchtoarray(contours *cons,int *sizes,int lenall)
{
    pitch *pit;
    contour *temp;
    pitch *temp_pitch;
    int acclen=0;

    pit=(pitch*)malloc(sizeof(pitch)*lenall);

    temp=cons->first;
    for(int i=0;i<cons->length;i++)
    {
        temp_pitch=temp->head;
        for(int j=0;j<temp->size;j++)
        {
            pit[acclen+j].salience=temp_pitch->salience;
            pit[acclen+j].binindex=temp_pitch->binindex;
            temp_pitch=temp_pitch->next;
        }
        acclen+=temp->size;
        temp=temp->next;
    }

    return pit;
}

contour_attribute* contourtoattributearray(contours* cons)
{
    contour_attribute *cona;
    contour *temp;
    cona=(contour_attribute*)malloc(sizeof(contour_attribute)*cons->length);

    temp=cons->first;
    for(int i=0;i<cons->length;i++)
    {
        cona[i].size=temp->size;
        cona[i].start=temp->start;
        cona[i].end=temp->end;

        temp=temp->next;
    }

    return cona;
}


void pitchtocontours(contours *cons,double *pitch_mean,double* pitch_std)
{
    contour *temp;
    int i=0;

    temp=cons->first;

    while(temp!=NULL)
    {

        
        temp=temp->next;
    }
}

void attrtocontours(contours *cons,double *salience_sum,double *salience_mean,double *salience_std,double *pitch_mean,double* pitch_std,double *vibrato)
{
    contour *temp;
    int i=0;

    temp=cons->first;

    while(temp!=NULL)
    {
        temp->salience_sum=salience_sum[i];
        temp->salience_mean=salience_mean[i];
        temp->salience_std=salience_std[i];

        temp->pitch_mean=pitch_mean[i];
        temp->pitch_std=pitch_std[i];   

        temp->vibrato=vibrato[i++];

        temp=temp->next;
    }
}

double contoursvibratomean(contours *cons)
{
    contour *temp;
    double mean=0;

    temp=cons->first;
    while(temp!=NULL)
    {
        mean+=temp->vibrato;
        temp=temp->next;
    }
    
    return mean/cons->length;
}

double contoursvibratostd(contours *cons,double vibratomean)
{
    contour *temp;
    double std=0;

    temp=cons->first;
    while(temp!=NULL)
    {
        std+=(temp->vibrato-vibratomean)*(temp->vibrato-vibratomean);
        temp=temp->next;
    }
    
    return sqrt(std/cons->length);
}

int octave(double p1,double p2)
{
    double result=fabsf(p1-p2);
    if(result>1150&&result<1250)return 1;
    else return 0;
}

int pitchoutlier(double p1,double p2)
{
    double result=fabsf(p1-p2);
    if(result>1250)return 1;
    else return 0;
}
