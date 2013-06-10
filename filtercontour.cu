
#include "filtercontour.h"

void deleteusingvoiced(contours *cons)
{
    contour *prev,*cur;

    prev=cons->first;
    cur=prev;

    //printf("threshold is %f\n",threshold);

    while(cur!=NULL)
    {
        if(cur->voiced==0)
        {
            //printf("cur salience mean is %f,pitch std is %f\n",cur->salience_mean,cur->pitch_std);
            prev->next=cur->next;
            //printf("con addr is %x\n",cur);
            if(cur==cons->first)
            {
                cons->first=cur->next;
                freecontour(cur);
                prev=cons->first;
                cur=prev;
            }
            else
            {
                freecontour(cur);
                cur=prev->next;
            }
            cons->length=cons->length-1;
        }
        else
        {
            prev=cur;
            cur=cur->next;
        }
    }
}

void deleteusingpitchstdandlength(contours *cons,double std_threshold,int length_threshold)
{
    contour *prev,*cur;

    prev=cons->first;
    cur=prev;

    //printf("threshold is %f\n",threshold);

    while(cur!=NULL)
    {
        if(cur->pitch_std<std_threshold&&cur->size>length_threshold)
        {
            //printf("cur salience mean is %f,pitch std is %f\n",cur->salience_mean,cur->pitch_std);
            prev->next=cur->next;
            //printf("con addr is %x\n",cur);
            if(cur==cons->first)
            {
                cons->first=cur->next;
                freecontour(cur);
                prev=cons->first;
                cur=prev;
            }
            else
            {
                freecontour(cur);
                cur=prev->next;
            }
            cons->length=cons->length-1;
        }
        else
        {
            prev=cur;
            cur=cur->next;
        }
    }
}

void deleteusingsaliencemean(contours* cons,double threshold)
{
    contour *prev,*cur;

    prev=cons->first;
    cur=prev;

    //printf("threshold is %f\n",threshold);

    while(cur!=NULL)
    {
        if(cur->salience_mean<threshold&&cur->pitch_std<40)
        {
            //printf("cur salience mean is %f,pitch std is %f\n",cur->salience_mean,cur->pitch_std);
            prev->next=cur->next;
            //printf("con addr is %x\n",cur);
            if(cur==cons->first)
            {
                cons->first=cur->next;
                freecontour(cur);
                prev=cons->first;
                cur=prev;
            }
            else
            {
                freecontour(cur);
                cur=prev->next;
            }
            cons->length=cons->length-1;
        }
        else
        {
            prev=cur;
            cur=cur->next;
        }
    }
}

void deleteusingpitchmean(contours *cons,double lthreshold)
{
    contour *prev,*cur;

    prev=cons->first;
    cur=prev;

    //printf("threshold is %f\n",threshold);

    while(cur!=NULL)
    {
        if(cur->pitch_mean<lthreshold)
        {
            //printf("cur salience mean is %f,pitch std is %f\n",cur->salience_mean,cur->pitch_std);
            prev->next=cur->next;
            //printf("con addr is %x\n",cur);
            if(cur==cons->first)
            {
                cons->first=cur->next;
                freecontour(cur);
                prev=cons->first;
                cur=prev;
            }
            else
            {
                freecontour(cur);
                cur=prev->next;
            }
            cons->length=cons->length-1;
        }
        else
        {
            prev=cur;
            cur=cur->next;
        }
    }

}


void deleteusinglength(contours* cons,int length)
{
    contour *prev,*cur;

    prev=cons->first;
    cur=prev;

    //printf("threshold is %f\n",threshold);

    while(cur!=NULL)
    {
        if(cur->size<length)
        {
            //printf("cur salience mean is %f,pitch std is %f\n",cur->salience_mean,cur->pitch_std);
            prev->next=cur->next;
            //printf("con addr is %x\n",cur);
            if(cur==cons->first)
            {
                cons->first=cur->next;
                freecontour(cur);
                prev=cons->first;
                cur=prev;
            }
            else
            {
                freecontour(cur);
                cur=prev->next;
            }
            cons->length=cons->length-1;
        }
        else
        {
            prev=cur;
            cur=cur->next;
        }
    }
}


void deleteusingmelodypitchmean(contours* cons,float *melodypitchmean,int start,double threshold)
{
    contour *prev,*cur;
    int s,e;
    double pitchmean=0;

    prev=cons->first;
    cur=prev;

    //printf("threshold is %f\n",threshold);

    while(cur!=NULL)
    {
        s=cur->start-start;
        e=cur->end-start;

        for(int i=s;i<=e;i++)
        {
            pitchmean+=melodypitchmean[i];
        }
        pitchmean/=(e-s+1);

        if(fabsf(cur->pitch_mean-pitchmean)>threshold)
        {
             prev->next=cur->next;
            //printf("con addr is %x\n",cur);
            if(cur==cons->first)
            {
                cons->first=cur->next;
                freecontour(cur);
                prev=cons->first;
                cur=prev;
            }
            else
            {
                freecontour(cur);
                cur=prev->next;
            }
            cons->length=cons->length-1;

        }
        else
        {
            prev=cur;
            cur=cur->next;
        }

        pitchmean=0;
    }
}


void deleteoctave(contours *cons,float *melodypitchmean,int mstart,int label)
{
    contour *prev,*cur;

    prev=cons->first;
    cur=prev;

    //printf("threshold is %f\n",threshold);

    while(cur!=NULL)
    {
        contour *before=cur;
        contour *p=cur->next;
        int deletecur=0;

        while(p!=NULL)
        {
            if(p->start<=cur->end)
            {
                int start=p->start>cur->start?p->start:cur->start;
                int end=p->end<cur->end?p->end:cur->end;
                int length=end-start+1;
                
                if(length<p->size*0.5&&length<cur->size*0.5)
                {
                    before=p;
                    p=p->next;
                    continue;
                }

                int r=octave(p->pitch_mean,cur->pitch_mean);
                if(r==1)
                {
                    double pitchmean=0,d1,d2;

                    start=start-mstart;
                    end=end-mstart;

                    for(int i=start;i<=end;i++)
                    {
                        pitchmean+=melodypitchmean[i];
                    }
                    pitchmean/=(end-start+1);

                    d1=fabsf(p->pitch_mean-pitchmean);
                    d2=fabsf(cur->pitch_mean-pitchmean);

                    if(d1<d2)
                    {
                        if(!label)
                        {
                            prev->next=cur->next;
                            if(cur==cons->first)
                            {
                                cons->first=cur->next;
                                freecontour(cur);
                                prev=cons->first;
                                cur=prev;
                            }
                            else
                            {
                                freecontour(cur);
                                cur=prev->next;
                            }
                            deletecur=1;
                            cons->length=cons->length-1;

                            break;
                        }
                        else
                        {
                            cur->voiced=0;
                            p->voiced=1;

                             
                            break;
                        }
                    }
                    else
                    {
                        if(!label)
                        {
                            before->next=p->next;
                            freecontour(p);
                            p=before;
                            cons->length=cons->length-1;
                        }
                        else
                        {
                            p->voiced=0;
                            cur->voiced=1;
                            before=before->next;
                            p=before;
                        }
                    }
                }
                else
                {
                    cur->voiced=1;
                    p->voiced=1;
                }
                before=p;
                p=p->next;
            }
            else
            {
                break;
            }
        }
        if(deletecur==0&&cur!=NULL)
        {
            prev=cur;
            cur=cur->next;
        }
    }
}


void deleteoctaveusinglength(contours *cons)
{
    contour *prev,*cur;

    prev=cons->first;
    cur=prev;

    //printf("threshold is %f\n",threshold);

    while(cur!=NULL)
    {
        contour *before=cur;
        contour *p=cur->next;
        int deletecur=0;

        while(p!=NULL)
        {
            if(p->start<=cur->end)
            {
                int start=p->start>cur->start?p->start:cur->start;
                int end=p->end<cur->end?p->end:cur->end;
                int length=end-start+1;
                
                if(length<p->size*0.5&&length<cur->size*0.5)
                {
                    before=p;
                    p=p->next;
                    continue;
                }

                int r=octave(p->pitch_mean,cur->pitch_mean);
                if(r==1)
                {
                    if(cur->size<p->size/3)
                    {
                        prev->next=cur->next;
                        if(cur==cons->first)
                        {
                            cons->first=cur->next;
                            freecontour(cur);
                            prev=cons->first;
                            cur=prev;
                        }
                        else
                        {
                            freecontour(cur);
                            cur=prev->next;
                        }
                        deletecur=1;
                        cons->length=cons->length-1;
                        
                        break;
                    }
                    if(p->size<cur->size/3)
                    {
                        before->next=p->next;
                        freecontour(p);
                        p=before;
                        cons->length=cons->length-1;
                    }
                }
                before=p;
                p=p->next;
            }
            else
            {
                break;
            }
        }
        if(deletecur==0&&cur!=NULL)
        {
            prev=cur;
            cur=cur->next;
        }
    }
}

void deletepitchoutlier(contours* cons,float *melodypitchmean,int start,int label)
{
    contour *prev,*cur;
    int s,e;
    float pitchmean=0;

    prev=cons->first;
    cur=prev;


    while(cur!=NULL)
    {
        s=cur->start-start;
        e=cur->end-start;

        for(int i=s;i<=e;i++)
        {
            pitchmean+=melodypitchmean[i];
        }
        pitchmean/=(e-s+1);

        int r=pitchoutlier(pitchmean,cur->pitch_mean);

        if(r==1)
        {
            if(!label)
            {
                prev->next=cur->next;
                //printf("con addr is %x\n",cur);
                if(cur==cons->first)
                {
                    cons->first=cur->next;
                    freecontour(cur);
                    prev=cons->first;
                    cur=prev;
                }
                else
                {
                    freecontour(cur);
                    cur=prev->next;
                }
                cons->length=cons->length-1;
            }
            else
            {
                cur->voiced=0;
                prev=cur;
                cur=cur->next;
            }

        }
        else
        {
            cur->voiced=1;
            prev=cur;
            cur=cur->next;
        }

        pitchmean=0;
    }
}

contour* deletepitchesatframe(contours *cons,int frame)
{
    contour *con=(contour *)malloc(sizeof(contour));
    contour *cur,*prev;

    con->next=0;
    con->size=0;

    cur=cons->first;
    prev=cur;
    while(cur!=NULL)
    {
        if(cur->start==frame)
        {
            pitch *temp=deletehead(cur);
            temp->salience=cur->salience_sum;
            insertaftertail(con,temp);
            if(cur->size==0)
            {
                prev->next=cur->next;
                if(cur==cons->first)
                {
                    cons->first=cur->next;
                    freecontour(cur);
                    prev=cons->first;
                    cur=prev;
                }
                else
                {
                    freecontour(cur);
                    cur=prev->next;
                }
                cons->length=cons->length-1;

            }
            else
            {
                prev=cur;
                cur=cur->next;
            }
        }
        else
        {
            break;
        }
    }
    
    return con;
}

int* ultimatemelodycontour(contours *cons,int framenumber)
{
    int *ultimatemelodypitch;

    ultimatemelodypitch=(int *)malloc(sizeof(int)*framenumber);

    for(int i=0;i<framenumber;i++)
    {
        contour *con=deletepitchesatframe(cons,i);
        if(con->size!=0)
        {
            pitch *temp;
            double saliencesum=0;
            int pitch=0;
            temp=deletehead(con);
            while(temp!=NULL)
            {
                if(temp->salience>saliencesum)
                {
                    pitch=temp->binindex*10;
                    saliencesum=temp->salience;
                }
                temp=deletehead(con);
            }

            ultimatemelodypitch[i]=pitch;
            freecontour(con);
        }
        else
        {
            ultimatemelodypitch[i]=0;
        }
    }

    return ultimatemelodypitch;
}

int* ultimatemelodycontour(contours *cons,float *melodypitchmean,int start,int framenumber)
{
    int *ultimatemelodypitch;

    ultimatemelodypitch=(int *)malloc(sizeof(int)*framenumber);

    for(int i=0;i<framenumber;i++)
    {
        contour *con=deletepitchesatframe(cons,i);
        if(con->size!=0)
        {
            pitch *temp;
            double max=6000;
            int pitch=0;
            temp=deletehead(con);
            while(temp!=NULL)
            {
                if(fabsf(temp->binindex*10-melodypitchmean[i-start])<max)
                {
                    pitch=temp->binindex*10;
                    max=fabsf(temp->binindex*10-melodypitchmean[i-start]);
                }
                temp=deletehead(con);
            }

            ultimatemelodypitch[i]=pitch;
            freecontour(con);
        }
        else
        {
            ultimatemelodypitch[i]=0;
        }
    }

    return ultimatemelodypitch;
}
