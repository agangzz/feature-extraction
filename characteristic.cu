
#include "characteristic.h"


void Pitchmeans(contours *cons)
{
    contour *temp;

    temp=cons->first;
    while(temp!=NULL)
    {
        pitchmean(temp);
        temp=temp->next;
    }
}

void Pitchstds(contours *cons)
{
    contour *temp;

    temp=cons->first;
    while(temp!=NULL)
    {
        if(temp->pitch_mean==0)//if we don't cal mean, cal first. used in std
        {
            pitchmean(temp);
        }
        pitchstd(temp);
        temp=temp->next;
    }
}

void Saliencesums(contours *cons)
{
    contour *temp;

    temp=cons->first;
    while(temp!=NULL)
    {
        saliencesum(temp);
        temp=temp->next;
    }

}

void Saliencemeans(contours *cons)
{
    contour *temp;

    temp=cons->first;
    while(temp!=NULL)
    {
        if(temp->salience_sum!=0)
        {
            saliencemean(temp,1);
        }
        else
        {
            saliencemean(temp,0);
        }
        temp=temp->next;
    }

}

void Saliencestds(contours *cons)
{
    contour *temp;

    temp=cons->first;
    while(temp!=NULL)
    {
        if(temp->salience_mean==0)
        {
            saliencemean(temp,1);
        }
        saliencestd(temp);
        
        temp=temp->next;
    }

}

double Contoursaliencemean(contours *cons)
{
    double sum=0;
    contour *temp;

    temp=cons->first;
    while(temp!=NULL)
    {
        sum+=temp->salience_mean;
        temp=temp->next;
    }

    return sum/cons->length;
    
}

double Contoursaliencestd(contours *cons,double mean)
{
    double sum=0;
    contour *temp;

    temp=cons->first;
    while(temp!=NULL)
    {
        sum+=(temp->salience_mean-mean)*(temp->salience_mean-mean);
        temp=temp->next;
    }

    return sqrt(sum/cons->length);

}

double Contourpitchmean(contours *cons)
{
    double sum=0;
    contour *temp;

    temp=cons->first;
    while(temp!=NULL)
    {
        sum+=temp->pitch_mean;
        temp=temp->next;
    }

    return sum/cons->length;
    
}

double Contourpitchstd(contours *cons,double mean)
{
    double sum=0;
    contour *temp;

    temp=cons->first;
    while(temp!=NULL)
    {
        sum+=(temp->pitch_mean-mean)*(temp->pitch_mean-mean);
        temp=temp->next;
    }

    return sqrt(sum/cons->length);

}

void pitchmean(contour *con)
{
    int sum=0;
    pitch *temp;

    temp=con->head;
    while(temp!=NULL)
    {
        sum+=temp->binindex*10;
        temp=temp->next;
    }

    con->pitch_mean=sum*1.0/con->size;
}

void pitchstd(contour *con)
{
    double sum=0;
    pitch *temp;

    temp=con->head;
    while(temp!=NULL)
    {
        sum+=(temp->binindex*10-con->pitch_mean)*(temp->binindex*10-con->pitch_mean);
        temp=temp->next;
    }

    con->pitch_std=sqrt(sum*1.0/con->size);
}

void saliencesum(contour *con)
{
    double sum=0;
    pitch *temp;

    temp=con->head;
    while(temp!=NULL)
    {
        sum+=temp->salience;
        temp=temp->next;
    }

    con->salience_sum=sum;

}

void saliencemean(contour *con,int flag)
{
    if(flag!=0)
    {
        con->salience_mean=con->salience_sum/con->size;
        return;
    }

    double sum=0;
    pitch *temp;

    temp=con->head;
    while(temp!=NULL)
    {
        sum+=temp->salience;
        temp=temp->next;
    }

    con->salience_mean=sum/con->size;

}

void saliencestd(contour *con)
{
    double sum=0;
    pitch *temp;

    temp=con->head;
    while(temp!=NULL)
    {
        sum+=(temp->salience-con->salience_mean)*(temp->salience-con->salience_mean);
        temp=temp->next;
    }

    con->salience_std=sqrt(sum/con->size);

}
