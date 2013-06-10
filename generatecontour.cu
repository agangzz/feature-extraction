#include "generatecontour.h"

#define HALF_PEAK_NUM 32
#define QTR_PEAK_NUM 16

contours* generateContours(float *filteredpeaks,int *filteredindex,int *filteredpeaks_num,float *leftpeaks,int *leftindex,int *leftpeaks_num,int framenumber,int contourgap)
{
    int leftsplus=framenumber;
    int i=0;
    int binthreshold=8;
    contours *cons=(contours*)malloc(sizeof(contours));
    cons->length=0;
    cons->maxcontoursize=0;

    //skip the zero elements in the front end
    while(leftpeaks_num[i]==0)
    {
        i++;
        leftsplus--;
    }

    while(leftsplus!=0)
    {
        if(leftpeaks_num[i]>0)//find the start point of a contour
        {
            contour* con=(contour*)malloc(sizeof(contour));
            con->next=0;//must have this statement, or there will be wrong
            con->voiced=1;
            con->size=0;
            con->start=i;

            pitch* pit=(pitch*)malloc(sizeof(pitch));
            pit->next=0;
            pit->salience=leftpeaks[i*QTR_PEAK_NUM+leftpeaks_num[i]-1];
            pit->binindex=leftindex[i*QTR_PEAK_NUM+leftpeaks_num[i]-1];

            int previndex=leftindex[i*QTR_PEAK_NUM+leftpeaks_num[i]-1];

            leftpeaks_num[i]--;

            insertaftertail(con,pit);//add the pitch to the head of the contour


            int j=i+1;
            int gaplen=0;

            while(j<framenumber&&leftpeaks_num[j]>=0)//search the other pitches in chronological order
            {
                int mind=512;
                int pos;

                //find the bin index which is nearest to the previous index
                for(int k=0;k<leftpeaks_num[j];k++)
                {
                    if(mind>abs(previndex-leftindex[j*QTR_PEAK_NUM+k]))
                    {
                        mind=abs(previndex-leftindex[j*QTR_PEAK_NUM+k]);
                        pos=k;
                    }
                }

                if(mind<binthreshold)//have found the most similar bin index
                {
                    pitch* pit=(pitch*)malloc(sizeof(pitch));
                    pit->next=0;
                    pit->salience=leftpeaks[j*QTR_PEAK_NUM+pos];
                    pit->binindex=leftindex[j*QTR_PEAK_NUM+pos];

                    previndex=leftindex[j*QTR_PEAK_NUM+pos];

                    if(pos!=leftpeaks_num[j]-1)//move the last element to the interval position
                    {
                        leftpeaks[j*QTR_PEAK_NUM+pos]=leftpeaks[j*QTR_PEAK_NUM+leftpeaks_num[j]-1];
                        leftindex[j*QTR_PEAK_NUM+pos]=leftindex[j*QTR_PEAK_NUM+leftpeaks_num[j]-1];

                    }

                    leftpeaks_num[j]--;

                    insertaftertail(con,pit);

                    j++;
                    gaplen=0;

                }
                else//not find the most similar bin index in left parts, so search the filtered parts
                {
                    if(gaplen<contourgap)
                    {
                        int mind=512;
                        int pos;
                        //find the bin index which is nearest to the previous index in filtered parts
                        for(int k=0;k<filteredpeaks_num[j];k++)
                        {
                            if(mind>abs(previndex-filteredindex[j*HALF_PEAK_NUM+k]))//Notice that the filtered parts have length HALF_PEAK_NUM
                            {
                                mind=abs(previndex-filteredindex[j*HALF_PEAK_NUM+k]);
                                pos=k;
                            }
                        }

                        if(mind<binthreshold)//have found the most similar bin index
                        {
                            pitch* pit=(pitch*)malloc(sizeof(pitch));
                            pit->next=0;
                            pit->salience=filteredpeaks[j*HALF_PEAK_NUM+pos];
                            pit->binindex=filteredindex[j*HALF_PEAK_NUM+pos];

                            previndex=filteredindex[j*HALF_PEAK_NUM+pos];

                            if(pos!=filteredpeaks_num[j]-1)
                            {
                                filteredpeaks[j*QTR_PEAK_NUM+pos]=filteredpeaks[j*HALF_PEAK_NUM+filteredpeaks_num[j]-1];
                                filteredindex[j*QTR_PEAK_NUM+pos]=filteredindex[j*HALF_PEAK_NUM+filteredpeaks_num[j]-1];

                            }

                            filteredpeaks_num[j]--;

                            insertaftertail(con,pit);

                            j++;
                            gaplen++;
                        }
                        else
                        {
                            break;
                        }
                    }
                    else//using so many elements in sminus
                    {
                        //printf("using the elements in sminus too much\n");
                        break;
                    }
                }//search the filtered parts
            }//while

            con->end=j-1;

            addcontour(cons,con);//finish one contour, add it to contours.
            if(cons->maxcontoursize<con->size)
            {
                cons->maxcontoursize=con->size;
            }
        }
        else
        {
            i++;
            leftsplus--;
        }
    }

    return cons;
}


