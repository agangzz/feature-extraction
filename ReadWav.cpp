#include "ReadWav.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CReadWav::CReadWav()
{

	pWav.FileSize=0;  
	pWav.wFormatLength=0;  
	pWav.nSamplesPerSec=0;      
	pWav.nAvgBytesPerSec=0;        
	pWav.datasize=0;      
	pWav.wFormatTag=0;   
	pWav.nChannels=0;    
	pWav.nBlockAlign=0; 
	pWav.wBitsPerSample=0; 
	for(int i=0;i<4;i++)
	{
		pWav.riff[i]=0;
		pWav.wave[i]=0;
		pWav.fmt[i]=0;
		pWav.data[i]=0;
	}

	databuf1=NULL;  
	databuf2=NULL; 


}
CReadWav::~CReadWav()
{


	delete[] databuf1;
	delete[] databuf2;

}
void CReadWav::readData(char *ad_wav)
{
	FILE *fp = NULL;
	fp=fopen(ad_wav,"rb");

	if(fp==NULL)
    {
		printf("failed to open this wave file!\n");
        return;
    }


	fread(pWav.riff,4,1,fp);
	fread(&(pWav.FileSize),1,4,fp);
	fread(pWav.wave,4,1,fp);
	fread(pWav.fmt,4,1,fp);
	fread(&(pWav.wFormatLength),1,4,fp);
	fread(&(pWav.wFormatTag),1,2,fp);
	fread(&(pWav.nChannels),1,2,fp);
	fread(&(pWav.nSamplesPerSec),1,4,fp);
	fread(&(pWav.nAvgBytesPerSec),1,4,fp);
	fread(&(pWav.nBlockAlign),1,2,fp);
	fread(&(pWav.wBitsPerSample),1,2,fp);
	//The format chunk may have two additional bytes created by software. If so, skip it. It has no use. 
	if (pWav.wFormatLength==18)
	{
		fseek(fp,2,SEEK_CUR);
		printf("skip two bytes\n");
	}
	fread(pWav.data,4,1,fp);
    
    //sometimes, the software may generate "fact" chunk, just ignore it.
	if (!strcmp(pWav.data,"fact"))
	{
		printf("chunk is %s\n",pWav.data);
		fread(&(pWav.datasize),1,4,fp);
		fseek(fp,pWav.datasize,SEEK_CUR);
		fread(pWav.data,4,1,fp);
	}
	fread(&(pWav.datasize),1,4,fp);
	if (pWav.nChannels==1)//mono wav
	{
		if(pWav.wBitsPerSample==16) 
		{
            //short is 2 bytes, so there is datasize/2 data.
			databuf2=(short*)malloc(pWav.datasize/2*sizeof(short));  
			fread(databuf2,pWav.datasize,1,fp);
		}
		else if(pWav.wBitsPerSample==8)
		{
			databuf1=(unsigned char*)malloc(pWav.datasize*sizeof(unsigned char));   
			fread(databuf1,pWav.datasize,1,fp);
		}
	}
	else//stereo wav
	{
		if(pWav.wBitsPerSample==16) 
		{
			short * temp=(short*)malloc(pWav.datasize/2*sizeof(short));   
			databuf2=(short *)malloc(pWav.datasize/4*sizeof(short));
			fread(temp,pWav.datasize,1,fp);
			for (int i=0;i<pWav.datasize/4;i++)
			{
				databuf2[i]=(temp[2*i]+temp[2*i+1])/2;
			}

			free(temp);
		}
		else if(pWav.wBitsPerSample==8)
		{
			unsigned char * temp=(unsigned char*)malloc(pWav.datasize*sizeof(unsigned char));  
			databuf1=(unsigned char*)malloc(pWav.datasize/2*sizeof(unsigned char));
			fread(temp,pWav.datasize,1,fp);
			for (int i=0;i<pWav.datasize/4;i++)
			{
				databuf1[i]=temp[2*i]/2+temp[2*i+1]/2;
			}
			free(temp);
		}
	}
    framelen=pWav.datasize/pWav.nChannels/(pWav.wBitsPerSample/8);

    data=(float *)malloc(sizeof(float)*framelen);
    if (pWav.wBitsPerSample==8)
    {
        for(int i=0;i<framelen;i++)
        {
                data[i]=(databuf1[i]*1.0-128)/128;
        }
    }
    else
    {
        for(int i=0;i<framelen;i++)
        {
            data[i]=databuf2[i]*1.0/32768;
        }

    }

    /*
    printf("wav data from 0 to 16 is: \n");
    for (int i=0;i<16;i++)
    {
        printf("%f\n",data[i]);
    }
    */

	fclose(fp);
}

