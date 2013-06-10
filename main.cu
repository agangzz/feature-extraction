#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "melodycontour.h"

#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <dirent.h>

#include <cutil_inline.h>

int isreg(char *filename)
{
    char *temp;

    temp=strrchr(filename,(int)'.');
    if(temp!=NULL) return 1;
    else return 0;
}

int iswav(char *filename)
{
    char *temp;

    temp=strstr(filename,".wav");
    if(temp!=NULL) return 1;
    else return 0;
}

int main(int argc, char **argv)
{
    if(argc!=3)
    {
        fprintf(stderr,"parameter should be:./melodycontour wavfile kind!\n");
        return -1;
    }

    //initialize GPU device
    cudaSetDevice( 1 );
    
    cudaThreadSynchronize();

    if(isreg(argv[1])&&iswav(argv[1]))
    {
        melodycontour(argv[1],atoi(argv[2]));
    }
    else
    {
        DIR *dp;
        struct dirent *entry;
        struct stat statbuf;

        if((dp=opendir(argv[1]))==NULL)
        {
            fprintf(stderr,"cannot open directory: %s\n",argv[1]);
        }

        chdir(argv[1]);

        while((entry=readdir(dp))!=NULL)
        {
            lstat(entry->d_name,&statbuf);
            if(S_ISREG(statbuf.st_mode)&&iswav(entry->d_name))
            {
                printf("calculate %s file\n",entry->d_name);
                melodycontour(entry->d_name,atoi(argv[2]));
                printf("\n\n");
            }
        }

    }
}
