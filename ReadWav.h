#ifndef _READ_WAV_
#define _READ_WAV_
struct wavhdr{
    char riff[4];
    unsigned long FileSize;
    char wave[4];
    char fmt[4];
    unsigned long wFormatLength;
    unsigned short wFormatTag;
    unsigned short nChannels;
    unsigned long nSamplesPerSec;
    unsigned long nAvgBytesPerSec;
    unsigned short nBlockAlign;
    unsigned short wBitsPerSample;
    char data[4];
    unsigned long datasize;
};
class CReadWav{
    public:
        CReadWav();
        ~CReadWav();

    public:
        wavhdr pWav;
        unsigned char *databuf1; //wav data for bitspersample 8
        short *databuf2; //wav data for bitspersampler 16
        float *data;//wav normalized data
        int framelen;

    public:
        void readData(char *ad_wav);

};

#endif



