all: melodycontour

#complier
CC = nvcc

OUTPUT_OPTION+= -O3
#include path

#lib path,this variable is first defined in ~/.bashrc file as an environment variable, the same as cflags.
LDFLAGS+= -lcutil -lm -lcufft -lcublas
CFLAGS+= -O3 --ptxas-options=-v
#NVCCFLAGS= -arch sm_13
NVCCFLAGS= -arch sm_13

melodycontour: main.o melodycontour.o characteristic.o ReadWav.o fft.o util.o hann.o statistic.o spectraltransform.o findpeaks.o saliencefunction.o saliencefilter.o generatecontour.o contour.o filtercontour.o 
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

#there is no need to use ldflags, and "-c" must be added to just compile the file.
statistic.o: statistic.cu
ifeq "$(cublas)" "v2"
	$(CC) $(NVCCFLAGS) $(CFLAGS) -DCUBLAS_V2  -o $@ -c $^ 
else
	$(CC) $(NVCCFLAGS) $(CFLAGS)  -o $@ -c $^ 
endif

%.o : %.cu
	$(CC) $(NVCCFLAGS) $(CFLAGS) -o $@ -c $^ 


clean:
	$(RM) *.o *.s *.i melodycontour
