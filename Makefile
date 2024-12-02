CPP = g++
CFLAGS = -lm
COPTFLAGS = -O3 -ffast-math

MPIFLAGS = -DMPI -lmpi
CUDAFLAGS = -DCUDA

NVCC = nvcc
NVCCFLAGS =

all: mpi gpu dp_serial naive_serial openmp

mpi: build/mpi
gpu: build/gpu
dp_serial: build/dp_serial
naive_serial: build/naive_serial
openmp: build/openmp

build/gpu: common/main.cpp gpu/gpu.cu
	$(NVCC) $^ -o $@ $(CUDAFLAGS)

build/dp_serial: common/main.cpp serial/dp_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS)

build/naive_serial: common/main.cpp serial/naive_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS)

build/mpi: common/main.cpp multithread/mpi.cpp
	$(CPP) $^ -o $@ $(MPIFLAGS) $(CFLAGS) $(COPTFLAGS)

build/openmp: common/main.cpp multithread/openmp.cpp
	$(CPP) $^ -o $@ -fopenmp $(CFLAGS) $(COPTFLAGS)

clean:
	rm -f build/*

.PHONY: all clean
