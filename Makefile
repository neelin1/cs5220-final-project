CPP = g++
CFLAGS = -lm
COPTFLAGS = -O3 -ffast-math

MPIFLAGS = -DMPI -lmpi
CUDAFLAGS = -DCUDA

NVCC = nvcc
NVCCFLAGS =

all: mpi gpu dp_serial naive_serial openmp

dp_serial: build/dp_serial
naive_serial: build/naive_serial
grid_openmp: build/dp_openmp
grid_mpi: build/mpi
grid_gpu: build/gpu

build/dp_serial: common/main.cpp serial/dp_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS)

build/naive_serial: common/main.cpp serial/naive_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS)

build/grid_mpi: common/main.cpp multithread/grid_mpi.cpp
	$(CPP) $^ -o $@ $(MPIFLAGS) $(CFLAGS) $(COPTFLAGS)

build/grid_openmp: common/main.cpp multithread/grid_openmp.cpp
	$(CPP) $^ -o $@ -fopenmp $(CFLAGS) $(COPTFLAGS)

build/grid_gpu: common/main.cpp gpu/grid_gpu.cu
	$(NVCC) $^ -o $@ $(CUDAFLAGS)

clean:
	rm -f build/*

.PHONY: all clean
