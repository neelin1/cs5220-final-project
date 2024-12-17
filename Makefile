CPP = g++
CFLAGS = -lm
COPTFLAGS = -O3 -ffast-math

MPIFLAGS = -DMPI -lmpi
CUDAFLAGS = -DCUDA

NVCC = nvcc
NVCCFLAGS =

all: dp_serial naive_serial wavefront_openmp wavefront_serial grid_mpi	grid_gpu

dp_serial: build/dp_serial
naive_serial: build/naive_serial
wavefront_serial: build/wavefront_serial
wavefront_openmp: build/wavefront_openmp
wavefrontblocking_openmp: build/wavefrontblocking_openmp
grid_openmp: build/grid_openmp
grid_mpi: build/mpi
grid_gpu: build/grid_gpu

build/dp_serial: common/main.cpp serial/dp_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS)

build/naive_serial: common/main.cpp serial/naive_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS)

build/wavefront_serial: common/main.cpp serial/wavefront_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS)

build/grid_mpi: common/main.cpp multithread/grid_mpi.cpp
	$(CPP) $^ -o $@ $(MPIFLAGS) $(CFLAGS) $(COPTFLAGS)

build/wavefront_openmp: common/main.cpp multithread/wavefront_openmp.cpp
	$(CPP) $^ -o $@ -fopenmp $(CFLAGS) $(COPTFLAGS)

build/wavefrontblocking_openmp: common/main.cpp multithread/wavefrontblocking_openmp.cpp
	$(CPP) $^ -o $@ -fopenmp $(CFLAGS) $(COPTFLAGS)

build/grid_openmp: common/main.cpp multithread/grid_openmp.cpp
	$(CPP) $^ -o $@ -fopenmp $(CFLAGS) $(COPTFLAGS)

build/grid_gpu: common/main.cpp gpu/grid_gpu.cu
	$(NVCC) $^ -o $@ $(CUDAFLAGS)

clean:
	rm -f build/*

.PHONY: all clean
