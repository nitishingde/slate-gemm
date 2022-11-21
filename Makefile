# Follow: https://bitbucket.org/icl/slate/wiki/machines/summit.md
CXX = mpicxx
SLATE_TOP = $(PWD)/slate/opt

# FIXME: set path to cuda root dir @tim
CUDA_TOP = /home/sci/nitish/.local/cuda-11.7

# FIXME: set path to intel mkl root dir @tim
SCL_LIB = /home/sci/nitish/intel/oneapi/mkl/2022.0.2/lib/intel64

OPTFLAGS = -O3 -fopenmp 
SNTFLAGS = -std=c++11 -fopenmp -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++11 -g $(OPTFLAGS) -I. -I$(SLATE_TOP)/include -DPIN_MATRICES -I$(CUDA_TOP)/include
LDFLAGS = -L$(CUDA_TOP)/lib64 -lcublas -lcuda -lcudart -lcusolver -L$(SCL_LIB) -lslate_scalapack_api -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64 -Wl,--copy-dt-needed-entries,-rpath=$(SLATE_TOP)/lib64 -L$(SLATE_TOP)/lib64 -lblaspp -llapackpp -lslate

OBJ = test_gemm.o
TARGET = test_gemm

all: $(TARGET)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $^

$(TARGET):  $(OBJ)
	$(CXX) $^ $(CXXFLAGS) -o $@ $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET)
