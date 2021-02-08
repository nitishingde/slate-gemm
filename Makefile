# Follow: https://bitbucket.org/icl/slate/wiki/machines/summit.md
CXX = mpicxx
SLATE_TOP = $(HOME)/builds/slate
CUDA_TOP = $(OLCF_CUDA_ROOT)
ESSL_TOP = $(OLCF_ESSL_ROOT)
SCL_LIB = /autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-8.1.1/netlib-scalapack-2.0.2-7x3lv7z2lzfbe5kfwlt2aajkx4hvmgdm/lib
OPTFLAGS = -O3 -fopenmp 
SNTFLAGS = -std=c++11 -fopenmp -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++11 -g $(OPTFLAGS) -I. -I$(SLATE_TOP)/include -DPIN_MATRICES
LDFLAGS = -L$(CUDA_TOP)/lib64 -lcublas -lcuda -lcudart -L$(SCL_LIB) -lscalapack -L$(ESSL_TOP)/lib64 -lessl -Wl,-rpath=$(SLATE_TOP)/lib64 -L$(SLATE_TOP)/lib64 -lblaspp -llapackpp -lslate

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
