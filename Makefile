# On Summit, following modules must be loaded:
# gcc/7.4.0
# essl/6.1.0-2
# openblas/0.3.6-omp
# cuda/10.1.243
# netlib-scalapack/2.0.2

CXX = mpicxx
SLATE_TOP = $(HOME)/builds/slate
CUDA_TOP = $(OLCF_CUDA_ROOT)
SCL_LIB = /autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-7.4.0/netlib-scalapack-2.0.2-7zlsy2iper5zkgvwplozog3xivudlfpe/lib
OPTFLAGS = -O3 -fopenmp 
SNTFLAGS = -std=c++11 -fopenmp -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++11 -g $(OPTFLAGS) -I. -I$(SLATE_TOP)/include
LDFLAGS = -L$(CUDA_TOP)/lib64 -lcublas -lcuda -lcudart -L$(SCL_LIB) -lscalapack -Wl,-rpath=$(SLATE_TOP)/lib64 -L$(SLATE_TOP)/lib64 -lblaspp -llapackpp -lslate

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
