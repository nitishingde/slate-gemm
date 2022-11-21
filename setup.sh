#!/usr/bin/sh

if [[ -d slate/build ]]; then
    rm -rf slate/build
fi
mkdir slate/build

if [[ ! -d slate/opt ]]; then
    mkdir slate/opt

    echo 'file(
        GLOB scalapack_api_src
        CONFIGURE_DEPENDS  # glob at build time
        scalapack_api/*.cc
    )
    # todo: getri not finished.
    list( FILTER scalapack_api_src EXCLUDE REGEX "getri" )
    message( DEBUG "scalapack_api_src = ${scalapack_api_src}" )

    add_library(
        slate_scalapack_api
        ${scalapack_api_src}
    )
    target_link_libraries( slate_scalapack_api PUBLIC slate )
    target_link_libraries( slate_scalapack_api PUBLIC mkl_scalapack_lp64 )' >> ./slate/CMakeLists.txt
fi

# build and install SLATE
cd slate/build

# FIXME: update the MKLROOT variable @tim
export MKLROOT=/home/sci/nitish/intel/oneapi/mkl/2022.0.2
export CXXFLAGS="-I${MKLROOT}/include -fopenmp"
export LDFLAGS="-L${MKLROOT}/lib/intel64 -Wl,-rpath,${MKLROOT}/lib/intel64 -fopenmp"
export LIBS="-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm"

cmake -DCMAKE_INSTALL_PREFIX=../opt/ -DCMAKE_PREFIX_PATH=../opt/ ..

make -j 8
make install

cp libslate_scalapack_api.so ../opt/lib64/
cp ../scalapack_api/scalapack_slate.hh ../opt/include/
