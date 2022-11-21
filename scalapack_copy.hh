// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_SCALAPACK_COPY_HH
#define SLATE_SCALAPACK_COPY_HH

#include "slate/Matrix.hh"
#include "slate/BaseTrapezoidMatrix.hh"
// #include "slate/internal/cublas.hh"

#include "lapack.hh"

#include "scalapack_wrappers.hh"

namespace slate {

//------------------------------------------------------------------------------
// const char* getCublasErrorName(cublasStatus_t status);

//------------------------------------------------------------------------------
/// Exception class for slate_cublas_call().
class CublasException : public Exception {
public:
    CublasException(const char* call,
                    cublasStatus_t code,
                    const char* func,
                    const char* file,
                    int line)
        : Exception()
    {
        // const char* name = getCublasErrorName(code);

        // what(std::string("SLATE CUBLAS ERROR: ")
        //      + call + " failed: " + name
        //      + " (" + std::to_string(code) + ")",
        //      func, file, line);
    }
};

/// Throws a CublasException if the CUBLAS call fails.
/// Example:
///
///     try {
///         slate_cublas_call( cublasCreate( &handle ) );
///     }
///     catch (CublasException& e) {
///         ...
///     }
///

#define slate_cublas_call(call) \
    do { \
        cublasStatus_t slate_cublas_call_ = call; \
        if (slate_cublas_call_ != CUBLAS_STATUS_SUCCESS) \
            throw slate::CublasException( \
                #call, slate_cublas_call_, __func__, __FILE__, __LINE__); \
    } while(0)

//------------------------------------------------------------------------------
/// Exception class for slate_cuda_call().
class CudaException : public Exception {
public:
    CudaException(const char* call,
                  cudaError_t code,
                  const char* func,
                  const char* file,
                  int line)
        : Exception()
    {
        const char* name = cudaGetErrorName(code);
        const char* string = cudaGetErrorString(code);

        what(std::string("SLATE CUDA ERROR: ")
             + call + " failed: " + string
             + " (" + name + "=" + std::to_string(code) + ")",
             func, file, line);
    }
};

/// Throws a CudaException if the CUDA call fails.
/// Example:
///
///     try {
///         slate_cuda_call( cudaSetDevice( device ) );
///     }
///     catch (CudaException& e) {
///         ...
///     }
///
#define slate_cuda_call(call) \
    do { \
        cudaError_t slate_cuda_call_ = call; \
        if (slate_cuda_call_ != cudaSuccess) \
            throw slate::CudaException( \
                #call, slate_cuda_call_, __func__, __FILE__, __LINE__); \
    } while(0)

} // namespace slate

//------------------------------------------------------------------------------
/// Indices for ScaLAPACK descriptor
/// 0:  dtype:   1 for dense
/// 1:  context: BLACS context handle
/// 2:  m:       global number of rows
/// 3:  n:       global number of cols
/// 4:  mb:      row blocking factor
/// 5:  nb:      col blocking factor
/// 6:  rowsrc:  process row over which the first row of array is distributed
/// 7:  colsrc:  process col over which the first col of array is distributed
/// 8:  ld:      local leading dimension

enum Descriptor {
    dtype   = 0,
    context = 1,
    m       = 2,
    n       = 3,
    mb      = 4,
    nb      = 5,
    rowsrc  = 6,
    colsrc  = 7,
    ld      = 8
};

//------------------------------------------------------------------------------
/// Copy tile (i, j) from SLATE matrix A to ScaLAPACK matrix B.
///
template <typename scalar_t>
void copyTile(
    slate::BaseMatrix<scalar_t>& A,
    scalar_t* B, lapack_int descB[9],
    int64_t i, int64_t j,
    int p, int q)
{
    int64_t mb  = descB[ Descriptor::mb ];
    int64_t nb  = descB[ Descriptor::nb ];
    int64_t ldb = descB[ Descriptor::ld ];

    int64_t ii_local = int64_t( i / p )*mb;
    int64_t jj_local = int64_t( j / q )*nb;
    if (A.tileIsLocal(i, j)) {
        int dev = A.tileDevice(i, j);
        if (A.tileExists(i, j) &&
            A.tileState(i, j) != slate::MOSI::Invalid)
        {
            // Copy from host tile, if it exists, to ScaLAPACK.
            auto Aij = A(i, j);
            lapack::lacpy(
                lapack::MatrixType::General,
                Aij.mb(), Aij.nb(),
                Aij.data(), Aij.stride(),
                &B[ ii_local + jj_local*ldb ], ldb );
        }
        else if (A.tileExists(i, j, dev) &&
                 A.tileState(i, j, dev) != slate::MOSI::Invalid)
        {
            // Copy from device tile, if it exists, to ScaLAPACK.
            auto Aij = A(i, j, dev);
            slate_cuda_call(
                cudaSetDevice(dev));
            slate_cublas_call(
                cublasGetMatrix(
                    Aij.mb(), Aij.nb(), sizeof(scalar_t),
                    Aij.data(), Aij.stride(),
                    &B[ ii_local + jj_local*ldb ], ldb ));
        }
        else {
            // todo: what to throw?
            throw std::runtime_error("missing tile");
        }
    }
}

//------------------------------------------------------------------------------
/// Copy tile (i, j) from ScaLAPACK matrix B to SLATE matrix A.
///
template <typename scalar_t>
void copyTile(
    scalar_t const* B, lapack_int descB[9],
    slate::BaseMatrix<scalar_t>& A,
    int64_t i, int64_t j,
    int p, int q)
{
    int64_t mb  = descB[ Descriptor::mb ];
    int64_t nb  = descB[ Descriptor::nb ];
    int64_t ldb = descB[ Descriptor::ld ];

    int64_t ii_local = int64_t( i / p )*mb;
    int64_t jj_local = int64_t( j / q )*nb;
    if (A.tileIsLocal(i, j)) {
        int dev = A.tileDevice(i, j);
        if (A.tileExists(i, j) &&
            A.tileState(i, j) != slate::MOSI::Invalid)
        {
            // Copy from ScaLAPACK to host tile, if it exists.
            A.tileGetForWriting(i, j, slate::LayoutConvert::ColMajor);
            auto Aij = A(i, j);
            lapack::lacpy(
                lapack::MatrixType::General,
                Aij.mb(), Aij.nb(),
                &B[ ii_local + jj_local*ldb ], ldb,
                Aij.data(), Aij.stride() );
        }
        else if (A.tileExists(i, j, dev) &&
                 A.tileState(i, j, dev) != slate::MOSI::Invalid)
        {
            // Copy from ScaLAPACK to device tile, if it exists.
            A.tileGetForWriting(i, j, dev, slate::LayoutConvert::ColMajor);
            auto Aij = A(i, j, dev);
            slate_cuda_call(
                cudaSetDevice(dev));
            slate_cublas_call(
                cublasSetMatrix(
                    Aij.mb(), Aij.nb(), sizeof(scalar_t),
                    &B[ ii_local + jj_local*ldb ], ldb,
                    Aij.data(), Aij.stride() ));
        }
        else {
            // todo: what to throw?
            throw std::runtime_error("missing tile");
        }
    }
}

//------------------------------------------------------------------------------
/// Copies the ScaLAPACK-style matrix B to SLATE general matrix A.
/// Assumes both matrices have the same 2D block cyclic distribution.
///
/// @param[in] B
///     Local ScaLAPACK matrix.
///
/// @param[in] descB
///     Descriptor for ScaLAPACK matrix B.
///
/// @param[in,out] A
///     Matrix to copy data to.
///     The tiles, on either host or device, must already exist.
///
template <typename scalar_t>
void copy(
    scalar_t* B, lapack_int descB[9],
    slate::Matrix<scalar_t>& A )
{
    // todo: verify A and B have same distribution.
    int p, q, myrow, mycol;
    Cblacs_gridinfo( descB[ Descriptor::context ], &p, &q, &myrow, &mycol );

    // Code assumes A is not transposed.
    if (A.op() != slate::Op::NoTrans)
        throw std::exception();

    #pragma omp parallel for
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            copyTile( B, descB, A, i, j, p, q );
        }
    }
}

//------------------------------------------------------------------------------
/// Copies SLATE general matrix A to ScaLAPACK-style matrix B.
/// Assumes both matrices have the same 2D block cyclic distribution.
///
/// @param[in] B
///     Local ScaLAPACK matrix.
///
/// @param[in] descB
///     Descriptor for ScaLAPACK matrix B.
///
/// @param[in,out] A
///     Matrix to copy data to.
///     The tiles, on either host or device, must already exist.
///
template <typename scalar_t>
void copy(
    slate::Matrix<scalar_t>& A,
    scalar_t* B, lapack_int descB[9] )
{
    // todo: verify A and B have same distribution.
    int p, q, myrow, mycol;
    Cblacs_gridinfo( descB[ Descriptor::context ], &p, &q, &myrow, &mycol );

    // Code assumes A is not transposed.
    if (A.op() != slate::Op::NoTrans)
        throw std::exception();

    #pragma omp parallel for
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            copyTile( A, B, descB, i, j, p, q );
        }
    }
}

//------------------------------------------------------------------------------
/// Copies the ScaLAPACK-style matrix B to SLATE trapezoid-storage matrix A.
/// Handles Trapezoid, Triangular, Symmetric, and Hermitian matrices.
/// Assumes both matrices have the same 2D block cyclic distribution.
///
/// @param[in] B
///     Local ScaLAPACK matrix.
///
/// @param[in] descB
///     Descriptor for ScaLAPACK matrix B.
///
/// @param[in,out] A
///     Matrix to copy data to.
///     The tiles, on either host or device, must already exist.
///
template <typename scalar_t>
void copy(
    scalar_t* B, lapack_int descB[9],
    slate::BaseTrapezoidMatrix<scalar_t>& A )
{
    // todo: verify A and B have same distribution.
    int p, q, myrow, mycol;
    Cblacs_gridinfo( descB[ Descriptor::context ], &p, &q, &myrow, &mycol );

    // Code assumes A is not transposed.
    if (A.op() != slate::Op::NoTrans)
        throw std::exception();

    bool lower = A.uplo() == slate::Uplo::Lower;
    #pragma omp parallel for
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t ibegin = (lower ? j : 0);
        int64_t iend   = (lower ? A.mt() : blas::min(j+1, A.mt()));
        for (int64_t i = ibegin; i < iend; ++i) {
            copyTile( B, descB, A, i, j, p, q );
        }
    }
}

//------------------------------------------------------------------------------
/// Copies SLATE trapezoid-storage matrix A to ScaLAPACK-style matrix B.
/// Handles Trapezoid, Triangular, Symmetric, and Hermitian matrices.
/// Assumes both matrices have the same 2D block cyclic distribution.
///
/// @param[in] B
///     Local ScaLAPACK matrix.
///
/// @param[in] descB
///     Descriptor for ScaLAPACK matrix B.
///
/// @param[in,out] A
///     Matrix to copy data to.
///     The tiles, on either host or device, must already exist.
///
template <typename scalar_t>
void copy(
    slate::BaseTrapezoidMatrix<scalar_t>& A,
    scalar_t* B, lapack_int descB[9] )
{
    // todo: verify A and B have same distribution.
    int p, q, myrow, mycol;
    Cblacs_gridinfo( descB[ Descriptor::context ], &p, &q, &myrow, &mycol );

    // Code assumes A is not transposed.
    if (A.op() != slate::Op::NoTrans)
        throw std::exception();

    bool lower = A.uplo() == slate::Uplo::Lower;
    #pragma omp parallel for
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t ibegin = (lower ? j : 0);
        int64_t iend   = (lower ? A.mt() : blas::min(j+1, A.mt()));
        for (int64_t i = ibegin; i < iend; ++i) {
            copyTile( A, B, descB, i, j, p, q );
        }
    }
}

#endif // SLATE_SCALAPACK_COPY_HH
