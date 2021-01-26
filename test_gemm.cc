// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "blas/flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include <unistd.h>
#include "utils.hh"

#undef PIN_MATRICES

static int dim_n, dim_m, dim_k;
static std::string origin = "d";
static int dnb = 128;
static int lnb = 16;
static std::string A_trans = "n";
static std::string B_trans = "n";
static bool is_int_t = false;
static bool is_float_t = true;
static bool is_double_t = false;
static bool is_complex_float_t = false;
static bool is_complex_double_t = false;
static double alpha = 1.0, beta = 0.0;

int parse_args(int argc, char* argv[]);

//------------------------------------------------------------------------------
// eventually accept all params: https://icl.bitbucket.io/slate/group__enum.html#gac97a2c5045464e6949b9a65a059b196a
template<typename scalar_t>
double test_gemm_work(Params& params)
{
    using real_t = blas::real_type<scalar_t>;
    using slate::Norm;

    // get & mark input values
    slate::Op transA = params.str2trans(A_trans);
    slate::Op transB = params.str2trans(B_trans);
    scalar_t alpha = params.alpha;
    scalar_t beta = params.beta;
    int64_t m = params.dims[0];
    int64_t n = params.dims[1];
    int64_t k = params.dims[2];
    int64_t nb = params.dnb;
    int64_t p = params.p;
    int64_t q = params.q;
    // TODO option to customize these parameters
    int64_t lookahead = 0;
    bool ref_only = false;
    slate::Norm norm = slate::Norm::One;
    bool check = false;
    bool ref = false;
    // one to one correspondence between origin and target, 
    // doesn't have to be, later have separate option for target
    slate::Origin origin = params.origin;
    slate::Target target = params.origin2target(origin);

    // Error analysis applies in these norms.
    slate_assert(norm == Norm::One || norm == Norm::Inf || norm == Norm::Fro);

    // sizes of A and B
    int64_t Am = (transA == slate::Op::NoTrans ? m : k);
    int64_t An = (transA == slate::Op::NoTrans ? k : m);
    int64_t Bm = (transB == slate::Op::NoTrans ? k : n);
    int64_t Bn = (transB == slate::Op::NoTrans ? n : k);
    int64_t Cm = m;
    int64_t Cn = n;

    // local values
    const int izero = 0, ione = 1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descB_tst[9], descC_tst[9], descC_ref[9];
    int iam = 0, nprocs = 1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(Am, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(An, nb, mycol, izero, npcol);
    scalapack_descinit(descA_tst, Am, An, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplrnt(&A_tst[0], Am, An, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc(Bm, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(Bn, nb, mycol, izero, npcol);
    scalapack_descinit(descB_tst, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
    slate_assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst(lldB*nlocB);
    scalapack_pplrnt(&B_tst[0], Bm, Bn, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 2);

    // matrix C, figure out local size, allocate, create descriptor, initialize
    int64_t mlocC = scalapack_numroc(m, nb, myrow, izero, nprow);
    int64_t nlocC = scalapack_numroc(n, nb, mycol, izero, npcol);
    scalapack_descinit(descC_tst, Cm, Cn, nb, nb, izero, izero, ictxt, mlocC, &info);
    slate_assert(info == 0);
    int64_t lldC = (int64_t)descC_tst[8];
    std::vector<scalar_t> C_tst(lldC*nlocC);
    scalapack_pplrnt(&C_tst[0], m, n, nb, nb, myrow, mycol, nprow, npcol, mlocC, iseed + 3);

    #ifdef PIN_MATRICES
    int cuerror;
    cuerror = cudaHostRegister(&A_tst[0], (size_t)size_A*sizeof(scalar_t), cudaHostRegisterDefault);
    cuerror = cudaHostRegister(&B_tst[0], (size_t)size_A*sizeof(scalar_t), cudaHostRegisterDefault);
    cuerror = cudaHostRegister(&C_tst[0], (size_t)size_A*sizeof(scalar_t), cudaHostRegisterDefault);
    #endif

    // if reference run is required, copy test data and create a descriptor for it
    std::vector<scalar_t> C_ref;
    slate::Matrix<scalar_t> C_ref_slate;
    if (check || ref) {
        C_ref = C_tst;
        scalapack_descinit(descC_ref, Cm, Cn, nb, nb, izero, izero, ictxt, mlocC, &info);
        slate_assert(info == 0);
        C_ref_slate = slate::Matrix<scalar_t>::fromScaLAPACK( m,  n, &C_ref[0], lldC, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    slate::Matrix<scalar_t> A, B, C;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = params.origin2target(origin);
        A = slate::Matrix<scalar_t>(Am, An, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);

        B = slate::Matrix<scalar_t>(Bm, Bn, nb, nprow, npcol, MPI_COMM_WORLD);
        B.insertLocalTiles(origin_target);
        copy(&B_tst[0], descB_tst, B);

        C = slate::Matrix<scalar_t>(Cm, Cn, nb, nprow, npcol, MPI_COMM_WORLD);
        C.insertLocalTiles(origin_target);
        copy(&C_tst[0], descC_tst, C);
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A = slate::Matrix<scalar_t>::fromScaLAPACK(Am, An, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
        B = slate::Matrix<scalar_t>::fromScaLAPACK(Bm, Bn, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
        C = slate::Matrix<scalar_t>::fromScaLAPACK( m,  n, &C_tst[0], lldC, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    if (transA == slate::Op::Trans)
        A = transpose(A);
    else if (transA == slate::Op::ConjTrans)
        A = conjTranspose(A);

    if (transB == slate::Op::Trans)
        B = transpose(B);
    else if (transB == slate::Op::ConjTrans)
        B = conjTranspose(B);

    slate_assert(A.mt() == C.mt());
    slate_assert(B.nt() == C.nt());
    slate_assert(A.nt() == B.mt());

    // if reference run is required, record norms to be used in the check/ref
    real_t A_norm=0, B_norm=0, C_orig_norm=0;
    if (check || ref) {
        A_norm = slate::norm(norm, A);
        B_norm = slate::norm(norm,B);
        C_orig_norm = slate::norm(norm, C_ref_slate);
    }

    // compute and save timing/performance
    double gflop = blas::Gflop<scalar_t>::gemm(m, n, k);
    double gflops;

    if (! ref_only) {

	    MPI_Barrier(MPI_COMM_WORLD);
	    double time = get_wtime();

	    //==================================================
	    // Run SLATE test.
	    // C = alpha A B + beta C.
	    //==================================================
	    slate::multiply(
			    alpha, A, B, beta, C, {
			    {slate::Option::Lookahead, 0},
			    {slate::Option::Target, target}
			    });

	    //---------------------
	    // Using traditional BLAS/LAPACK name
	    // slate::gemm(
	    //     alpha, A, B, beta, C, {
	    //         {slate::Option::Lookahead, lookahead},
	    //         {slate::Option::Target, target}
	    //     });

	    MPI_Barrier(MPI_COMM_WORLD);

	    // compute and save timing/performance
	    double time_tst = get_wtime() - time;
	    gflops = gflop / time_tst;
    }

    

    #ifdef PIN_MATRICES
    cuerror = cudaHostUnregister(&A_tst[0]);
    cuerror = cudaHostUnregister(&B_tst[0]);
    cuerror = cudaHostUnregister(&C_tst[0]);
    #endif

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering

   return gflops;
}

// ----------------------------------------------------------------------------- //
int main(int argc, char* argv[])
{
    int mpi_rank = 0, mpi_size = 0, provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int status = 0;
    std::string msg;
    try {
        if (provided < MPI_THREAD_MULTIPLE)
            throw std::runtime_error("SLATE requires MPI_THREAD_MULTIPLE");

        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        bool print = (mpi_rank == 0);

        parse_args(argc, argv);

        // print input so running `test [input] > out.txt` documents input
        if (print) {
            char buf[100];
            std::string args = buf;

            // Input line.
            args += "input:";
            for (int i = 0; i < argc; ++i) {
                args += ' ';
                args += argv[i];
            }
            args += "\n";

            // Date and time, MPI, OpenMP, CUDA specs.
            std::time_t now = std::time(nullptr);
            char nowstr[100];
            std::strftime(nowstr, sizeof(nowstr), "%F %T", std::localtime(&now));
            args += nowstr;
            args += ", MPI size " + std::to_string(mpi_size);
            args += ", OpenMP threads " + std::to_string(omp_get_max_threads());
            int num_devices = 0;
            cudaGetDeviceCount(&num_devices);
            if (num_devices > 0)
                args += ", CUDA devices available " + std::to_string(num_devices);
            args += "\n";

            printf("%s", args.c_str());
        }

        // save passed params
        Params params({dim_m, dim_n, dim_k}, nb, origin);

        // Make default p x q grid as square as possible.
        // Worst case is p=1, q=mpi_size.
        int p = 1, q = 1;
        for (p = int(sqrt(mpi_size)); p > 0; --p) {
            q = int(mpi_size / p);
            if (p*q == mpi_size)
                break;
        }
        std::array<int, 2> pts = {p, q}; 
        params.set_grid(pts);
        assert((params.p * params.q) == mpi_size);
        params.alpha = alpha;
        params.beta = beta;
        double gflops;
 
        if (is_float_t) {
            params.type = slate::Type::FLOAT;
            gflops = test_gemm_work<float>(params);
        }
        else if (is_double_t) {
            params.type = slate::Type::DOUBLE;
            gflops = test_gemm_work<double>(params);
        }
        else if (is_complex_float_t) {
            params.type = slate::Type::COMPLEX_FLOAT;
            gflops = test_gemm_work<std::complex<float>>(params);
        }
        else if (is_complex_double_t) {
            params.type = slate::Type::COMPLEX_DOUBLE;
            gflops = test_gemm_work<std::complex<double>>(params);
        }
        else { 
            params.type = slate::Type::DOUBLE;
            gflops = test_gemm_work<double>(params);
	}

	double pe_gflops = 0.0;
	MPI_Reduce(&gflops, &pe_gflops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (print)
		std::cout << "GFlops: " << pe_gflops << std::endl;
    }
    catch (const std::exception& ex) {
        msg = ex.what();
    }
    int err = print_reduce_error(msg, mpi_rank, MPI_COMM_WORLD);
    if (err)
        status = -1;

    MPI_Finalize();

    if (mpi_rank == 0)
        return status;
    else
        return 0;
}

int parse_args(int argc, char* argv[])
{
  int ret;

  while ((ret = getopt(argc, argv, "m:n:k:a:b:l:r:uvo:isdcz")) != -1) 
  {
    switch (ret) {
    case 'm':
      dim_m = atol(optarg);
      break;
    case 'n':
      dim_n = atol(optarg);
      break;
    case 'k':
      dim_k = atol(optarg);
      break;
    case 'a':
      alpha = atof(optarg);
      break;
    case 'b':
      beta = atof(optarg);
      break;
    case 'l':
      lnb = atoi(optarg);
      break;
    case 'r':
      dnb = atoi(optarg);
      break;
    case 'u':
      A_trans.assign(optarg); 
      break;
    case 'v':
      B_trans.assign(optarg); 
      break;
    case 'o':
      origin.assign(optarg);
      break;
    case 'i':
      is_int_t = true;
    case 's':
      is_float_t = true;
      break;
    case 'd':
      is_double_t = true;
      break;
    case 'c':
      is_complex_float_t = true;
      break;
    case 'z':
      is_complex_double_t = true;
      break;
    default:
      assert(0 && "Should not reach here!!");
      break;
    }
  }
}
