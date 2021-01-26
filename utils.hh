#ifndef UTILS_HH
#define UTILS_HH

#include <stdint.h>
#include <vector>
#include <array>
#include <initializer_list>

namespace slate {

	enum class Origin {
		Host,
		ScaLAPACK,
		Devices,
	};

	enum class Dist {
		Row,
		Col,
	};
	
        enum class Type {
                INT,
		FLOAT,
		DOUBLE,
                COMPLEX_FLOAT,
                COMPLEX_DOUBLE,
	};
} // namespace slate

int print_reduce_error(
    const std::string& msg, int mpi_rank, MPI_Comm comm)
{
    // reduction to determine first rank with an error
    typedef struct { int err, rank; } err_rank_t;
    int err = ! msg.empty();
    err_rank_t err_rank = { err, mpi_rank };
    err_rank_t err_first = { 0, 0 };
    MPI_Allreduce(&err_rank, &err_first, 1, MPI_2INT, MPI_MAXLOC, comm);

    if (err_first.err) {
        // count ranks with an error
        int root = 0;
        int cnt = 0;
        MPI_Reduce(&err, &cnt, 1, MPI_INT, MPI_SUM, root, comm);

        // first rank with error sends msg to root
        char buf[ 255 ];
        if (mpi_rank == err_first.rank) {
            snprintf(buf, sizeof(buf), "%s", msg.c_str());
            // if rank == root, nothing to send
            if (mpi_rank != root) {
                slate_mpi_call(
                    MPI_Send(buf, sizeof(buf), MPI_CHAR, root, 0, comm));
            }
        }
        else if (mpi_rank == root) {
            MPI_Status status;
            slate_mpi_call(
                MPI_Recv(buf, sizeof(buf), MPI_CHAR, err_first.rank, 0, comm,
                         &status));
        }

        // root prints msg
        if (mpi_rank == root) {
            fprintf(stderr,
                    "\n%s%sError on rank %d: %s. (%d ranks had some error.)%s\n",
                    "", "",
                    err_first.rank, buf, cnt,
                    "");
        }
    }

    return err_first.err;
}

class Params 
{
public:
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double pi  = 3.141592653589793;
    const double e   = 2.718281828459045;
    double alpha     = 1.0;
    double beta      = 0.0;
    int dnb          = 128;
    int lnb          = 16;  
    std::vector<int> dims;
    slate::Origin origin;
    slate::Target target;
    slate::Type type;
    int p, q;


    Params(const std::initializer_list<int>& dim3, int blocksz, std::string origin)
    {
        dims.insert(dims.end(), dim3.begin(), dim3.end());
        dnb = blocksz;
        set_origin(origin);
    }

    ~Params() { dims.clear(); }

    inline void set_origin(std::string orig)
    {
	    std::string origin_ = orig;
	    std::transform(origin_.begin(), origin_.end(), origin_.begin(), ::tolower);
	    if (origin_ == "d" || origin_ == "dev" || origin_ == "device" ||
			    origin_ == "devices")
		    origin = slate::Origin::Devices;
	    else if (origin_ == "h" || origin_ == "host")
		    origin = slate::Origin::Host;
	    else if (origin_ == "s" || origin_ == "scalapack")
		    origin = slate::Origin::ScaLAPACK;
	    else
		    throw slate::Exception("unknown origin");
    }

    inline slate::Op str2trans(std::string targ)
    {
	    std::string target_ = targ;
	    std::transform(target_.begin(), target_.end(), target_.begin(), ::tolower);
	    if (target_ == "t" || target_ == "trans" || target_ == "transpose")
		    return slate::Op::Trans;
	    else if (target_ == "n" || target_ == "notrans" || target_ == "notranspose")
		    return slate::Op::NoTrans;
	    else if (target_ == "c" || target_ == "conjtrans" || target_ == "conjtranspose")
		    return slate::Op::ConjTrans;
	    else
		    throw slate::Exception("unknown transpose string");
    }

    inline slate::Target origin2target(slate::Origin orig)
    {
	    switch (orig) {
		    case slate::Origin::Host:
		    case slate::Origin::ScaLAPACK:
			    return slate::Target::Host;

		    case slate::Origin::Devices:
			    return slate::Target::Devices;

		    default:
			    throw slate::Exception("unknown origin");
	    }
    }

    inline void set_target(std::string targ)
    {
	    std::string target_ = targ;
	    std::transform(target_.begin(), target_.end(), target_.begin(), ::tolower);
	    if (target_ == "t" || target_ == "task")
		    target = slate::Target::HostTask;
	    else if (target_ == "n" || target_ == "nest")
		    target = slate::Target::HostNest;
	    else if (target_ == "b" || target_ == "batch")
		    target = slate::Target::HostBatch;
	    else if (target_ == "d" || target_ == "dev" || target_ == "device" ||
			    target_ == "devices")
		    target = slate::Target::Devices;
	    else if (target_ == "h" || target_ == "host")
		    target = slate::Target::Host;
	    else
		    throw slate::Exception("unknown target");
    }

    inline void set_grid(const std::array<int,2>& coord)
    { p = coord[0]; q = coord[1]; }
};


// -----------------------------------------------------------------------------
// Compare a == b, bitwise. Returns true if a and b are both the same NaN value,
// unlike (a == b) which is false for NaNs.
template <typename T>
inline bool same( T a, T b )
{ return (memcmp( &a, &b, sizeof(T) ) == 0); }

template< typename T >
inline T roundup(T x, T y)
{ return T((x + y - 1) / y)*y; }

double get_wtime()
{
    #ifdef _OPENMP
        return omp_get_wtime();
    #else
        struct timeval tv;
        gettimeofday( &tv, nullptr );
        return tv.tv_sec + tv.tv_usec*1e-6;
    #endif
}
//------------------------------------------------------------------------------
// Similar to ScaLAPACK numroc (number of rows or columns).
inline int64_t localRowsCols(int64_t n, int64_t nb, int iproc, int mpi_size)
{
    int64_t nblocks = n / nb;
    int64_t num = (nblocks / mpi_size) * nb;
    int64_t extra_blocks = nblocks % mpi_size;
    if (iproc < extra_blocks) {
        // extra full blocks
        num += nb;
    }
    else if (iproc == extra_blocks) {
        // last partial block
        num += n % nb;
    }
    return num;
}

//------------------------------------------------------------------------------
// Similar to BLACS gridinfo
// (local row ID and column ID in 2D block cyclic distribution).
inline int64_t whoismyrow(int mpi_rank, int64_t p)
{
    return (mpi_rank % p);
}

inline int64_t whoismycol(int mpi_rank, int64_t p)
{
    return (mpi_rank / p);
}

#endif // UTILS_HH
