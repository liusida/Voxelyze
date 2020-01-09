#if !defined(TI_UTILS_H)
#define TI_UTILS_H
#include <stdexcept>

#ifdef __CUDACC__
    #ifndef gpuErrchk
        #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
        inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
        {
            if (code != cudaSuccess)
            {
                //fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

                if (abort) {
                    char buffer[200];
                    snprintf(buffer, sizeof(buffer), "GPUassert error in CUDA kernel: %s %s %d\n", cudaGetErrorString(code), file, line);
                    std::string buffer_string = buffer;
                    throw std::runtime_error(buffer_string);
                    exit(code);
                }
            }
        }
    #endif
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE
#endif

#include "types.h"
#include "const.h"
#include "TI_vector.h"
#include "TI_Vec3D.h"
#include "TI_Quat3D.h"

#endif // TI_UTILS_H
