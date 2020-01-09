#if !defined(VX3_H)
#define VX3_H
#include <string>
#include <stdexcept>

#ifdef __CUDA_ARCH__
    #ifndef gpuErrchk
        #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
        inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
        {
            if (code != cudaSuccess)
            {
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

#endif // VX3_H
