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


#define COLORCODE_RED "\033[0;31m" 
#define COLORCODE_BOLD_RED "\033[1;31m" 
#define COLORCODE_GREEN "\033[0;32m" 
#define COLORCODE_BLUE "\033[0;34m" 
#define COLORCODE_RESET "\033[0m" 

#endif // VX3_H
