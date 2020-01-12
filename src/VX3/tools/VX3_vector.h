//TODO: Unit test before use

//Instead of reallocate every time, use over-allocation to reduce memory copy.

#if !defined(TI_VECTOR_H)
#define TI_VECTOR_H
#include <vector>

template <typename T>
class VX3_hdVector {
/* It should be initialized in host, and pass to kernel directly, and it should be freed after kernel returns. */
public:
    VX3_hdVector<T>(const std::vector<T>& p, cudaStream_t stream) {
        num_main = p.size();
        VcudaMalloc( &main , num_main*sizeof(T) );
        T* temp = (T *)malloc(num_main*sizeof(T));
        // VcudaHostAlloc((void **)&temp, num_main*sizeof(T), cudaHostAllocWriteCombined);
        for (unsigned i=0;i<num_main;i++) {
            temp[i] = p[i];
        }
        VcudaMemcpyAsync(main, temp, num_main*sizeof(T), VcudaMemcpyHostToDevice, stream);
        // cudaFreeHost(temp);
        delete temp;
    }
    void free() {
        // cannot use ~ method, becuase passing parameter to kernel triggers ~ method.
        VcudaFree(main);
    }
    __device__ inline T &operator[] (unsigned index) {return main[index];}
    __device__ inline T get (unsigned index) {return main[index];}
    __device__ inline unsigned size() {return num_main;}
    unsigned sizeof_chunk;
    unsigned num_main;
    T* main;
};

#define DEFAULT_CHUNK_SIZE 32
template <typename T>
class VX3_dVector {
/* It should be initialized in dev, and do push_back() and get(), and it should be freed after use. 
   if size<Default, use GPU local memory memory[Default], otherwise use malloc and main and store things in GPU global memory. (malloc is slow in GPU)*/
public:
    __device__ VX3_dVector<T>() {
        default_chunk_size = DEFAULT_CHUNK_SIZE;
        init();
    }
    __device__ ~VX3_dVector<T>() {
        if (!defaultMemory())
            delete main;
    }
    __device__ void inline init() {
        sizeof_chunk = default_chunk_size;
        num_main = 0;
    }
    __device__ void inline push_back(T t) {
        if (defaultMemory()) {
            if (sizeof_chunk == default_chunk_size) { //use memory
                default_memory[num_main] = t;
            } else { //use main
                main[num_main] = t;
            }
        } else { //need allocation
            T* new_main;
            new_main = (T*)malloc(sizeof_chunk * 2 * sizeof(T));
            memcpy(new_main, main, sizeof_chunk * sizeof(T));
            delete main;
            main = new_main;
            main[num_main] = t;
            sizeof_chunk *= 2;
        }
        num_main++;
    }
    __device__ inline bool defaultMemory() { return num_main<sizeof_chunk; }
    __device__ inline unsigned size() {return num_main;};
    __device__ inline T &operator[] (unsigned index) {return get(index);};
    __device__ inline T get (unsigned index) {
        if (defaultMemory()) 
            return default_memory[index];
        else
            return main[index];
    }
    __device__ void clear() {
        num_main = 0;
        delete main;
        sizeof_chunk = default_chunk_size;
    }
    __device__ inline bool find(T value) {
        if (defaultMemory()) {
            for (unsigned i=0;i<num_main;i++) {
                if (default_memory[i]==value) {
                    return true;
                }
            }
            return false;
        } else {
            for (unsigned i=0;i<num_main;i++) {
                if (main[i]==value) {
                    return true;
                }
            }
            return false;
        }
    }
    T* main;
    T default_memory[DEFAULT_CHUNK_SIZE];
    size_t default_chunk_size;
    unsigned sizeof_chunk;
    unsigned num_main;
};


// Just copy VX_dVector and make it larger
#define DEFAULT_CHUNK_SIZE_LARGER 128
template <typename T>
class VX3_dVector_Larger {
/* It should be initialized in dev, and do push_back() and get(), and it should be freed after use. 
   if size<Default, use GPU local memory memory[Default], otherwise use malloc and main and store things in GPU global memory. (malloc is slow in GPU)*/
public:
    __device__ VX3_dVector_Larger<T>() {
        default_chunk_size = DEFAULT_CHUNK_SIZE_LARGER;
        init();
    }
    __device__ ~VX3_dVector_Larger<T>() {
        if (!defaultMemory())
            delete main;
    }
    __device__ void inline init() {
        sizeof_chunk = default_chunk_size;
        num_main = 0;
    }
    __device__ void inline push_back(T t) {
        if (defaultMemory()) {
            if (sizeof_chunk == default_chunk_size) { //use memory
                default_memory[num_main] = t;
            } else { //use main
                main[num_main] = t;
            }
        } else { //need allocation
            T* new_main;
            new_main = (T*)malloc(sizeof_chunk * 2 * sizeof(T));
            memcpy(new_main, main, sizeof_chunk * sizeof(T));
            delete main;
            main = new_main;
            main[num_main] = t;
            sizeof_chunk *= 2;
        }
        num_main++;
    }
    __device__ inline bool defaultMemory() { return num_main<sizeof_chunk; }
    __device__ inline unsigned size() {return num_main;};
    __device__ inline T &operator[] (unsigned index) {return get(index);};
    __device__ inline T get (unsigned index) {
        if (defaultMemory()) 
            return default_memory[index];
        else
            return main[index];
    }
    __device__ void clear() {
        num_main = 0;
        delete main;
        sizeof_chunk = default_chunk_size;
    }
    __device__ inline bool find(T value) {
        if (defaultMemory()) {
            for (unsigned i=0;i<num_main;i++) {
                if (default_memory[i]==value) {
                    return true;
                }
            }
            return false;
        } else {
            for (unsigned i=0;i<num_main;i++) {
                if (main[i]==value) {
                    return true;
                }
            }
            return false;
        }
    }
    T* main;
    T default_memory[DEFAULT_CHUNK_SIZE_LARGER];
    size_t default_chunk_size;
    unsigned sizeof_chunk;
    unsigned num_main;
};


#endif // TI_VECTOR_H
