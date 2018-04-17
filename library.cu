
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <device_launch_parameters.h>

#include <cusolverSp.h>

#include "helper_math.h"


#ifndef _TIMES_H

#include "sys/times.h"

#endif







namespace cudamat{

    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }
    //==========================

    //return single processor core count
    inline int get_sp_cores(cudaDeviceProp devProp)
    {
        int cores = 0;
        int mp = devProp.multiProcessorCount;
        switch (devProp.major){
            case 2: // Fermi
                if (devProp.minor == 1) cores = mp * 48;
                else cores = mp * 32;
                break;
            case 3: // Kepler
                cores = mp * 192;
                break;
            case 5: // Maxwell
                cores = mp * 128;
                break;
            case 6: // Pascal
                if (devProp.minor == 1) cores = mp * 128;
                else if (devProp.minor == 0) cores = mp * 64;
                else printf("Unknown device type\n");
                break;
            case 7: // Volta
                if (devProp.minor == 0) cores = mp * 64;
                else printf("Unknown device type\n");
                break;
            default:
                printf("Unknown device type\n");
                break;
        }
        return cores;
    }


    extern "C" float rand_float_0_1(){
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

        return r;
    }

    extern "C" float rand_float(float min, float max){
        float norm = rand_float_0_1();

        return norm * (max - min) + min;
    }

    extern "C" int gen_rand_csr_matrix(int n, int m, std::vector<float> *A, std::vector<int> *IA, std::vector<int> *JA, float probability_of_zero, float min, float max){
        IA->push_back(0);
        int row_count = 0; //NNZ
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                bool zero = rand_float_0_1() <= probability_of_zero;
                if(!zero){
                    auto r = rand_float(min, max);
                    if(r != 0.0F){
                        A->push_back(r);
                        JA->push_back(j);
                        row_count += 1;
                    }

                }

            }
            IA->push_back(row_count);
        }

        return row_count;
    }




    extern "C" void gen_rand_vector(int n, float *vector, float probability_of_zero, float min, float max){
        for (int i = 0; i < n; ++i) {
            vector[i] = rand_float_0_1() <= probability_of_zero ? 0.0F : rand_float(min, max);
        }
    }

    void dump_sparse_matrix_row(std::ostringstream &stream, int i, int m, float *A, int *IA, int *JA){

        stream << "(";

        int last_col_num = -1;

        for (int j = IA[i]; j <= IA[i + 1] - 1; ++j) {

            int col_num = JA[j];

            for (int k = 0; k < col_num - last_col_num - 1; ++k) {
                stream << "0.0 ";
            }

            last_col_num = col_num;

            stream << std::to_string(A[j]) + " ";

            //printf("row %d, non zero %f, col num %d\n", i, A[j], col_num);
        }

        for (int k = 0; k < m - last_col_num - 1; ++k) {
            stream << "0.0 ";
        }

        stream << ")";

    }

    extern "C" void dump_sparse_matrix(std::ostringstream &stream, int n, int m, float *A, int *IA, int *JA){


        for (int i = 0; i < n; ++i) {
            dump_sparse_matrix_row(stream, i, m, A, IA, JA);
            stream << "\n";
        }

    }

    extern "C" void dump_vector(std::ostringstream &stream, int n, float *vector){
        stream << "(";
        for (int i = 0; i < n; ++i) {
            stream << std::to_string(vector[i]) + " ";
        }
        stream << ")";

    }

    inline std::string cusolve_get_err_string(int status){
        switch(status){
            case CUSOLVER_STATUS_SUCCESS: return "the operation completed successfully.";
            case CUSOLVER_STATUS_NOT_INITIALIZED: return "the library was not initialized.";
            case CUSOLVER_STATUS_ALLOC_FAILED: return "the resources could not be allocated.";
            case CUSOLVER_STATUS_INVALID_VALUE: return "invalid parameters were passed (m,nnz<=0), base index is not 0 or 1.";
            case CUSOLVER_STATUS_ARCH_MISMATCH: return "the device only supports compute capability 2.0 and above.";
            case CUSOLVER_STATUS_INTERNAL_ERROR: return "an internal operation failed.";
            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "the matrix type is not supported.";
            default: return "Unknown error";
        }
    }


    extern "C" long int cur_time_ms(){
        struct timeval tp;
        gettimeofday(&tp, NULL);
        long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

        return ms;
    }


    extern "C" void print_device_info(){
        int            deviceCount;
        cudaDeviceProp devProp;

        gpuErrchk(cudaGetDeviceCount ( &deviceCount ));

        printf ( "Found %d devices\n", deviceCount );


        for (int i = 0; i < deviceCount; ++i) {
            printf("======================== DEVICE %d =============================\n", i);
            cudaGetDeviceProperties ( &devProp, i );

            printf("Device name:                %s\n", devProp.name);
            printf("Major revision number:      %d\n", devProp.major);
            printf("Minor revision Number:      %d\n", devProp.minor);
            printf("Total Global Memory:        %u\n", devProp.totalGlobalMem);
            printf("Total shared mem per block: %u\n", devProp.sharedMemPerBlock);
            printf("Total const mem size:       %u\n", devProp.totalConstMem);
            printf("Warp size:                  %d\n", devProp.warpSize);
            printf("Maximum block dimensions:   %d x %d x %d\n", devProp.maxThreadsDim[0], \
                                                                                                          devProp.maxThreadsDim[1], \
                                                                                                          devProp.maxThreadsDim[2]);

            printf("Maximum grid dimensions:    %d x %d x %d\n", devProp.maxGridSize[0], \
                                                                                                          devProp.maxGridSize[1], \
                                                                                                          devProp.maxGridSize[2]);
            printf("Clock Rate:                 %d\n", devProp.clockRate);
            printf("Number of muliprocessors:   %d\n", devProp.multiProcessorCount);

            printf("Number of cores %d\n", get_sp_cores(devProp));
        }
    }


    struct CudaMatHandle{
        cusolverSpHandle_t cusolver_handle;
        cudaStream_t stream;
    };

    extern "C" void init(CudaMatHandle *handle_out){
        cusolverSpHandle_t handle;
        auto status = cusolverSpCreate(&handle);

        if(status != CUSOLVER_STATUS_SUCCESS){
            std::cerr << "failed to initialize CUSOLVER" << std::endl;
            exit(-1);
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cusolverSpSetStream(handle, stream);

        handle_out->cusolver_handle = handle;
        handle_out->stream = stream;
    }

    extern "C" void destroy(CudaMatHandle handle){
        cusolverSpDestroy(handle.cusolver_handle);
    }



    //CSR matrix format
    //solves given linear system `Ax = b`, where {NNZ,A,IA,JA} is N x N sparse csr matrix

    //INPUT:
    //handle - cusolver handle gathered by calling `init()`
    //N - one dimension of square input matrix
    //NNZ - number of nonzero elements in array A
    //A - array of length NNZ, holds all the nonzero entries of input matrix in left-to-right top-to-bottom ("row-major") order.
    //IA - array of length m + 1. It is defined by this recursive definition:
       //IA[0] = 0
       //IA[i] = IA[i − 1] + (number of nonzero elements on the (i-1)-th row in the original matrix)
    //JA - array, contains the column index in original matrix of each element of A and hence is of length NNZ as well.
    //b - array, containers entries of vector `b` of the linear system
    //tolerance - a positive floating point value, used to truncate singular values whose absolute value is below given tolerance (used in SVD decomposition)

    //OUTPUT:
    //x - pointer to the array representing the actual solution
    //if the system is singular then no memory will be allocated for the solution otherwise
    //new array will be allocated and `x` set pointing the this array
    //the array must be freed manually then

    //RETURNS:
    //whether there is a solution for the system with given tolerance

    //see https://wikipedia.org/wiki/Sparse_matrix for more info



    extern "C" bool solve_linear_system_qr_svd(CudaMatHandle handle, size_t N, size_t NNZ, float* A, int* IA, int* JA, float* b, float tolerance, float** x){
        cusparseMatDescr_t descr;
        cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

        // === device memory ===
        float *A_d;
        int *IA_d;
        int *JA_d;
        float *X_d;
        float *b_d;
        // =====================

        gpuErrchk(cudaMalloc(&A_d, sizeof(float) * NNZ));
        gpuErrchk(cudaMalloc(&IA_d, sizeof(float) * (N+1)));
        gpuErrchk(cudaMalloc(&JA_d, sizeof(float) * NNZ));
        gpuErrchk(cudaMalloc(&X_d, sizeof(float) * N));
        gpuErrchk(cudaMalloc(&b_d, sizeof(float) * N));


        gpuErrchk(cudaMemcpy(A_d, A, sizeof(float) * NNZ, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(IA_d, IA, sizeof(float) * (N+1), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(JA_d, JA, sizeof(float) * NNZ, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(b_d, b, sizeof(float) * N, cudaMemcpyHostToDevice));

        int singularity;
        auto status = cusolverSpScsrlsvqr(handle.cusolver_handle, N, NNZ, descr, A_d, IA_d, JA_d, b_d, tolerance, 0, X_d, &singularity);

        if(status != CUSOLVER_STATUS_SUCCESS){
            std::cerr << cusolve_get_err_string(status) << std::endl;
            exit(-1);
        }else{
            //solved

            if(singularity == -1){
                *x = static_cast<float *>(malloc(sizeof(float) * N));
                gpuErrchk(cudaMemcpy(*x, X_d, sizeof(float)*N, cudaMemcpyDeviceToHost));
            }


        }

        gpuErrchk(cudaFree(A_d));
        gpuErrchk(cudaFree(IA_d));
        gpuErrchk(cudaFree(JA_d));
        gpuErrchk(cudaFree(X_d));
        gpuErrchk(cudaFree(b_d));

        return singularity == -1;

    }



}

