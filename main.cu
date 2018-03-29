

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <string>
#include <iostream>
#include <device_launch_parameters.h>
#include "helper_math.h"
#include <vector>
#include <algorithm>

#include <cusolverSp.h>
#include <sys/time.h>



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


int getSPcores(cudaDeviceProp devProp)
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


float rand_float_0_1(){
    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    return r;
}

float rand_float(float min, float max){
    float norm = rand_float_0_1();

    return norm * (max - min) + min;
}

int gen_rand_csr_matrix(int n, int m, std::vector<float> *A, std::vector<int> *IA, std::vector<int> *JA, float probability_of_zero, float min, float max){
    IA->push_back(0);
    int row_count = 0; //NNZ
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            bool zero = rand_float_0_1() <= probability_of_zero;
            if(!zero){
                A->push_back(rand_float(min, max));
                JA->push_back(j);
                row_count += 1;
            }

        }
        IA->push_back(row_count);
    }

    return row_count;
}




void gen_rand_vector(int n, float *vector, float probability_of_zero, float min, float max){
    for (int i = 0; i < n; ++i) {
        vector[i] = rand_float_0_1() <= probability_of_zero ? 0.0F : rand_float(min, max);
    }
}

std::string dump_sparse_matrix_row(int i, int m, float *A, int *IA, int *JA){
    std::string str = "( ";

    int last_col_num = -1;

    for (int j = IA[i]; j <= IA[i + 1] - 1; ++j) {

        int col_num = JA[j];

        for (int k = 0; k < col_num - last_col_num - 1; ++k) {
            str += "0.0 ";
        }

        last_col_num = col_num;

        str += std::to_string(A[j]) + " ";

        //printf("row %d, non zero %f, col num %d\n", i, A[j], col_num);
    }

    for (int k = 0; k < m - last_col_num - 1; ++k) {
        str += "0.0 ";
    }

    str += ")";

    return str;
}

std::string dump_sparse_matrix(int n, int m, float *A, int *IA, int *JA){

    std::string str = "";

    for (int i = 0; i < n; ++i) {
        str += dump_sparse_matrix_row(i, m, A, IA, JA) + "\n";
    }

    return str;
}

std::string dump_vector(int n, float *vector){
    std::string str = "( ";
    for (int i = 0; i < n; ++i) {
        str += std::to_string(vector[i]) + " ";
    }
    str += ")";

    return str;
}

std::string cusolve_get_err_string(int status){
    switch(status){
        case CUSOLVER_STATUS_SUCCESS: return "the operation completed successfully.";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "the library was not initialized.";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "the resources could not be allocated.";
        case CUSOLVER_STATUS_INVALID_VALUE: return "invalid parameters were passed (m,nnz<=0), base index is not 0 or 1.";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "the device only supports compute capability 2.0 and above.";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "an internal operation failed.";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "the matrix type is not supported.";

    }
}


long int cur_time_ms(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

    return ms;
}

int main(){


    int            deviceCount;
    cudaDeviceProp devProp;

    cudaGetDeviceCount ( &deviceCount );

    printf ( "Found %d devices\n", deviceCount );


    for (int i = 0; i < deviceCount; ++i) {
        printf("======================== DEVICE %d =============================\n", i);
        cudaGetDeviceProperties ( &devProp, i );

        printf("Device name:                %s\n", devProp.name);
        printf("Major revision number:      %d\n", devProp.major);
        printf("Minor revision Number:      %d\n", devProp.minor);
        printf("Total Global Memory:        %d\n", devProp.totalGlobalMem);
        printf("Total shared mem per block: %d\n", devProp.sharedMemPerBlock);
        printf("Total const mem size:       %d\n", devProp.totalConstMem);
        printf("Warp size:                  %d\n", devProp.warpSize);
        printf("Maximum block dimensions:   %d x %d x %d\n", devProp.maxThreadsDim[0], \
                                                                                                          devProp.maxThreadsDim[1], \
                                                                                                          devProp.maxThreadsDim[2]);

        printf("Maximum grid dimensions:    %d x %d x %d\n", devProp.maxGridSize[0], \
                                                                                                          devProp.maxGridSize[1], \
                                                                                                          devProp.maxGridSize[2]);
        printf("Clock Rate:                 %d\n", devProp.clockRate);
        printf("Number of muliprocessors:   %d\n", devProp.multiProcessorCount);

        printf("Number of cores %d\n", getSPcores(devProp));
    }


    cusolverSpHandle_t handle;
    auto status = cusolverSpCreate(&handle);

    if(status != CUSOLVER_STATUS_SUCCESS){
        std::cerr << "failed to initialize CUSOLVER" << std::endl;
        exit(-1);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cusolverSpSetStream(handle, stream);


    std::vector<float> A;
    std::vector<int> IA;
    std::vector<int> JA;



    float *A_d;
    int *IA_d;
    int *JA_d;
    float *X_d;
    float *b_d;

    int n = 100;
    float prob = 0.75;
    float min = -1;
    float max = 1;
    float tol = 0.005;

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    std::vector<float> b(n);

    int NNZ = gen_rand_csr_matrix(n, n, &A, &IA, &JA, prob, min, max);
    gen_rand_vector(n, &b[0], 0, -1, 1);
    std::cout << dump_sparse_matrix(n,n, &A[0], &IA[0], &JA[0]) << std::endl;

    std::cout << dump_vector(n, &b[0]) << std::endl;


    cudaMalloc(&A_d, sizeof(float) * A.size());
    cudaMalloc(&IA_d, sizeof(float) * IA.size());
    cudaMalloc(&JA_d, sizeof(float) * JA.size());
    cudaMalloc(&X_d, sizeof(float) * n);
    cudaMalloc(&b_d, sizeof(float) * n);


    cudaMemcpy(A_d, &A[0], sizeof(float)*A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(IA_d, &IA[0], sizeof(float)*IA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(JA_d, &JA[0], sizeof(float)*JA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, &b[0], sizeof(float)*b.size(), cudaMemcpyHostToDevice);


    int singularity;


    auto t1 = cur_time_ms();
    status = cusolverSpScsrlsvqr(handle, n, A.size(), descr, A_d, IA_d, JA_d, b_d, tol, 0, X_d, &singularity);
    auto t2 = cur_time_ms();

    printf("delta time %d ms\n", t2 - t1);


    if(status != CUSOLVER_STATUS_SUCCESS){
        std::cerr << cusolve_get_err_string(status) << std::endl;
    }else{
        //solved

        printf("singularity(-1 for non-singular) : %d\n", singularity);

        std::vector<float> X(n);

        cudaMemcpy(&X[0], X_d, sizeof(float)*n, cudaMemcpyDeviceToHost);

        std::cout << "resulting vector:\n";
        std::cout << dump_vector(n, &X[0]) << std::endl;

    }

    cudaFree(A_d);
    cudaFree(IA_d);
    cudaFree(JA_d);
    cudaFree(X_d);
    cudaFree(b_d);

    cusolverSpDestroy(handle);


    return 0;
}