
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
#include "paralution.hpp"

#ifndef _TIMES_H

#include "sys/times.h"

#endif





using namespace paralution;

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



    extern "C" long int cur_time_ms(){
        struct timeval tp;
        gettimeofday(&tp, NULL);
        long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

        return ms;
    }


    extern "C" void print_device_info(){
        info_paralution();
    }

    extern "C" void select_device(int n){

        set_device_paralution(n);
    }

    extern "C" void init(){
     init_paralution();
    }

    extern "C" void destroy(){
        stop_paralution();
    }



    //CSR matrix format
    //solves given linear system `Ax = b`, where {NNZ,A,IA,JA} is N x N sparse csr matrix

    //INPUT:
    //N - one dimension of square input matrix
    //NNZ - number of nonzero elements in array A
    //A - array of length NNZ, holds all the nonzero entries of input matrix in left-to-right top-to-bottom ("row-major") order.
    //IA - array of length m + 1. It is defined by this recursive definition:
       //IA[0] = 0
       //IA[i] = IA[i − 1] + (number of nonzero elements on the (i-1)-th row in the original matrix)
    //JA - array, contains the column index in original matrix of each element of A and hence is of length NNZ as well.
    //b - array, containers entries of vector `b` of the linear system


    //A,IA,JA,b will be MOVED ! (data is released)

    //OUTPUT:
    //x - pointer to the array representing the actual solution
    //needs to be freed manually


    //see https://wikipedia.org/wiki/Sparse_matrix for more info



    extern "C" void solve_linear_system_fixedPoint_jacobi(size_t N, size_t NNZ, float* A, int* IA, int* JA, float* b, float** x, double *dt_micro_sec){
        LocalVector<float> lx;
        LocalVector<float> lb;
        LocalMatrix<float> lmat;

        lmat.SetDataPtrCSR(&IA, &JA, &A, "A", static_cast<const int>(NNZ), N, N);
        lb.SetDataPtr(&b, "b", N);
        lx.Allocate("x", lmat.get_nrow());

        FixedPoint<LocalMatrix<float>, LocalVector<float>, float > ls ;
        ls.Init(1e-10, 1e-8, 1e8, 10000);
        Jacobi<LocalMatrix<float>, LocalVector<float>, float > p ;

        ls.SetRelaxation(1.3);


        double tick, tack;

        lx.Zeros();

        lmat.MoveToAccelerator();
        lx.MoveToAccelerator();
        lb.MoveToAccelerator();


        ls.SetOperator(lmat);
        ls.SetPreconditioner(p);
        ls.Verbose(0);
        ls.Build();


        tick = paralution_time();
        ls.Solve(lb, &lx);
        tack = paralution_time();
        *dt_micro_sec = tack - tick;

        ls.Clear();

        lx.MoveToHost();


        lx.LeaveDataPtr(x);
    }



}

