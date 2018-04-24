//
// Created by russoul on 17.04.18.
//

#ifndef CUDA_MAT_LIBRARY_H
#define CUDA_MAT_LIBRARY_H

#include <vector>
#include <string>
#include <sstream>



namespace cudamat{



    extern "C"{
        float rand_float_0_1();
        float rand_float(float min, float max);
        int gen_rand_csr_matrix(int n, int m, std::vector<float> *A, std::vector<int> *IA, std::vector<int> *JA, float probability_of_zero, float min, float max);
        void gen_rand_vector(int n, float *vector, float probability_of_zero, float min, float max);
        void dump_sparse_matrix(std::ostringstream &stream, int n, int m, float *A, int *IA, int *JA);
        void dump_vector(std::ostringstream &stream, int n, float *vector);
        long int cur_time_ms();
        void print_device_info();

        //select particular GPU
        void select_device(int);
        void init();
        void destroy();
        void solve_linear_system_fixedPoint_jacobi(size_t N, size_t NNZ, float* A, int* IA, int* JA, float* b, float** x, double* dt_micro_sec);

    }
}

#endif //CUDA_MAT_LIBRARY_H
