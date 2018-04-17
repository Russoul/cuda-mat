//
// Created by russoul on 17.04.18.
//
#include <string>
#include <iostream>
#include <vector>
#include <sstream>

#include "library.h"

int main(){


    std::vector<float> A;
    std::vector<int> IA;
    std::vector<int> JA;

    int n;
    float prob;
    float probVector;
    float min;
    float max;
    float tol = 0.005;
    bool print;



    std::cout << "Input N: ";
    std::cin >> n;
    std::cout << "Input probability of zero while generating sparse matrix: ";
    std::cin >> prob;
    std::cout << "Input probability of zero while generating sparse vector: ";
    std::cin >> probVector;
    std::cout << "Input min value: ";
    std::cin >> min;
    std::cout << "Input max value: ";
    std::cin >> max;
    std::cout << "Output random sparse matrix and vector to the console ?(0/1): ";
    std::cin >> print;

    if(n <= 1){
        std::cerr << "for N must hold: N > 1" << std::endl;
        return 0;
    }

    if(prob < 0 || prob > 1 || probVector < 0 || probVector > 1){
        std::cerr << "for probability must hold: 0 <= P <= 1" << std::endl;
        return 0;
    }

    if(min > max){
        std::cerr << "for min/max values must hold: min <= max" << std::endl;
    }


    cudamat::CudaMatHandle handle;
    cudamat::init(&handle);

    std::vector<float> b(n);

    int NNZ = cudamat::gen_rand_csr_matrix(n, n, &A, &IA, &JA, prob, min, max);
    cudamat::gen_rand_vector(n, &b[0], probVector, min, max);


    if(print){

        std::ostringstream str;

        cudamat::dump_sparse_matrix(str, n,n, &A[0], &IA[0], &JA[0]);

        str << std::endl;

        cudamat::dump_vector(str, n, &b[0]);

        str << std::endl;

        std::cout << str.str();
    }

    float *x = nullptr;

    auto t1 = cudamat::cur_time_ms();
    auto non_singular = cudamat::solve_linear_system_qr_svd(handle, n, A.size(), &A[0], &IA[0], &JA[0], &b[0], tol, &x);
    auto t2 = cudamat::cur_time_ms();



    if(non_singular){
        std::ostringstream str;

        str << "solution(" << (t2 - t1) << " ms): " << std::endl;

        cudamat::dump_vector(str, n, &x[0]);

        str << std::endl;

        std::cout << str.str();

        free(x);

    }else{
        std::cout << "Ax = b has no solution." << std::endl;
    }

    cudamat::destroy(handle);


    return 0;
}
