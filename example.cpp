//
// Created by russoul on 17.04.18.
//
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <fstream>

#include "library.h"

int main(int argc, char* argv[]){

    std::vector<float> A;
    std::vector<int> IA;
    std::vector<int> JA;

    int n;
    int dev;
    float prob;
    float probVector;
    float min;
    float max;
    bool print;
    std::string filename;

    cudamat::print_device_info();



    std::cout << "Device number: ";
    std::cin >> dev;
    cudamat::select_device(dev);
    cudamat::init();
    cudamat::print_device_info();
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
    std::cout << "Output result ?(0/1): ";
    std::cin >> print;
    std::cout << "File output(filename): ";
    std::cin >> filename;





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

    std::vector<float> b(n);

    int NNZ = cudamat::gen_rand_csr_matrix(n, n, &A, &IA, &JA, prob, min, max);
    cudamat::gen_rand_vector(n, &b[0], probVector, min, max);




    int *IAptr  = new int[IA.size()]; //gets moved
    int *JAptr  = new int[JA.size()]; //gets moved
    float *Aptr = new float[A.size()]; //gets moved
    float *bptr = new float[b.size()]; //gets moved

    memcpy(IAptr, &IA[0], sizeof(int) * IA.size());
    memcpy(JAptr, &JA[0], sizeof(int) * JA.size());
    memcpy(bptr, &b[0], sizeof(float) * b.size());
    memcpy(Aptr, &A[0], sizeof(float) * A.size());

    std::ofstream file;

    file.open(filename);
    std::ostringstream str;
    file << "=========================== MATRIX ========================================" << std::endl;
    cudamat::dump_sparse_matrix(str, n, n, Aptr, IAptr, JAptr);
    file << str.str() << std::endl;
    file << "=========================== VECTOR ========================================" << std::endl;
    std::ostringstream str2;
    cudamat::dump_vector(str2, n, bptr);
    file << str2.str() << std::endl;


    float* x;

    double dt_micro;
    cudamat::solve_linear_system_fixedPoint_jacobi(n, NNZ, Aptr, IAptr, JAptr, bptr, &x, &dt_micro);

    std::cout << dt_micro << " micro seconds" << std::endl;

    if(print){
        std::cout << "solution" << std::endl;
        for (int i = 0; i < n; ++i) {
            std::cout << x[i] << std::endl;
        }
    }

    file << "=========================== RESULT ========================================" << std::endl;
    std::ostringstream str3;
    cudamat::dump_vector(str3, n, x);
    file << str3.str();
    file.close();

    cudamat::destroy();
    delete[] x;


    return 0;
}
