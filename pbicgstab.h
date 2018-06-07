/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma once

#include <typeinfo> // for usage of C++ typeid
#include <cuda_runtime.h>
#include <vector>
#include <conio.h>
#include <sstream>
#include <iostream>
#include <functional>


enum Base{
    Base0 = 0,
    Base1 = 1
};

double rand_float_0_1();

double rand_float(double min, double max);

template<Base base>
int gen_rand_csr_matrix(int n, int m, std::vector<double> *A, std::vector<int> *IA, std::vector<int> *JA, double probability_of_zero, double min, double max, double eps) {
    IA->push_back(base);//base
    int row_count = base; //NNZ, base
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            bool zero = rand_float_0_1() <= probability_of_zero;
            if (!zero) {
                auto r = rand_float(min, max);
                while (abs(r) < eps) {
                    r = rand_float(min, max);
                }

                A->push_back(r);
                JA->push_back(j + base);//base
                row_count += 1;

            }
        }
        IA->push_back(row_count);
    }

    return A->size();
}

template<Base base>
int fill_csr_matrix(int n, int m, std::vector<double> *A, std::vector<int> *IA, std::vector<int> *JA,
                    std::function<double(int, int)> f, double eps) {
    IA->push_back(base);//base
    int row_count = base; //NNZ, base
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double el = f(i, j);
            if (abs(el) > eps) {
                A->push_back(el);
                JA->push_back(j + base);//base
                row_count += 1;

            }
        }
        IA->push_back(row_count);
    }

    return A->size();
}

void gen_rand_vector(int n, double *vector, double probability_of_zero, double min, double max);

template <typename T>
void dump_vector(std::ostringstream &stream, int n, T *vector) {
    stream << "(";
    for (int i = 0; i < n; ++i) {
        stream << std::to_string(vector[i]) + " ";
    }
    stream << ")";

}


void toDenseVector(int n, int nnz, double* A, int* IA, double* out);

#define IN  //function argument marked as `input`
#define OUT //function argument marked as `output`

//common arguments

//n - dimension of input matrix `A` of equation `Ax = b`
//nnz - number of non zero elements in matrix `A`
//A - array of non zero elements of size `nnz`
//iA - array of row sizes of size `n + 1`, iA[0] is always `0` in case of `base 0` or 1 in case of `base 1`
//by default .mtx files use `base 1`
//jA - array of column indices of non zero elements of size `nnz` (indexing starts from `0` in case of `base 0` and from `1` otherwise)
//b - dense vector of right hand side of equation
//maxit - maximum iterations before halting the algorithm
//debug - whether to print debug info or not
//x - output result, solution to equation
//dtAlg - output delta time of GPU part of the algorithm (I/O is not accounted)

//matrix A can be represented as A = A0 + I*d (can be used only without preconditioner)

//solve Ax = b, no preconditioner
bool bicgstab(int n, int nnz, double IN(*A), int IN(*iA), int IN(*jA), double IN(*b), int maxit, double tol, bool debug, double OUT(*x), double OUT(*dtAlg));

//solve (A0 + I*d)x = b, no preconditioner
bool bicgstab(int n, int nnz, double IN(*A0), int IN(*iA0), int IN(*jA0), double IN(*d), double IN(*x0), double IN(*b), int maxit, double tol, bool debug, double OUT(*x), double OUT(*dtAlg));

//solve Ax = b, LU preconditioner, for i = j must hold: A[i,j] != 0
bool bicgstab_lu_precond(int n, int nnz, double IN(*A), int IN(*iA), int IN(*jA), double IN(*b),
                  int maxit, double tol, bool debug, double OUT(*x), double OUT(*dtAlg));
