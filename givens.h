//
// Created by russoul on 16.03.18.
//

#ifndef CUDA_MAT_GIVENS_H
#define CUDA_MAT_GIVENS_H

#include <iostream>
#include <string>
#include <cmath>
#include "Matrix.h"


using namespace std;


template<class T>
struct QR{
    Matrix<T> q;
    Matrix<T> r;
};

template<class T>
Matrix<T> givens_matrix(int n, int _i, int _j, T &c, T &s){
    T zero = 0;
    auto res = Matrix<T>(n, n, zero);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            T el;
            if(i == j){
                if(i != _i && i != _j){
                    el = 1;
                }else{
                    el = c;
                }
            }else{
                if(i == _i && j == _j || i == _j && j == _i){
                    if(i > j){
                        el = s;
                    }else{
                        el = -s;
                    }
                }else{
                    el = 0;
                }
            }

            res.set(i, j, el);
        }
    }

    return res;
}

template<class T>
QR<T> qr(const Matrix<T> &A, T eps){

    auto R = A;
    auto Q = Matrix<T>::identity(R.n);

    for (int j = 0; j < R.m; ++j) {
        for (int i = j + 1; i < R.n; ++i) { //i > j
            auto b = R.get(i, j);
            if(abs(b) >= eps){ //not zero
                auto a = R.get(j, j);

                auto r = sqrt(a * a + b * b);

                auto c = a / r;
                auto s = -b / r;

                auto givens = givens_matrix(R.n, i, j, c, s);

                R = mul(givens, R);

                Q = mul(Q, Matrix<T>::transpose(givens));

            }
        }
    }

    return {Q,R};
}


template<class T>
int rank_row_echelon(const Matrix<T> &A, const T eps){
    auto rank = A.n;

    for (int i = A.n - 1; i >= 0; --i) {
        if(Matrix<T>::is_zero(A.row(i), eps))
            rank -= 1;
    }

    return rank;
}

//Kronecker-Capelli theorem
template<class T>
bool is_consistent_row_echelon(const Matrix<T> &augmented, const T eps, int *out_rank = nullptr){


    auto A = Matrix<T>::without_column(augmented, augmented.m - 1);

    auto rank_A = rank_row_echelon(A, eps);

    if(out_rank)
        *out_rank = rank_A;

    return rank_A == rank_row_echelon(augmented, eps);
}

bool has_singular_root(int rank, int m){
    return rank == m;
}

template<class T>
Matrix<T> back_substitution_singular_root(const Matrix<T> &A){
    auto res = Matrix<T>(A.n, 1, 0);

    for (int i = A.n - 1; i >= 0; --i) {
        auto b = A.get(i, i + 1);
        auto sum = 0;

        for (int j = A.n - 1; j > i; ++j) {
            sum += res.get(j, 0) * A.get(i,j);
        }

        res.set(i, 0, (b - sum)/A.get(i,i));
    }

    return res;
}

#endif //CUDA_MAT_GIVENS_H
