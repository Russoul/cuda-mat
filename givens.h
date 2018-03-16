//
// Created by russoul on 16.03.18.
//

#ifndef CUDA_MAT_GIVENS_H
#define CUDA_MAT_GIVENS_H

#include <iostream>
#include <string>
#include "Matrix.h"


using namespace std;

template<class T>
Matrix<T> givens_matrix(int n, int m, int _i, int _j, T &c, T &s){
    T zero = 0;
    auto res = Matrix<T>(n, m, zero);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
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

#endif //CUDA_MAT_GIVENS_H
