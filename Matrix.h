//
// Created by russoul on 16.03.18.
//

#ifndef CUDA_MAT_MATRIX_H
#define CUDA_MAT_MATRIX_H

#include <initializer_list>
#include "util.h"


template<class T>
class Matrix {


    T* array; //T must have a default constructor

private:
    Matrix(int n, int m) : n(n), m(m) {

        array = new T[n * m];
    }

public:

    const int n;
    const int m;

    Matrix(int n, int m, T &def) : Matrix(n,m){
        for (int i = 0; i < n * m; ++i) {
            array[i] = def;
        }
    }

    Matrix(int n, int m, std::initializer_list<T>);

    Matrix(const Matrix<T> &copy) : n(copy.n), m(copy.m){
        array = new T[n * m];

        for (int i = 0; i < n * m; ++i) {
            array[i] = copy.array[i];
        }
    }

    Matrix &operator=(const Matrix<T> &copy){
        if(n != copy.n || m != copy.m) throw std::runtime_error("A = B called on matrices of different sizes");

        for (int i = 0; i < n * m; ++i) {
            array[i] = copy.array[i];
        }
    }

    ~Matrix(){
        delete[] array;
    }

    const T& get(int i, int j) const{
        return array[i * m + j];
    }

    Matrix &set(int i, int j, T &e){
        array[i * m + j] = e;
        return *this;
    }

    Matrix<T> row(int i){
        auto ret = Matrix<T>(1, m);

        for (int j = 0; j < m; ++j) {
            ret.set(i, j, this->get(i, j));
        }

        return ret;
    }

    Matrix<T> column(int j){
        auto ret = Matrix<T>(n, 1);

        for (int i = 0; i < m; ++i) {
            ret.set(i, j, this->get(i, j));
        }

        return ret;
    }

    template<class A>
    friend Matrix<A> mul(const Matrix<A>, const Matrix<A> b);

};

template <class T>
Matrix<T> mul(const Matrix<T> a, const Matrix<T> b){
    if(a.m != b.n) throw std::runtime_error("mul(A,B) called on incompatible matrices");
    auto ret = Matrix<T>(a.n, b.m);

    for (int i = 0; i < a.n; ++i) {
        for (int j = 0; j < b.m; ++j) {
            T sum = 0;
            for (int k = 0; k < a.n; ++k) {
                sum += a.get(i,k) * b.get(k,j);
            }
            ret.set(i, j, sum);
        }
    }

    return ret;
}

template<class T>
Matrix<T>::Matrix(int n, int m, std::initializer_list<T> list) : n(n), m(m) {
    array = new T[n * m];
    auto *l = list.begin();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            array[i * m + j] = *l;
            ++l;
        }
    }
}

template<Show T>
std::string to_string(Matrix<T> a) {
    std::string str = "";
    for (int i = 0; i < a.n; ++i) {
        str += "( ";
        for (int j = 0; j < a.m; ++j) {
            str += show(a.get(i,j)) + " ";
        }
        str += ")\n";
    }

    return str;
}


#endif //CUDA_MAT_MATRIX_H
