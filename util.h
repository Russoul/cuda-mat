//
// Created by russoul on 16.03.18.
//

#ifndef CUDA_MAT_UTIL_H
#define CUDA_MAT_UTIL_H

#include <iostream>
#include <string>

using namespace std;

template<typename T>
concept bool Show = requires(T a){
    {to_string(a)} -> string;
};

template<Show T>
auto show(T a){
    return to_string(a);
}


#endif //CUDA_MAT_UTIL_H
