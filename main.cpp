#include <iostream>
#include "Matrix.h"
#include "givens.h"

int main() {


    auto m1 = Matrix(2,2, {1,2,
                          3,4});

    auto m2 = Matrix(2,2, {0,-1,
                           2, 3});

    auto m3 = mul(m1, m2);

    std::cout << show(m3) << std::endl;


    auto c = 2;
    auto s = 3;

    std::cout << show(givens_matrix(3,3, 2,1, c, s)) << std::endl;

    return 0;
}