#include <iostream>
#include "Matrix.h"
#include "givens.h"


//CPU Ax = b solver
int main() {


    auto m1 = Matrix(2,2, {1.0F,2.0F,
                          3.0F,4.0F});

    auto m2 = Matrix(2,2, {0.0F,-1.0F,
                           2.0F, 3.0F});

    auto m3 = mul(m1, m2);

    std::cout << "m1 x m2" << std::endl;
    std::cout << show(m3) << std::endl;




    auto c = 2;
    auto s = 3;


    auto r = sqrt(10.0F);
    auto c1 = 1 / r;
    auto s1 = -3 / r;

    auto givens2 = givens_matrix(2, 1,0, c1, s1);

    std::cout << "givens2" << std::endl;
    std::cout << show(givens2) << std::endl;

    auto R = mul(givens2, m1);

    std::cout << "givens multiplied" << std::endl;
    std::cout << show(R) << std::endl;


    auto QR2 = qr(m1, 0.01F);

    std::cout << "result of algorithm 2" << std::endl;
    std::cout << show(QR2.r) << std::endl;

    auto mat3 = Matrix(3,4, {6.0F, 5.0F, 0.0F, 1.0F,
                             5.0F, 1.0F, 4.0F, 2.0F,
                             5.0F, 1.0F, 5.0F, 3.0F});

    auto QR3 = qr(mat3, 0.01F);
    std::cout << "result of alg 3" << std::endl;

    std::cout << "R:" << std::endl;
    std::cout << show(QR3.r) << std::endl;

    std::cout << "Q:" << std::endl;
    std::cout << show(QR3.q) << std::endl;


    int rank;
    std::cout << "consistency" << std::endl;
    std::cout << is_consistent_row_echelon(QR3.r, 0.01F, &rank) << std::endl; //TODO upper triangular matrix may not always be in row echelon form

    std::cout << "singular root" << std::endl;
    std::cout << has_singular_root(rank, QR3.r.m - 1) << std::endl;

    //TODO test back substitution and general solver of consistent and uniquely determined SLE

    return 0;
}