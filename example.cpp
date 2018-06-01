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

#include <typeinfo> // for usage of C++ typeid
#include <cuda_runtime.h>
#include <vector>
#include <conio.h>
#include <sstream>
#include <iostream>

#include "cublas_v2.h"
#include "cusparse_v2.h"
#include "helper_cusolver.h"
#include "mmio.h"

#include "mmio_wrapper.h"

#include "helper_cuda.h"

#include "pbicgstab.h"
 



int main (int argc, char *argv[]){
    int status = EXIT_FAILURE;
    char *matrix_filename = NULL;
	char *vector_filename = NULL;
    bool debug=false;
    double prob_of_zero_mat = 0.1;
    double prob_of_zero_vec = 0.0;
    int dim = 4;
	bool print = false;


	const int maxit = 2000;
    const double tol= 0.0000001;



    /* WARNING: it is assumed that the matrices are stores in Matrix Market format */
    printf("WARNING: it is assumed that the matrices are stored in Matrix Market format with double as element type\n Usage: ./BiCGStab -M[matrix.mtx] -V[vector.mtx] [-D] -R[prob of zero] -N[dim] [-P] [device=<num>]\n"
		   "By default matrix will be random, N = 4, P(X = 0)=0.1, vector will be random, P(X = 0)=0.1\n"
           "example usage:\n"
           "./example.exe -M\"mat10000.mtx\"\n"
           "./example.exe -M\"mat3.mtx\" -V\"vec3.mtx\" -D -P\n"
		   "./example.exe -N\"40\" -R\"0.5\" -D\n"
		   );

    int i=0;
    int temp_argc = argc;
    while (argc) {
        if (*argv[i] == '-') {
            switch (*(argv[i]+1)) { 
            case 'M':
                matrix_filename = argv[i]+2;  
                break;
			case 'V':
				vector_filename = argv[i] + 2;
				break;
            case 'D':
                debug = true;
                break;    
			case 'R':
				prob_of_zero_mat = std::stod(argv[i] + 2);
				break;
			case 'P':
				print = true;
				break;
			case 'N':
				dim = std::stoi(argv[i] + 2);
				break;
            default:
                fprintf (stderr, "Unknown switch '-%s'\n", argv[i]+1);
                return status;
            }
        }
        argc--;
        i++;
    }

    argc = temp_argc;

    if (matrix_filename != NULL){
		printf("Using matrix input file [%s]\n", matrix_filename);
    }


	if (vector_filename != NULL) {
		printf("Using vector input file [%s]\n", vector_filename);
	}


    findCudaDevice(argc, (const char **)argv);

	int n;
	int nnz;
	double *A;
	int *iA;
	int *jA;
	double *b = nullptr;
	double *x = nullptr;

	if (matrix_filename != nullptr){

		int matrixN;
		int matrixM;

		if (loadMMSparseMatrix(matrix_filename, 'd', true, &matrixM, &matrixN, &nnz, &A, &iA, &jA)) {
			fprintf(stderr, "!!!! cusparseLoadMMSparseMatrix FAILED\n");
			return EXIT_FAILURE;
		}

		if(matrixN != matrixM){
			fprintf(stderr, "!!!! square matrix is expected\n");
			return EXIT_FAILURE;
		}

		n = matrixN;


	}else{

		std::vector<double> _A;
		std::vector<int> _IA;
		std::vector<int> _JA;



		nnz = gen_rand_csr_matrix(dim, dim, &_A, &_IA, &_JA, prob_of_zero_mat, 1.0, 5.0);
		n = dim;



		A = static_cast<double *>(malloc(sizeof(double) * nnz));
		iA = static_cast<int *>(malloc(sizeof(int) * (n + 1)));
		jA = static_cast<int *>(malloc(sizeof(int) * nnz));

		if(_A.empty()){
			fprintf(stderr, "!!!! all random elements of the random matrix are zeros !\n");
			return EXIT_FAILURE;
		}

		memcpy(A, &_A[0], sizeof(double)*nnz);
		memcpy(iA, &_IA[0], sizeof(int)*(n + 1));
		memcpy(jA, &_JA[0], sizeof(int)*nnz);


	}

	if(vector_filename != nullptr){

		int vN;
		int vM;
		int vnnz;
		double *vA = nullptr;
		int* vIA = nullptr;
		int* vJA = nullptr;

		if (loadMMSparseMatrix(vector_filename, 'd', true, &vM, &vN, &vnnz, &vA, &vIA, &vJA)) {
			fprintf(stderr, "!!!! cusparseLoadMMSparseMatrix FAILED\n");
			return EXIT_FAILURE;
		}

		if (vN != 1) {
			fprintf(stderr, "b must be a vector !\n");
			return EXIT_FAILURE;
		}

		if (vM != n) {
			fprintf(stderr, "incorrect dim\n");
			return EXIT_FAILURE;
		}

		b = (double*)malloc(sizeof(double) * n);

		toDenseVector(vM, vnnz, vA, vIA, b);

		free(vA);
		free(vIA);
		free(vJA);
	}else{
		b = (double*)malloc(sizeof(double) * n);
		gen_rand_vector(n, b, prob_of_zero_vec, 1, 5.0);

	}

	x = static_cast<double *>(malloc(sizeof(double) * n));

	std::cout << "nnz=" << nnz << std::endl;


	double dtAlg;
	auto t1 = second();
	bool solved = bicgstab(n, nnz, A, iA, jA, b, maxit, tol, debug, x, &dtAlg);
	auto t2 = second();



	if(solved){
		std::cout << "success" << std::endl;
		if(print){
			std::cout << "result:" << std::endl;
			std::ostringstream s;
			dump_vector(s, n, x);
			std::cout << s.str() << std::endl;
		}

		std::cout << "algorithm delta time = " << dtAlg << " s" << std::endl;
		std::cout << "total delta time = " << t2 - t1 << " s" << std::endl;
	}else{
		std::cerr << "method failed" << std::endl;
	}

	free(x);
	free(b);
	free(A);
	free(iA);
	free(jA);


    return status;
}

