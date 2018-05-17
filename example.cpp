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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
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
    char * matrix_filename = NULL;
	char * vector_filename = NULL;

    int symmetrize=0;
    int debug=0;
    int maxit = 2000; //5; //2000; //1000;  //50; //5; //50; //100; //500; //10000;
    double tol= 0.0000001; //0.000001; //0.00001; //0.00000001; //0.0001; //0.001; //0.00000001; //0.1; //0.001; //0.00000001;
    double damping= 0.75;
	bool random = false;
	double probOfZero = 0.75;
	int dimRand = 3;
	bool print = false;
	
	

    /* WARNING: it is assumed that the matrices are stores in Matrix Market format */
    printf("WARNING: it is assumed that the matrices are stored in Matrix Market format with double as element type\n Usage: ./BiCGStab -M[matrix.mtx] -V[vector.mtx] [-E] [-D] -R[prob of zero] -N[dim] [-P] [device=<num>]\n");

    printf("Starting [%s]\n", argv[0]);
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
            case 'E':
                symmetrize = 1;
                break;     
            case 'D':
                debug = 1;
                break;    
			case 'R':
				random = true;
				probOfZero = std::stod(argv[i] + 2);
				break;
			case 'P':
				print = true;
				break;
			case 'N':
				dimRand = std::stoi(argv[i] + 2);
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

    // Use default input file
    if (matrix_filename == NULL)
    {
        printf("argv[0] = %s", argv[0]);
		//matrix_filename = "gr_900_900_crg.mtx";
		matrix_filename = "mat3.mtx";

        if (matrix_filename != NULL)
        {
            printf("Using default matrix input file [%s]\n", matrix_filename);
        }
        else
        {
            printf("Could not find input file = %s\n", matrix_filename);
            return EXIT_FAILURE;
        }
    }
    else
    {
        printf("Using matrix input file [%s]\n", matrix_filename);
    }

	// Use default input file
	if (vector_filename == NULL)
	{
		printf("argv[0] = %s", argv[0]);
		vector_filename = "vec3.mtx";

		if (vector_filename != NULL)
		{
			printf("Using default vector input file [%s]\n", vector_filename);
		}
		else
		{
			printf("Could not find input file = %s\n", vector_filename);
			return EXIT_FAILURE;
		}
	}
	else
	{
		printf("Using vector input file [%s]\n", vector_filename);
	}

    findCudaDevice(argc, (const char **)argv);

	int matrixM;
	int matrixN;
	int nnz;
	double* Aval;
	int* ArowsIndex;
	int* AcolsIndex;
	double* b = 0;
	

	if (random) {
		std::vector<double> A;
		std::vector<int> IA;
		std::vector<int> JA;

		matrixN = dimRand;
		b = (double*)malloc(sizeof(double) * matrixN);
		memset(b, 0, sizeof(double)*matrixN);
		gen_rand_csr_matrix(matrixN, matrixN, &A, &IA, &JA, probOfZero, -10.0, 10.0);
		gen_rand_vector(matrixN, b, 0.0, -10.0, 10.0);
		nnz = A.size();
		Aval = &A[0];
		ArowsIndex = &IA[0];
		AcolsIndex = &JA[0];

		
		

		status = test_bicgstab(matrixN, nnz, Aval, ArowsIndex, AcolsIndex, b, debug, damping, maxit, tol,
			DBICGSTAB_MAX_ULP_ERR, DBICGSTAB_EPS, print);


		std::cout.flush();
		//_getch();

		free(b);

		return status;
	}
	else {
		if (loadMMSparseMatrix(matrix_filename, 'd', true, &matrixM, &matrixN, &nnz, &Aval, &ArowsIndex, &AcolsIndex, symmetrize)) {
			fprintf(stderr, "!!!! cusparseLoadMMSparseMatrix FAILED\n");
			return EXIT_FAILURE;
		}

		int vN;
		int vM;
		int vnnz;
		double* vA;
		int* vIA;
		int* vJA;

		if (loadMMSparseMatrix(vector_filename, 'd', true, &vM, &vN, &vnnz, &vA, &vIA, &vJA, 0)) {
			fprintf(stderr, "!!!! cusparseLoadMMSparseMatrix FAILED\n");
			return EXIT_FAILURE;
		}



		if (matrixN != matrixM) {
			fprintf(stderr, "Matrix A must be a square matrix !\n");
			return EXIT_FAILURE;
		}

		if (vN != 1) {
			fprintf(stderr, "b must be a vector !\n");
			return EXIT_FAILURE;
		}

		b = (double*)malloc(sizeof(double) * matrixN);
		//memset(b, 0, sizeof(double)*matrixN);

		//gen_rand_vector(matrixN, b, 0.0, -10, 10);

		toDenseVector(vM, vnnz, vA, vIA, b);

		status = test_bicgstab(matrixN, nnz, Aval, ArowsIndex, AcolsIndex, b, debug, damping, maxit, tol,
			DBICGSTAB_MAX_ULP_ERR, DBICGSTAB_EPS, print);

		std::cout.flush();
		//_getch();

		free(b);
		free(vA);
		free(vIA);
		free(vJA);

		return status;


	}



    return 0;
}

