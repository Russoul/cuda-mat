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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <vector>
#include <conio.h>
#include <sstream>
#include <iostream>

  

 //profiling the code
#define TIME_INDIVIDUAL_LIBRARY_CALLS

#define DBICGSTAB_MAX_ULP_ERR   100
#define DBICGSTAB_EPS           1.E-14f //9e-2

double rand_float_0_1();

double rand_float(double min, double max);

int gen_rand_csr_matrix(int n, int m, std::vector<double> *A, std::vector<int> *IA, std::vector<int> *JA, double probability_of_zero, double min, double max);

void gen_rand_vector(int n, double *vector, double probability_of_zero, double min, double max);

void dump_vector(std::ostringstream &stream, int n, double *vector);

void toDenseVector(int n, int nnz, double* A, int* IA, double* out);

int test_bicgstab(int matrixN, int nnz, double* Aval, int* ArowsIndex, int* AcolsIndex, double* b,
	int debug, double damping, int maxit, double tol,
	float err, float eps, bool print);