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


#define CLEANUP()                       \
do {                                    \
    if (x)          free (x);           \
    if (f)          free (f);           \
    if (r)          free (r);           \
    if (rw)         free (rw);          \
    if (p)          free (p);           \
    if (pw)         free (pw);          \
    if (s)          free (s);           \
    if (t)          free (t);           \
    if (v)          free (v);           \
    if (tx)         free (tx);          \
    if (Aval)       free(Aval);         \
    if (AcolsIndex) free(AcolsIndex);   \
    if (ArowsIndex) free(ArowsIndex);   \
    if (Mval)       free(Mval);         \
    if (devPtrX)    checkCudaErrors(cudaFree (devPtrX));                    \
    if (devPtrF)    checkCudaErrors(cudaFree (devPtrF));                    \
    if (devPtrR)    checkCudaErrors(cudaFree (devPtrR));                    \
    if (devPtrRW)   checkCudaErrors(cudaFree (devPtrRW));                   \
    if (devPtrP)    checkCudaErrors(cudaFree (devPtrP));                    \
    if (devPtrS)    checkCudaErrors(cudaFree (devPtrS));                    \
    if (devPtrT)    checkCudaErrors(cudaFree (devPtrT));                    \
    if (devPtrV)    checkCudaErrors(cudaFree (devPtrV));                    \
    if (devPtrAval) checkCudaErrors(cudaFree (devPtrAval));                 \
    if (devPtrAcolsIndex) checkCudaErrors(cudaFree (devPtrAcolsIndex));     \
    if (devPtrArowsIndex) checkCudaErrors(cudaFree (devPtrArowsIndex));     \
    if (devPtrMval)       checkCudaErrors(cudaFree (devPtrMval));           \
    if (stream)           checkCudaErrors(cudaStreamDestroy(stream));       \
    if (cublasHandle)     checkCudaErrors(cublasDestroy(cublasHandle));     \
    if (cusparseHandle)   checkCudaErrors(cusparseDestroy(cusparseHandle)); \
    fflush (stdout);                                    \
} while (0)


static void gpu_pbicgstab(cublasHandle_t cublasHandle, cusparseHandle_t cusparseHandle, int m, int n, int nnz, 
                          const cusparseMatDescr_t descra, /* the coefficient matrix in CSR format */  
                          double *a, int *ia, int *ja, 
                          const cusparseMatDescr_t descrm, /* the preconditioner in CSR format, lower & upper triangular factor */
                          double *vm, int *im, int *jm, 
                          cusparseSolveAnalysisInfo_t info_l, cusparseSolveAnalysisInfo_t info_u, /* the analysis of the lower and upper triangular parts */
                          double *f, double *r, double *rw, double *p, double *pw, double *s, double *t, double *v, double *x, 
                          int maxit, double tol, double ttt_sv)
{
    double rho, rhop, beta, alpha, negalpha, omega, negomega, temp, temp2;
    double nrmr, nrmr0;
    rho = 0.0;
    double zero = 0.0;
    double one  = 1.0;
    double mone = -1.0;
    int i=0;
    int j=0;
    double ttl,ttl2,ttu,ttu2,ttm,ttm2;
    double ttt_mv=0.0;

    //WARNING: Analysis is done outside of the function (and the time taken by it is passed to the function in variable ttt_sv)

    //compute initial residual r0=b-Ax0 (using initial guess in x)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
    checkCudaErrors(cudaDeviceSynchronize());
    ttm = second();
#endif

    checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, x, &zero, r));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
    cudaDeviceSynchronize();
    ttm2= second();
    ttt_mv += (ttm2-ttm);
    //printf("matvec %f (s)\n",ttm2-ttm);
#endif
    checkCudaErrors(cublasDscal(cublasHandle, n, &mone, r, 1));
    checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, f, 1, r, 1));
    //copy residual r into r^{\hat} and p
    checkCudaErrors(cublasDcopy(cublasHandle, n, r, 1, rw, 1));
    checkCudaErrors(cublasDcopy(cublasHandle, n, r, 1, p, 1)); 
    checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr0));
    //printf("gpu, init residual:norm %20.16f\n",nrmr0); 

    for (i=0; i<maxit; ){
        rhop = rho;
        checkCudaErrors(cublasDdot(cublasHandle, n, rw, 1, r, 1, &rho));

        if (i > 0){
            beta= (rho/rhop) * (alpha/omega);
            negomega = -omega;
            checkCudaErrors(cublasDaxpy(cublasHandle,n, &negomega, v, 1, p, 1));
            checkCudaErrors(cublasDscal(cublasHandle,n, &beta, p, 1));
            checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, r, 1, p, 1));
        }
        //preconditioning step (lower and upper triangular solve)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        checkCudaErrors(cudaDeviceSynchronize());
        ttl  = second();
#endif
        checkCudaErrors(cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_LOWER));
        checkCudaErrors(cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_UNIT));
        checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,&one,descrm,vm,im,jm,info_l,p,t));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        checkCudaErrors(cudaDeviceSynchronize());
        ttl2 = second();
        ttu  = second();
#endif
        checkCudaErrors(cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_UPPER));
        checkCudaErrors(cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_NON_UNIT));
        checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,&one,descrm,vm,im,jm,info_u,t,pw));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        checkCudaErrors(cudaDeviceSynchronize());
        ttu2 = second();
        ttt_sv += (ttl2-ttl)+(ttu2-ttu);
        //printf("solve lower %f (s), upper %f (s) \n",ttl2-ttl,ttu2-ttu);
#endif

        //matrix-vector multiplication
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        checkCudaErrors(cudaDeviceSynchronize());
        ttm = second();
#endif

        checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, pw, &zero, v));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        checkCudaErrors(cudaDeviceSynchronize());
        ttm2= second();
        ttt_mv += (ttm2-ttm);
        //printf("matvec %f (s)\n",ttm2-ttm);
#endif

        checkCudaErrors(cublasDdot(cublasHandle,n, rw, 1, v, 1,&temp));
        alpha= rho / temp;
        negalpha = -(alpha);
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &negalpha, v, 1, r, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &alpha,        pw, 1, x, 1));
        checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr));

        if (nrmr < tol*nrmr0){
            j=5;
            break;
        }

        //preconditioning step (lower and upper triangular solve)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        checkCudaErrors(cudaDeviceSynchronize());
        ttl  = second();
#endif
        checkCudaErrors(cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_LOWER));
        checkCudaErrors(cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_UNIT));
        checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n, &one,descrm,vm,im,jm,info_l,r,t));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        checkCudaErrors(cudaDeviceSynchronize());
        ttl2 = second();
        ttu  = second();
#endif
        checkCudaErrors(cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_UPPER));
        checkCudaErrors(cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_NON_UNIT));
        checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n, &one,descrm,vm,im,jm,info_u,t,s));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        checkCudaErrors(cudaDeviceSynchronize());
        ttu2 = second();
        ttt_sv += (ttl2-ttl)+(ttu2-ttu);
        //printf("solve lower %f (s), upper %f (s) \n",ttl2-ttl,ttu2-ttu);
#endif
        //matrix-vector multiplication
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        checkCudaErrors(cudaDeviceSynchronize());
        ttm = second();
#endif

        checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, s, &zero, t));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        checkCudaErrors(cudaDeviceSynchronize());
        ttm2= second();
        ttt_mv += (ttm2-ttm);
        //printf("matvec %f (s)\n",ttm2-ttm);
#endif

        checkCudaErrors(cublasDdot(cublasHandle,n, t, 1, r, 1,&temp));
        checkCudaErrors(cublasDdot(cublasHandle,n, t, 1, t, 1,&temp2));
        omega= temp / temp2;
        negomega = -(omega);
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &omega, s, 1, x, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &negomega, t, 1, r, 1));

        checkCudaErrors(cublasDnrm2(cublasHandle,n, r, 1,&nrmr));

        if (nrmr < tol*nrmr0){
            i++;
            j=0;
            break;
        }
        i++;
    }  

#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
    printf("gpu total solve time %f (s), matvec time %f (s)\n",ttt_sv,ttt_mv);
#endif
}

double rand_float_0_1() {
	float r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

	return r;
}

double rand_float(double min, double max) {
	double norm = rand_float_0_1();

	return norm * (max - min) + min;
}

//base 1 !
int gen_rand_csr_matrix(int n, int m, std::vector<double> *A, std::vector<int> *IA, std::vector<int> *JA, double probability_of_zero, double min, double max) {
	IA->push_back(1);//base 1
	int row_count = 1; //NNZ, base 1
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			bool zero = rand_float_0_1() <= probability_of_zero;
			if (!zero) {
				auto r = rand_float(min, max);
				if (r != 0.0F) {
					A->push_back(r);
					JA->push_back(j + 1);//base 1
					row_count += 1;

					std::cout << "i=" << (i + 1) << " j=" << (j + 1) << " " << r << std::endl;
				}

			}

		}
		IA->push_back(row_count);
	}

	return row_count;
}

void gen_rand_vector(int n, double *vector, double probability_of_zero, double min, double max) {
	for (int i = 0; i < n; ++i) {
		vector[i] = rand_float_0_1() <= probability_of_zero ? 0.0 : rand_float(min, max);
	}
}

void dump_vector(std::ostringstream &stream, int n, double *vector) {
	stream << "(";
	for (int i = 0; i < n; ++i) {
		stream << std::to_string(vector[i]) + " ";
	}
	stream << ")";

}

void toDenseVector(int n, int nnz, double* A, int* IA, double* out) {
	int sum = 0;
	int count = 0;
	for (int i = 0; i < n; ++i) {
		if (IA[i + 1] - sum > 0) {
			out[count++] = A[i];
		}
		else {
			out[count++] = 0.0;
		}
	}
}


int test_bicgstab(int matrixN, int nnz, double* Aval, int* ArowsIndex, int* AcolsIndex, double* b, 
	int debug, double damping, int maxit, double tol, 
                  float err, float eps, bool print){
    cublasHandle_t cublasHandle  = 0;
    cusparseHandle_t cusparseHandle  = 0;
    cusparseMatDescr_t descra= 0;
    cusparseMatDescr_t descrm= 0;
    cudaStream_t stream = 0;
    cusparseSolveAnalysisInfo_t info_l = 0;
    cusparseSolveAnalysisInfo_t info_u = 0;
    cusparseStatus_t status1, status2, status3;
	int matrixM = matrixN;
    double *devPtrAval       = 0;
    int    *devPtrAcolsIndex = 0;  
    int    *devPtrArowsIndex = 0;    
    double *devPtrMval       = 0;
    int    *devPtrMcolsIndex = 0;  
    int    *devPtrMrowsIndex = 0;
    double *devPtrX = 0;
    double *devPtrF = 0;
    double *devPtrR = 0;
    double *devPtrRW= 0;
    double *devPtrP = 0;
    double *devPtrPW= 0;
    double *devPtrS = 0;
    double *devPtrT = 0;
    double *devPtrV = 0;     
    double *Mval       =0;   
    int    *MrowsIndex =0;
    int    *McolsIndex =0;
    double *x  = 0;
    double *tx = 0;
    double *f  = b;
    double *r  = 0;
    double *rw = 0;
    double *p  = 0;
    double *pw = 0;
    double *s  = 0;    
    double *t  = 0;
    double *v  = 0;
    int matrixSizeAval, matrixSizeAcolsIndex, matrixSizeArowsIndex, mSizeAval, mSizeAcolsIndex, mSizeArowsIndex;
    int arraySizeX, arraySizeF, arraySizeR, arraySizeRW, arraySizeP,  arraySizePW, arraySizeS, arraySizeT, arraySizeV, mNNZ;
    long long flops;    
    double start, stop;
    int num_iterations, nbrTests, count, base, mbase;
    cusparseOperation_t trans;
    double alpha;
    double ttt_sv=0.0;

    
    alpha = damping;
    trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

    /* load the coefficient matrix */
    /*if (loadMMSparseMatrix(matrix_filename, *element_type, true, &matrixM, &matrixN, &nnz, &Aval, &ArowsIndex, &AcolsIndex, symmetrize)){
        CLEANUP();
        fprintf (stderr, "!!!! cusparseLoadMMSparseMatrix FAILED\n");
        return EXIT_FAILURE;
    }*/

    matrixSizeAval       = nnz;
    matrixSizeAcolsIndex = matrixSizeAval;
    matrixSizeArowsIndex = matrixM + 1;   
    base                 = ArowsIndex[0];    
    /*if (matrixM != matrixN){
        fprintf( stderr, "!!!! matrix MUST be square, error: m=%d != n=%d\n",matrixM,matrixN);
        return EXIT_FAILURE;
    }*/
    printf( "M=%d, N=%d, nnz=%d\n", matrixM, matrixN, nnz);

	/*for (size_t i = 0; i < matrixN; i++)
	{
		std::cout << b[i] << std::endl;
	}

	std::cout << "----" << std::endl;

	for (size_t i = 0; i < nnz; i++)
	{
		std::cout << Aval[i] << std::endl;
	}

	std::cout << "----" << std::endl;

	for (size_t i = 0; i < matrixN + 1; i++)
	{
		std::cout << ArowsIndex[i] << std::endl;
	}

	std::cout << "----" << std::endl;

	for (size_t i = 0; i < nnz; i++)
	{
		std::cout << AcolsIndex[i] << std::endl;
	}*/

    /* set some extra parameters for lower triangular factor */
    mNNZ            = ArowsIndex[matrixM]-ArowsIndex[0];
    mSizeAval       = mNNZ;
    mSizeAcolsIndex = mSizeAval;
    mSizeArowsIndex = matrixM + 1;   
    mbase           = ArowsIndex[0];    
    
    /* compressed sparse row */
    arraySizeX = matrixN;
    arraySizeF = matrixM;
    arraySizeR = matrixM;
    arraySizeRW= matrixM;
    arraySizeP = matrixN;
    arraySizePW= matrixN;
    arraySizeS = matrixM;
    arraySizeT = matrixM;
    arraySizeV = matrixM;
        
    /* initialize cublas */
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        fprintf( stderr, "!!!! CUBLAS initialization error\n" );
        return EXIT_FAILURE;
    }
    /* initialize cusparse */
    status1 = cusparseCreate(&cusparseHandle);    
    if (status1 != CUSPARSE_STATUS_SUCCESS) {
        fprintf( stderr, "!!!! CUSPARSE initialization error\n" );
        return EXIT_FAILURE;
    }
    /* create three matrix descriptors */
    status1 = cusparseCreateMatDescr(&descra); 
    status2 = cusparseCreateMatDescr(&descrm); 
    if ((status1 != CUSPARSE_STATUS_SUCCESS) ||
        (status2 != CUSPARSE_STATUS_SUCCESS)){
        fprintf( stderr, "!!!! CUSPARSE cusparseCreateMatDescr (coefficient matrix or preconditioner) error\n" );
        return EXIT_FAILURE;
    }

    /* allocate device memory for csr matrix and vectors */
    checkCudaErrors(cudaMalloc ((void**)&devPtrX, sizeof(devPtrX[0]) * arraySizeX));
    checkCudaErrors(cudaMalloc ((void**)&devPtrF, sizeof(devPtrF[0]) * arraySizeF));
    checkCudaErrors(cudaMalloc ((void**)&devPtrR, sizeof(devPtrR[0]) * arraySizeR));
    checkCudaErrors(cudaMalloc ((void**)&devPtrRW,sizeof(devPtrRW[0])* arraySizeRW));
    checkCudaErrors(cudaMalloc ((void**)&devPtrP, sizeof(devPtrP[0]) * arraySizeP));
    checkCudaErrors(cudaMalloc ((void**)&devPtrPW,sizeof(devPtrPW[0])* arraySizePW));
    checkCudaErrors(cudaMalloc ((void**)&devPtrS, sizeof(devPtrS[0]) * arraySizeS));
    checkCudaErrors(cudaMalloc ((void**)&devPtrT, sizeof(devPtrT[0]) * arraySizeT));
    checkCudaErrors(cudaMalloc ((void**)&devPtrV, sizeof(devPtrV[0]) * arraySizeV));
    checkCudaErrors(cudaMalloc ((void**)&devPtrAval, sizeof(devPtrAval[0]) * matrixSizeAval));
    checkCudaErrors(cudaMalloc ((void**)&devPtrAcolsIndex, sizeof(devPtrAcolsIndex[0]) * matrixSizeAcolsIndex));
    checkCudaErrors(cudaMalloc ((void**)&devPtrArowsIndex, sizeof(devPtrArowsIndex[0]) * matrixSizeArowsIndex));
    checkCudaErrors(cudaMalloc ((void**)&devPtrMval, sizeof(devPtrMval[0]) * mSizeAval));

    /* allocate host memory for  vectors */
    x  = (double *)malloc (arraySizeX * sizeof(x[0]));
    //f  = (double *)malloc (arraySizeF * sizeof(f[0]));
    r  = (double *)malloc (arraySizeR * sizeof(r[0]));
    rw = (double *)malloc (arraySizeRW* sizeof(rw[0]));
    p  = (double *)malloc (arraySizeP * sizeof(p[0]));
    pw = (double *)malloc (arraySizePW* sizeof(pw[0]));
    s  = (double *)malloc (arraySizeS * sizeof(s[0]));
    t  = (double *)malloc (arraySizeT * sizeof(t[0]));
    v  = (double *)malloc (arraySizeV * sizeof(v[0]));
    tx = (double *)malloc (arraySizeX * sizeof(tx[0]));
    Mval=(double *)malloc (matrixSizeAval*sizeof(Mval[0]));
    if ((!Aval) || (!AcolsIndex) || (!ArowsIndex) || (!Mval) ||
        (!x) || (!f) || (!r) || (!rw) || (!p) || (!pw) || (!s) || (!t) || (!v) || (!tx)) {
        CLEANUP();
        fprintf (stderr, "!!!! memory allocation error\n");
        return EXIT_FAILURE;
    }
    /* use streams */
    int useStream =0;
    if (useStream) {

        checkCudaErrors(cudaStreamCreate(&stream));

        if (cublasSetStream(cublasHandle, stream) != CUBLAS_STATUS_SUCCESS) {
            CLEANUP();
            fprintf (stderr, "!!!! cannot set CUBLAS stream\n");
            return EXIT_FAILURE;
        }
        status1 = cusparseSetStream(cusparseHandle, stream);        
        if (status1 != CUSPARSE_STATUS_SUCCESS) {
            CLEANUP();
            fprintf (stderr, "!!!! cannot set CUSPARSE stream\n");
            return EXIT_FAILURE;
        }      
    }

    /* clean memory */
    checkCudaErrors(cudaMemset((void *)devPtrX,         0, sizeof(devPtrX[0])          * arraySizeX));
    checkCudaErrors(cudaMemset((void *)devPtrF,         0, sizeof(devPtrF[0])          * arraySizeF));
    checkCudaErrors(cudaMemset((void *)devPtrR,         0, sizeof(devPtrR[0])          * arraySizeR));
    checkCudaErrors(cudaMemset((void *)devPtrRW,        0, sizeof(devPtrRW[0])         * arraySizeRW));
    checkCudaErrors(cudaMemset((void *)devPtrP,         0, sizeof(devPtrP[0])          * arraySizeP));
    checkCudaErrors(cudaMemset((void *)devPtrPW,        0, sizeof(devPtrPW[0])         * arraySizePW));
    checkCudaErrors(cudaMemset((void *)devPtrS,         0, sizeof(devPtrS[0])          * arraySizeS));
    checkCudaErrors(cudaMemset((void *)devPtrT,         0, sizeof(devPtrT[0])          * arraySizeT));
    checkCudaErrors(cudaMemset((void *)devPtrV,         0, sizeof(devPtrV[0])          * arraySizeV));
    checkCudaErrors(cudaMemset((void *)devPtrAval,      0, sizeof(devPtrAval[0])       * matrixSizeAval));
    checkCudaErrors(cudaMemset((void *)devPtrAcolsIndex,0, sizeof(devPtrAcolsIndex[0]) * matrixSizeAcolsIndex));
    checkCudaErrors(cudaMemset((void *)devPtrArowsIndex,0, sizeof(devPtrArowsIndex[0]) * matrixSizeArowsIndex));
    checkCudaErrors(cudaMemset((void *)devPtrMval,      0, sizeof(devPtrMval[0])       * mSizeAval));

    memset(x,         0, arraySizeX           * sizeof(x[0]));
    //memset(f,         0, arraySizeF           * sizeof(f[0]));
    memset(r,         0, arraySizeR           * sizeof(r[0]));
    memset(rw,        0, arraySizeRW          * sizeof(rw[0]));
    memset(p,         0, arraySizeP           * sizeof(p[0]));
    memset(pw,        0, arraySizePW          * sizeof(pw[0]));
    memset(s,         0, arraySizeS           * sizeof(s[0]));
    memset(t,         0, arraySizeT           * sizeof(t[0]));
    memset(v,         0, arraySizeV           * sizeof(v[0]));
    memset(tx,        0, arraySizeX           * sizeof(tx[0]));

    /* create the test matrix and vectors on the host */
    checkCudaErrors(cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL));
    if (base) {
        checkCudaErrors(cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ONE));
    }
    else{
        checkCudaErrors(cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO));
    } 
    checkCudaErrors(cusparseSetMatType(descrm,CUSPARSE_MATRIX_TYPE_GENERAL));
    if (mbase) {
        checkCudaErrors(cusparseSetMatIndexBase(descrm,CUSPARSE_INDEX_BASE_ONE));
    }
    else{
        checkCudaErrors(cusparseSetMatIndexBase(descrm,CUSPARSE_INDEX_BASE_ZERO));
    }

    //compute the right-hand-side f=A*e, where e=[1, ..., 1]'
    for (int i=0; i<arraySizeP; i++) {
        p[i]=1.0;
    }

    /* copy the csr matrix and vectors into device memory */
    double start_matrix_copy, stop_matrix_copy, start_preconditioner_copy, stop_preconditioner_copy;

    start_matrix_copy = second();

    checkCudaErrors(cudaMemcpy (devPtrAval,       Aval,       (size_t)(matrixSizeAval       * sizeof(Aval[0])),       cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrAcolsIndex, AcolsIndex, (size_t)(matrixSizeAcolsIndex * sizeof(AcolsIndex[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrArowsIndex, ArowsIndex, (size_t)(matrixSizeArowsIndex * sizeof(ArowsIndex[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrMval,       devPtrAval, (size_t)(matrixSizeAval       * sizeof(devPtrMval[0])), cudaMemcpyDeviceToDevice));

    stop_matrix_copy = second();

    fprintf (stdout, "Copy matrix from CPU to GPU, time(s) = %10.8f\n",stop_matrix_copy-start_matrix_copy);

    checkCudaErrors(cudaMemcpy (devPtrX, x, (size_t)(arraySizeX * sizeof(devPtrX[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrF, f, (size_t)(arraySizeF * sizeof(devPtrF[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrR, r, (size_t)(arraySizeR * sizeof(devPtrR[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrRW,rw,(size_t)(arraySizeRW* sizeof(devPtrRW[0])),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrP, p, (size_t)(arraySizeP * sizeof(devPtrP[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrPW,pw,(size_t)(arraySizePW* sizeof(devPtrPW[0])),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrS, s, (size_t)(arraySizeS * sizeof(devPtrS[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrT, t, (size_t)(arraySizeT * sizeof(devPtrT[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrV, v, (size_t)(arraySizeV * sizeof(devPtrV[0])), cudaMemcpyHostToDevice));

    /* --- GPU --- */
    /* create the analysis info (for lower and upper triangular factors) */
    checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_l));
    checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_u));

    /* analyse the lower and upper triangular factors */
    double ttl = second();
    checkCudaErrors(cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_LOWER));
    checkCudaErrors(cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_UNIT));
    checkCudaErrors(cusparseDcsrsv_analysis(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,matrixM,nnz,descrm,devPtrAval,devPtrArowsIndex,devPtrAcolsIndex,info_l));
    checkCudaErrors(cudaDeviceSynchronize());
    double ttl2 = second();

    double ttu = second();
    checkCudaErrors(cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_UPPER));
    checkCudaErrors(cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_NON_UNIT));
    checkCudaErrors(cusparseDcsrsv_analysis(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,matrixM,nnz,descrm,devPtrAval,devPtrArowsIndex,devPtrAcolsIndex,info_u));
    checkCudaErrors(cudaDeviceSynchronize());
    double ttu2= second();
    ttt_sv += (ttl2-ttl)+(ttu2-ttu);
    printf("analysis lower %f (s), upper %f (s) \n",ttl2-ttl,ttu2-ttu);

    /* compute the lower and upper triangular factors using CUSPARSE csrilu0 routine (on the GPU) */
    double start_ilu, stop_ilu;
    printf("CUSPARSE csrilu0 ");  
    start_ilu= second();
    devPtrMrowsIndex = devPtrArowsIndex; 
    devPtrMcolsIndex = devPtrAcolsIndex;
    checkCudaErrors(cusparseDcsrilu0(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,matrixM,descra,devPtrMval,devPtrArowsIndex,devPtrAcolsIndex,info_l));
    checkCudaErrors(cudaDeviceSynchronize());
    stop_ilu = second();
    fprintf (stdout, "time(s) = %10.8f \n",stop_ilu-start_ilu);

    /* run the test */
    num_iterations=1; //10; 
    start = second()/num_iterations;
    for (count=0; count<num_iterations; count++) {
        
        gpu_pbicgstab(cublasHandle, cusparseHandle, matrixM, matrixN, nnz,
                              descra, devPtrAval, devPtrArowsIndex, devPtrAcolsIndex,
                              descrm, devPtrMval, devPtrMrowsIndex, devPtrMcolsIndex,
                              info_l, info_u,
                              devPtrF,devPtrR,devPtrRW,devPtrP,devPtrPW,devPtrS,devPtrT,devPtrV,devPtrX, maxit, tol, ttt_sv);

        checkCudaErrors(cudaDeviceSynchronize());
    }
    stop = second()/num_iterations;

    /* destroy the analysis info (for lower and upper triangular factors) */
    checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_l));
    checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_u));

    /* copy the result into host memory */
    checkCudaErrors(cudaMemcpy (tx, devPtrX, (size_t)(arraySizeX * sizeof(tx[0])), cudaMemcpyDeviceToHost));

	if (print) {
		std::ostringstream str;
		dump_vector(str, matrixN, tx);
		std::cout << str.str() << std::endl;
	}

    return EXIT_SUCCESS;
}

