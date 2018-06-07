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


#include <device_launch_parameters.h>

__global__ void mult_spec(int n, double *a, double*b, double k, double *c){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i >= n) return;

    c[i] = a[i] * b[i] * k;
}


static void gpu_pbicgstab(cublasHandle_t cublasHandle, cusparseHandle_t cusparseHandle, int n, int nnz,
                          const cusparseMatDescr_t descra, /* the coefficient matrix in CSR format */
                          double *a, int *ia, int *ja,
                          const cusparseMatDescr_t descrm, /* the preconditioner in CSR format, lower & upper triangular factor */
                          double *vm, int *im, int *jm,
                          cusparseSolveAnalysisInfo_t info_l, cusparseSolveAnalysisInfo_t info_u, /* the analysis of the lower and upper triangular parts */
                          double *f, double *r, double *rw, double *p, double *pw, double *s, double *t, double *v, double *x,
                          int maxit, double tol, bool debug)
{
    double rho, rhop, beta, alpha, negalpha, omega, negomega, temp, temp2;
    double nrmr, nrmr0;
    rho = 0.0;
    double zero = 0.0;
    double one  = 1.0;
    double mone = -1.0;
    int i=0;

    //WARNING: Analysis is done outside of the function (and the time taken by it is passed to the function in variable ttt_sv)

    //compute initial residual r0=b-Ax0 (using initial guess in x)


    checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, x, &zero, r));

    checkCudaErrors(cublasDscal(cublasHandle, n, &mone, r, 1));
    checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, f, 1, r, 1));
    //copy residual r into r^{\hat} and p
    checkCudaErrors(cublasDcopy(cublasHandle, n, r, 1, rw, 1));
    checkCudaErrors(cublasDcopy(cublasHandle, n, r, 1, p, 1));
    checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr0));

    if(debug)
        printf("gpu, init residual:norm %20.16f\n",nrmr0);

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

        checkCudaErrors(cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_LOWER));
        checkCudaErrors(cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_UNIT));
        checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,&one,descrm,vm,im,jm,info_l,p,t));

        checkCudaErrors(cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_UPPER));
        checkCudaErrors(cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_NON_UNIT));
        checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,&one,descrm,vm,im,jm,info_u,t,pw));


        //matrix-vector multiplication


        checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, pw, &zero, v));

        checkCudaErrors(cublasDdot(cublasHandle,n, rw, 1, v, 1,&temp));
        alpha= rho / temp;
        negalpha = -(alpha);
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &negalpha, v, 1, r, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &alpha,        pw, 1, x, 1));
        checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr));

        if(debug)
            std::cout << "i = " << i << ", residual norm (before precond) = " << nrmr << std::endl;

        if (nrmr < tol*nrmr0){
            break;
        }

        //preconditioning step (lower and upper triangular solve)
        checkCudaErrors(cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_LOWER));
        checkCudaErrors(cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_UNIT));
        checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n, &one,descrm,vm,im,jm,info_l,r,t));

        checkCudaErrors(cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_UPPER));
        checkCudaErrors(cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_NON_UNIT));
        checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n, &one,descrm,vm,im,jm,info_u,t,s));

        //matrix-vector multiplication


        checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, s, &zero, t));


        checkCudaErrors(cublasDdot(cublasHandle,n, t, 1, r, 1,&temp));
        checkCudaErrors(cublasDdot(cublasHandle,n, t, 1, t, 1,&temp2));
        omega= temp / temp2;
        negomega = -(omega);
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &omega, s, 1, x, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &negomega, t, 1, r, 1));

        checkCudaErrors(cublasDnrm2(cublasHandle,n, r, 1,&nrmr));

        if(debug)
            std::cout << "i = " << i << ", residual norm = " << nrmr << std::endl;

        if (nrmr < tol*nrmr0){
            i++;
            break;
        }
        i++;
    }

}


bool bicgstab_lu_precond(int matrixN, int nnz, double* Aval, int* ArowsIndex, int* AcolsIndex, double* b,
                  int maxit, double tol, bool debug, double *res, double *dtAlg){
    cublasHandle_t cublasHandle  = 0;
    cusparseHandle_t cusparseHandle  = 0;
    cusparseMatDescr_t descra= 0;
    cusparseMatDescr_t descrm= 0;
    cusparseSolveAnalysisInfo_t info_l = 0;
    cusparseSolveAnalysisInfo_t info_u = 0;
    cusparseStatus_t status1, status2;
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
    double *x  = 0;
    double *f  = b;
    double *r  = 0;
    double *rw = 0;
    double *p  = 0;
    double *pw = 0;
    double *s  = 0;
    double *t  = 0;
    double *v  = 0;
    int matrixSizeAval, matrixSizeAcolsIndex, matrixSizeArowsIndex, mSizeAval;
    int arraySizeX, arraySizeF, arraySizeR, arraySizeRW, arraySizeP,  arraySizePW, arraySizeS, arraySizeT, arraySizeV, mNNZ;
    int base;




    matrixSizeAval       = nnz;
    matrixSizeAcolsIndex = matrixSizeAval;
    matrixSizeArowsIndex = matrixM + 1;
    base                 = ArowsIndex[0];

    if(debug)
        printf( "N=%d, nnz=%d\n", matrixN, nnz);


    /* set some extra parameters for lower triangular factor */
    mNNZ            = ArowsIndex[matrixM]-ArowsIndex[0];
    mSizeAval       = mNNZ;

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
        return false;
    }
    /* initialize cusparse */
    status1 = cusparseCreate(&cusparseHandle);
    if (status1 != CUSPARSE_STATUS_SUCCESS) {
        fprintf( stderr, "!!!! CUSPARSE initialization error\n" );
        return false;
    }
    /* create three matrix descriptors */
    status1 = cusparseCreateMatDescr(&descra);
    status2 = cusparseCreateMatDescr(&descrm);
    if ((status1 != CUSPARSE_STATUS_SUCCESS) ||
        (status2 != CUSPARSE_STATUS_SUCCESS)){
        fprintf( stderr, "!!!! CUSPARSE cusparseCreateMatDescr (coefficient matrix or preconditioner) error\n" );
        return false;
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

    /* create the test matrix and vectors on the host */
    checkCudaErrors(cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL));
    if (base) {
        checkCudaErrors(cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ONE));
        checkCudaErrors(cusparseSetMatIndexBase(descrm,CUSPARSE_INDEX_BASE_ONE));
    }
    else{
        checkCudaErrors(cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO));
        checkCudaErrors(cusparseSetMatIndexBase(descrm,CUSPARSE_INDEX_BASE_ZERO));
    }
    checkCudaErrors(cusparseSetMatType(descrm,CUSPARSE_MATRIX_TYPE_GENERAL));

    for (int i=0; i<arraySizeX; i++) {
        x[i]=1.0; //x0
    }

    /* copy the csr matrix and vectors into device memory */


    checkCudaErrors(cudaMemcpy (devPtrAval,       Aval,       (size_t)(matrixSizeAval       * sizeof(Aval[0])),       cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrAcolsIndex, AcolsIndex, (size_t)(matrixSizeAcolsIndex * sizeof(AcolsIndex[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrArowsIndex, ArowsIndex, (size_t)(matrixSizeArowsIndex * sizeof(ArowsIndex[0])), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy (devPtrMval,       devPtrAval, (size_t)(matrixSizeAval       * sizeof(devPtrMval[0])), cudaMemcpyDeviceToDevice));


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

    if(debug)
        printf("analysis lower %f (s), upper %f (s) \n",ttl2-ttl,ttu2-ttu);

    /* compute the lower and upper triangular factors using CUSPARSE csrilu0 routine (on the GPU) */
    double start_ilu, stop_ilu;
    if(debug)
        printf("CUSPARSE csrilu0 ");
    start_ilu= second();
    devPtrMrowsIndex = devPtrArowsIndex;
    devPtrMcolsIndex = devPtrAcolsIndex;
    checkCudaErrors(cusparseDcsrilu0(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,matrixM,descra,devPtrMval,devPtrArowsIndex,devPtrAcolsIndex,info_l));
    checkCudaErrors(cudaDeviceSynchronize());
    stop_ilu = second();
    if(debug)
        fprintf (stdout, "time(s) = %10.8f \n",stop_ilu-start_ilu);

    auto t1 = second();
    gpu_pbicgstab(cublasHandle, cusparseHandle, matrixN, nnz,
                  descra, devPtrAval, devPtrArowsIndex, devPtrAcolsIndex,
                  descrm, devPtrMval, devPtrMrowsIndex, devPtrMcolsIndex,
                  info_l, info_u,
                  devPtrF,devPtrR,devPtrRW,devPtrP,devPtrPW,devPtrS,devPtrT,devPtrV,devPtrX, maxit, tol, debug);

    checkCudaErrors(cudaDeviceSynchronize());
    auto t2 = second();
    *dtAlg = t2 - t1;

    /* destroy the analysis info (for lower and upper triangular factors) */
    checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_l));
    checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_u));

    /* copy the result into host memory */
    checkCudaErrors(cudaMemcpy (res, devPtrX, (size_t)(arraySizeX * sizeof(double)), cudaMemcpyDeviceToHost));


    free (x);
    free (r);
    free (rw);
    free (p);
    free (pw);
    free (s);
    free (t);
    free (v);
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
    if (cublasHandle)     checkCudaErrors(cublasDestroy(cublasHandle));     \
    if (cusparseHandle)   checkCudaErrors(cusparseDestroy(cusparseHandle)); \


    return true;
}



double rand_float_0_1() {
    double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

    return r;
}

double rand_float(double min, double max) {
    double norm = rand_float_0_1();

    return norm * (max - min) + min;
}

static bool gpu_pbicgstab2(
        cublasHandle_t cublasHandle,
        cusparseHandle_t cusparseHandle,
        int n,
        int nnz,
        const cusparseMatDescr_t descrA,
	    double *A,
        int *iA,
        int *jA,
        double* x0,
        double* b,

        int maxit,
        double tol,
        bool debug,

        double* x, //out

        /*work, all zeroed*/
        double* r0,
        double* r,
        double* r_,
        double* v,
        double* v_,
        double* p,
        double* p_,
        double* s,
        double* t,
        double* h){


	//v = v_ = p = p_ = [0,0,0....0]
	double omega = 1;
	double alpha = 1;
	double beta = 0;
	double rho = 1;
	double rho_ = rho;

	double norm0;

	double one = 1;
	double mone = -1.0;
	double zero = 0;

	checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &mone, descrA, A, iA, jA, x0, &zero, r));

   /* double r_host[3];
    cudaMemcpy(r_host, r, 3 * sizeof(double), cudaMemcpyDeviceToHost);
    std::ostringstream st;
    dump_vector(st, 3, r_host);
    std::cout << st.str() << std::endl;

    checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, b, 1, r, 1));
	checkCudaErrors(cudaMemcpy(r0, r, sizeof(double) * n, cudaMemcpyDeviceToDevice));*/


	checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &norm0));


	if(debug){
        std::cout << "initial norm = " << norm0 << std::endl;
	}

	for(size_t i = 0; i < maxit; i++)
	{

        checkCudaErrors(cublasDdot(cublasHandle, n, r0, 1, r, 1, &rho_));
        beta = (rho_ / rho) * (alpha / omega);
        double momega = -omega;
        checkCudaErrors(cudaMemcpy(p_, v, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cublasDscal(cublasHandle, n, &momega, p_, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, p, 1, p_, 1));
        checkCudaErrors(cublasDscal(cublasHandle, n, &beta, p_, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, r, 1, p_, 1));


		checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descrA, A, iA, jA, p_, &zero, v_));

       /* double *temp = static_cast<double *>(malloc(sizeof(double) * n));
        cudaMemcpy(temp, v_, sizeof(double)*n, cudaMemcpyDeviceToHost);
        std::cout << "temp" << std::endl;

        std::ostringstream ss;
        dump_vector(ss, n, temp);
        std::cout << ss.str() << std::endl;*/

		double dot_r_v;

		checkCudaErrors(cublasDdot(cublasHandle, n, r0, 1, v_, 1, &dot_r_v));
		alpha = rho_ / dot_r_v;
		double malpha = -alpha;

		//std::cout << "alpha=" << alpha << std::endl;

        checkCudaErrors(cudaMemcpy(h, p_, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cublasDscal(cublasHandle, n, &alpha, h, 1));
		checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, x0, 1, h, 1));

		checkCudaErrors(cudaMemcpy(s, v_, sizeof(double) * n, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cublasDscal(cublasHandle, n, &malpha, s, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, r, 1, s, 1));


		checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descrA, A, iA, jA, s, &zero, t));

		double num;
		double denum;
		checkCudaErrors(cublasDdot(cublasHandle, n, t, 1, s, 1, &num));
		checkCudaErrors(cublasDdot(cublasHandle, n, t, 1, t, 1, &denum));
		omega = num / denum;
		momega = -omega;


        checkCudaErrors(cudaMemcpy(x, s, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cublasDscal(cublasHandle, n, &omega, x, 1));
		checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, h, 1, x, 1));

        checkCudaErrors(cudaMemcpy(r_, t, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cublasDscal(cublasHandle, n, &momega, r_, 1));
		checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, s, 1, r_, 1));

		double norm;
		checkCudaErrors(cublasDnrm2(cublasHandle, n, r_, 1, &norm));


        if(debug){
            std::cout << "k = " << i << ", norm = " << norm << std::endl;
        }

		if(norm < tol * norm0){
			return true;
		}


		if(abs(omega) < 1e-5 || isnan(omega)){
            if(debug){
                std::cout << "omega is close to zero, cannot continue" << std::endl;
                std::cout << "omega = " << omega << std::endl;
            }

            return false;
        }

		checkCudaErrors(cudaMemcpy(r, r_, sizeof(double) * n, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(p, p_, sizeof(double) * n, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(v, v_, sizeof(double) * n, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(x0, x, sizeof(double) * n, cudaMemcpyDeviceToDevice));
		rho = rho_;
	}

	return false;


}

//TODO stupid code duplication
static bool gpu_pbicgstab2(
        cublasHandle_t cublasHandle,
        cusparseHandle_t cusparseHandle,
        int n,
        int nnz,
        const cusparseMatDescr_t descrA,
        double *A0,
        int *iA0,
        int *jA0,
        double *d,
        double* x0,
        double* b,

        int maxit,
        double tol,
        bool debug,

        double* x, //out

        /*work, all zeroed*/
        double* r0,
        double* r,
        double* r_,
        double* v,
        double* v_,
        double* p,
        double* p_,
        double* s,
        double* t,
        double* h){


    //v = v_ = p = p_ = [0,0,0....0]
    double omega = 1;
    double alpha = 1;
    double beta = 0;
    double rho = 1;
    double rho_ = rho;

    double norm0;

    double one = 1;
    double mone = -1.0;
    double zero = 0;

    const int blockDim = 512;

    int blockCount;
    if(n % blockDim == 0){
        blockCount = n / blockDim;
    }else{
        blockCount = n / blockDim + 1;
    }





//    double r_host[3];
//    cudaMemcpy(r_host, r, 3 * sizeof(double), cudaMemcpyDeviceToHost);
//    std::ostringstream st;
//    dump_vector(st, 3, r_host);
//    std::cout << st.str() << std::endl;

    mult_spec<<<blockDim, blockCount>>>(n, x0, d, -1, r);
    checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &mone, descrA, A0, iA0, jA0, x0, &one, r));


    checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, b, 1, r, 1));


    checkCudaErrors(cudaMemcpy(r0, r, sizeof(double) * n, cudaMemcpyDeviceToDevice));


    checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &norm0));


    if(debug){
        std::cout << "initial norm = " << norm0 << std::endl;
    }

    for(size_t i = 0; i < maxit; i++)
    {

        checkCudaErrors(cublasDdot(cublasHandle, n, r0, 1, r, 1, &rho_));
        beta = (rho_ / rho) * (alpha / omega);
        double momega = -omega;
        checkCudaErrors(cudaMemcpy(p_, v, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cublasDscal(cublasHandle, n, &momega, p_, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, p, 1, p_, 1));
        checkCudaErrors(cublasDscal(cublasHandle, n, &beta, p_, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, r, 1, p_, 1));


        mult_spec<<<blockDim, blockCount>>>(n, p_, d, 1, v_);
        checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descrA, A0, iA0, jA0, p_, &one, v_));

        /* double *temp = static_cast<double *>(malloc(sizeof(double) * n));
         cudaMemcpy(temp, v_, sizeof(double)*n, cudaMemcpyDeviceToHost);
         std::cout << "temp" << std::endl;

         std::ostringstream ss;
         dump_vector(ss, n, temp);
         std::cout << ss.str() << std::endl;*/

        double dot_r_v;

        checkCudaErrors(cublasDdot(cublasHandle, n, r0, 1, v_, 1, &dot_r_v));
        alpha = rho_ / dot_r_v;
        double malpha = -alpha;

        //std::cout << "alpha=" << alpha << std::endl;

        checkCudaErrors(cudaMemcpy(h, p_, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cublasDscal(cublasHandle, n, &alpha, h, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, x0, 1, h, 1));

        checkCudaErrors(cudaMemcpy(s, v_, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cublasDscal(cublasHandle, n, &malpha, s, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, r, 1, s, 1));


        mult_spec<<<blockDim, blockCount>>>(n, s, d, 1, t);
        checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descrA, A0, iA0, jA0, s, &one, t));

        double num;
        double denum;
        checkCudaErrors(cublasDdot(cublasHandle, n, t, 1, s, 1, &num));
        checkCudaErrors(cublasDdot(cublasHandle, n, t, 1, t, 1, &denum));
        omega = num / denum;
        momega = -omega;


        checkCudaErrors(cudaMemcpy(x, s, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cublasDscal(cublasHandle, n, &omega, x, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, h, 1, x, 1));

        checkCudaErrors(cudaMemcpy(r_, t, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cublasDscal(cublasHandle, n, &momega, r_, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, s, 1, r_, 1));

        double norm;
        checkCudaErrors(cublasDnrm2(cublasHandle, n, r_, 1, &norm));


        if(debug){
            std::cout << "k = " << i << ", norm = " << norm << std::endl;
        }

        if(norm < tol * norm0){
            return true;
        }


        if(abs(omega) < 1e-5 || isnan(omega)){
            if(debug){
                std::cout << "omega is close to zero, cannot continue" << std::endl;
                std::cout << "omega = " << omega << std::endl;
            }

            return false;
        }

        checkCudaErrors(cudaMemcpy(r, r_, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(p, p_, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(v, v_, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(x0, x, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        rho = rho_;
    }

    return false;


}

bool bicgstab(int n, int nnz, double *A, int *iA, int *jA, double *b, int maxit, double tol, bool debug, double *x, double *dtAlg){


    cublasHandle_t cublasHandle  = nullptr;
    cusparseHandle_t cusparseHandle  = nullptr;
    cusparseMatDescr_t descrA= nullptr;
    cusparseStatus_t status1;
    double *dev_A       = nullptr;
    int    *dev_iA = nullptr;
    int    *dev_jA = nullptr;
    double *dev_x0 = nullptr;
    double *dev_b = nullptr;
    double *dev_x = nullptr;
    double *dev_r0 = nullptr;
    double *dev_r = nullptr;
    double *dev_r_ = nullptr;
    double *dev_v = nullptr;
    double *dev_v_ = nullptr;
    double *dev_p = nullptr;
    double *dev_p_ = nullptr;
    double *dev_s = nullptr;
    double *dev_t = nullptr;
    double *dev_h = nullptr;



    int base = iA[0];

    /* initialize cublas */
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        fprintf( stderr, "!!!! CUBLAS initialization error\n" );
        return false;
    }
    /* initialize cusparse */
    status1 = cusparseCreate(&cusparseHandle);
    if (status1 != CUSPARSE_STATUS_SUCCESS) {
        fprintf( stderr, "!!!! CUSPARSE initialization error\n" );
        return false;
    }
    /* create three matrix descriptors */
    status1 = cusparseCreateMatDescr(&descrA);
    if ((status1 != CUSPARSE_STATUS_SUCCESS)){
        fprintf( stderr, "!!!! CUSPARSE cusparseCreateMatDescr (coefficient matrix) error\n" );
        return false;
    }

    /* allocate device memory for csr matrix and vectors */
    checkCudaErrors(cudaMalloc ((void**)&dev_A,  sizeof(double) * nnz));
    checkCudaErrors(cudaMalloc ((void**)&dev_iA, sizeof(int) * (n + 1)));
    checkCudaErrors(cudaMalloc ((void**)&dev_jA, sizeof(int) * nnz));
    checkCudaErrors(cudaMalloc ((void**)&dev_x0, sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_b,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_x,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_r0, sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_r,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_r_, sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_v,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_v_, sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_p,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_p_, sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_s,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_t,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_h,  sizeof(double) * n));


    checkCudaErrors(cudaMemcpy(dev_A, A, sizeof(double) * nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_iA, iA, sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_jA, jA, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_b, b, sizeof(double) * n, cudaMemcpyHostToDevice));


    double *x0 = new double[n]; //x0 = [1,1,1,...1]
    for (int i = 0; i < n; ++i) {
        x0[i] = 1;
    }
    checkCudaErrors(cudaMemcpy(dev_x0, x0, sizeof(double) * n, cudaMemcpyHostToDevice));
    delete[] x0;


    //checkCudaErrors(cudaMemset((void *)dev_x0,          0, sizeof(double)* n)); //x0 is zero vector
    checkCudaErrors(cudaMemset((void *)dev_x,           0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_r0,          0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_r,           0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_r_,          0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_v,           0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_v_,          0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_p,           0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_p_,          0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_s,           0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_t,           0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_h,           0, sizeof(double)* n));



    /* create the test matrix and vectors on the host */
    checkCudaErrors(cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL));
    if (base) {
        checkCudaErrors(cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ONE));
    }
    else{
        checkCudaErrors(cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO));
    }


    auto t1 = second();
    auto res = gpu_pbicgstab2(cublasHandle,
                              cusparseHandle,
                              n,
                              nnz,
                              descrA,
                              dev_A,
                              dev_iA,
                              dev_jA,
                              dev_x0,
                              dev_b,
                              maxit,
                              tol,
                              debug,
                              dev_x,
                              dev_r0,
                              dev_r,
                              dev_r_,
                              dev_v,
                              dev_v_,
                              dev_p,
                              dev_p_,
                              dev_s,
                              dev_t,
                              dev_h
    );
    auto t2 = second();
    *dtAlg = t2 - t1;

    checkCudaErrors(cudaDeviceSynchronize());


    /* copy the result into host memory */
    checkCudaErrors(cudaMemcpy (x, dev_x, sizeof(double) * n, cudaMemcpyDeviceToHost));


    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);



    checkCudaErrors(cudaFree(dev_A));
    checkCudaErrors(cudaFree(dev_iA));
    checkCudaErrors(cudaFree(dev_jA));
    checkCudaErrors(cudaFree(dev_x0));
    checkCudaErrors(cudaFree(dev_b));
    checkCudaErrors(cudaFree(dev_x));
    checkCudaErrors(cudaFree(dev_r0));
    checkCudaErrors(cudaFree(dev_r));
    checkCudaErrors(cudaFree(dev_r_));
    checkCudaErrors(cudaFree(dev_v));
    checkCudaErrors(cudaFree(dev_v_));
    checkCudaErrors(cudaFree(dev_p));
    checkCudaErrors(cudaFree(dev_p_));
    checkCudaErrors(cudaFree(dev_s));
    checkCudaErrors(cudaFree(dev_t));
    checkCudaErrors(cudaFree(dev_h));



    return res;
}


//TODO another stupid duplication
bool bicgstab(int n, int nnz, double *A0, int *iA0, int *jA0, double *d, double *x0, double *b, int maxit, double tol, bool debug, double *x, double *dtAlg){


    cublasHandle_t cublasHandle  = nullptr;
    cusparseHandle_t cusparseHandle  = nullptr;
    cusparseMatDescr_t descrA= nullptr;
    cusparseStatus_t status1;
    double *dev_A       = nullptr;
    int    *dev_iA = nullptr;
    int    *dev_jA = nullptr;
    double *dev_x0 = nullptr;
    double *dev_b = nullptr;
    double *dev_x = nullptr;
    double *dev_r0 = nullptr;
    double *dev_r = nullptr;
    double *dev_r_ = nullptr;
    double *dev_v = nullptr;
    double *dev_v_ = nullptr;
    double *dev_p = nullptr;
    double *dev_p_ = nullptr;
    double *dev_s = nullptr;
    double *dev_t = nullptr;
    double *dev_h = nullptr;
    double *dev_d = nullptr;



    int base = iA0[0];

    /* initialize cublas */
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        fprintf( stderr, "!!!! CUBLAS initialization error\n" );
        return false;
    }
    /* initialize cusparse */
    status1 = cusparseCreate(&cusparseHandle);
    if (status1 != CUSPARSE_STATUS_SUCCESS) {
        fprintf( stderr, "!!!! CUSPARSE initialization error\n" );
        return false;
    }
    /* create three matrix descriptors */
    status1 = cusparseCreateMatDescr(&descrA);
    if ((status1 != CUSPARSE_STATUS_SUCCESS)){
        fprintf( stderr, "!!!! CUSPARSE cusparseCreateMatDescr (coefficient matrix) error\n" );
        return false;
    }

    /* allocate device memory for csr matrix and vectors */
    checkCudaErrors(cudaMalloc ((void**)&dev_A,  sizeof(double) * nnz));
    checkCudaErrors(cudaMalloc ((void**)&dev_iA, sizeof(int) * (n + 1)));
    checkCudaErrors(cudaMalloc ((void**)&dev_jA, sizeof(int) * nnz));
    checkCudaErrors(cudaMalloc ((void**)&dev_x0, sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_b,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_x,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_r0, sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_r,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_r_, sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_v,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_v_, sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_p,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_p_, sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_s,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_t,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_h,  sizeof(double) * n));
    checkCudaErrors(cudaMalloc ((void**)&dev_d,  sizeof(double) * n));

    checkCudaErrors(cudaMemcpy(dev_A, A0, sizeof(double) * nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_iA, iA0, sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_jA, jA0, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_b, b, sizeof(double) * n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_d, d, sizeof(double) * n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_x0, x0, sizeof(double) * n, cudaMemcpyHostToDevice));


    //checkCudaErrors(cudaMemset((void *)dev_x0,          0, sizeof(double)* n)); //x0 is zero vector
    checkCudaErrors(cudaMemset((void *)dev_x,           0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_r0,          0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_r,           0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_r_,          0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_v,           0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_v_,          0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_p,           0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_p_,          0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_s,           0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_t,           0, sizeof(double)* n));
    checkCudaErrors(cudaMemset((void *)dev_h,           0, sizeof(double)* n));



    /* create the test matrix and vectors on the host */
    checkCudaErrors(cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL));
    if (base) {
        checkCudaErrors(cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ONE));
    }
    else{
        checkCudaErrors(cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO));
    }


    auto t1 = second();
    auto res = gpu_pbicgstab2(cublasHandle,
                              cusparseHandle,
                              n,
                              nnz,
                              descrA,
                              dev_A,
                              dev_iA,
                              dev_jA,
                              dev_d,
                              dev_x0,
                              dev_b,
                              maxit,
                              tol,
                              debug,
                              dev_x,
                              dev_r0,
                              dev_r,
                              dev_r_,
                              dev_v,
                              dev_v_,
                              dev_p,
                              dev_p_,
                              dev_s,
                              dev_t,
                              dev_h
    );
    auto t2 = second();
    *dtAlg = t2 - t1;

    checkCudaErrors(cudaDeviceSynchronize());


    /* copy the result into host memory */
    checkCudaErrors(cudaMemcpy (x, dev_x, sizeof(double) * n, cudaMemcpyDeviceToHost));


    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);



    checkCudaErrors(cudaFree(dev_A));
    checkCudaErrors(cudaFree(dev_iA));
    checkCudaErrors(cudaFree(dev_jA));
    checkCudaErrors(cudaFree(dev_x0));
    checkCudaErrors(cudaFree(dev_b));
    checkCudaErrors(cudaFree(dev_x));
    checkCudaErrors(cudaFree(dev_r0));
    checkCudaErrors(cudaFree(dev_r));
    checkCudaErrors(cudaFree(dev_r_));
    checkCudaErrors(cudaFree(dev_v));
    checkCudaErrors(cudaFree(dev_v_));
    checkCudaErrors(cudaFree(dev_p));
    checkCudaErrors(cudaFree(dev_p_));
    checkCudaErrors(cudaFree(dev_s));
    checkCudaErrors(cudaFree(dev_t));
    checkCudaErrors(cudaFree(dev_h));



    return res;
}




void gen_rand_vector(int n, double *vector, double probability_of_zero, double min, double max) {
	for (int i = 0; i < n; ++i) {
		vector[i] = rand_float_0_1() <= probability_of_zero ? 0.0 : rand_float(min, max);
	}
}



void toDenseVector(int n, int nnz, double* A, int* IA, double* out) {
    int sum = IA[0]; //base
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (IA[i + 1] - sum > 0) {
            out[i] = A[count++];
            sum = IA[i + 1];

        }
        else {
            out[i] = 0.0;
        }

    }
}






