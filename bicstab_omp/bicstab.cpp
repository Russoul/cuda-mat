#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <memory.h>

#define BICG_OK 0
#define FLOAT_TYPE double
#define EPSILON 0.000001

typedef struct CrsMatrix {
    int N; // Размер матрицы (N x N)
    int NZ; // Кол-во ненулевых элементов
    FLOAT_TYPE *Value; // Массив значений (размер NZ)
    int *Col; // Массив номеров столбцов (размер NZ)
    int *RowIndex; // Массив индексов строк (размер N +1)
} crsMatrix;

int FreeMatrix(crsMatrix &mtx) {
    mtx.N = 0;
    mtx.NZ = 0;
    delete[] mtx.Value;
    delete[] mtx.Col;
    delete[] mtx.RowIndex;
}

int InitializeMatrix(int N, int NZ, crsMatrix &mtx) {
    mtx.Value = new FLOAT_TYPE[NZ];
    mtx.Col = new int[NZ];
    mtx.RowIndex = new int[N + 1];
    mtx.N = N;
    mtx.NZ = NZ;
}

double Transpose2(crsMatrix A, crsMatrix &AT) {
    
    int i, j, N = A.N, S, tmp, j1, j2, Col, V, RIndex, IIndex;
    
    InitializeMatrix(A.N, A.NZ, AT);
    
    memset(AT.RowIndex, 0, (N + 1) * sizeof(int));
    for (i = 0; i < A.NZ; i++)
        AT.RowIndex[A.Col[i] + 1]++;
    
    S = 0;
    for (i = 1; i <= A.N; i++) {
        tmp = AT.RowIndex[i];
        AT.RowIndex[i] = S;
        S = S + tmp;
    }
    
    for (i = 0; i < A.N; i++) {
        j1 = A.RowIndex[i];
        j2 = A.RowIndex[i + 1];
        Col = i; // Столбец в AT - строка в А
        for (j = j1; j < j2; j++) {
            V = A.Value[j]; // Значение
            RIndex = A.Col[j]; // Строка в AT
            IIndex = AT.RowIndex[RIndex + 1];
            AT.Value[IIndex] = V;
            AT.Col[IIndex] = Col;
            AT.RowIndex[RIndex + 1]++;
        }
    }
    return 0;
}


int MatrixVectorMult(crsMatrix A, FLOAT_TYPE *x, FLOAT_TYPE *b) {

#pragma omp parallel for
    for (int i = 0; i < A.N; i++) {
        b[i] = 0.0;
        for (int j = A.RowIndex[i]; j < A.RowIndex[i + 1]; j++) {
            b[i] += A.Value[j] * x[A.Col[j]];
        }
    }
    
    return BICG_OK;
}


double scalarProduct(int n, FLOAT_TYPE *a, FLOAT_TYPE *b) {
    FLOAT_TYPE sum = 0.0;
    int i;
#pragma omp parallel for reduction(+:sum)
    for (i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

int BiCG(crsMatrix A, FLOAT_TYPE *b, FLOAT_TYPE *x, int
CountIteration, int &iter) {
    // Для ускорения вычислений вычислим
    // транспонированную матрицу А
    crsMatrix At;
    
    Transpose2(A, At);
    // массивы для хнанения невязки
    // текущего и следующего приближения
    FLOAT_TYPE *R, *biR;
    FLOAT_TYPE *nR, *nbiR;
    
    R = new FLOAT_TYPE[A.N];
    biR = new FLOAT_TYPE[A.N];
    nR = new FLOAT_TYPE[A.N];
    nbiR = new FLOAT_TYPE[A.N];
    
    // массивы для хранения текущего и следующего вектора
    // направления шага метода
    FLOAT_TYPE *P, *biP;
    FLOAT_TYPE *nP, *nbiP;
    
    P = new FLOAT_TYPE[A.N];
    biP = new FLOAT_TYPE[A.N];
    nP = new FLOAT_TYPE[A.N];
    nbiP = new FLOAT_TYPE[A.N];
    // указатель, для смены указателей на вектора текущего
    // и следующего шага метода
    FLOAT_TYPE *tmp;
    
    // массивы для хранения произведения матрицы на вектор
    //направления и бисопряженный к нему
    FLOAT_TYPE *multAP, *multAtbiP;
    multAP = new FLOAT_TYPE[A.N];
    multAtbiP = new FLOAT_TYPE[A.N];
    // beta и alfa - коэффициенты расчетных формул
    FLOAT_TYPE alfa, beta;
    // числитель и знаменатель коэффициентов beta и alfa
    FLOAT_TYPE numerator, denominator;
    // переменные для вычисления
    // точности текущего приближения
    FLOAT_TYPE check, norm;
    norm = sqrt(scalarProduct(A.N, b, b));
    //задание начального приближения
    int i;
    int n = A.N;
    for (i = 0; i < n; i++)
        x[i] = 1.0;
    //инициализация метода
    MatrixVectorMult(A, x, multAP);
    for (i = 0; i < n; i++)
        R[i] = biR[i] = P[i] = biP[i] = b[i] - multAP[i];
    // реализация метода
    for (iter = 0; iter < CountIteration; iter++) {
        MatrixVectorMult(A, P, multAP);
        MatrixVectorMult(At, biP, multAtbiP);
        numerator = scalarProduct(A.N, biR, R);
        denominator = scalarProduct(A.N, biP, multAP);
        alfa = numerator / denominator;
        for (i = 0; i < n; i++)
            nR[i] = R[i] - alfa * multAP[i];
        for (i = 0; i < n; i++)
            nbiR[i] = biR[i] - alfa * multAtbiP[i];
        denominator = numerator;
        numerator = scalarProduct(A.N, nbiR, nR);
        beta = numerator / denominator;
        for (i = 0; i < n; i++)
            nP[i] = nR[i] + beta * P[i];
        for (i = 0; i < n; i++)
            nbiP[i] = nbiR[i] + beta * biP[i];
        // контроль достежения необходимой точности
        check = sqrt(scalarProduct(n, R, R)) / norm;
        if (check < EPSILON)
            break;
        for (i = 0; i < n; i++)
            x[i] += alfa * P[i];
        // меняем массивы текущего и следующего шага местами
        tmp = R;
        R = nR;
        nR = tmp;
        tmp = P;
        P = nP;
        nP = tmp;
        tmp = biR;
        biR = nbiR;
        nbiR = tmp;
        tmp = biP;
        biP = nbiP;
        nbiP = tmp;
    }
    // освобождение памяти
    FreeMatrix(At);
    delete[] R;
    delete[] biR;
    delete[] nR;
    delete[] nbiR;
    delete[] P;
    delete[] biP;
    delete[] nP;
    delete[] nbiP;
    delete[] multAP;
    delete[] multAtbiP;
    return BICG_OK;
}

int ReadMatrix(crsMatrix &mtx, const char *fileName) {
    int N, NZ;
    if (fileName == NULL) return -1;
    FILE *f = fopen(fileName, "r");
    if (f == NULL) return -1;
    fscanf(f, "%d", &NZ);
    fscanf(f, "%d", &N);
    InitializeMatrix(N, NZ, mtx);
    for (int i = 0; i < NZ; i++) {
        fscanf(f, "%lf %i", &(mtx.Value[i]), &(mtx.Col[i]));
    }
    for (int i = 0; i < N + 1; i++) {
        fscanf(f, "%d", &(mtx.RowIndex[i]));
    }
    fclose(f);
    return 0;
}

int ReadVector(FLOAT_TYPE **a, int &N, const char *fileName) {
    if (fileName == NULL) return -1;
    std::ifstream f(fileName);
    if (!f.is_open()) return -1;
    f >> N;
    *a = new FLOAT_TYPE[N];
    for (int i = 0; i < N; i++) {
        f >> ((*a)[i]);
    }
    f.close();
    return 0;
}


int main() {
    std::string vec, mat;
    std::cin >> mat >> vec;
    FLOAT_TYPE *v, *res;
    int n;
    crsMatrix mtx;
    ReadVector(&v, n, vec.data());
    ReadMatrix(mtx, mat.data());
    res = new FLOAT_TYPE[n];
    int iter;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    
    start = std::chrono::high_resolution_clock::now();
    
    BiCG(mtx, v, res, 2000, iter);
    
    end = std::chrono::high_resolution_clock::now();
    
    std::cout << "dim: "<< mtx.N << std::endl;
    std::cout << "not null elements: " << mtx.NZ << std::endl;
    std::cout << "iteration to solve: " << iter << std::endl;
    std::cout << "time to solve(mics): "
                 << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()
              << std::endl;
    
    for (int i = 0; i < n; ++i)
        std::cout << res[i] << " ";
    
    return 0;
}