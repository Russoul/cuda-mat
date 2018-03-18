#ifndef CSR_MATRIX
#define CSR_MATRIX

#include <iostream>
#include <iomanip>
#include <fstream>
#include <list>
#include <algorithm>

template<typename T>
class CSR;//RR(C)U

struct TypePortrain {
    int *jptr;
    int size_j;
    int *iptr;
    int size_i;
};

template<typename T>
CSR<T> operator*(const CSR<T> &matrix, T l);

template<typename T>
CSR<T> operator*(T l, const CSR<T> &matrix);

template<typename T>
std::istream &operator>>(std::istream &stream, CSR<T> &obj);

template<typename T>
std::ostream &operator<<(std::ostream &stream, const CSR<T> &obj);

template<typename T>
CSR<T> operator*(const CSR<T> &matrix_a, const CSR<T> &matrix_b);

template<typename T>
TypePortrain mult_portrain(const CSR<T> &matrix_a, const CSR<T> &matrix_b);

template<typename T>
class CSR {
    T *aelem;
    int *jptr;
    int *iptr;
    int lenght;
    int rows, collumns;
    
    void NewMem(int a_size_not_nil, int rows_matrix, int collumns_matrix);
    
    friend TypePortrain mult_portrain<>(const CSR &matrix_a, const CSR &matrix_b);

public:
    ~CSR();
    
    CSR(int a_size_not_nil = 3, int rows_matrix = 3, int collumns_matrix = 3, T start = 1);
    
    CSR(const CSR<T> &matrix);
    
    friend CSR operator*<>(const CSR &matrix, T l);
    
    friend CSR operator*<>(T l, const CSR &matrix);
    
    int GetSize();
    
    int GetLenght();
    
    friend std::ostream &operator<<<>(std::ostream &stream, const CSR &obj);
    
    friend std::istream &operator>><>(std::istream &stream, CSR &obj);
    
    friend CSR operator*<>(const CSR &matrix_a, const CSR &matrix_b);
    
    T get(int i, int j);
};

template<typename T>
CSR<T>::~CSR() {
    if (lenght != 0) {
        delete[] aelem;
        delete[] jptr;
    }
    delete[] iptr;
    lenght = 0;
    rows = 0;
    collumns = 0;
}

template<typename T>
CSR<T>::CSR(const CSR<T> &matrix) {
    lenght = matrix.lenght;
    rows = matrix.rows;
    collumns = matrix.collumns;
    aelem = new T[lenght];
    jptr = new int[lenght];
    iptr = new int[rows + 1];
    for (int i = 0; i < lenght; i++) {
        aelem[i] = matrix.aelem[i];
        jptr[i] = matrix.jptr[i];
    }
    for (int i = 0; i <= rows; i++) {
        iptr[i] = matrix.iptr[i];
    }
}

template<typename T>
CSR<T>::CSR(int a_size_not_nil, int rows_matrix, int collumns_matrix, T start) {
    aelem = new T[a_size_not_nil];
    jptr = new int[a_size_not_nil];
    iptr = new int[rows_matrix + 1];
    lenght = a_size_not_nil;
    rows = rows_matrix;
    collumns = collumns_matrix;
    for (int i = 0; i < lenght; i++) {
        aelem[i] = start;
        jptr[i] = 0;
    }
    for (int i = 0; i <= rows_matrix; i++)
        iptr[i] = 0;
}

template<typename T>
int CSR<T>::GetSize() {
    return rows * collumns;
}

template<typename T>
int CSR<T>::GetLenght() {
    return lenght;
}

template<typename T>
CSR<T> operator*(const CSR<T> &matrix, T l) {
    CSR<T> res(matrix);
    for (int i = 0; i < matrix.lenght; i++)
        res.aelem[i] *= l;
    return res;
}

template<typename T>
CSR<T> operator*(T l, const CSR<T> &matrix) {
    return operator*(matrix, l);
}

template<typename T>
std::istream &operator>>(std::istream &stream, CSR<T> &obj) {
    int count = 0;
    std::list<int> jptr_l(0);
    std::list<T> aelem_l(0);
    T tmp;
    stream >> obj.rows >> obj.collumns;
    obj.NewMem(count, obj.rows, obj.collumns);
    for (int i = 0; i < obj.rows; i++) {
        obj.iptr[i] = count;
        for (int j = 0; j < obj.collumns; j++) {
            stream >> std::setw(7) >> std::fixed >> std::setprecision(3) >> tmp;
            if (tmp != 0.) {
                obj.lenght++;
                aelem_l.push_back(tmp);
                jptr_l.push_back(j);
                count++;
            }
        }
    }
    obj.iptr[obj.rows] = count;
    obj.aelem = new T[obj.lenght];
    std::copy(aelem_l.begin(), aelem_l.end(), obj.aelem);
    obj.jptr = new int[obj.lenght];
    std::copy(jptr_l.begin(), jptr_l.end(), obj.jptr);
    return stream;
}

template<typename T>
std::ostream &operator<<(std::ostream &stream, const CSR<T> &obj) {
    stream << std::fixed << obj.rows << " " << std::fixed << obj.collumns << std::endl;
    /*
    int k = 0, i = 0, j = 0;
    for (i = 0; i < obj.rows; i++) {
        for (j = 0; j < obj.collumns; j++) {
            while (k < obj.iptr[i + 1] && j < obj.collumns) {
                for (int io = j; io < obj.jptr[k]; io++, j++)
                    stream << std::setw(7) << std::fixed << std::setprecision(3) << T(0.);
                if (j < obj.collumns) {
                    stream << std::setw(7) << std::fixed << std::setprecision(3) << obj.aelem[k];
                    k++;
                    j++;
                }
            }
            while (j < obj.collumns) {
                j++;
                stream << std::setw(7) << std::fixed << std::setprecision(3) << T(0.);
            }
        }
        stream << std::endl;
    }*/ // Old for RR(C)O
    
    for(int i = 0; i < obj.rows; ++i) {
        stream << std::setw(3) << std::fixed << i << ": ";
        for(int j = obj.iptr[i]; j < obj.iptr[i + 1]; ++j){
            stream << std::setw(7) << std::fixed << std::setprecision(3) << obj.aelem[j]
            << "(" << obj.jptr[j] << ") ";
        }
        stream << std::endl;
    }
    return stream;
}

template<typename T>
void CSR<T>::NewMem(int a_size_not_nil, int rows_matrix, int collumns_matrix) {
    delete[] aelem;
    delete[] jptr;
    delete[] iptr;
    aelem = new T[a_size_not_nil];
    jptr = new int[a_size_not_nil];
    iptr = new int[rows_matrix + 1];
    lenght = a_size_not_nil;
    rows = rows_matrix;
    collumns = collumns_matrix;
}

template<typename T>
CSR<T> operator*(const CSR<T> &matrix_a, const CSR<T> &matrix_b) {
    TypePortrain portrain = mult_portrain(matrix_a, matrix_b);
    CSR<T> res;
    int ica, icb, jaa, jab, j_val, iba, ibb, k_val;
    T a_val;
    res.iptr = portrain.iptr;
    res.jptr = portrain.jptr;
    res.lenght = portrain.size_j;
    res.rows = matrix_a.rows;
    res.collumns = matrix_b.collumns;
    res.aelem = new T[res.lenght];
    T x[res.collumns];
    if (portrain.size_j != 0) {
        for (int i = 0; i < res.rows; ++i) {
            ica = res.iptr[i];
            icb = res.iptr[i + 1] - 1;
            if (ica > icb) continue;
            for (int j = ica; j <= icb; ++j) {
                x[res.jptr[j]] = T(0.);
            }
            jaa = matrix_a.iptr[i];
            jab = matrix_a.iptr[i + 1] - 1;
            for (int k = jaa; k <= jab; ++k) {
                j_val = matrix_a.jptr[k];
                a_val = matrix_a.aelem[k];
                iba = matrix_b.iptr[j_val];
                ibb = matrix_b.iptr[j_val + 1] - 1;
                if (iba > ibb) continue;
                for (int l = iba; l <= ibb; ++l) {
                    k_val = matrix_b.jptr[l];
                    x[k_val] = x[k_val] + a_val * matrix_b.aelem[l];
                }
            }
            for (int j = ica; j <= icb; ++j) {
                res.aelem[j] = x[res.jptr[j]];
            }
        }
    }
    return res;
}

template<typename T>
TypePortrain mult_portrain(const CSR<T> &matrix_a, const CSR<T> &matrix_b) {
    TypePortrain res;
    res.size_i = matrix_a.rows + 1;
    res.iptr = new int[res.size_i];
    std::list<int> jptr_l(0);
    int jaa, jab, val_j, iba, ibb, val_j_b;
    int ix[matrix_a.rows];
    int ip = 0;
    for (int i = 0; i < matrix_b.collumns; ++i) {
        ix[i] = -1;
    }
    for (int j = 0; j <= matrix_a.rows; ++j) {
        res.iptr[j] = ip;
        jaa = matrix_a.iptr[j];
        jab = matrix_a.iptr[j + 1] - 1;
        if (jaa > jab) continue;
        for (int k = jaa; k <= jab; ++k) {
            val_j = matrix_a.jptr[k];
            iba = matrix_b.iptr[val_j];
            ibb = matrix_b.iptr[val_j + 1] - 1;
            if (iba > ibb) continue;
            for (int l = iba; l <= ibb; ++l) {
                val_j_b = matrix_b.jptr[l];
                if (ix[val_j_b] == j) continue;
                jptr_l.push_back(val_j_b);
                ip++;
                ix[val_j_b] = j;
            }
        }
    }
    if (ip != 0) {
        res.jptr = new int[ip];
        std::copy(jptr_l.begin(), jptr_l.end(), res.jptr);
        res.size_j = ip;
    } else {
        res.size_j = 0;
    }
    return res;
}

#endif
