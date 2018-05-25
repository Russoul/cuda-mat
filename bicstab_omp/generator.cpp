#include <iostream>
#include <vector>

float rand_float_0_1() {
    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    
    return r;
}

float rand_float(float min, float max) {
    float norm = rand_float_0_1();
    
    return norm * (max - min) + min;
}

int gen_rand_coord_matrix(long long n, float probability_of_zero, float min, float max) {
    int count = 0, lenght = 0;
    std::vector<int> jptr(0);
    double tmp;
    std::vector<double> aelem(0);
    
    int *iptr = new int[n+1];
    for (int i = 0; i < n; i++) {
        iptr[i] = count;
        for (int j = 0; j < n; j++) {
            tmp = rand_float(min, max);
            if (tmp != 0. && probability_of_zero <= rand_float_0_1()) {
                jptr.push_back(j);
                aelem.push_back(tmp);
                lenght++;
                count++;
            }
        }
    }
    iptr[n] = count;
    
    
    
    //std::cout << "%%MatrixMarket matrix coordinate real general" << std::endl;
    std::cout << count << " " << n << std::endl;
    
    for(int i = 0; i < lenght; ++i)
        std::cout << aelem[i] << " " << jptr[i] << " ";
    
    for(int i = 0; i <= n; ++i)
        std::cout << iptr[i] << " ";
    
    return count;
}

void gen_rand_vector(long long n, float probability_of_zero, float min, float max){
    std::cout << n << " ";
    for (long long i = 0; i < n; ++i) {
        std:: cout << (rand_float_0_1() <= probability_of_zero ? 0.0F : rand_float(min, max)) << " ";
    }
}

int main() {
    bool mat_vec;
    long long dim1;
    float min,max, probability_of_zero;
    std::cin >> mat_vec >> dim1 >> min >> max >> probability_of_zero;
    if(mat_vec)
        gen_rand_coord_matrix(dim1, probability_of_zero, min, max);
    else
        gen_rand_vector(dim1, probability_of_zero, min, max);
    return 0;
}