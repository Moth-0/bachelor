#include "qm/matrix.h"
#include <iostream>

int main () {
    qm::rmat A = {{1, 2, 3}, {4, 5, 6}};
    std::cout << A << std::endl;
    FOR_MAT(A) std::cout << A(j, i) << std::endl;
} 