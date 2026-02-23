#include<iostream>
#include"matrix.h"
#include"gaussian.h"

int main () {
    qm::matrix A = {{1,2}, {3,4}};
    qm::matrix s = {{0,0}, {0,0}};
    qm::matrix r = {{1,0}, {0,1}};
    //std::cout << A << std::endl;
    qm::gaus g(A, s);
    //std::cout << g(r) << std::endl; 
    qm::matrix B = {{1,2,3,4},{4,5,6,7},{2,2,3,4},{4,0,4,0}};
    B.T();
    std::cout << B.determinat() << std::endl;
return 0;} 