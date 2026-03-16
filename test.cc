#include"qm/matrix.h"
#include"qm/jacobi.h"
#include"qm/gaussian.h"
#include<random>

int main () {
    std::mt19937 rng(42);
    qm::JacobiSystem sys({333.0, 330.0, 500.0}, {"n", "p", "sigma"});
    qm::random_gaussian_bare(sys, 3.0, 0.1, rng);
    return 0;
}