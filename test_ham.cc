#include <iostream>
#include <cmath>
#include <cassert>

#include "qm/hamiltonian.h"
#include "qm/gaussian.h"
#include "qm/matrix.h"

using namespace qm;

bool approx_eq(long double a, long double b, long double epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
}

// Helper to wipe the randomized initializations
void clear_gaus(gaus& g) {
    for (size_t i = 0; i < g.dim(); ++i) {
        for (size_t j = 0; j < g.dim(); ++j) g.A(i, j) = 0.0;
        for (size_t d = 0; d < 3; ++d)       g.s(i, d) = 0.0;
    }
}

void test_W_scalar() {
    std::cout << "--- Testing W Operator (Scalar Coupling / Sigma) ---\n";
    hamiltonian H;
    
    gaus g_bare(1, 0.1, 10.0);
    clear_gaus(g_bare);
    g_bare.A(0,0) = 1.0;
    
    gaus g_clothed(2, 0.1, 10.0);
    clear_gaus(g_clothed);
    g_clothed.A(0,0) = 1.0; 
    g_clothed.A(1,1) = 1.0;
    
    matrix Omega(2, 2);
    Omega(0,0) = 0.5; 
    Omega(1,1) = 0.5;
    
    long double alpha = 2.0;
    vector beta(3); // {0, 0, 0}
    
    long double w_val = H.W(g_bare, g_clothed, Omega, 1, alpha, beta);
    
    gaus g_prom = promote(g_bare, 2);
    g_prom.A = g_prom.A + Omega;
    long double expected_M = overlap(g_prom, g_clothed);
    
    assert(approx_eq(w_val, alpha * expected_M));
    std::cout << "[PASS] W Operator Scalar Coupling\n";
}

void test_W_pwave() {
    std::cout << "--- Testing W Operator (P-wave Coupling / Pion) ---\n";
    hamiltonian H;
    
    gaus g_bare(1, 0.1, 10.0);
    clear_gaus(g_bare);
    g_bare.A(0,0) = 1.0; 
    
    gaus g_clothed(2, 0.1, 10.0);
    clear_gaus(g_clothed);
    g_clothed.A(0,0) = 1.0; 
    g_clothed.A(1,1) = 1.0;
    
    g_clothed.s(0, 0) = 1.0; 
    g_clothed.s(1, 0) = 2.0; 
    
    matrix Omega(2, 2);
    Omega(0,0) = 0.5; 
    Omega(1,1) = 0.5;
    
    long double alpha = 0.0;
    vector beta(3);
    beta[0] = 3.0; 
    
    long double w_val_0 = H.W(g_bare, g_clothed, Omega, 0, alpha, beta);
    long double w_val_1 = H.W(g_bare, g_clothed, Omega, 1, alpha, beta);
    
    gaus g_prom = promote(g_bare, 2);
    g_prom.A = g_prom.A + Omega;
    long double expected_M = overlap(g_prom, g_clothed);
    
    long double expected_u0 = 0.2;
    long double expected_u1 = 2.0 / 3.0;
    
    assert(approx_eq(w_val_0, beta[0] * expected_u0 * expected_M));
    assert(approx_eq(w_val_1, beta[0] * expected_u1 * expected_M));
    
    std::cout << "[PASS] W Operator P-wave Coupling\n";
}

int main() {
    std::cout << "Starting Hamiltonian W-Operator tests...\n\n";
    test_W_scalar();
    test_W_pwave();
    std::cout << "\nAll tests passed successfully!\n";
    return 0;
}