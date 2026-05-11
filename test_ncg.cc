#include "qm/gevp_ncg_solver.h"
#include <iostream>
#include <iomanip>

using namespace qm;

int main() {
    // Small test: 3×3 GEVP
    int n = 3;
    
    // Create a simple test problem
    MatrixXcld H(n, n), N(n, n);
    
    // H = [[2, 0.1i, 0], [-0.1i, 3, 0.2], [0, 0.2, 4]]
    H << cld(2, 0),     cld(0, 0.1),   cld(0, 0),
         cld(0, -0.1),  cld(3, 0),     cld(0.2, 0),
         cld(0, 0),     cld(0.2, 0),   cld(4, 0);
    
    // N = [[1, 0.05i, 0], [-0.05i, 1.1, 0.1], [0, 0.1, 1.2]]
    N << cld(1, 0),      cld(0, 0.05),  cld(0, 0),
         cld(0, -0.05),  cld(1.1, 0),   cld(0.1, 0),
         cld(0, 0),      cld(0.1, 0),   cld(1.2, 0);
    
    // Initial guess
    VectorXcld c_init(n);
    c_init << cld(1, 0), cld(1, 0), cld(1, 0);
    
    // Solve
    auto result = solve_gevp_ncg(H, N, c_init, 1e-12, 1000, true);
    
    std::cout << "Result:\n";
    std::cout << "  Eigenvalue: " << std::setprecision(15) << result.eigenvalue << "\n";
    std::cout << "  Converged: " << (result.converged ? "YES" : "NO") << "\n";
    std::cout << "  Iterations: " << result.iterations_used << "\n";
    
    return 0;
}
