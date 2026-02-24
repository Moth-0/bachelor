#include<iostream>
#include"matrix.h"
#include"eigen.h"
#include"gaussian.h"
#include"hamiltonian.h"
#include"jacobian.h"


int main () {
    // Particle masses in eV
    long double m_n  = 939565420.52; 
    long double m_p  = 938272088.16; 
    long double m_pi = 139570390.0;

    // 1. Initialize the vector with 8 default/empty elements
    std::vector<qm::gaus> basis(8); 

    // 2. Generate a unique random Gaussian for each element
    for (int i = 0; i < 8; i++) {
        basis[i] = qm::gaus(2);
    }

    qm::hamiltonian h_calc;
    qm::jacobian J = {m_n, m_p, m_pi};

    // 1. Calculate the matrices
    qm::matrix H = h_calc.hamiltonian_matrix(basis, J, true);
    qm::matrix N = h_calc.overlap_matrix(basis);

    // 2. Factorize N = L * L^T
    qm::matrix L = qm::cholesky(N);
    qm::matrix L_inv = L.inverse(); // Assuming you have .inverse() and .transpose()
    qm::matrix L_inv_T = L_inv.transpose();

    // 3. Transform H -> H'
    qm::matrix H_prime = L_inv * H * L_inv_T;

    // 4. Solve the standard eigenproblem H' v = E v
    qm::vector energies = qm::jacobi_eigenvalues(H_prime);

    // 5. Find the ground state (the lowest eigenvalue)
    long double E_0 = energies[0];
    for (size_t i = 1; i < energies.size(); i++) {
        if (energies[i] < E_0) E_0 = energies[i];
    }

    std::cout << "Current Ground State Energy: " << E_0 << " eV" << std::endl;

return 0;} 