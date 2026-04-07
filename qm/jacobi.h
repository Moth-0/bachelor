/*
╔════════════════════════════════════════════════════════════════════════════════╗
║              jacobi.h - N-BODY COORDINATE TRANSFORMATIONS                      ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Transforms between N physical particle positions and N-1 relative            ║
║   (Jacobi) coordinates plus center-of-mass. Automatically computes             ║
║   reduced masses and density-of-states factors needed for kinetic energy.      ║
║                                                                                ║
║ PHYSICS MOTIVATION:                                                            ║
║   In an N-body system, 3 coordinates describe CM motion (which we ignore).     ║
║   The remaining 3(N-1) coordinates are internal degrees of freedom:            ║
║                                                                                ║
║   Bare state (N=2):  1 relative coordinate (PN separation)                     ║
║   Dressed state (N=3): 2 relative coordinates (e.g., pion-COM and one pair)    ║
║                                                                                ║
║ JACOBI COORDINATES:                                                            ║
║   Step 1: r₁ = r₁ - r₂  [particle 1 relative to 2]                             ║
║   Step 2: r₂ = r₃ - (m₁r₁ + m₂r₂)/(m₁+m₂)  [particle 3 relative to COM of 1,2] ║
║   Step N-1: r_N-1 = r_N - COM(1...N-1)                                         ║
║   Step N: R_CM = (m₁r₁ + ... + m_Nr_N) / M_total  [center of mass]             ║
║                                                                                ║
║                                                                                ║
║ REDUCED MASSES:                                                                ║
║   μ_i = m_i × M_cumulative / (m_i + M_cumulative)                              ║
║                                                                                ║
║   where m_i is particle i, M_cumulative = m₁ + ... + m_i                       ║
║                                                                                ║
║ KINETIC ENERGY CONSEQUENCE:                                                    ║
║   T = Σ p_i²/(2m_i) [physical basis]                                           ║
║     = Σ p_i²/(2μ_i) [Jacobi basis where p_i are conjugates to r_i]             ║
║                                                                                ║
║   This is why we store reduced_masses: essential for kinetic energy calc.      ║
║                                                                                ║
║ USAGE:                                                                         ║
║   Jacobian jac({938.27, 939.565});        // Create 2-body (proton, neutron)   ║
║   Jacobian jac3({938.27, 939.565, 135});  // 3-body (PN + pion)                ║
║                                                                                ║
║   jac.reduced_masses[0]  // μ for PN pair  (~469 MeV)                          ║
║   jac.dim                // N-1 = 1 (2-body has 1 internal coordinate)         ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#pragma once

#include <vector>
#include "matrix.h"

namespace qm {

struct Jacobian {
    size_t N;            // Number of particles
    rvec masses;         // m_1, m_2, ..., m_N
    rvec reduced_masses; // μ_1, ..., μ_N-1
    rmat J;              // The Jacobi Matrix (N x N)
    rmat U_trans;        // The Transpose of the Inverse (J^-1)^T for coordinate vectors
    size_t dim;          // N-1 dimension

    // Constructor:
    Jacobian(const rvec& m_list) {
        N = m_list.size();
        dim = N - 1;
        masses = m_list;
        
        J = rmat(N, N); 
        
        // Calculate cumulative masses M and reduced masses mu 
        rvec M(N + 1); 
        M[0] = 0.0;
        reduced_masses.resize(dim); 

        for(size_t i = 0; i < N; ++i) {
            // Accumulate the mass for the current particle
            M[i+1] = M[i] + m_list[i];
            
            // Calculate the reduced mass for the Jacobi coordinate
            if (i < dim) {
                // M[i+1] is the cumulative mass so far (M_current)
                // m_list[i+1] is the mass of the next particle (m_next)
                reduced_masses[i] = (M[i+1] * m_list[i+1]) / (M[i+1] + m_list[i+1]);
            }
        }

        // Fill Jacobi Matrix J
        for (size_t i = 0; i < N; ++i) {
            if (i < N - 1) {
                // Relative coordinate rows
                for (size_t j = 0; j <= i; ++j) {
                    J(i, j) = m_list[j] / M[i+1];
                }
                J(i, i+1) = -1.0;
            } else {
                // Last row: Center of Mass coordinates
                for (size_t j = 0; j < N; ++j) {
                    J(i, j) = m_list[j] / M[N];
                }
            }
        }
        
        // Precompute U^T for physical space transformations: w_i -> U^T * w_i
        U_trans = J.inverse().transpose();

        
    }

    ~Jacobian() = default;

    rvec transform_w(size_t particle_index) const {
        rvec w(N); 
        w[particle_index] = 1.0;
        return U_trans * w; 
    }


    rvec transform_k(size_t particle_index) const {
        rvec k(N); 
        k[particle_index] = 1.0;
        return J * k;
    }

    // Gets the c-vector for an internal Jacobi coordinate.
    rvec get_c_internal(size_t jacobi_idx) const {
        rvec c(dim); 
        c[jacobi_idx] = 1.0;
        return c;
    }
};

} // namespace qm