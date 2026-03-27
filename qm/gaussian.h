#pragma once

#include <cmath>
#include <vector>
#include <random>
#include "matrix.h"
#include "jacobi.h"

using namespace qm; 

inline long double random_ld(long double lo, long double hi) {
    thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<long double> dist(lo, hi);
    return dist(rng);
}

// Generates the Van der Corput sequence for base 2
inline ld van_der_corput(size_t n, size_t base = 2) {
    ld q = 0.0;
    ld bk = 1.0 / static_cast<ld>(base);
    while (n > 0) {
        q += (n % base) * bk;
        n /= base;
        bk /= base;
    }
    return q;
}


struct Gaussian {
    rmat A; // Correlation matrix (positive definite)
    rmat s; // Shift matrix (N-1 x 3)

    Gaussian() = default;
    Gaussian(const rmat& A_in, const rmat& s_in) : A(A_in), s(s_in) {}

    ~Gaussian() = default;

    // r is now an (N-1) x 3 matrix representing the 3D Jacobi vectors
    ld evaluate(const rmat& r) const {
        ld rAr = 0.0;
        ld sr = 0.0;
        
        // Loop over x (0), y (1), z (2) columns
        for (size_t c = 0; c < 3; ++c) {
            rAr += dot_no_conj(r[c], A * r[c]);
            sr  += dot_no_conj(s[c], r[c]);
        }
        
        return std::exp(-rAr + sr);
    }

    void set(const rmat& A_in, const rmat& s_in) { SELF.A = A_in; SELF.s = s_in; }
};


struct SpatialWavefunction {
    rmat A;
    rmat s;
    int parity_sign; // +1 for symmetric (pn), -1 for antisymmetric (pn pi)

    // FIXED: s_ is now an rmat
    SpatialWavefunction(const rmat& A_, const rmat& s_, int parity) 
        : A(A_), s(s_), parity_sign(parity) {}

    SpatialWavefunction() = default;
    ~SpatialWavefunction() = default;

    // --- UPDATED RANDOMIZE FUNCTION ---
    void randomize(const Jacobian& jac, ld b_range) {
        size_t N = jac.N;           // Number of physical particles
        size_t dim = N - 1;         // Internal Jacobi dimensions (ignoring CM)
        
        rmat A_new(dim, dim); 

        // We start at 1 because n=0 gives 0.0, which breaks the logarithm!
        thread_local size_t vdc_counter = 1;

        // 1. Loop over pairs (i < j), not matrix dimensions!
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                
                // Get full N-dimensional physical selection vectors
                rvec w_i = jac.transform_w(i);
                rvec w_j = jac.transform_w(j);
                rvec w_ij_full = w_i - w_j;
                
                // 2. Truncate the Center of Mass coordinate (drop the last element)
                rvec w_ij(dim);
                for (size_t k = 0; k < dim; ++k) {
                    w_ij[k] = w_ij_full[k];
                }
                
                // The outer product is now safely (N-1) x (N-1)
                rmat outer = outer_no_conj(w_ij, w_ij);
                
                // 3. Stochastically pick b_ij using the Van der Corput sequence!
                ld u = van_der_corput(vdc_counter++, 2); 
                
                // Safety bound just in case to prevent log(0)
                if (u < 1e-10) u = 1e-10;
                
                ld b_ij = -std::log(u) * b_range;
                
                A_new += outer / (b_ij * b_ij);
            }
        }
        
        SELF.A = A_new;

        rmat r0(dim, 3);
        // 4. Randomize the shift vector 
        // Small shifts can remain standard pseudo-random, as they just jitter the spatial center.
        ld range = 0.2 * b_range; 
        FOR_MAT(r0) r0(j, i) = random_ld(-range, range);
        
        SELF.s = A_new * r0 * 2.0L;
    }

    ld evaluate(const rmat& r) const {
        ld rAr = 0.0;
        ld sr = 0.0;
        
        size_t dim = A.size1(); // This is correctly N-1
        
        // Loop over x (0), y (1), z (2) columns
        for (size_t c = 0; c < 3; ++c) {
            // Safely extract only the first 'dim' elements of the column
            rvec r_c_internal(dim);
            for (size_t i = 0; i < dim; ++i) {
                r_c_internal[i] = r[c][i];
            }
            
            // Now do the math safely!
            rAr += dot_no_conj(r_c_internal, A * r_c_internal);
            sr  += dot_no_conj(s[c], r_c_internal);
        }
        
        ld g_plus  = std::exp(-rAr + sr);
        ld g_minus = std::exp(-rAr - sr);
        
        return g_plus + parity_sign * g_minus;
    }
};

// 1. Primitive Overlap: <A1, s1 | A2, s2>
inline ld gaussian_overlap(const rmat& A1, const rmat& s1, const rmat& A2, const rmat& s2) {
    rmat B = A1 + A2;
    rmat v = s1 + s2; // matrix.h natively adds the (N-1)x3 matrices!
    size_t dim = B.size1();

    ld detB = B.determinant();
    if (std::abs(detB) < 1e-25) return 0.0; // Singular, reject

    ld pi_to_d = std::pow(M_PI, dim);
    ld prefactor = std::pow(pi_to_d / detB, 1.5);

    rmat B_inv = B.inverse();
    ld vBv = 0.0;

    // Evaluate v^T B^-1 v for x, y, and z columns
    for (size_t c = 0; c < 3; ++c) {
        vBv += dot_no_conj(v[c], B_inv * v[c]);
    }

    return prefactor * std::exp(0.25 * vBv);
}

// 2. Full Spatial Overlap incorporating Parity
inline ld spactial_overlap(const SpatialWavefunction& g1, const SpatialWavefunction& g2) {
    // A SpatialWavefunction is |+s> + p |-s>. 
    // Expanding <g1 | g2> gives 4 terms:
    
    // <+s1 | +s2>
    ld term1 = gaussian_overlap(g1.A, g1.s, g2.A, g2.s);
    
    // p2 * <+s1 | -s2>
    ld term2 = g2.parity_sign * gaussian_overlap(g1.A, g1.s, g2.A, -1.0L * g2.s);
    
    // p1 * <-s1 | +s2>
    ld term3 = g1.parity_sign * gaussian_overlap(g1.A, -1.0L * g1.s, g2.A, g2.s);
    
    // p1 * p2 * <-s1 | -s2>
    ld term4 = (g1.parity_sign * g2.parity_sign) * gaussian_overlap(g1.A, -1.0L * g1.s, g2.A, -1.0L * g2.s);

    return term1 + term2 + term3 + term4;
}

template <typename Func>
auto apply_basis_expansion(const SpatialWavefunction& bra, const SpatialWavefunction& ket, Func operation) {
    rmat B = bra.A + ket.A;
    
    // Deduce the return type dynamically (ld or cld)
    using RetType = decltype(operation(Gaussian(bra.A, bra.s), Gaussian(ket.A, ket.s), 0.0L, B));
    
    if (std::abs(B.determinant()) < ZERO_LIMIT) return RetType{0}; 
    
    rmat R = B.inverse();
    int num_bra_terms = (bra.parity_sign == 0) ? 1 : 2;
    int num_ket_terms = (ket.parity_sign == 0) ? 1 : 2;

    rmat s_bra[2] = {bra.s, -bra.s};
    rmat s_ket[2] = {ket.s, -ket.s};
    int p_bra[2] = {1, bra.parity_sign}; 
    int p_ket[2] = {1, ket.parity_sign};

    RetType total_value{0};

    for (int i = 0; i < num_bra_terms; ++i) {
        for (int j = 0; j < num_ket_terms; ++j) {
            int parity_factor = p_bra[i] * p_ket[j];
            Gaussian g_b(bra.A, s_bra[i]);
            Gaussian g_k(ket.A, s_ket[j]);
            ld M_term = gaussian_overlap(g_b.A, g_b.s, g_k.A, g_k.s);
            
            if (std::abs(M_term) > 1e-25) {
                // Multiply the scalar parity_factor by the RetType result
                total_value += static_cast<ld>(parity_factor) * operation(g_b, g_k, M_term, R);
            }
        }
    }
    return total_value;
}

// Promotes a bare state (e.g., pn) to the dressed dimension (pn + pion)
inline Gaussian promote_and_absorb(const Gaussian& g_bare, size_t target_dim, 
                                   const rvec& w_piN, ld b) 
{
    // 1. Promote A^(d) by padding it with zeros up to target_dim
    rmat A_tilde = zeros<ld>(target_dim, target_dim);
    for (size_t i = 0; i < g_bare.A.size1(); ++i) {
        for (size_t j = 0; j < g_bare.A.size2(); ++j) {
            A_tilde(i, j) = g_bare.A(i, j);
        }
    }

    // 2. Promote the shift vector s by padding it with zeros
    rmat s_promoted = zeros<ld>(target_dim, 3);
    for (size_t i = 0; i < g_bare.s.size1(); ++i) {
        for (size_t col = 0; col < 3; ++col) {
            s_promoted(i, col) = g_bare.s(i, col);
        }
    }

    // 3. Absorb the spatial form factor into the padded matrix!
    // A_tilde = A_promoted + (1 / b^2) * w_piN * w_piN^T
    ld inv_b_sq = 1.0 / (b * b);
    A_tilde += outer_no_conj(w_piN, w_piN) * inv_b_sq;

    // Return the new fully prepped Gaussian
    return Gaussian(A_tilde, s_promoted);
}

// Flattens a wavefunction into a 1D rvec for Nelder-Mead
rvec pack_wavefunction(const SpatialWavefunction& psi) {
    rvec p;
    // 1. Pack upper triangle of A
    for (size_t i = 0; i < psi.A.size1(); ++i) {
        for (size_t j = i; j < psi.A.size2(); ++j) {
            p.push_back(psi.A(i, j));
        }
    }
    // 2. Pack the entire s matrix
    for (size_t i = 0; i < psi.s.size1(); ++i) {
        for (size_t j = 0; j < psi.s.size2(); ++j) {
            p.push_back(psi.s(i, j));
        }
    }
    return p;
}

// Rebuilds the wavefunction from the 1D rvec
void unpack_wavefunction(SpatialWavefunction& psi, const rvec& p) {
    size_t idx = 0;
    // 1. Unpack A (Enforcing symmetry: A(i,j) == A(j,i))
    for (size_t i = 0; i < psi.A.size1(); ++i) {
        for (size_t j = i; j < psi.A.size2(); ++j) {
            psi.A(i, j) = p[idx];
            psi.A(j, i) = p[idx]; 
            idx++;
        }
    }
    // 2. Unpack s
    for (size_t i = 0; i < psi.s.size1(); ++i) {
        for (size_t j = 0; j < psi.s.size2(); ++j) {
            psi.s(i, j) = p[idx];
            idx++;
        }
    }
}