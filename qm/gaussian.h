/*
╔════════════════════════════════════════════════════════════════════════════════╗
║               gaussian.h - GAUSSIAN BASIS FUNCTIONS & OVERLAPS                 ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Implements Correlated Shifted Gaussians (CSG) as spatial basis functions     ║
║   for nuclear wavefunctions. Includes analytical overlap integral              ║
║   calculations and basis state randomization.                                  ║
║                                                                                ║
║ PHYSICS BACKGROUND:                                                            ║
║   CSG spacial wave functions have the form:                                    ║
║     ψ(r) = [exp(-r·A·r + s·r) + P·exp(-r·A·r - s·r)]                           ║
║                                                                                ║
║   where:                                                                       ║
║     • A = (N-1)×(N-1) correlation matrix (positive definite)                   ║
║     • s = (N-1)×3 shift matrix (translates center)                             ║
║     • P = ±1 parity sign (symmetric or antisymmetric under exchange)           ║
║     • r = relative Jacobi coordinates                                          ║
║                                                                                ║
║                                                                                ║
║ KEY FUNCTIONS:                                                                 ║
║   • gaussian_overlap():   <G1|G2> elementary Gaussian overlap (4 terms + P)    ║
║   • spatial_overlap():    <ψ1|ψ2> including parity structure                   ║
║   • randomize():          Gaussian method - generate random A,s                ║
║   • promote_and_absorb(): Embed 2-body Gaussian into 3-body space              ║
║                                                                                ║
║ NUMERICAL METHOD:                                                              ║
║   Gaussian overlap is computed analytically using the formula:                 ║
║     <g1|g2> = (π/det(A1+A2))^(3/2) exp(v^T(A1+A2)^-1 v/4)                      ║
║                                                                                ║
║   where v = s1 + s2. This avoids numerical integration entirely.               ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#pragma once

#include <cmath>
#include <vector>
#include <random>
#include <atomic>
#include "matrix.h"
#include "jacobi.h"

namespace qm {

inline ld random_ld(long double lo, long double hi) {
    thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<long double> dist(lo, hi);
    return dist(rng);
}

// Generates the Van der Corput sequence for base 2
inline ld van_der_corput(size_t n, size_t base = 2) {
    ld q = 0.0;
    ld bk = 1.0 / base;
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

    // r is an (N-1) x 3 matrix representing the 3D Jacobi vectors
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

    // Randomize A and s parameters using Van der Corput sequence and pseudo-random numbers.
    //
    // Uses Jacobi coordinate information to build correlation matrix A:
    //   A = Σ (w_ij w_ij^T) / b_ij²
    // where w_ij are Jacobi coordinate difference vectors and b_ij are stochastically chosen.
    //
    // Parameters:
    //   - jac: Jacobian object with N particles, dimension N-1
    //   - b_range: Search space for widths (fm); larger → explores wider spatial scales
    
    void randomize(const Jacobian& jac, ld b_range, ld b_form) {
        size_t N = jac.N;           // Number of physical particles
        size_t dim = N - 1;         // Internal Jacobi dimensions (ignoring CM)

        rmat A_new(dim, dim);

        static std::atomic<size_t> global_vdc_counter{1};

        // 1. Pull ONE ticket for this entire Gaussian state
        size_t current_n = global_vdc_counter.fetch_add(1, std::memory_order_relaxed);

        // 2. Array of prime numbers for the Halton sequence bases
        const size_t primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
        size_t pair_idx = 0; // Tracks which pair we are on

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {

                // Get full N-dimensional physical selection vectors
                rvec w_i = jac.transform_w(i);
                rvec w_j = jac.transform_w(j);
                rvec w_ij_full = w_i - w_j;

                // Truncate the Center of Mass coordinate 
                rvec w_ij(dim);
                for (size_t k = 0; k < dim; ++k) {
                    w_ij[k] = w_ij_full[k];
                }

                rmat outer = outer_no_conj(w_ij, w_ij);

                // -------------------------------------------------------------
                // 3. THE HALTON GENERATOR
                // Use a different prime base for each physical pair!
                size_t base = primes[pair_idx % 10]; 
                ld u = van_der_corput(current_n, base);
                pair_idx++;
                // -------------------------------------------------------------

                // Prevent u from hitting pure 0 or 1 boundaries
                if (u < ZERO_LIMIT) u = ZERO_LIMIT;
                if (u > 1.0 - ZERO_LIMIT) u = 1.0 - ZERO_LIMIT;

                ld b_ij = -std::log(u) * b_range;
                ld b2 = b_ij * b_ij;

                // Safety constraints
                if (b2 < 0.01) b2 = 0.01;
                if (b2 > (b_range * b_range)) b2 = (b_range * b_range);

                A_new += outer / b2;
            }
        }

        SELF.A = A_new;

        rmat r0(dim, 3);
        ld range = b_form;
        FOR_MAT(r0) {
            // The pion gets a random physical shift
            r0(j, i) = random_ld(-range, range);
        }
        
        // 1. Matrix multiply (this will mix the coordinates!)
        SELF.s = A_new * r0 * 2.0L;

        // 2. THE LOCK: Mathematically sever the NN shift from the pion shift
        for (size_t col = 0; col < 3; ++col) {
            SELF.s(0, col) = 0.0; 
        }
    }
};

struct SpatialWavefunction {
    rmat A;
    rmat s;
    int parity_sign; // +1 for symmetric (pn), -1 for antisymmetric (pn pi)

    // Constructor with full parameters
    SpatialWavefunction(const rmat& A_, const rmat& s_, int parity)
        : A(A_), s(s_), parity_sign(parity) {}

    // Constructor with just parity (initializes A, s to empty matrices)
    explicit SpatialWavefunction(int parity)
        : A(), s(), parity_sign(parity) {}

    SpatialWavefunction() = default;
    ~SpatialWavefunction() = default;

    /// Set A and s from a Gaussian basis function.
    ///
    /// This allows separate randomization of a Gaussian, then setting
    /// the wavefunction's parameters all at once.
    ///
    /// Example:
    ///   Gaussian g;
    ///   g.randomize(jac, b_range);
    ///   SpatialWavefunction psi(1);  // Initialize with parity +1
    ///   psi.set_from_gaussian(g);     // Copy A and s
    void set_from_gaussian(const Gaussian& g) {
        SELF.A = g.A;
        SELF.s = g.s;
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

        return 1/sqrt(2.0L) * (g_plus + parity_sign * g_minus);
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

    // THE FIX: Apply the 1/sqrt(2) normalization factors!
    ld norm_factor = 1.0L / std::sqrt(static_cast<ld>(num_bra_terms * num_ket_terms));

    return norm_factor * total_value;
}

// 2. Full Spatial Overlap incorporating Parity
inline ld spactial_overlap(const SpatialWavefunction& g1, const SpatialWavefunction& g2) {
    return apply_basis_expansion(g1, g2, [](const Gaussian& g_b, const Gaussian& g_k, ld M_term, const rmat& R) {
        // The operation for a pure overlap is literally just returning the M_term!
        return M_term; 
    });
}

// Promotes a bare state (e.g., pn) to the dressed dimension (pn + pion)
inline Gaussian promote_and_absorb(const Gaussian& g_bare, size_t target_dim,
                                   const rvec& w_piN, const rvec& w_nn, ld alpha)
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
    // A_tilde = A_promoted + (1 / b^2) * (w_piN * w_piN^T + w_nn * w_nn^T)
    A_tilde += (outer_no_conj(w_piN, w_piN) + outer_no_conj(w_nn, w_nn)) * alpha;

    // Return the new fully prepped Gaussian
    return Gaussian(A_tilde, s_promoted);
}

// Flattens a wavefunction into a 1D rvec for Nelder-Mead
// Packs lower-triangular Cholesky factor L where A = L * L^T
rvec pack_wavefunction(const SpatialWavefunction& psi, bool optimize_shift) {
    rvec p;
    size_t dim = psi.A.size1();

    // 1. Compute Cholesky decomposition: A = L * L^T (L is lower triangular)
    // Standard algorithm
    rmat L = zeros<ld>(dim, dim);
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            ld sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += L(i, k) * L(j, k);
            }
            if (i == j) {
                ld diag_val = psi.A(i, i) - sum;
                if (diag_val <= 0.0) {
                    // Matrix not PD - should not happen for valid states, but safeguard
                    L(i, i) = 1e-10;
                } else {
                    L(i, i) = std::sqrt(diag_val);
                }
            } else {
                if (L(j, j) < 1e-15) {
                    L(i, j) = 0.0;
                } else {
                    L(i, j) = (psi.A(i, j) - sum) / L(j, j);
                }
            }
        }
    }

    // 2. Pack lower triangle of L (including diagonal)
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            p.push_back(L(i, j));
        }
    }

    // 3. Pack the non-NN shifts only
    if (optimize_shift) {
        for (size_t i = 1; i < psi.s.size1(); ++i) {
            for (size_t j = 0; j < psi.s.size2(); ++j) {
                p.push_back(psi.s(i, j));
            }
        }
    }
    return p;
}

// Pack ALL basis states into a single vector for global Nelder-Mead optimization
// NOTE: This is defined in SVM.h since it needs BasisState

// Unpack a single vector back into all basis states
// NOTE: This is defined in SVM.h since it needs BasisState



// Rebuilds the wavefunction from the 1D rvec
// Reconstructs A from lower-triangular Cholesky factor: A = L * L^T (guaranteed PD!)
void unpack_wavefunction(SpatialWavefunction& psi, const rvec& p, bool optimize_shift) {
    size_t idx = 0;
    size_t dim = psi.A.size1();

    // 1. Unpack lower triangle to reconstruct L
    rmat L = zeros<ld>(dim, dim);
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            L(i, j) = p[idx++];
        }
    }

    // 2. Reconstruct A = L * L^T (guaranteed positive-definite!)
    psi.A = L * L.transpose();

    // 3. Unpack shifts (or force to zero)
    if (optimize_shift) {
        for (size_t i = 1; i < psi.s.size1(); ++i) {
            for (size_t j = 0; j < psi.s.size2(); ++j) {
                psi.s(i, j) = p[idx++];
            }
        }
    }

    // Always zero out the first row (NN shift locked to zero) for PN states only
    // For pion states (3 Jacobi coords), row 0 is not NN - keep it!
    // Only lock if this is a 1D state (PN)
    if (dim == 1) {
        for (size_t j = 0; j < psi.s.size2(); ++j) {
            psi.s(0, j) = 0.0;
        }
    }
}

} // namespace qm
