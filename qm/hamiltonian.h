#pragma once
#include <cmath>
#include <array>
//#include <vector>
#include <functional>
#include <limits>
#include "matrix.h"
#include "gaussian.h"
#include "jacobian.h"

// ============================================================
//  hamiltonian.h
//
//  All quantum mechanical operator matrix elements between
//  correlated shifted Gaussians.
//
//  Design principles:
//  - All functions are pure: no state except hbar_c
//  - No particle-type logic; only masses and coordinates
//  - The unified W function handles both scalar (s-wave)
//    and P-wave couplings via (alpha, beta) parameterisation
//
//  Units: energies in MeV, lengths in fm, hbar*c = 197.327 MeV·fm
// ============================================================

namespace qm {

struct hamiltonian {
    long double hbar_c = 197.3269804L; // MeV·fm

    // --------------------------------------------------------
    //  OVERLAP (convenience wrapper; see gaussian.h)
    // --------------------------------------------------------
    long double ovlp(const gaus& gi, const gaus& gj) const {
        return overlap(gi, gj);
    }

    // --------------------------------------------------------
    //  CLASSICAL KINETIC ENERGY
    //
    //  K_cla = (hbar^2 / 2*mu) < gi | -nabla_c^2 | gj >
    //
    //  where nabla_c differentiates along the Jacobi direction
    //  selected by vector c (a unit vector in Jacobi space).
    //
    //  Analytic result using the "eta" formalism:
    //    eta   = (A_i B^{-1} A_j - A_j B^{-1} A_i) s difference
    //    gamma = c^T (A_i B^{-1} A_j) c
    //
    //  The final expression is (with eta = |c^T(B^{-1}(A_i s_i - A_j s_j))|):
    //    K = N_ij * (hbar^2/2mu) * (3/(2*gamma) - eta^2) * (per spatial dir)
    //    ... summed over x,y,z automatically via the 3D overlap prefactor.
    // --------------------------------------------------------
    long double K_cla(const gaus& gi, const gaus& gj,
                      const vector& c, long double mass) const {
        long double ov = overlap(gi, gj);
        if (std::abs(ov) < ZERO_LIMIT) return 0.0L;

        matrix B     = gi.A + gj.A;
        matrix B_inv = B.inverse();

        // gamma = c^T (B^{-1}) c  (scalar, same for all spatial directions)
        long double gamma = 0.0L;
        for (size_t i = 0; i < c.size(); ++i)
            for (size_t j = 0; j < c.size(); ++j)
                gamma += c[i] * B_inv(i,j) * c[j];

        if (std::abs(gamma) < ZERO_LIMIT) return 0.0L;

        long double prefactor = (hbar_c * hbar_c) / (2.0L * mass);

        // eta_d = |c^T B^{-1} (A_j s_j - A_i s_i)|_d  per spatial direction
        // Then sum eta_d^2 over d=x,y,z
        long double eta_sq = 0.0L;
        matrix Ai_Binv = gi.A * B_inv; // precompute
        matrix Aj_Binv = gj.A * B_inv;

        for (size_t d = 0; d < 3; ++d) {
            // diff_d = (A_j s_j - A_i s_i)[:,d]  (a dim-vector)
            long double cBdiff = 0.0L;
            for (size_t k = 0; k < gi.dim(); ++k) {
                long double diff_k = 0.0L;
                for (size_t l = 0; l < gi.dim(); ++l)
                    diff_k += Aj_Binv(k,l) * gj.s(l,d)
                            - Ai_Binv(k,l) * gi.s(l,d);
                // project onto c
                cBdiff += c[k] * diff_k;
            }
            eta_sq += cBdiff * cBdiff;
        }

        // Final result: ov * prefactor * 3 * (1/(2*gamma) - eta_sq/3)
        // = ov * prefactor * (3/(2*gamma) - eta_sq)
        return ov * prefactor * (1.5L / gamma - eta_sq);
    }

    // --------------------------------------------------------
    //  RELATIVISTIC KINETIC ENERGY
    //
    //  T_rel = sqrt(p^2 c^2 + m^2 c^4) - m c^2
    //
    //  Matrix element computed via the integral representation:
    //
    //    sqrt(p^2 + m^2) - m = (1/sqrt(pi)) integral_0^inf
    //      dt/sqrt(t) * [m^2 e^{-t m^2} - (p^2+m^2) e^{-t(p^2+m^2)}]
    //
    //  In position space, this becomes a 1D Gaussian quadrature
    //  over t, where each integrand is an analytic Gaussian matrix
    //  element with a modified A matrix.
    //
    //  Parameters:
    //    n_quad  : number of quadrature points (default 64)
    //    t_max   : upper limit of the t-integral (in 1/MeV^2)
    // --------------------------------------------------------
    long double K_rel(const gaus& gi, const gaus& gj,
                      const vector& c, long double mass,
                      int n_quad = 64, long double t_max = 20.0L) const {

        // Gauss-Laguerre-inspired grid: map t in [0, t_max] via t = u^2
        // so dt/sqrt(t) = 2 du, and the integrand becomes smooth.
        auto integrand = [&](long double t) -> long double {
            // At integration parameter t, the kinetic operator is:
            // (p^2 + m^2) exp(-t(p^2+m^2)) - m^2 exp(-t m^2)
            // In position space for a Gaussian, acting with exp(-t*p^2)
            // on phi_j shifts A_j -> A_j + t * (c c^T) (in Jacobi space).

            long double m = mass / hbar_c; // convert to fm^{-1}
            long double m2 = m * m;

            // Build modified Gaussian: A_j + t * c c^T
            gaus gj_mod = gj;
            for (size_t i = 0; i < gj.dim(); ++i)
                for (size_t k = 0; k < gj.dim(); ++k)
                    gj_mod.A(i,k) += t * hbar_c * hbar_c * c[i] * c[k];

            long double ov_mod = overlap(gi, gj_mod);
            long double ov_0   = overlap(gi, gj);

            return (m2 + 1.0L/t) * ov_mod - m2 * ov_0;
        };

        // Trapezoidal integration in u = sqrt(t), so t = u^2, dt = 2u du
        long double u_max = std::sqrt(t_max);
        long double du    = u_max / n_quad;
        long double result = 0.0L;

        for (int k = 1; k < n_quad; ++k) {
            long double u = k * du;
            long double t = u * u;
            result += 2.0L * u * du * integrand(t); // dt = 2u du
        }
        // End-point half-weight corrections (trapezoidal rule)
        // k=0: u=du,    t=du^2
        result += du * du * integrand(du * du);
        // k=n_quad: u=u_max, t=t_max
        result += u_max * du * integrand(t_max);

        // Prefactor: 1/sqrt(pi) from the integral representation
        // Subtract rest mass: the integral gives sqrt(p^2+m^2), we want T = ... - m
        long double ov_0 = overlap(gi, gj);
        return (result / std::sqrt(pi)) - (mass * ov_0);
    }

    // --------------------------------------------------------
    //  COULOMB POTENTIAL (for testing with hydrogen)
    //
    //  V_cou = < gi | 1/r_c | gj >
    //
    //  where r_c = |c^T x| is the distance along Jacobi coord c.
    // --------------------------------------------------------
    long double V_cou(const gaus& gi, const gaus& gj,
                      const vector& c) const {
        long double ov = overlap(gi, gj);
        if (std::abs(ov) < ZERO_LIMIT) return 0.0L;

        matrix B     = gi.A + gj.A;
        matrix B_inv = B.inverse();

        // beta = 1 / (c^T B^{-1} c)
        long double cBc = 0.0L;
        for (size_t i = 0; i < c.size(); ++i)
            for (size_t j = 0; j < c.size(); ++j)
                cBc += c[i] * B_inv(i,j) * c[j];
        long double beta = 1.0L / cBc;

        // q_d = c^T (A_j B^{-1} s_j + A_i B^{-1} s_i)  per direction
        long double q_sq = 0.0L;
        for (size_t d = 0; d < 3; ++d) {
            long double qd = 0.0L;
            for (size_t k = 0; k < c.size(); ++k) {
                long double term = 0.0L;
                for (size_t l = 0; l < c.size(); ++l)
                    term += B_inv(k,l) * (gj.A(l,k) * gj.s(l,d)
                                        + gi.A(l,k) * gi.s(l,d));
                qd += c[k] * term;
            }
            q_sq += qd * qd;
        }
        long double q = std::sqrt(q_sq);

        long double J = (q < 1e-12L)
            ? 2.0L * std::sqrt(beta / pi)
            : std::erf(std::sqrt(beta) * q) / q;

        return ov * J;
    }

    // --------------------------------------------------------
    //  UNIFIED MESON-EXCHANGE COUPLING (W operator)
    //
    //  Computes the off-diagonal block matrix element:
    //
    //    W_ij = < g_bare | (alpha + beta . x_{coord}) * 
    //                      exp(-x^T Omega x) | g_clothed >
    //
    //  where:
    //    g_bare    : dim1-dimensional Gaussian (bare sector)
    //    g_clothed : dim2-dimensional Gaussian (clothed sector, dim2 > dim1)
    //    Omega     : dim2 x dim2 kernel matrix (Gaussian form factor)
    //    coord_idx : which Jacobi coordinate carries the linear factor
    //                (= jac_clothed.meson_index() for the meson coordinate)
    //    alpha     : scalar coupling strength (use for sigma/s-wave)
    //    beta      : 3-vector coupling (use for pion/p-wave; = S * <chi|sigma|chi>)
    //
    //  This is the UNIFIED formula:
    //    W = (alpha + beta . u_{coord}) * M
    //
    //  where M = overlap after promotion+Omega, and
    //    u_{coord}^d = [0.5 * B^{-1} * v^d]_{coord_idx}
    //    B = A_promoted_with_Omega + A_clothed
    //    v^d = s_promoted[:,d] + s_clothed[:,d]
    //
    //  Special cases:
    //    Sigma meson (scalar): beta = {0,0,0}, returns alpha * M
    //    Pion (P-wave):        alpha = 0,      returns (beta.u) * M
    // --------------------------------------------------------
    long double W(const gaus& g_bare,
                  const gaus& g_clothed,
                  const matrix& Omega,
                  size_t coord_idx,
                  long double alpha,
                  const std::array<long double,3>& beta) const {

        size_t d1 = g_bare.dim();
        size_t d2 = g_clothed.dim();
        assert(d2 > d1);
        assert(coord_idx < d2);
        assert(Omega.size1() == d2 && Omega.size2() == d2);

        // 1. Promote g_bare to d2 (pad with zeros)
        gaus g_prom = promote(g_bare, d2);

        // 2. Absorb Omega into the promoted Gaussian's A matrix
        g_prom.A = g_prom.A + Omega;

        // 3. Scalar overlap M (integrates the Gaussian kernel)
        long double M = overlap(g_prom, g_clothed);
        if (std::abs(M) < ZERO_LIMIT) return 0.0L;

        // 4. If pure scalar coupling, we are done
        bool pure_scalar = (beta[0] == 0.0L && beta[1] == 0.0L && beta[2] == 0.0L);
        if (pure_scalar) return alpha * M;

        // 5. P-wave part: compute u_{coord_idx} for each spatial direction
        //    B = A_promoted_with_Omega + A_clothed  (already in g_prom.A)
        matrix B     = g_prom.A + g_clothed.A;
        matrix B_inv = B.inverse();

        long double beta_dot_u = 0.0L;
        for (size_t d = 0; d < 3; ++d) {
            if (std::abs(beta[d]) < ZERO_LIMIT) continue;

            // v^d = s_prom[:,d] + s_clothed[:,d]  (a d2-vector)
            vector v_d(d2);
            for (size_t k = 0; k < d2; ++k)
                v_d[k] = g_prom.s(k,d) + g_clothed.s(k,d);

            // u^d = 0.5 * B^{-1} * v^d  (the Gaussian mean in Jacobi space)
            vector u_d = B_inv * v_d;
            u_d = u_d * 0.5L;

            // Extract the meson Jacobi coordinate component
            beta_dot_u += beta[d] * u_d[coord_idx];
        }

        return (alpha + beta_dot_u) * M;
    }

    // --------------------------------------------------------
    //  DIAGONAL BLOCK BUILDER
    //
    //  Fills H and N matrices for a single Fock sector.
    //  Each sector has its own Jacobian (different masses).
    //  The meson rest mass contribution is added separately
    //  in the main build function (not here) so this stays general.
    // --------------------------------------------------------
    void build_diagonal_block(
        const std::vector<gaus>& basis,
        const jacobian& jac,
        bool relativistic,
        matrix& H_block,
        matrix& N_block) const
    {
        size_t n = basis.size();
        H_block = matrix(n, n);
        N_block = matrix(n, n);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                long double N_ij = overlap(basis[i], basis[j]);
                N_block(i,j) = N_ij;

                long double K_ij = 0.0L;
                for (size_t coord = 0; coord < jac.dim(); ++coord) {
                    if (relativistic)
                        K_ij += K_rel(basis[i], basis[j],
                                      jac.c(coord), jac.mu(coord));
                    else
                        K_ij += K_cla(basis[i], basis[j],
                                      jac.c(coord), jac.mu(coord));
                }
                H_block(i,j) = K_ij;
            }
        }
    }
};

} // namespace qm