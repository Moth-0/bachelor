#pragma once
#include <cmath>
#include <array>
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
    //  HELPER: R = (A_i + A_j)^{-1}  (shared by K_cla + K_rel)
    // --------------------------------------------------------
    matrix calc_R(const gaus& gi, const gaus& gj) const {
        return (gi.A + gj.A).inverse();
    }

    // --------------------------------------------------------
    //  HELPER: gamma = 0.25 / (c^T (A_j R A_i) c)
    //
    //  This is the effective inverse-width of the kinetic
    //  energy integrand in momentum space.
    // --------------------------------------------------------
    long double calc_gamma(const gaus& gi, const gaus& gj,
                           const vector& c) const {
        matrix R   = calc_R(gi, gj);
        matrix BRA = gj.A * R * gi.A;
        long double cBRAc = 0.0L;
        for (size_t i = 0; i < c.size(); ++i)
            for (size_t j = 0; j < c.size(); ++j)
                cBRAc += c[i] * BRA(i,j) * c[j];
        if (std::abs(cBRAc) < ZERO_LIMIT) return 0.0L;
        return 0.25L / cBRAc;
    }

    // --------------------------------------------------------
    //  HELPER: eta  (shift-dependent displacement)
    //
    //  Uses the harmonic mean R_h = (A_j^{-1} + A_i^{-1})^{-1}
    //  to build the cross-term between the two shift vectors.
    //
    //  For s=0 (all our current sectors): eta = 0 exactly.
    //  For P-wave sectors (pion): eta encodes the shift direction.
    //
    //  Returns a scalar eta summed over all 3 spatial directions,
    //  matching the structure of the 3D integral.
    // --------------------------------------------------------
    long double calc_eta(const gaus& gi, const gaus& gj,
                         const vector& c) const {
        // Harmonic mean of A matrices
        matrix Rh = (gj.A.inverse() + gi.A.inverse()).inverse();
        size_t n  = gj.A.size1();

        long double eta = 0.0L;

        // Sum over spatial directions (x, y, z)
        for (size_t d = 0; d < 3; ++d) {
            // Ra_d = Rh * s_i[:,d],  Rb_d = Rh * s_j[:,d]
            vector Ra(n), Rb(n);
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j) {
                    Ra[i] += Rh(i,j) * gi.s(j,d);
                    Rb[i] += Rh(i,j) * gj.s(j,d);
                }

            // diff_d = A_i * Rb - A_j * Ra
            vector diff(n);
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    diff[i] += gi.A(i,j) * Rb[j] - gj.A(i,j) * Ra[j];

            long double eta_d = 0.0L;
            for (size_t i = 0; i < n; ++i)
                eta_d += c[i] * diff[i];

            eta += eta_d; // accumulate (sign matters for sin in K_rel)
        }
        return eta;
    }

    // --------------------------------------------------------
    //  CLASSICAL KINETIC ENERGY
    //
    //  K_cla = overlap * (hbar^2/2mu) * J_cla
    //
    //  When eta = 0  (s-wave, all shifts zero):
    //    J_cla = 6 * c^T(A_j R A_i)c  =  1.5 / gamma
    //
    //  General (eta != 0):
    //    J_cla = 1.5/gamma - eta^2
    // --------------------------------------------------------
    long double K_cla(const gaus& gi, const gaus& gj,
                      const vector& c, long double mass) const {
        long double ov = overlap(gi, gj);
        if (std::abs(ov) < ZERO_LIMIT) return 0.0L;

        long double prefactor = (hbar_c * hbar_c) / (2.0L * mass);
        long double eta = calc_eta(gi, gj, c);

        if (std::abs(eta) < ZERO_LIMIT) {
            // s=0 branch: J = 6 * c^T(A_j R A_i)c
            matrix R   = calc_R(gi, gj);
            matrix BRA = gj.A * R * gi.A;
            long double cBRAc = 0.0L;
            for (size_t i = 0; i < c.size(); ++i)
                for (size_t j = 0; j < c.size(); ++j)
                    cBRAc += c[i] * BRA(i,j) * c[j];
            return ov * prefactor * 6.0L * cBRAc;
        }

        long double gamma = calc_gamma(gi, gj, c);
        if (std::abs(gamma) < ZERO_LIMIT) return 0.0L;

        return ov * prefactor * (1.5L / gamma - eta * eta);
    }

    // --------------------------------------------------------
    //  RELATIVISTIC KINETIC ENERGY
    //
    //  T_rel = sqrt(p^2 c^2 + m^2 c^4) - m c^2
    //
    //  Matrix element (eq. A.25 from your notes):
    //
    //    K_rel = overlap * (gamma/pi)^{3/2} * 2*pi
    //            * exp(-gamma*eta^2)
    //            * integral_0^{x_max} x f(p) exp(-gamma x^2)
    //                * [2x  if eta~0  else
    //                   exp(gamma eta^2)/(gamma eta) * sin(2 gamma eta x)]
    //              dx
    //
    //  where f(p) = p^2 / (sqrt(p^2 + m^2) + m)  [conjugate trick]
    //  and p = hbar_c * x.
    //
    //  Integrated with Simpson's rule, 200 points.
    //  x_max = 6 / sqrt(gamma)  covers > 6 sigma of the Gaussian.
    // --------------------------------------------------------
    long double K_rel(const gaus& gi, const gaus& gj,
                      const vector& c, long double mass) const {
        long double ov = overlap(gi, gj);
        if (std::abs(ov) < ZERO_LIMIT) return 0.0L;

        long double gamma = calc_gamma(gi, gj, c);
        if (std::abs(gamma) < ZERO_LIMIT) return 0.0L;

        long double eta = calc_eta(gi, gj, c);

        // Integrand: x * f(p) * exp(-gamma x^2) * [angular factor]
        auto integrand = [&](long double x) -> long double {
            long double p     = hbar_c * x;
            // Conjugate trick: sqrt(p^2+m^2) - m = p^2/(sqrt(p^2+m^2)+m)
            long double f_val = (p * p) / (std::sqrt(p*p + mass*mass) + mass);
            long double base  = x * f_val * std::exp(-gamma * x * x);

            if (std::abs(gamma * eta) < ZERO_LIMIT) {
                return 2.0L * x * base;           // eq A.26 (eta->0 limit)
            } else {
                return std::exp(gamma * eta * eta) / (gamma * eta)
                       * base * std::sin(2.0L * gamma * eta * x); // eq A.25
            }
        };

        // Simpson's rule
        const int    n_pts = 200;
        long double  x_max = 6.0L / std::sqrt(gamma);
        long double  h     = x_max / n_pts;
        long double  sum   = integrand(0.0L) + integrand(x_max);

        for (int i = 1; i < n_pts; ++i)
            sum += (i % 2 == 1 ? 4.0L : 2.0L) * integrand(i * h);

        long double front = std::pow(gamma / pi, 1.5L) * 2.0L * pi;
        return ov * (h / 3.0L) * sum * front;
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
    //    W_ij = < g_bare | (alpha + beta . (c_coord^T x)) *
    //                      exp(-x^T Omega x) | g_clothed >
    //
    //  where:
    //    g_bare    : dim1-dimensional Gaussian (bare sector)
    //    g_clothed : dim2-dimensional Gaussian (clothed sector, dim2 > dim1)
    //    Omega     : dim2 x dim2 kernel matrix (Gaussian form factor)
    //    c_coord   : Jacobi-space projection vector for the physical distance.
    //                For sigma:  c = e_{meson_idx}  (single Jacobi coord)
    //                For pion:   c = [±m1/(m0+m1), 1] so that
    //                            c^T x = r_{pi} - r_{emitting nucleon}
    //    alpha     : scalar coupling strength (use for sigma/s-wave)
    //    beta      : 3-vector coupling (use for pion/p-wave; = S * <chi|sigma|chi>)
    //
    //  This is the UNIFIED formula:
    //    W = (alpha + beta . (c_coord^T u)) * M
    //
    //  where M = overlap after promotion+Omega, and
    //    u^d = 0.5 * B^{-1} * v^d   (Gaussian mean vector in Jacobi space)
    //    B = A_promoted_with_Omega + A_clothed
    //    v^d = s_promoted[:,d] + s_clothed[:,d]
    //
    //  Special cases:
    //    Sigma meson (scalar): beta = {0,0,0}, returns alpha * M
    //    Pion (P-wave):        alpha = 0,      returns (beta . (c^T u)) * M
    // --------------------------------------------------------
    long double W(const gaus& g_bare,
                  const gaus& g_clothed,
                  const matrix& Omega,
                  const vector& c_coord,
                  long double alpha,
                  const vector& beta) const {

        size_t d1 = g_bare.dim();
        size_t d2 = g_clothed.dim();
        assert(d2 > d1);
        assert(c_coord.size() == d2);
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

        // 5. P-wave part: compute c_coord^T u for each spatial direction
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

            // Project onto the physical coordinate: c_coord^T u_d
            // For pion: this gives r_{pi -> emitting nucleon} in direction d
            long double cu = 0.0L;
            for (size_t k = 0; k < d2; ++k)
                cu += c_coord[k] * u_d[k];
            beta_dot_u += beta[d] * cu;
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

    // --------------------------------------------------------
    //  SQUARED PROTON-NEUTRON DISTANCE (Observable)
    //
    //  Calculates < gi | r_pn^2 | gj >
    //  r_pn is always the 0th Jacobi coordinate in our sectors.
    // --------------------------------------------------------
    long double R2_matrix_element(const gaus& a, const gaus& b) const {
        long double ov = overlap(a, b);
        if (std::abs(ov) < ZERO_LIMIT) return 0.0L;

        matrix B = a.A + b.A;
        matrix B_inv = B.inverse();

        // 1. The pure width contribution (Trace over the 3 spatial dimensions)
        // Since r_pn is Jacobi coordinate 0, we only want the (0,0) element of the inverse
        long double term1 = 1.5L * B_inv(0, 0);

        // 2. The shift contribution 
        // We calculate the Gaussian mean shift 'u' for the 0th coordinate
        long double term2 = 0.0L;
        size_t d = a.dim();
        
        for (size_t k = 0; k < 3; ++k) { // Loop over x, y, z directions
            long double v0_k = 0.0L; 
            for (size_t j = 0; j < d; ++j) {
                long double vj = a.s(j, k) + b.s(j, k);
                v0_k += B_inv(0, j) * vj; // Matrix-vector multiplication for coord 0
            }
            long double u0_k = 0.5L * v0_k;
            term2 += u0_k * u0_k; // Add to squared vector length
        }

        return ov * (term1 + term2);
    }
};

} // namespace qm