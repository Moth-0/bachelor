#pragma once
// hamiltonian.h  --  matrix elements of the Hamiltonian between correlated
// shifted Gaussians.
//
// All spatial integrals (K_cla, K_rel, V_cou, W) are evaluated analytically
// or by 1D numerical quadrature.  The W function returns a COMPLEX number
// because the pion emission vertex (sigma.r) has complex spherical components:
//
//   r[0] = z          (real)
//   r[1] = x + iy     (complex, spin-flip down)
//   r[2] = x - iy     (complex, spin-flip up)
//
// The full W matrix element between bare Gaussian g_i and clothed Gaussian g_j
// is the sum over all emitting nucleons k of the spatial factor for that nucleon
// times the spin-isospin coefficient stored in VertexTerms from particle.h.
//
// Reference for matrix element formulae: D.V. Fedorov,
// "Analytic matrix elements and gradients with shifted correlated Gaussians".

#include <cmath>
#include <cassert>
#include <complex>
#include <vector>
#include <array>
#include <functional>
#include <limits>
#include "matrix.h"
#include "gaussian.h"
#include "jacobian.h"
#include "particle.h"

namespace qm {

// ---------------------------------------------------------------------------
// NucleonCoupling
//
// Bundles everything needed to compute one nucleon's contribution to the W
// matrix element:
//   c_coord : projection vector such that c^T x = r_pion - r_{nucleon_k}
//             (in Jacobi coordinates of the clothed sector)
//   terms   : list of VertexTerms from apply_vertex(ket_nucleon_k, pion)
//   strength: S_pion / b_pion  (form-factor strength, same for all k)
//
// W contribution from this coupling:
//   sum_t  sum_m  strength * t.coeff[m] * (c^T u)_sph[m] * M
// ---------------------------------------------------------------------------
struct NucleonCoupling {
    vector                   c_coord;   // size = clothed Jacobi dim
    std::vector<VertexTerm>  terms;     // from apply_vertex
    long double              strength;  // S_pion / b_pion
};

// ---------------------------------------------------------------------------
// hamiltonian
// ---------------------------------------------------------------------------
struct hamiltonian {
    long double hbar_c = 197.3269804L;  // MeV * fm

    long double ovlp(const gaus& a, const gaus& b) const { return overlap(a,b); }

    // -----------------------------------------------------------------------
    // Helper: B^{-1} where B = A_i + A_j
    // -----------------------------------------------------------------------
    matrix calc_B_inv(const gaus& gi, const gaus& gj) const {
        return (gi.A + gj.A).inverse();
    }

    // -----------------------------------------------------------------------
    // gamma = 1/(4 c^T A_j B^{-1} A_i c)
    // Sets the width of the momentum-space Gaussian for kinetic energy.
    // -----------------------------------------------------------------------
    long double calc_gamma(const gaus& gi, const gaus& gj, const vector& c) const {
        matrix B_inv = calc_B_inv(gi, gj);
        matrix BRA   = gj.A * B_inv * gi.A;
        long double cBc = 0.0L;
        for (size_t i = 0; i < c.size(); ++i)
            for (size_t j = 0; j < c.size(); ++j)
                cBc += c[i] * BRA(i,j) * c[j];
        if (std::abs(cBc) < ZERO_LIMIT) return 0.0L;
        return 0.25L / cBc;
    }

    // -----------------------------------------------------------------------
    // eta: shift correction to kinetic energy (zero when all shifts = 0)
    // -----------------------------------------------------------------------
    long double calc_eta(const gaus& gi, const gaus& gj, const vector& c) const {
        matrix Rh = (gj.A.inverse() + gi.A.inverse()).inverse();
        size_t n  = gj.A.size1();
        long double eta = 0.0L;
        for (size_t d = 0; d < 3; ++d) {
            vector Ra(n), Rb(n);
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j) {
                    Ra[i] += Rh(i,j) * gi.s(j,d);
                    Rb[i] += Rh(i,j) * gj.s(j,d);
                }
            vector diff(n);
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    diff[i] += gi.A(i,j)*Rb[j] - gj.A(i,j)*Ra[j];
            for (size_t i = 0; i < n; ++i) eta += c[i] * diff[i];
        }
        return eta;
    }

    // -----------------------------------------------------------------------
    // K_cla: classical kinetic energy  <gi| -(hbar^2/2mu) nabla_{c^T x}^2 |gj>
    //
    // K = (hbar_c^2 / 2mu) * overlap * J
    // J = 1.5/gamma - eta^2
    // (for S-wave with no shifts: eta=0 and J = 6 c^T A_j B^{-1} A_i c)
    // -----------------------------------------------------------------------
    long double K_cla(const gaus& gi, const gaus& gj,
                      const vector& c, long double mu) const {
        long double ov = overlap(gi, gj);
        if (std::abs(ov) < ZERO_LIMIT) return 0.0L;
        long double gamma = calc_gamma(gi, gj, c);
        if (gamma < ZERO_LIMIT) return 0.0L;
        long double eta = calc_eta(gi, gj, c);
        long double J   = 1.5L / gamma - eta * eta;
        return (hbar_c * hbar_c / (2.0L * mu)) * ov * J;
    }

    // -----------------------------------------------------------------------
    // K_rel: semirelativistic kinetic energy  <gi| sqrt(p^2 + m^2) - m |gj>
    //
    // Computed in momentum space via 1D numerical integration (Simpson's rule).
    // The Gaussian in momentum space has effective width gamma (fm^{-2}).
    // The integrand (without the angular part) is:
    //
    //   f(p) = p^2 * T(p) * exp(-p^2 / (4 gamma))
    //
    // where T(p) = sqrt((hbar_c p)^2 + mu^2) - mu  [in MeV, p in fm^{-1}].
    //
    // front = (gamma/pi)^{3/2} * 4pi  (from d^3p angular integration and
    //         conversion from the Gaussian normalisation).
    // -----------------------------------------------------------------------
    long double K_rel(const gaus& gi, const gaus& gj,
                      const vector& c, long double mu) const {
        long double ov = overlap(gi, gj);
        if (std::abs(ov) < ZERO_LIMIT) return 0.0L;
        long double gamma = calc_gamma(gi, gj, c);
        if (gamma < ZERO_LIMIT) return 0.0L;

        // p_max ~ 10 sigma in momentum space (in fm^{-1})
        const int   n_steps = 200;
        long double p_max   = 10.0L * std::sqrt(gamma);
        long double h       = p_max / n_steps;

        // T(p) in MeV, p in fm^{-1}
        auto integrand = [&](long double p) -> long double {
            long double pc   = hbar_c * p;           // hbar_c * p  [MeV]
            long double T    = std::sqrt(pc*pc + mu*mu) - mu;
            return p * p * T * std::exp(-p*p / (4.0L * gamma));
        };

        // Simpson's rule
        long double sum = integrand(0.0L) + integrand(p_max);
        for (int i = 1; i < n_steps; ++i)
            sum += ((i % 2 == 0) ? 2.0L : 4.0L) * integrand(i * h);

        long double front = std::pow(gamma / pi, 1.5L) * 2.0L * pi;
        return ov * (h / 3.0L) * sum * front;
    }

    // -----------------------------------------------------------------------
    // V_cou: Coulomb potential  <gi| 1/r_c |gj>,  r_c = |c^T x|
    //
    // V = overlap * erf(sqrt(beta) * q) / q
    // where beta = 1/(c^T B^{-1} c)  and  q = |mean shift along c|.
    // -----------------------------------------------------------------------
    long double V_cou(const gaus& gi, const gaus& gj, const vector& c) const {
        long double ov = overlap(gi, gj);
        if (std::abs(ov) < ZERO_LIMIT) return 0.0L;
        matrix B_inv = calc_B_inv(gi, gj);

        long double cBc = 0.0L;
        for (size_t i = 0; i < c.size(); ++i)
            for (size_t j = 0; j < c.size(); ++j)
                cBc += c[i] * B_inv(i,j) * c[j];
        long double beta = 1.0L / cBc;

        long double q_sq = 0.0L;
        for (size_t d = 0; d < 3; ++d) {
            long double qd = 0.0L;
            for (size_t k = 0; k < c.size(); ++k) {
                long double t = 0.0L;
                for (size_t l = 0; l < c.size(); ++l)
                    t += B_inv(k,l) * (gj.A(l,k)*gj.s(l,d) + gi.A(l,k)*gi.s(l,d));
                qd += c[k] * t;
            }
            q_sq += qd * qd;
        }
        long double q = std::sqrt(q_sq);
        long double J = (q < 1e-12L) ? 2.0L * std::sqrt(beta / pi)
                                      : std::erf(std::sqrt(beta) * q) / q;
        return ov * J;
    }

    // -----------------------------------------------------------------------
    // W: pion-emission matrix element
    //   <g_bare| W_1 + W_2 + ... |g_clothed>
    //
    // where W_k = strength_k * (tau_k . pi)(sigma_k . r_k) * exp(-r_k^2/b^2)
    //
    // Parameters:
    //   g_bare    : bare-sector Gaussian (lower dimension)
    //   g_clothed : clothed-sector Gaussian (higher dimension)
    //   Omega     : form-factor kernel matrix  Omega_{ij} = c_i c_j / b^2
    //               (absorbed into g_bare after promotion)
    //   couplings : one NucleonCoupling per emitting nucleon.  Each carries:
    //               - c_coord : pion-nucleon projection (clothed Jacobi dim)
    //               - terms   : VertexTerms from apply_vertex
    //               - strength: S_pion / b_pion
    //
    // Returns: complex<long double> because r[1]=x+iy and r[2]=x-iy.
    //
    // Algorithm:
    //   1. Promote g_bare to clothed dimension; absorb Omega into A.
    //   2. Compute overlap M = <g_prom | g_clothed>.
    //   3. For each coupling k:
    //        a. Compute mean position matrix u (dim x 3):
    //           u[:,d] = (1/2) B^{-1} v[:,d],  v = s_prom + s_clothed.
    //        b. Project onto c_k to get c^T u for each Cartesian direction:
    //           ctu_x, ctu_y, ctu_z  (all real).
    //        c. Build spherical projections:
    //           (c^T u)_sph[0] = ctu_z           (real)
    //           (c^T u)_sph[1] = ctu_x + i ctu_y (complex)
    //           (c^T u)_sph[2] = ctu_x - i ctu_y (complex)
    //        d. Accumulate: sum_t sum_m  term.coeff[m] * (c^T u)_sph[m]
    //   4. Return total * M.
    // -----------------------------------------------------------------------
    cld W(const gaus& g_bare, const gaus& g_clothed,
          const matrix& Omega,
          const std::vector<NucleonCoupling>& couplings) const {

        size_t d2 = g_clothed.dim();
        assert(g_bare.dim() < d2);
        assert(Omega.size1() == d2);

        // Promote bare Gaussian to clothed dimension and absorb form factor
        gaus g_prom   = promote(g_bare, d2);
        g_prom.A      = g_prom.A + Omega;

        long double M = overlap(g_prom, g_clothed);
        if (std::abs(M) < ZERO_LIMIT) return cld(0);

        // Shared B^{-1}: same for all nucleon couplings
        matrix B_inv = (g_prom.A + g_clothed.A).inverse();

        // Mean position u[i][d] = (1/2) (B^{-1} v)_i  for Cartesian d
        // where v[i,d] = s_prom[i,d] + s_clothed[i,d]
        // We store u as a (d2 x 3) matrix of long doubles.
        matrix u_mat(d2, 3);
        for (size_t d = 0; d < 3; ++d) {
            vector v(d2);
            for (size_t i = 0; i < d2; ++i)
                v[i] = g_prom.s(i,d) + g_clothed.s(i,d);
            vector Bu = B_inv * v * 0.5L;
            for (size_t i = 0; i < d2; ++i) u_mat(i,d) = Bu[i];
        }

        cld total(0);

        for (const auto& coup : couplings) {
            if (coup.terms.empty()) continue;
            assert(coup.c_coord.size() == d2);

            // Cartesian projections: ctu_d = c^T u[:,d]  (all real)
            long double ctu[3] = {0.0L, 0.0L, 0.0L};
            for (size_t d = 0; d < 3; ++d)
                for (size_t i = 0; i < d2; ++i)
                    ctu[d] += coup.c_coord[i] * u_mat(i,d);

            // Spherical projections of c^T u:
            //   sph[0] = z       = ctu[2]          (real axis in Cartesian: x=0,y=1,z=2)
            //   sph[1] = x + iy  = ctu[0] + i*ctu[1]
            //   sph[2] = x - iy  = ctu[0] - i*ctu[1]
            std::array<cld, 3> sph = {
                cld(ctu[2], 0.0L),
                cld(ctu[0],  ctu[1]),
                cld(ctu[0], -ctu[1])
            };

            // Accumulate spin-isospin weighted spatial projections
            cld contrib(0);
            for (const auto& t : coup.terms) {
                for (size_t m = 0; m < 3; ++m)
                    contrib += t.coeff[m] * sph[m];
            }
            total += cld(coup.strength, 0.0L) * contrib;
        }

        return total * cld(M, 0.0L);
    }

    // -----------------------------------------------------------------------
    // R2_matrix_element: <gi| (c^T x)^2 |gj>  (used for charge radius)
    //
    // <r^2> = N_ij * (3/2 * B^{-1}_{00} + |u_0|^2)
    // -----------------------------------------------------------------------
    long double R2_matrix_element(const gaus& a, const gaus& b) const {
        long double ov = overlap(a, b);
        if (std::abs(ov) < ZERO_LIMIT) return 0.0L;
        matrix B_inv = calc_B_inv(a, b);
        long double term1 = 1.5L * B_inv(0,0);
        long double term2 = 0.0L;
        size_t d = a.dim();
        for (size_t k = 0; k < 3; ++k) {
            long double u0 = 0.0L;
            for (size_t j = 0; j < d; ++j)
                u0 += B_inv(0,j) * (a.s(j,k) + b.s(j,k));
            term2 += (0.5L * u0) * (0.5L * u0);
        }
        return ov * (term1 + term2);
    }
};

} // namespace qm