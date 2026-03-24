#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// hamiltonian.h
//
// Matrix element evaluation for the coupled meson-nucleon Hamiltonian.
// System-agnostic: works for sigma (scalar), pion (pseudoscalar), or any
// other meson, provided the caller supplies the correct Jacobi vectors and
// spin/isospin coefficients.
//
// This file provides:
//   1.  Kinetic energy blocks  (NR and relativistic, diagonal blocks)
//   2.  W-operator matrix elements  (off-diagonal coupling blocks)
//   3.  Pion mass / rest energy contribution to diagonal
//
// Conventions
// ───────────
//   n_bare  : number of Jacobi coordinates in the bare N-nucleon sector
//   n_dress : number of Jacobi coordinates in the dressed (N-nucleon + meson) sector
//   All A matrices and s vectors live in their respective Jacobi spaces.
//   "Promoted" means a bare-sector A padded to dress-sector size with zeros.
//
// Coupling operator for a pseudoscalar meson (pion):
//   W = S * (τ·π)(σ·w_πN^T r) * exp(-r^T C r)
// where C = w_πN w_πN^T / b^2 is absorbed into the effective A matrix.
// The scalar S and range b are free parameters fitted to experiment.
//
// The matrix element then factorises into:
//   <g_bare | W_channel | g_dress> = S * (isospin coeff) * (spin-spatial ME)
//
// The spin-spatial ME for each channel reduces to a position matrix element
// <g_bare_eff | w_coord^T r | g_dress>  times  the overlap  M_eff,
// where w_coord is z (no spin flip), x or y (spin flip channels).
// ─────────────────────────────────────────────────────────────────────────────

#include "gaussian.h"
#include "matrix.h"

using namespace qm;
// ─────────────────────────────────────────────────────────────────────────────
// §1  Kinetic energy matrix element (bare sector, NR or relativistic)
//
// For a system with multiple particles, the total kinetic energy is a sum
// over all particles. Here we provide both NR and relativistic versions.
//
// The kinetic-energy matrix K (diagonal, in Jacobi space) has entries
//   K[i][i] = hbar^2 / (2 mu_i)
// where mu_i is the reduced mass for Jacobi coordinate i.
//
// 'particles_c' : list of momentum extraction vectors c_i in Jacobi space
//                 (one per particle whose KE we include)
// 'masses'      : rest masses in natural units (for relativistic KE)
// 'K_matrix'    : diagonal matrix hbar^2/(2mu) in Jacobi space (for NR)
// ─────────────────────────────────────────────────────────────────────────────

// NR kinetic energy block element <g'_i | K_NR | g_j>
// K_matrix has entries K(i,i) = hbar^2/(2*mu_i) for each Jacobi coordinate
inline ld ke_NR_element(const rmat& A, const rvec& s,
                        const rmat& Ap, const rvec& sp,
                        const rmat& K_matrix)
{
    size_t n = A.size1();
    auto ov = gauss::overlap(n, A, s, Ap, sp);
    return gauss::kinetic_NR(A, Ap, K_matrix, ov);
}

// Relativistic KE block element — computes overlap ONCE, then loops over
// all Jacobi coordinates. This avoids redundant matrix inversions.
inline ld ke_rel_element(const rmat& A, const rvec& s,
                         const rmat& Ap, const rvec& sp,
                         const std::vector<rvec>& c_vecs,
                         const std::vector<ld>& red_masses,
                         int n_gl = 64)
{
    size_t n = A.size1();
    assert(c_vecs.size() == red_masses.size());
    // Compute overlap (and B, Binv) once — reused for every kinetic term
    auto ov = gauss::overlap(n, A, s, Ap, sp);
    ld result = 0;
    for (size_t k = 0; k < c_vecs.size(); k++)
        result += gauss::kinetic_rel(A, Ap, ov, c_vecs[k], red_masses[k], n_gl);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  Pion rest-mass contribution
//
// In the dressed sector the diagonal Hamiltonian contains m_pi (the pion
// rest mass). This is simply m_pi * <g'|g> for each pair of dressed states.
// ─────────────────────────────────────────────────────────────────────────────
inline ld meson_mass_element(const rmat& A, const rvec& s,
                             const rmat& Ap, const rvec& sp,
                             ld m_meson)
{
    size_t n = A.size1();
    auto ov = gauss::overlap(n, A, s, Ap, sp);
    return m_meson * ov.value;
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  W-operator matrix element
//
// For a SCALAR meson (sigma), W = S * exp(-r^T C r), and the matrix element
// is (Fedorov paper eq. 11, 24, 25):
//   <A_bare | W | A_dress> = S * <A_tilde | A_dress>
// where A_tilde = promote(A_bare) + w_piN w_piN^T / b^2.
// ─────────────────────────────────────────────────────────────────────────────

// Scalar meson W matrix element (no spin/isospin structure, e.g. sigma meson).
// A_bare  : correlation matrix in bare sector (n_bare × n_bare)
// A_dress : correlation matrix in dressed sector (n_dress × n_dress)
// s_bare, s_dress : shift vectors (set to zero for unshifted)
// w_piN   : pion-nucleon relative coordinate vector in dressed Jacobi space
// b       : form factor range
// S       : coupling strength
inline ld W_scalar(const rmat& A_bare,  const rvec& s_bare,
                   const rmat& A_dress, const rvec& s_dress,
                   const rvec& w_piN,
                   ld b, ld S)
{
    size_t n_dress = A_dress.size1();

    // Promote A_bare to dressed space
    rmat A_tilde = gauss::promote_A(A_bare, n_dress);

    // Absorb form factor: A_tilde += w_piN w_piN^T / b^2
    A_tilde = gauss::absorb_formfactor(A_tilde, w_piN, b);

    // Promote s_bare to dressed space (pad with zeros)
    rvec s_tilde(n_dress);
    for (size_t i = 0; i < s_bare.size(); i++) s_tilde[i] = s_bare[i];

    // Matrix element = S * <A_tilde | A_dress>
    auto ov = gauss::overlap(n_dress, A_tilde, s_tilde, A_dress, s_dress);
    return S * ov.value;
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Pseudoscalar (pion) W-operator matrix elements
//
// For a pseudoscalar meson, the coupling operator for nucleon i is:
//   W_i = (τ_i · π)(σ_i · w_πi^T r) f(w_πi^T r)
// where f is the Gaussian form factor.
//
// After absorbing the form factor into A_tilde (same as scalar case), the
// spin-spatial part becomes a POSITION matrix element:
//   <g_bare_eff | w_coord^T r | g_dress>
//
// which by eq.(4) in your bachelor equals:
//   (w_coord^T u_eff) * M_eff
//
// The three spatial channels are:
//   z-channel  (no spin flip)   : w_coord = w_z  (z-component of w_πN in Jacobi)
//   x-channel  (spin flip)      : w_coord = w_x  (x-component, gives r+)
//   y-channel  (spin flip)      : w_coord = w_y  (y-component, gives r-)
//
// In practice since we work with real Gaussians we separate into:
//   - No spin flip  : coefficient * (w_πN^T u) * M   [z-component]
//   - Spin flip     : coefficient * (w_πN^T u) * M   [x or y component]
// but since the Gaussians are spherically symmetric, the x, y, z contributions
// are equal in magnitude and handled by a factor of sqrt(2) from the spherical
// basis (see section 3.5 of your bachelor).
//
// PionChannel: specifies which spin-isospin channel this W element is for.
// ─────────────────────────────────────────────────────────────────────────────

// Spin-spatial W matrix element for a pseudoscalar meson, single channel.
//
// A_bare  : bare sector (n_bare × n_bare)
// A_dress : dressed sector (n_dress × n_dress)
// s_bare, s_dress : shift vectors
// w_piN   : pion-nucleon relative coordinate in dressed Jacobi space
//           (also used as the spatial direction for σ·r)
// b       : form factor range
// S       : coupling strength
// iso_coeff : isospin coefficient for this channel (+1, -1, or sqrt(2))
// spatial_component : 0=z (no flip), 1=x, 2=y (spin flip channels)
//                    For unshifted real Gaussians, x and y give the same
//                    result as z (by spherical symmetry), so this parameter
//                    is used as a label; the actual computation uses w_piN.
//
// Returns: iso_coeff * S * (w_piN^T u_eff) * M_eff
//          where u_eff and M_eff come from the effective overlap with A_tilde.
//
// IMPORTANT: The caller must multiply by the spin coefficient (1 or sqrt(2))
// for spin-flip vs no-flip channels, as described in section 3.5 of your bachelor.
inline ld W_pseudoscalar_spatial(const rmat& A_bare,  const rvec& s_bare,
                                 const rmat& A_dress, const rvec& s_dress,
                                 const rvec& w_piN,
                                 ld b, ld S, ld iso_coeff)
{
    size_t n_dress = A_dress.size1();

    // Promote and absorb form factor
    rmat A_tilde = gauss::promote_A(A_bare, n_dress);
    A_tilde = gauss::absorb_formfactor(A_tilde, w_piN, b);

    rvec s_tilde(n_dress);
    for (size_t i = 0; i < s_bare.size(); i++) s_tilde[i] = s_bare[i];

    // Effective overlap
    auto ov = gauss::overlap(n_dress, A_tilde, s_tilde, A_dress, s_dress);

    // Position matrix element: (w_piN^T u) * M
    ld pos = gauss::position_element(w_piN, ov);

    return iso_coeff * S * pos * ov.value;
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Overlap matrix element (needed for building the N matrix)
// ─────────────────────────────────────────────────────────────────────────────
inline ld overlap_element(const rmat& A, const rvec& s,
                          const rmat& Ap, const rvec& sp)
{
    size_t n = A.size1();
    return gauss::overlap(n, A, s, Ap, sp).value;
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  Charge radius matrix element
//
// <A | r^T w_i w_i^T r | A'> = (3/2) w_i^T B^{-1} w_i * <A|A'>
// (Fedorov paper eq. 37)
//
// Returns the matrix element (without the charge Z_i factor).
// To get total charge radius: sum over charged particles Z_i * this.
// ─────────────────────────────────────────────────────────────────────────────
inline ld charge_radius_element(const rmat& A, const rvec& s,
                                const rmat& Ap, const rvec& sp,
                                const rvec& w_i)
{
    size_t n = A.size1();
    auto ov = gauss::overlap(n, A, s, Ap, sp);
    // w_i^T Binv w_i
    ld wBw = 0;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            wBw += w_i[i] * ov.Binv(i,j) * w_i[j];
    return 1.5L * wBw * ov.value;
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  Build the kinetic energy matrix K (diagonal, in Jacobi space)
//     for both NR and relativistic cases.
//
// For n Jacobi coordinates with reduced masses mu[0]...mu[n-1]:
//   K(i,i) = hbar^2 / (2 mu[i])
//
// In natural units with hbar=c=1 and lengths in fm, masses in MeV:
//   hbar*c = 197.3269804 MeV*fm => hbar^2/c^2 in MeV^2 * fm^2
//   K(i,i) = (hbar*c)^2 / (2 mu[i] c^2)  [MeV * fm^2 ... but we work in fm^{-2}]
//
// Actually since NR KE = p^2/(2m) and p is in fm^{-1} * hbar:
//   K(i,i) = hbar^2 / (2 mu_i)  =  (hbar*c)^2 / (2 mu_i c^2)
// in units where distances are fm and energies are MeV:
//   K(i,i) = 197.3269804^2 / (2 * mu_MeV)  [MeV * fm^2]
// ─────────────────────────────────────────────────────────────────────────────
constexpr ld hbar_c = 197.3269804L;   // MeV * fm

inline rmat build_K_matrix(const std::vector<ld>& reduced_masses_MeV, size_t n)
{
    rmat K(n, n);  // zero-initialised
    for (size_t i = 0; i < std::min(n, reduced_masses_MeV.size()); i++)
        K(i,i) = (hbar_c * hbar_c) / (2.0L * reduced_masses_MeV[i]);
    return K;
}

// Reduced mass for two particles of mass m1, m2 (in MeV)
inline ld reduced_mass(ld m1, ld m2) { return m1 * m2 / (m1 + m2); }

// Reduced mass for meson vs (n-body system of total mass M)
inline ld reduced_mass_meson(ld m_meson, ld M_nucleons)
{
    return m_meson * M_nucleons / (m_meson + M_nucleons);
}

} // namespace ham