#pragma once

#include "matrix.h"
#include "jacobi.h"
#include "gaussian.h"
#include <cmath>
#include <cassert>
#include <functional>
#include <iostream>
#include <iomanip>
#include <omp.h>

// ─────────────────────────────────────────────────────────────────────────────
// hamiltonian.h  —  Matrix elements for a general multi-channel Hamiltonian
//
// Provides the building blocks for the (9K × 9K) block Hamiltonian.
// The specific channel layout (which pion, which isospin coefficient, which
// spin type) is defined in main.cc via the Channel struct — this header is
// deliberately general.
//
// ── What lives here ──────────────────────────────────────────────────────────
//
//   KineticParams       —  γ and η for one Jacobi mode
//   ke_classical()      —  ⟨g'|(c·p)²|g⟩  (thesis eq.8–9)
//   ke_relativistic()   —  ⟨g'|√((c·p)²+m²)−m|g⟩  (thesis eq.11–12)
//   kinetic_energy()    —  sum over all Jacobi modes
//   SpinType            —  NO_FLIP / SPIN_FLIP
//   Channel             —  descriptor for one block-channel
//   w_matrix_element()  —  ⟨g'_dressed|(σ·r_πN)f(r)|g_bare⟩
//   HamiltonianBuilder  —  assembles the full complex H and N matrices
//
// ── Units ────────────────────────────────────────────────────────────────────
//
//   Masses and energies : MeV
//   Lengths             : fm
//   A matrix            : fm⁻²   (so that r^T A r is dimensionless)
//   Momenta             : fm⁻¹   (ħ = 1 convention inside integrals)
//   ħc                  : 197.3269804 MeV·fm
//
//   Because A is in fm⁻², the integration variable in ke_relativistic is
//   also in fm⁻¹.  The particle mass must therefore be converted to fm⁻¹
//   before entering the dispersion relation, and the result converted back
//   to MeV via ħc.  See ke_relativistic() for the explicit conversion.
// ─────────────────────────────────────────────────────────────────────────────

namespace qm {

// ─────────────────────────────────────────────────────────────────────────────
// KineticParams  —  γ and η for the kinetic energy integral of mode c
//
// For a Jacobi mode described by the unit vector c, given a GaussianPair:
//
//   1/γ = 4 c^T A_bra B⁻¹ A_ket c
//   η   = c^T ( A_bra B⁻¹ s_ket  −  A_ket B⁻¹ s_bra )
//
// Both classical and relativistic KE integrals are parameterised by γ and η.
// For zero-shift Gaussians (s = s' = 0) η = 0 exactly.
//
// Construction:
//   KineticParams kp(gp, c);    // gp is a GaussianPair, c is the mode vector
// ─────────────────────────────────────────────────────────────────────────────
struct KineticParams {
    ld gamma;      // γ
    ld inv_gamma;  // 1/γ  =  4 c^T A_bra B⁻¹ A_ket c
    ld eta;        // η

    KineticParams() : gamma(0), inv_gamma(0), eta(0) {}

    KineticParams(const GaussianPair& gp, const rvec& c)
    {
        assert(c.size() == gp.dim);

        // 1/γ = 4 c^T A_bra ( B⁻¹ A_ket c )
        rvec Binv_Aket_c = gp.Binv * (gp.ket.A * c);
        inv_gamma = ld{4} * dot_no_conj(c, gp.bra.A * Binv_Aket_c);
        gamma     = (inv_gamma > ld{1e-30L}) ? ld{1} / inv_gamma : ld{0};

        // η = c^T ( A_bra B⁻¹ s_ket  −  A_ket B⁻¹ s_bra )
        rvec term = gp.bra.A * (gp.Binv * gp.ket.s)
                  - gp.ket.A * (gp.Binv * gp.bra.s);
        eta = dot_no_conj(c, term);
    }

    bool is_zero_shift() const { return std::abs(eta) < ld{1e-12L}; }
};


// ─────────────────────────────────────────────────────────────────────────────
// Numerical integration  (32-point Gauss-Legendre on [0, x_max])
//
// Gauss-Legendre places its sample points at the roots of the 32nd Legendre
// polynomial, with matching weights chosen so the rule is exact for all
// polynomials up to degree 63.  For smooth, exponentially-decaying integrands
// (like our Gaussian-weighted integrals) 32 points gives ~machine precision.
//
// The standard rule lives on [−1, 1].  We map it to [0, x_max] by
// substituting t = x_max * (1 + ξ)/2  (ξ ∈ [−1,1], t ∈ [0, x_max]).
// Each node and weight pair below is the pre-mapped version on [0, 1];
// the integrate() function then scales to [0, x_max].
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

// Initialise 32 Gauss-Legendre nodes and weights on [0, 1]
inline void gauss_legendre_nodes(std::vector<ld>& x, std::vector<ld>& w) {
    // Positive roots of P_32 (symmetric about 0), standard tables
    static const double nodes_half[] = {
        0.0483076656877383162,  0.1444719615827964935,
        0.2392873622521370745,  0.3318686022821276498,
        0.4213512761306353454,  0.5068999089322293901,
        0.5877157572407623291,  0.6630442669302152009,
        0.7321821187402896804,  0.7944837959679424070,
        0.8493676137325699701,  0.8963211557660521239,
        0.9349060759377396892,  0.9647622555875064307,
        0.9856115115452684055,  0.9972638618494815636
    };
    static const double weights_half[] = {
        0.0965400885147278006,  0.0956387200792748594,
        0.0938443990808045654,  0.0911738786957638847,
        0.0876520930044038111,  0.0833119242269467552,
        0.0781938957870703065,  0.0723457941088485062,
        0.0658222227763618468,  0.0586840934785355471,
        0.0509980592623761762,  0.0428358980222266807,
        0.0342738629130214331,  0.0253920653092620595,
        0.0162743947309056706,  0.0070186100094701000
    };
    x.resize(32); w.resize(32);
    for (int i = 0; i < 16; i++) {
        // Map from [−1,1] to [0,1]: t = (1 + node)/2
        x[i]    = ld{0.5} * (ld{1} - static_cast<ld>(nodes_half[15-i]));
        x[31-i] = ld{0.5} * (ld{1} + static_cast<ld>(nodes_half[i]));
        w[i]    = ld{0.5} * static_cast<ld>(weights_half[15-i]);
        w[31-i] = ld{0.5} * static_cast<ld>(weights_half[i]);
    }
}

// Integrate f over [0, x_max] using 32-point Gauss-Legendre.
// f is passed as a std::function<ld(ld)> — any lambda or callable works.
inline ld integrate(std::function<ld(ld)> f, ld x_max) {
    // Static storage so nodes/weights are computed only once per program run
    static std::vector<ld> nodes, weights;
    static bool initialised = false;
    if (!initialised) {
        #pragma omp critical 
        {
            if (!initialised) {
            gauss_legendre_nodes(nodes, weights);
            initialised = true;
            }
        }
    }
    ld sum = ld{0};
    for (size_t i = 0; i < nodes.size(); i++)
        sum += weights[i] * f(nodes[i] * x_max);
    return sum * x_max;  // Jacobian of the [0,1] → [0,x_max] mapping
}

} // namespace detail


// ─────────────────────────────────────────────────────────────────────────────
// Classical kinetic energy matrix element  (thesis eq.8–9)
//
//   ⟨g'| (c·p)² |g⟩
//
// Shifted (η ≠ 0):
//   = M * [ (3/2)(1/γ) − η² ]
//
// Unshifted (η = 0):
//   = M * (3/2)(1/γ)  =  M * 6 c^T A_bra B⁻¹ A_ket c
//
// Returns M * spatial_factor.  The caller multiplies by (ħc)²/(2μ) to get MeV.
//
// Why no (ħc)²/(2μ) here?  This function returns the pure spatial integral.
// kinetic_energy() applies the physical prefactor when summing over modes.
// ─────────────────────────────────────────────────────────────────────────────
inline ld ke_classical(const GaussianPair&  gp,
                       const KineticParams& kp)
{
    return gp.M * (ld{1.5L} * kp.inv_gamma - kp.eta * kp.eta);
}


// ─────────────────────────────────────────────────────────────────────────────
// Relativistic kinetic energy matrix element  (thesis eq.11–12)
//
//   ⟨g'| √((c·p)² + m²) − m |g⟩
//
// Unshifted (η = 0):
//   = M (γ/π)^{3/2} 4π  ∫₀^∞ x² e^{−γx²} [√(x²+m_nat²) − m_nat] dx
//
// Shifted (η ≠ 0):
//   = M (γ/π)^{3/2} 2π [exp(γη²)/(γη)]
//     ∫₀^∞ x e^{−γx²} sin(2γηx) [√(x²+m_nat²) − m_nat] dx
//
// Unit conversion:
//   The integration variable x is in fm⁻¹ (because γ is in fm²).
//   The dispersion relation needs m in the same units:
//     m_nat [fm⁻¹] = mass_MeV [MeV] / (ħc [MeV·fm])
//   The integral result is in fm⁻¹; multiplying by ħc gives MeV.
//
//   x_max = 6/√γ  captures the Gaussian envelope to ~e^{−36} ≈ 10^{−16}.
// ─────────────────────────────────────────────────────────────────────────────
inline ld ke_relativistic(const GaussianPair&  gp,
                           const KineticParams& kp,
                           ld                   mass_MeV)
{
    if (kp.gamma < ld{1e-30L}) return ld{0};

    // Convert mass to natural momentum units (fm⁻¹)
    ld m_nat = mass_MeV / phys::hbar_c;
    ld x_max = ld{6} / std::sqrt(kp.gamma);

    // Relativistic dispersion in fm⁻¹:  √(x² + m_nat²) − m_nat
    // Defined as a lambda so we write it once and reuse below.
    // [&] captures m_nat from the enclosing scope.
    // (ld x) is the integration variable.
    // -> ld declares the return type explicitly.
    auto disp = [&](ld x) -> ld {
        return std::sqrt(x*x + m_nat*m_nat) - m_nat;
    };

    ld prefactor, integral;

    if (kp.is_zero_shift()) {
        // Unshifted formula (thesis eq.12)
        prefactor = gp.M
                  * std::pow(kp.gamma / static_cast<ld>(M_PI), ld{1.5L})
                  * ld{4} * static_cast<ld>(M_PI);

        // Lambda for the integrand: x² e^{−γx²} [√(x²+m²)−m]
        // [&] captures kp and disp.
        integral = detail::integrate([&](ld x) -> ld {
            return x * x * std::exp(-kp.gamma * x * x) * disp(x);
        }, x_max);

    } else {
        // Shifted formula (thesis eq.11)
        ld geta = kp.gamma * kp.eta;
        prefactor = gp.M
                  * std::pow(kp.gamma / static_cast<ld>(M_PI), ld{1.5L})
                  * ld{2} * static_cast<ld>(M_PI)
                  * std::exp(kp.gamma * kp.eta * kp.eta)
                  / geta;

        // Lambda for the integrand: x e^{−γx²} sin(2γηx) [√(x²+m²)−m]
        integral = detail::integrate([&](ld x) -> ld {
            return x * std::exp(-kp.gamma * x * x)
                     * std::sin(ld{2} * geta * x)
                     * disp(x);
        }, x_max);
    }

    // Convert integral [fm⁻¹] → MeV by multiplying by ħc
    return phys::hbar_c * prefactor * integral;
}


// ─────────────────────────────────────────────────────────────────────────────
// Total kinetic energy  ⟨g'|T|g⟩  summed over all Jacobi modes
//
// Classical:
//   T = Σ_α  (ħc)² / (2μ_α)  *  ke_classical(gp, kp_α)
//
// Relativistic:
//   T = Σ_α  ke_relativistic(gp, kp_α, μ_α)
//   (the (ħc)² / (2μ) factor is implicit inside ke_relativistic)
//
// Parameters:
//   gp           — GaussianPair (both bra and ket must have dim = N−1)
//   sys          — JacobiSystem (provides c_vec[α] and μ[α])
//   relativistic — true → K^rel,  false → K^cla
// ─────────────────────────────────────────────────────────────────────────────
inline ld kinetic_energy(const GaussianPair& gp,
                          const JacobiSystem& sys,
                          bool                relativistic)
{
    assert(gp.dim == sys.N - 1);
    ld total = ld{0};
    for (size_t alpha = 0; alpha < gp.dim; alpha++) {
        KineticParams kp(gp, sys.c_vec[alpha]);
        if (relativistic) {
            total += ke_relativistic(gp, kp, sys.mu[alpha]);
        } else {
            total += (phys::hbar_c2 / (ld{2} * sys.mu[alpha]))
                     * ke_classical(gp, kp);
        }
    }
    return total;
}


// ─────────────────────────────────────────────────────────────────────────────
// SpinType — which spin component of the W operator
//
// NO_FLIP  :  σ_z component  →  z_πN              (real, no spin flip)
// SPIN_FLIP:  σ_+ component  →  x_πN + i y_πN     (complex, spin flip)
// SCALAR   :  no spin/position factor              (real, scalar-isoscalar)
//             W = C_iso · S · exp(−Σ_α x_α² / b²)
//             Used for the σ-meson model (Fedorov 2020).
//             The form factor penalises ALL (N−1) Jacobi coordinates, not
//             just the pion-nucleon separation, so build_A_tilde() is not
//             used; instead (1/b²)·I is added to the promoted matrix.
// ─────────────────────────────────────────────────────────────────────────────
enum class SpinType { NO_FLIP, SPIN_FLIP, SCALAR };


// ─────────────────────────────────────────────────────────────────────────────
// Channel  —  physical descriptor for one block-channel in the Hamiltonian
//
// Defined in main.cc via build_channels() for the specific p+n+π system.
// This struct is used by HamiltonianBuilder to assemble the matrix.
// ─────────────────────────────────────────────────────────────────────────────
struct Channel {
    int      index;      // channel index 0..8
    bool     is_bare;    // true only for the bare pn channel (index 0)
    ld       pion_mass;  // 0 for bare,  m_π for dressed  (MeV)
    ld       iso_coeff;  // isospin coefficient from τ·π algebra
    rvec     w_piN;      // Jacobi vector r_π − r_nucleon  (dim = N−1)
    SpinType spin_type;  // NO_FLIP or SPIN_FLIP
    size_t   dim;        // 1 (bare) or N−1 (dressed)
};


// ─────────────────────────────────────────────────────────────────────────────
// W-operator matrix element  ⟨g'_dressed | (σ·r_πN) f(r_πN) | g_bare⟩
//
// The pion–nucleon coupling operator is:
//   W = C_iso * S * (σ·r_πN) * exp(−r_πN² / b²)
//
// The form factor exp(−r_πN²/b²) is absorbed into the bare Gaussian's A
// matrix by JacobiSystem::build_A_tilde(), producing an effective ket.
//
// The spin operator (σ·r_πN) acting on |↑⟩:
//   NO_FLIP:   σ_z|↑⟩ = |↑⟩  →  factor z_πN             (real)
//   SPIN_FLIP: σ_+|↑⟩ = |↓⟩  →  factor (x_πN + i y_πN)  (complex)
//
// Both cases extract the same spatial magnitude via dot(w_piN, u)*M.
// The spin-flip case contributes to both real (x) and imaginary (y) parts
// equally, since x and y are rotationally equivalent.
//
// Returns a complex number:  real + i*imag
// ─────────────────────────────────────────────────────────────────────────────
inline cld w_matrix_element(const Gaussian&     g_bra_dressed,
                             const Gaussian&     g_ket_bare,
                             const JacobiSystem& sys,
                             const rvec&         w_piN,
                             ld                  b,
                             ld                  S,
                             ld                  C_iso,
                             SpinType            spin_type)
{
    assert(g_bra_dressed.dim == sys.N - 1);
    assert(g_ket_bare.dim    == 1);

    // Promote bare A to the full (N−1)-dim space and build the shift vector.
    rmat A_prom = sys.promote(g_ket_bare.A, g_ket_bare.dim);
    rvec s_full(sys.N - 1);
    for (size_t k = 0; k < g_ket_bare.dim; k++) s_full[k] = g_ket_bare.s[k];

    // ── SCALAR branch  (σ-meson / scalar-isoscalar coupling) ─────────────────
    // W = C_iso · S · exp(−(x₀² + x₁² + …) / b²)
    //
    // The form factor couples all (N−1) Jacobi coordinates symmetrically.
    // Absorb it by adding (1/b²)·I to the promoted matrix — equivalent to
    // A_tilde = A_prom + (1/b²) I  — rather than the pion-style rank-1 update.
    // There is no gradient/position prefactor; the matrix element is a pure
    // (modified) overlap:  C_iso · S · ⟨g_bra | g_ket_eff⟩.
    if (spin_type == SpinType::SCALAR) {
        rmat A_tilde = A_prom;
        ld inv_b2 = ld{1} / (b * b);
        for (size_t k = 0; k < sys.N - 1; k++)
            A_tilde(k, k) += inv_b2;
        Gaussian g_ket_eff(A_tilde, s_full);
        GaussianPair gp(g_bra_dressed, g_ket_eff);
        return cld{C_iso * S * gp.M, ld{0}};
    }

    // ── Pion (gradient-coupling) branch ──────────────────────────────────────
    // W = C_iso · S · (σ·r_πN) · exp(−r_πN² / b²)
    //
    // Absorb the form factor via the rank-1 update  A_tilde = A_prom + w·w^T/b².
    rmat A_tilde = sys.build_A_tilde(A_prom, w_piN, b);
    Gaussian g_ket_eff(A_tilde, s_full);
    GaussianPair gp(g_bra_dressed, g_ket_eff);

    // Spatial factor: ⟨g'|r_πN|g_eff⟩ = dot(w_piN, u) · M
    ld r_piN = dot_no_conj(w_piN, gp.u) * gp.M;

    ld coeff = C_iso * S;
    switch (spin_type) {
        case SpinType::NO_FLIP:
            // σ_z contribution: purely real
            return cld{coeff * r_piN, ld{0}};
        case SpinType::SPIN_FLIP:
            // σ_+ contribution: x + iy (x and y components equal by symmetry)
            return cld{coeff * r_piN, coeff * r_piN};
        default:
            return cld{0, 0};  // unreachable
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// HamiltonianBuilder  —  assembles the full complex H and N matrices
//
// Takes a vector of Channel descriptors (built in main.cc) and two basis
// vectors: one for the bare channel (dim=1) and one for all dressed channels
// (dim=N-1, shared across channels 1..8).
//
// Block layout in the (9K × 9K) matrix:
//   Rows/cols 0       ..  K_bare-1          : bare channel
//   Rows/cols K_bare  ..  K_bare+K_dress-1  : dressed channel 1
//   Rows/cols K_bare + K_dress .. etc        : dressed channels 2..8
//
// Diagonal blocks:  kinetic energy + (pion rest mass × overlap) for dressed
// Off-diagonal blocks 0↔a (a≥1):  W coupling and its adjoint W†
// All other off-diagonal blocks:  zero
// ─────────────────────────────────────────────────────────────────────────────
class HamiltonianBuilder {
public:
    const JacobiSystem&          sys;
    const std::vector<Channel>&  channels;
    const std::vector<Gaussian>& basis_bare;
    const std::vector<Gaussian>& basis_dressed;
    ld   b;             // form-factor range (fm)
    ld   S;             // coupling strength (MeV)
    bool relativistic;  // true → K^rel,  false → K^cla

    size_t K_bare;
    size_t K_dress;
    size_t n_channels;
    size_t total_dim;

    HamiltonianBuilder(const JacobiSystem&          sys_,
                       const std::vector<Channel>&  ch_,
                       const std::vector<Gaussian>& b_bare,
                       const std::vector<Gaussian>& b_dress,
                       ld b_, ld S_, bool rel_)
        : sys(sys_), channels(ch_),
          basis_bare(b_bare), basis_dressed(b_dress),
          b(b_), S(S_), relativistic(rel_),
          K_bare(b_bare.size()), K_dress(b_dress.size()),
          n_channels(ch_.size()),
          total_dim(b_bare.size() + (ch_.size()-1) * b_dress.size())
    {}

    // Row offset of channel a in the full matrix
    size_t offset(size_t a) const {
        return (a == 0) ? 0 : K_bare + (a - 1) * K_dress;
    }

    // Basis size for channel a
    size_t block_size(size_t a) const {
        return (a == 0) ? K_bare : K_dress;
    }

    // ── Overlap matrix N ─────────────────────────────────────────────────────
    cmat build_N() const {
        cmat N = zeros<cld>(total_dim, total_dim);
        for (size_t a = 0; a < n_channels; a++) {
            size_t off = offset(a);
            size_t sz  = block_size(a);
            const auto& bas = (a == 0) ? basis_bare : basis_dressed;

            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t i = 0; i < sz; i++)
                for (size_t j = 0; j < sz; j++) {
                    GaussianPair gp(bas[i], bas[j]);
                    N(off+i, off+j) = cld{gp.M, ld{0}};
                }
        }
        return N;
    }

    // ── Hamiltonian matrix H ─────────────────────────────────────────────────
    cmat build_H() const {
        cmat H = zeros<cld>(total_dim, total_dim);

        // ── Diagonal blocks: kinetic energy (+ pion rest mass for dressed) ──
        for (size_t a = 0; a < n_channels; a++) {
            size_t off = offset(a);
            size_t sz  = block_size(a);
            const auto& bas = (a == 0) ? basis_bare : basis_dressed;

            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t i = 0; i < sz; i++) {
                for (size_t j = 0; j < sz; j++) {
                    GaussianPair gp(bas[i], bas[j]);
                    ld ke;

                    if (a == 0) {
                        // Bare channel: 1D, only the pn relative mode (μ_0)
                        rvec c0_bare(1); c0_bare[0] = ld{1};
                        KineticParams kp(gp, c0_bare);
                        if (relativistic)
                            ke = ke_relativistic(gp, kp, sys.mu[0]);
                        else
                            ke = (phys::hbar_c2 / (ld{2} * sys.mu[0]))
                                 * ke_classical(gp, kp);
                    } else {
                        // Dressed channel: full KE over all modes
                        ke = kinetic_energy(gp, sys, relativistic);
                        // Add pion rest mass contribution: m_π * ⟨g'|g⟩
                        ke += channels[a].pion_mass * gp.M;
                    }

                    H(off+i, off+j) += cld{ke, ld{0}};
                }
            }
        }

        // ── Off-diagonal W-coupling blocks: bare ↔ each dressed channel ─────
        for (size_t a = 1; a < n_channels; a++) {
            const Channel& ch = channels[a];
            size_t off_0 = offset(0);
            size_t off_a = offset(a);

            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t i = 0; i < K_dress; i++) {    // dressed index
                for (size_t j = 0; j < K_bare; j++) { // bare index
                    cld w = w_matrix_element(
                        basis_dressed[i], basis_bare[j],
                        sys, ch.w_piN, b, S,
                        ch.iso_coeff, ch.spin_type
                    );
                    H(off_a + i, off_0 + j) += w;             // W
                    H(off_0 + j, off_a + i) += std::conj(w);  // W†
                }
            }
        }

        return H;
    }
};

} // namespace qm