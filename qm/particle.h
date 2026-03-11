#pragma once
// particle.h  --  particle types, spin/isospin algebra, and the pion-emission
// vertex operator W = (tau.pi)(sigma.r) f(r).
//
// The key struct is VertexTerm, which encodes the spin-isospin coefficient for
// one nucleon's contribution to W.  The spatial part is handled separately in
// hamiltonian.h via NucleonCoupling.
//
// Spherical components of (sigma . r):
//
//   sigma.r = sigma_z * r[0]  +  sigma_-  * r[1]  +  sigma_+ * r[2]
//
// where:
//   r[0] = z              (real)
//   r[1] = x + iy         (complex, sigma_- lowers spin: |up> -> |down>)
//   r[2] = x - iy         (complex, sigma_+ raises spin: |down> -> |up>)
//
// A VertexTerm stores coeff[m] for m in {0,1,2}.  Given the mean Gaussian
// position u and pion projection vector c, the spatial contribution is:
//
//   sum_m  coeff[m] * (c^T u)_sph[m]
//
// where  (c^T u)_sph[0] = c^T u_z,
//        (c^T u)_sph[1] = c^T u_x + i * c^T u_y,
//        (c^T u)_sph[2] = c^T u_x - i * c^T u_y.

#include <string>
#include <complex>
#include <cmath>
#include <vector>
#include <array>

namespace qm {

using cld = std::complex<long double>;

// ---------------------------------------------------------------------------
// Ladder operator result
// ---------------------------------------------------------------------------
struct OpResult {
    long double coef   = 0.0;
    bool        valid  = false;
    double      new_sz = 0.0;
    double      new_tz = 0.0;
};

// ---------------------------------------------------------------------------
// Base particle
// ---------------------------------------------------------------------------
struct Particle {
    std::string name;
    long double mass;
    int         charge;
    double s, sz;  // spin quantum numbers
    double t, tz;  // isospin quantum numbers

    Particle() = default;
    Particle(std::string n, long double m, int c,
             double spin, double spin_z, double isospin, double iso_z)
        : name(n), mass(m), charge(c), s(spin), sz(spin_z), t(isospin), tz(iso_z) {}

    // |j, m+1> coefficient
    OpResult ladder_plus(double j, double m) const {
        if (m + 1.0 > j + 0.001) return {0.0, false, sz, tz};
        return {std::sqrt(j*(j+1.0) - m*(m+1.0)), true, sz, tz};
    }
    // |j, m-1> coefficient
    OpResult ladder_minus(double j, double m) const {
        if (m - 1.0 < -j - 0.001) return {0.0, false, sz, tz};
        return {std::sqrt(j*(j+1.0) - m*(m-1.0)), true, sz, tz};
    }
};

// ---------------------------------------------------------------------------
// Nucleon  (spin-1/2, isospin-1/2)
// ---------------------------------------------------------------------------
struct Nucleon : public Particle {
    Nucleon(std::string n, long double m, int c, double spin_z, double iso_z)
        : Particle(n, m, c, 0.5, spin_z, 0.5, iso_z) {}

    static Nucleon Proton (double sz = 0.5) { return {"proton",  938.272046, +1, sz, +0.5}; }
    static Nucleon Neutron(double sz = 0.5) { return {"neutron", 939.565379,  0, sz, -0.5}; }

    // --- isospin: tau = 2T ---
    OpResult apply_tau_z()     const { return {2.0*tz, true, sz, tz}; }
    OpResult apply_tau_plus()  const { auto r=ladder_plus (t,tz); if(r.valid) r.new_tz=tz+1; return r; }
    OpResult apply_tau_minus() const { auto r=ladder_minus(t,tz); if(r.valid) r.new_tz=tz-1; return r; }

    // --- spin: sigma = 2S ---
    OpResult apply_sigma_z()     const { return {2.0*sz, true, sz, tz}; }
    OpResult apply_sigma_plus()  const { auto r=ladder_plus (s,sz); if(r.valid) r.new_sz=sz+1; return r; }
    OpResult apply_sigma_minus() const { auto r=ladder_minus(s,sz); if(r.valid) r.new_sz=sz-1; return r; }
};

// ---------------------------------------------------------------------------
// Pion  (isospin-1 triplet, spin-0)
// ---------------------------------------------------------------------------
struct Pion : public Particle {
    Pion(std::string n, long double m, int c, double iso_z)
        : Particle(n, m, c, 0.0, 0.0, 1.0, iso_z) {}

    static Pion PiPlus () { return {"pi+", 139.5704, +1, +1.0}; }
    static Pion PiZero () { return {"pi0", 134.9768,  0,  0.0}; }
    static Pion PiMinus() { return {"pi-", 139.5704, -1, -1.0}; }
};

// ---------------------------------------------------------------------------
// Scalar isoscalar meson (sigma)
// ---------------------------------------------------------------------------
struct Meson : public Particle {
    Meson(std::string n, long double m) : Particle(n, m, 0, 0.0, 0.0, 0.0, 0.0) {}
    static Meson Sigma() { return {"sigma", 500.0}; }
};

// ---------------------------------------------------------------------------
// VertexTerm
//
// Represents the spin-isospin coefficient of ONE nucleon's contribution to the
// W operator: W_k = (tau_k . pi)(sigma_k . r_k) f(r_k).
//
// coeff[m] = (tau.pi coefficient) * (sigma_m matrix element)
// for spherical component m in {0,1,2}:
//   m=0: sigma_z  component  (r[0] = z,     real)
//   m=1: sigma_-  component  (r[1] = x+iy,  complex)
//   m=2: sigma_+  component  (r[2] = x-iy,  complex)
//
// bra_sz / bra_tz: output (bra) spin/isospin of this nucleon after the operator acts.
// These must match the bra sector to contribute non-zero.
// ---------------------------------------------------------------------------
struct VertexTerm {
    std::array<cld, 3> coeff = {cld(0), cld(0), cld(0)};
    double bra_sz = 0.5;
    double bra_tz = 0.5;

    bool is_zero() const {
        return std::abs(coeff[0]) < 1e-14 &&
               std::abs(coeff[1]) < 1e-14 &&
               std::abs(coeff[2]) < 1e-14;
    }
};

// ---------------------------------------------------------------------------
// apply_vertex
//
// Computes all VertexTerms from (tau.pi)(sigma.r) acting on ket_nuc emitting
// the given pion.
//
// The isospin vertex (tau.pi):
//   tau.pi = tau_z pi0 + sqrt(2) tau_- pi+ + sqrt(2) tau_+ pi-
//
// For each isospin-allowed transition, the spin part gives three possible
// terms (one per spherical component of sigma.r):
//
//   m=0: <bra_sz| sigma_z |ket_sz>  = 2*ket_sz  (preserves spin)
//   m=1: <bra_sz| sigma_- |ket_sz>  (lowers spin: ket_sz -> ket_sz - 1)
//   m=2: <bra_sz| sigma_+ |ket_sz>  (raises spin: ket_sz -> ket_sz + 1)
//
// Returns a list of VertexTerms (empty if the emission is forbidden by isospin).
// ---------------------------------------------------------------------------
inline std::vector<VertexTerm> apply_vertex(const Nucleon& ket, const Pion& pi) {

    // Step 1: isospin coefficient from tau.pi
    long double iso_coeff = 0.0;
    double      out_tz    = ket.tz;

    if (std::abs(pi.tz) < 0.1) {
        // pi0: tau_z eigenvalue = 2*tz
        iso_coeff = ket.apply_tau_z().coef;
        out_tz    = ket.tz;
    } else if (pi.tz > 0.5) {
        // pi+: tau_- (lowers nucleon isospin: proton -> neutron)
        auto r = ket.apply_tau_minus();
        if (!r.valid) return {};
        iso_coeff = std::sqrt(2.0L) * r.coef;
        out_tz    = r.new_tz;
    } else {
        // pi-: tau_+ (raises nucleon isospin: neutron -> proton)
        auto r = ket.apply_tau_plus();
        if (!r.valid) return {};
        iso_coeff = std::sqrt(2.0L) * r.coef;
        out_tz    = r.new_tz;
    }

    if (std::abs(iso_coeff) < 1e-14) return {};

    // Step 2: spin matrix elements for each spherical component of sigma.r
    std::vector<VertexTerm> terms;

    // m=0: sigma_z, spin-preserving, <sz|sigma_z|sz> = 2*sz
    {
        long double c = ket.apply_sigma_z().coef;
        if (std::abs(c) > 1e-14) {
            VertexTerm t;
            t.coeff[0] = cld(iso_coeff * c, 0.0L);
            t.bra_sz   = ket.sz;
            t.bra_tz   = out_tz;
            terms.push_back(t);
        }
    }

    // m=1: sigma_-, lowers spin by 1: ket_sz -> ket_sz - 1
    // r[1] = x + iy  (the complex coordinate)
    {
        auto r = ket.apply_sigma_minus();
        if (r.valid && std::abs(r.coef) > 1e-14) {
            VertexTerm t;
            t.coeff[1] = cld(iso_coeff * r.coef, 0.0L);
            t.bra_sz   = r.new_sz;
            t.bra_tz   = out_tz;
            terms.push_back(t);
        }
    }

    // m=2: sigma_+, raises spin by 1: ket_sz -> ket_sz + 1
    // r[2] = x - iy  (the complex conjugate coordinate)
    {
        auto r = ket.apply_sigma_plus();
        if (r.valid && std::abs(r.coef) > 1e-14) {
            VertexTerm t;
            t.coeff[2] = cld(iso_coeff * r.coef, 0.0L);
            t.bra_sz   = r.new_sz;
            t.bra_tz   = out_tz;
            terms.push_back(t);
        }
    }

    return terms;
}

// ---------------------------------------------------------------------------
// Legacy: VertexResult + apply_pion_emission (isospin only, kept for
// Jacobian construction in main.cc)
// ---------------------------------------------------------------------------
struct VertexResult {
    long double coefficient;
    bool        allowed;
    Nucleon     resulting_nucleon;
};

inline VertexResult apply_pion_emission(const Nucleon& n, const Pion& pi) {
    if (std::abs(pi.tz) < 0.1)
        return {n.apply_tau_z().coef, true, Nucleon(n.name, n.mass, n.charge, n.sz, n.tz)};
    if (pi.tz > 0.5) {
        auto r = n.apply_tau_minus();
        if (!r.valid) return {0.0, false, n};
        return {std::sqrt(2.0L)*r.coef, true, Nucleon("neutron",939.565379,0,n.sz,r.new_tz)};
    }
    auto r = n.apply_tau_plus();
    if (!r.valid) return {0.0, false, n};
    return {std::sqrt(2.0L)*r.coef, true, Nucleon("proton",938.272046,+1,n.sz,r.new_tz)};
}

} // namespace qm