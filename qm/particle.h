#pragma once
#include <string>
#include <cmath>

namespace qm {

// Result of a ladder operator application. valid=false means annihilation.
struct OpResult {
    long double coef   = 0.0;
    bool        valid  = false;
    double      new_sz = 0.0;
    double      new_tz = 0.0;
};

struct Particle {
    std::string name;
    long double mass;   // MeV/c^2
    int charge;         // Electric charge
    double s, sz;       // spin, spin projection
    double t, tz;       // isospin, isospin projection

    Particle() = default;
    Particle(std::string n, long double m, int c,
             double spin, double spin_z, double isospin, double iso_z)
        : name(n), mass(m), charge(c), s(spin), sz(spin_z), t(isospin), tz(iso_z) {}
    ~Particle() = default;

    // J_+ |j,m> = sqrt(j(j+1) - m(m+1)) |j,m+1>
    OpResult ladder_plus(double j, double m) const {
        if (m + 1.0 > j + 0.001) return {0.0, false, sz, tz};
        return {std::sqrt(j * (j + 1.0) - m * (m + 1.0)), true, sz, tz};
    }

    // J_- |j,m> = sqrt(j(j+1) - m(m-1)) |j,m-1>
    OpResult ladder_minus(double j, double m) const {
        if (m - 1.0 < -j - 0.001) return {0.0, false, sz, tz};
        return {std::sqrt(j * (j + 1.0) - m * (m - 1.0)), true, sz, tz};
    }
};

// Nucleon: proton (tz=+1/2) or neutron (tz=-1/2), spin-1/2, isospin-1/2
struct Nucleon : public Particle {
    Nucleon(std::string n, long double m, int c, double spin_z, double iso_z)
        : Particle(n, m, c, 0.5, spin_z, 0.5, iso_z) {}

    static Nucleon Proton (double spin_z = 0.5) { return Nucleon("proton",  938.272046, +1, spin_z, +0.5); }
    static Nucleon Neutron(double spin_z = 0.5) { return Nucleon("neutron", 939.565379,  0, spin_z, -0.5); }

    // ---------------------------------------------------------------
    // Isospin operators: tau = 2T
    // ---------------------------------------------------------------
    OpResult apply_tau_z() const { return {2.0 * tz, true, sz, tz}; }

    OpResult apply_tau_plus() const {
        auto r = ladder_plus(t, tz);
        if (r.valid) r.new_tz = tz + 1.0;
        return r;
    }
    OpResult apply_tau_minus() const {
        auto r = ladder_minus(t, tz);
        if (r.valid) r.new_tz = tz - 1.0;
        return r;
    }

    // ---------------------------------------------------------------
    // Spin operators: sigma = 2S
    // ---------------------------------------------------------------
    OpResult apply_sigma_z() const { return {2.0 * sz, true, sz, tz}; }

    OpResult apply_sigma_plus() const {
        auto r = ladder_plus(s, sz);
        if (r.valid) r.new_sz = sz + 1.0;
        return r;
    }
    OpResult apply_sigma_minus() const {
        auto r = ladder_minus(s, sz);
        if (r.valid) r.new_sz = sz - 1.0;
        return r;
    }

    // ---------------------------------------------------------------
    // Spin coupling coefficients for the (sigma . r) operator
    //
    // W = (tau.pi)(sigma.r)f(r) has two spin channels:
    //
    //   Chan A  (spin-preserving):   sigma_z r_z
    //   Chan B  (spin-flipping):     sigma_x r_x + sigma_y r_y
    //
    // coef_z    = <sz| sigma_z |sz>   -- coefficient for chan_A (beta[2])
    // coef_perp = combined transverse coefficient for chan_B (beta[0])
    //             The factor sqrt(2) comes from rotational symmetry:
    //             |sigma_x r_x + sigma_y r_y|^2 = 2|sigma_x r_x|^2
    //             so both x and y are captured by a single x-channel
    //             weighted by sqrt(2).
    // ---------------------------------------------------------------
    struct SpinCoupling {
        long double coef_z;     // for sigma_z r_z  (chan_A, spin-preserving)
        long double coef_perp;  // for combined (sigma_x r_x + sigma_y r_y) (chan_B, spin-flip)
    };

    SpinCoupling spin_dot_r_coefficients() const {
        // Chan A: sigma_z eigenvalue on the reference spin state
        long double cz = apply_sigma_z().coef;   // = 2*sz = +1.0 for spin-up reference

        // Chan B: the transverse flip amplitude, multiplied by sqrt(2) to
        // account for both the x and y directions by rotational symmetry.
        // sigma_minus acting on |spin-up> gives the flip amplitude.
        long double cp = std::sqrt(2.0L) * ladder_minus(s, sz).coef;

        return {cz, cp};
    }
};

// Pion: isospin-1 triplet (pi+, pi0, pi-)
struct Pion : public Particle {
    Pion(std::string n, long double m, int c, double iso_z)
        : Particle(n, m, c, 0.0, 0.0, 1.0, iso_z) {}

    static Pion PiPlus () { return Pion("pi+", 139.5704, +1, +1.0); }
    static Pion PiZero () { return Pion("pi0", 134.9768,  0,  0.0); }
    static Pion PiMinus() { return Pion("pi-", 139.5704, -1, -1.0); }

    OpResult apply_T_z() const { return {tz, true, sz, tz}; }
    OpResult apply_T_plus() const {
        auto r = ladder_plus(t, tz);
        if (r.valid) r.new_tz = tz + 1.0;
        return r;
    }
    OpResult apply_T_minus() const {
        auto r = ladder_minus(t, tz);
        if (r.valid) r.new_tz = tz - 1.0;
        return r;
    }
};

// Scalar isoscalar meson (sigma)
struct Meson : public Particle {
    Meson(std::string n, long double m) : Particle(n, m, 0, 0.0, 0.0, 0.0, 0.0) {}
    static Meson Sigma() { return Meson("sigma", 500.0); }
};

// Result of a pion emission vertex (tau.pi coupling).
// coefficient: the isospin Clebsch-Gordan coefficient from (tau.pi)
// allowed:     false if the transition is forbidden by isospin conservation
// resulting_nucleon: the nucleon state after emission
struct VertexResult {
    long double coefficient;
    bool        allowed;
    Nucleon     resulting_nucleon;
};

// Pion emission vertex: W = (tau.pi)(sigma.r)f(r)
// This function computes the isospin factor (tau.pi) only.
// The spin factor (sigma.r) is obtained separately via spin_dot_r_coefficients().
//
// tau.pi = tau_z pi0 + sqrt(2) tau_- pi+ + sqrt(2) tau_+ pi-
//
// Emission rules:
//   proton  + pi0  -> proton   (coef = +1,     tau_z  eigenvalue on proton)
//   neutron + pi0  -> neutron  (coef = -1,     tau_z  eigenvalue on neutron)
//   proton  + pi+  -> neutron  (coef = +sqrt2, tau_- lowers proton to neutron)
//   neutron + pi-  -> proton   (coef = +sqrt2, tau_+ raises neutron to proton)
VertexResult apply_pion_emission(const Nucleon& n, const Pion& emitted_pi) {
    if (std::abs(emitted_pi.tz) < 0.1) {
        // pi0: coefficient is the tau_z eigenvalue of the nucleon
        return {n.apply_tau_z().coef, true, Nucleon(n.name, n.mass, n.charge, n.sz, n.tz)};
    }
    if (emitted_pi.tz > 0.5) {
        // pi+: nucleon loses one unit of isospin (proton -> neutron via tau_-)
        auto r = n.apply_tau_minus();
        if (!r.valid) return {0.0, false, n};
        return {std::sqrt(2.0) * r.coef, true,
                Nucleon("neutron", 939.565379, 0, n.sz, r.new_tz)};
    }
    if (emitted_pi.tz < -0.5) {
        // pi-: nucleon gains one unit of isospin (neutron -> proton via tau_+)
        auto r = n.apply_tau_plus();
        if (!r.valid) return {0.0, false, n};
        return {std::sqrt(2.0) * r.coef, true,
                Nucleon("proton", 938.272046, +1, n.sz, r.new_tz)};
    }
    return {0.0, false, n};
}

} // namespace qm