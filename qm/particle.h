#pragma once
#include <string>
#include <cmath>

namespace qm {

// ============================================================
//  particle.h
//
//  Physical particle definitions with quantum numbers and
//  ladder operator algebra for spin and isospin.
//
//  Design: Nucleon and Pion are subclasses of Particle.
//  The apply_pion_emission() free function encodes the
//  (tau . pi) vertex and returns the isospin coupling weight
//  and the resulting nucleon after the transition.
//
//  This is used in main.cc to determine:
//    - which Fock sector each W block connects to
//    - the isospin weight multiplying W_spatial
// ============================================================

// Result of applying a ladder operator.
// valid=false means the operator annihilated the state.
struct OpResult {
    long double coef  = 0.0;
    bool        valid = false;
    double      new_sz = 0.0;
    double      new_tz = 0.0;
};

// ------------------------------------------------------------
struct Particle {
    std::string name;
    long double mass;   // MeV/c^2
    int         charge;

    // Quantum numbers
    double s;   // total spin
    double sz;  // spin projection
    double t;   // total isospin
    double tz;  // isospin projection

    Particle() = default;
    Particle(std::string n, long double m, int c,
             double spin, double spin_z,
             double isospin, double iso_z)
        : name(n), mass(m), charge(c),
          s(spin), sz(spin_z), t(isospin), tz(iso_z) {}

    virtual ~Particle() = default;

    // J_+ |j,m> = sqrt(j(j+1) - m(m+1)) |j,m+1>
    OpResult ladder_plus(double j, double m) const {
        if (m + 1.0 > j + 0.001) return {0.0, false, sz, tz};
        long double coef = std::sqrt(j*(j+1.0) - m*(m+1.0));
        return {coef, true, sz, tz};
    }

    // J_- |j,m> = sqrt(j(j+1) - m(m-1)) |j,m-1>
    OpResult ladder_minus(double j, double m) const {
        if (m - 1.0 < -j - 0.001) return {0.0, false, sz, tz};
        long double coef = std::sqrt(j*(j+1.0) - m*(m-1.0));
        return {coef, true, sz, tz};
    }
};

// ------------------------------------------------------------
//  Nucleon: proton (tz=+1/2) or neutron (tz=-1/2)
// ------------------------------------------------------------
struct Nucleon : public Particle {
    Nucleon(std::string n, long double m, int c,
            double spin_z, double iso_z)
        : Particle(n, m, c, 0.5, spin_z, 0.5, iso_z) {}

    static Nucleon Proton (double spin_z = 0.5)
        { return Nucleon("proton",  938.272046L, +1, spin_z, +0.5); }
    static Nucleon Neutron(double spin_z = 0.5)
        { return Nucleon("neutron", 939.565379L,  0, spin_z, -0.5); }

    // Pauli spin operators: sigma = 2 * S
    OpResult apply_sigma_z() const
        { return {2.0L * sz, true, sz, tz}; }

    OpResult apply_sigma_plus() const {
        auto res = ladder_plus(s, sz);
        if (res.valid) res.new_sz = sz + 1.0;
        return res;
    }
    OpResult apply_sigma_minus() const {
        auto res = ladder_minus(s, sz);
        if (res.valid) res.new_sz = sz - 1.0;
        return res;
    }

    // Isospin operators: tau = 2 * T
    OpResult apply_tau_z() const
        { return {2.0L * tz, true, sz, tz}; }

    OpResult apply_tau_plus() const {
        auto res = ladder_plus(t, tz);
        if (res.valid) res.new_tz = tz + 1.0;
        return res;
    }
    OpResult apply_tau_minus() const {
        auto res = ladder_minus(t, tz);
        if (res.valid) res.new_tz = tz - 1.0;
        return res;
    }
};

// ------------------------------------------------------------
//  Pion: isospin T=1 triplet (π+, π0, π-)
// ------------------------------------------------------------
struct Pion : public Particle {
    Pion(std::string n, long double m, int c, double iso_z)
        : Particle(n, m, c, 0.0, 0.0, 1.0, iso_z) {}

    static Pion PiPlus () { return Pion("pi+", 139.5704L, +1, +1.0); }
    static Pion PiZero () { return Pion("pi0", 134.9768L,  0,  0.0); }
    static Pion PiMinus() { return Pion("pi-", 139.5704L, -1, -1.0); }

    // T=1 isospin operators on the pion
    OpResult apply_T_z() const
        { return {tz, true, sz, tz}; }  // T_z for T=1 is just tz

    OpResult apply_T_plus() const {
        auto res = ladder_plus(t, tz);
        if (res.valid) res.new_tz = tz + 1.0;
        return res;
    }
    OpResult apply_T_minus() const {
        auto res = ladder_minus(t, tz);
        if (res.valid) res.new_tz = tz - 1.0;
        return res;
    }
};

// ------------------------------------------------------------
//  Meson: scalar/isoscalar (sigma meson for tests)
// ------------------------------------------------------------
struct Meson : public Particle {
    Meson(std::string n, long double m)
        : Particle(n, m, 0, 0.0, 0.0, 0.0, 0.0) {}

    // 500 MeV scalar sigma used in Fedorov's model
    static Meson Sigma() { return Meson("sigma", 500.0L); }
};

// ------------------------------------------------------------
//  Pion emission vertex: (tau . pi) coupling
//
//  Evaluates which nucleon results from emitting a given pion
//  and returns the isospin weight from the spherical tensor
//  expansion of (tau . pi):
//
//    (tau . pi) = tau_z pi0  -  (1/sqrt(2)) tau_+ pi-
//                           +  (1/sqrt(2)) tau_- pi+
//
//  Coupling weights (from Clebsch-Gordan / tau algebra):
//    pi0  : weight = tau_z eigenvalue = +1 (proton) or -1 (neutron)
//    pi+  : tau_- on proton → neutron,  weight = sqrt(2) * 1
//    pi-  : tau_+ on neutron → proton,  weight = sqrt(2) * 1
//
//  Returns {weight, allowed, resulting_nucleon}.
// ------------------------------------------------------------
struct VertexResult {
    long double coefficient;
    bool        allowed;
    Nucleon     resulting_nucleon;
};

inline VertexResult apply_pion_emission(const Nucleon& n,
                                         const Pion& emitted_pi)
{
    // --- pi0: tau_z coupling ---
    if (std::abs(emitted_pi.tz) < 0.1) {
        OpResult res = n.apply_tau_z();
        return {res.coef, true,
                Nucleon(n.name, n.mass, n.charge, n.sz, n.tz)};
    }

    // --- pi+: proton emits pi+, becomes neutron (tau_-) ---
    if (emitted_pi.tz > 0.5) {
        OpResult res = n.apply_tau_minus();
        if (!res.valid) return {0.0L, false, n};
        long double w = std::sqrt(2.0L) * res.coef;
        return {w, true,
                Nucleon("neutron", 939.565379L, 0, n.sz, res.new_tz)};
    }

    // --- pi-: neutron emits pi-, becomes proton (tau_+) ---
    if (emitted_pi.tz < -0.5) {
        OpResult res = n.apply_tau_plus();
        if (!res.valid) return {0.0L, false, n};
        long double w = std::sqrt(2.0L) * res.coef;
        return {w, true,
                Nucleon("proton", 938.272046L, +1, n.sz, res.new_tz)};
    }

    return {0.0L, false, n};
}

} // namespace qm