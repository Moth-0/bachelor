#pragma once
#include <string>

namespace qm {

// ============================================================
//  particle.h
//
//  Pure data structs for physical particles.
//  No operator logic here — that belongs in main.cc where
//  you know which specific system you are building.
//  Masses are in MeV/c^2, charges in units of e.
// ============================================================

struct Particle {
    std::string name;
    long double mass;   // MeV/c^2
    int         charge; // units of e
    double      sz;     // spin-z quantum number
    double      tz;     // isospin-z quantum number

    Particle() : mass(0), charge(0), sz(0), tz(0) {}
    Particle(std::string n, long double m, int c, double spin_z, double iso_z)
        : name(n), mass(m), charge(c), sz(spin_z), tz(iso_z) {}
};

// --------------- Convenience factory functions ---------------
// All masses in MeV/c^2 from PDG 2022.

inline Particle make_proton(double sz = 0.5) {
    return Particle("proton", 938.272046, +1, sz, +0.5);
}

inline Particle make_neutron(double sz = 0.5) {
    return Particle("neutron", 939.565379, 0, sz, -0.5);
}

inline Particle make_pi0() {
    return Particle("pi0", 134.9768, 0, 0.0, 0.0);
}

inline Particle make_pi_plus() {
    return Particle("pi+", 139.5704, +1, 0.0, +1.0);
}

inline Particle make_pi_minus() {
    return Particle("pi-", 139.5704, -1, 0.0, -1.0);
}

inline Particle make_sigma() {
    // Sigma meson (scalar, isoscalar) used for testing.
    // Mass is model-dependent; 500 MeV is a common choice.
    return Particle("sigma", 500.0, 0, 0.0, 0.0);
}

} // namespace qm