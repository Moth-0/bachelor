#pragma once

#include <string>
#include <cmath>
#include <iostream>

namespace qm {

// Forward declaration of Particle to use in OpResult
struct Particle;

// Struct to hold the result of applying an operator
// If valid == false, the operator annihilated the state (e.g., raising a spin-up state).
struct OpResult {
    long double coef;
    bool valid;
    double new_sz;
    double new_tz;
};

// Base Particle Struct
struct Particle {
    std::string name;
    long double mass; // in eV/c^2
    int charge;
    
    // Quantum Numbers
    double s;   // Total spin (1/2 for nucleons, 0 for pions)
    double sz;  // Spin projection
    double t;   // Total isospin (1/2 for nucleons, 1 for pions)
    double tz;  // Isospin projection

    Particle() = default;
    Particle(std::string n, long double m, int c, double spin, double spin_z, double isospin, double iso_z)
        : name(n), mass(m), charge(c), s(spin), sz(spin_z), t(isospin), tz(iso_z) {}

    virtual ~Particle() = default;

    // --- Core Ladder Operator Math ---
    // J_+ |j, m> = sqrt(j(j+1) - m(m+1)) |j, m+1>
    OpResult ladder_plus(double j, double m) const {
        if (m + 1.0 > j + 0.001) return {0.0, false, sz, tz};
        long double coef = std::sqrt(j * (j + 1.0) - m * (m + 1.0));
        return {coef, true, sz, tz};
    }

    // J_- |j, m> = sqrt(j(j+1) - m(m-1)) |j, m-1>
    OpResult ladder_minus(double j, double m) const {
        if (m - 1.0 < -j - 0.001) return {0.0, false, sz, tz};
        long double coef = std::sqrt(j * (j + 1.0) - m * (m - 1.0));
        return {coef, true, sz, tz};
    }
};

// --- NUCLEON SUBCLASS ---
struct Nucleon : public Particle {
    Nucleon(std::string n, long double m, int c, double spin_z, double iso_z)
        : Particle(n, m, c, 0.5, spin_z, 0.5, iso_z) {}

    // Convenience initializers
    static Nucleon Proton(double spin_z)  { return Nucleon("Proton", 938.272e6, 1, spin_z, 0.5); }
    static Nucleon Neutron(double spin_z) { return Nucleon("Neutron", 939.565e6, 0, spin_z, -0.5); }

    // Apply Pauli Spin Operators (sigma_z, sigma_+, sigma_-)
    OpResult apply_sigma_z() const {
        return {2.0 * sz, true, sz, tz}; // sigma_z |sz> = 2*sz |sz>
    }
    
    OpResult apply_sigma_plus() const {
        OpResult res = ladder_plus(s, sz);
        if (res.valid) res.new_sz = sz + 1.0;
        return res;
    }

    OpResult apply_sigma_minus() const {
        OpResult res = ladder_minus(s, sz);
        if (res.valid) res.new_sz = sz - 1.0;
        return res;
    }

    // Apply Isospin Operators (tau_z, tau_+, tau_-)
    OpResult apply_tau_z() const {
        return {2.0 * tz, true, sz, tz}; 
    }

    OpResult apply_tau_plus() const {
        OpResult res = ladder_plus(t, tz);
        if (res.valid) res.new_tz = tz + 1.0;
        return res;
    }

    OpResult apply_tau_minus() const {
        OpResult res = ladder_minus(t, tz);
        if (res.valid) res.new_tz = tz - 1.0;
        return res;
    }
};

// --- PION SUBCLASS ---
struct Pion : public Particle {
    Pion(std::string n, long double m, int c, double iso_z)
        : Particle(n, m, c, 0.0, 0.0, 1.0, iso_z) {}

    // Convenience initializers
    static Pion PiPlus()  { return Pion("Pi+", 139.570e6, 1, 1.0); }
    static Pion PiZero()  { return Pion("Pi0", 134.976e6, 0, 0.0); }
    static Pion PiMinus() { return Pion("Pi-", 139.570e6, -1, -1.0); }

    // Pions have spin 0, so Pauli matrices annihilate them
    
    // Apply Isospin Operators T=1
    OpResult apply_T_z() const {
        return {tz, true, sz, tz}; // Note: T_z for T=1 is just tz, not 2*tz
    }

    OpResult apply_T_plus() const {
        OpResult res = ladder_plus(t, tz);
        if (res.valid) res.new_tz = tz + 1.0;
        return res;
    }

    OpResult apply_T_minus() const {
        OpResult res = ladder_minus(t, tz);
        if (res.valid) res.new_tz = tz - 1.0;
        return res;
    }
};


// Represents the result of a pion-nucleon interaction vertex
struct VertexResult {
    long double coefficient;
    bool allowed;
    Nucleon resulting_nucleon;
};

// Evaluates the tau * pi coupling: tau_z * pi_0 + sqrt(2)*tau_- * pi_+ + sqrt(2)*tau_+ * pi_-
VertexResult apply_pion_emission(const Nucleon& n, const Pion& emitted_pi) {
    // If it emits a pi0 (tz = 0), we use the tau_z operator
    if (std::abs(emitted_pi.tz) < 0.1) {
        OpResult res = n.apply_tau_z();
        // tau_z applied to proton yields +1, applied to neutron yields -1
        return {res.coef, true, Nucleon(n.name, n.mass, n.charge, n.sz, n.tz)};
    }
    
    // If it emits a pi+ (tz = +1), the nucleon must lower its isospin (tau_-) to conserve charge
    if (emitted_pi.tz > 0.5) {
        OpResult res = n.apply_tau_minus();
        if (!res.valid) return {0.0, false, n}; // Cannot lower a neutron
        
        // Multiply by sqrt(2) from the spherical tensor expansion
        long double final_coef = std::sqrt(2.0) * res.coef;
        return {final_coef, true, Nucleon("Neutron", 939.565e6, 0, n.sz, res.new_tz)};
    }
    
    // If it emits a pi- (tz = -1), the nucleon must raise its isospin (tau_+)
    if (emitted_pi.tz < -0.5) {
        OpResult res = n.apply_tau_plus();
        if (!res.valid) return {0.0, false, n}; // Cannot raise a proton
        
        long double final_coef = std::sqrt(2.0) * res.coef;
        return {final_coef, true, Nucleon("Proton", 938.272e6, 1, n.sz, res.new_tz)};
    }

    return {0.0, false, n};
}

} // namespace qm