/*
╔════════════════════════════════════════════════════════════════════════════════╗
║              plot_wavefunction.cc - Reconstruct & plot ground state            ║
║                                                                                ║
║ Reads basis_final.txt and generates CSV with:                                 ║
║   - r: radial coordinate (fm)                                                 ║
║   - psi: wavefunction |ψ(r)| evaluated from basis expansion                   ║
║   - asymptotic: expected asymptotic form e^(-κr)                              ║
║   - ratio: psi / asymptotic (should approach 1 at large r)                    ║
║                                                                                ║
║ Usage: ./plot_wavefunction [basis_file] [output_csv]                          ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <sstream>
#include <complex>

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "deuterium.h"
#include "qm/serialization.h"

using namespace qm;

// Evaluate wavefunction at radius r using basis expansion
ld evaluate_wavefunction(const std::vector<BasisState>& basis, 
                         const rvec& coefficients,
                         ld r) {
    ld psi_r = 0.0;
    
    // Create coordinate matrix: r in z-direction (spherically symmetric)
    // SpatialWavefunction::evaluate expects an (N-1)x3 matrix where:
    //   - column 0 is x coords
    //   - column 1 is y coords
    //   - column 2 is z coords
    // For spherical symmetry, we only need one coordinate set (0,0,r)
    
    rmat coord(1, 3);  // 1 row (since basis is 2-body, dimension is 2-1=1)
    coord(0, 0) = 0.0;
    coord(0, 1) = 0.0;
    coord(0, 2) = r;
    
    // Evaluate each basis state and accumulate
    for (size_t i = 0; i < basis.size() && i < coefficients.size(); ++i) {
        // Evaluate Gaussian basis at coordinate
        ld psi_i = basis[i].psi.evaluate(coord);
        psi_r += coefficients[i] * psi_i;
    }
    
    return std::abs(psi_r);
}

int main(int argc, char* argv[]) {
    std::string basis_file = "basis_final.txt";
    std::string output_csv = "wavefunction.csv";
    
    if (argc > 1) basis_file = argv[1];
    if (argc > 2) output_csv = argv[2];
    
    std::cout << "Reading basis from: " << basis_file << "\n";
    
    // Load basis
    std::vector<BasisState> basis;
    rvec coefficients;
    ld energy = 0, radius = 0;
    
    try {
        auto [b, c, e, r, k] = load_basis_state(basis_file);
        basis = b;
        coefficients = c;
        energy = e;
        radius = r;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR loading basis: " << ex.what() << "\n";
        return 1;
    }
    
    std::cout << "Loaded " << basis.size() << " basis states\n";
    std::cout << "Energy: " << energy << " MeV\n";
    std::cout << "Charge Radius: " << radius << " fm\n";
    
    // Calculate decay constant κ = sqrt(2μE_binding/ℏ²)
    // E_binding = -(energy) in our convention (energy is negative for bound states)
    // μ = m_p * m_n / (m_p + m_n) ≈ 469.4 MeV (reduced mass for deuteron)
    // ℏ = 197.327 MeV·fm (in natural units)
    
    ld m_p = 938.27;  // MeV
    ld m_n = 939.56;  // MeV
    ld hbar = 197.327;  // MeV·fm
    
    ld mu = (m_p * m_n) / (m_p + m_n);  // Reduced mass
    ld E_binding = -energy;  // Positive binding energy
    
    // κ from E = -ℏ²κ²/(2μ)  =>  κ = sqrt(2μE/ℏ²)
    ld kappa = std::sqrt(2.0 * mu * E_binding) / hbar;
    
    std::cout << "Reduced mass μ: " << mu << " MeV\n";
    std::cout << "Binding energy: " << E_binding << " MeV\n";
    std::cout << "Decay constant κ: " << kappa << " fm⁻¹\n";
    std::cout << "\nGenerating wavefunction at radial points...\n";
    
    // Generate output CSV
    std::ofstream csv(output_csv);
    if (!csv.is_open()) {
        std::cerr << "ERROR: Cannot open output file: " << output_csv << "\n";
        return 1;
    }
    
    csv << std::setprecision(10);
    csv << "r_fm,psi_abs,asymptotic_form,ratio_psi_to_asymptotic\n";
    
    // Evaluate wavefunction at r from 0.1 to 20 fm
    ld r_min = 0.1;
    ld r_max = 20.0;
    int num_points = 500;
    
    for (int i = 0; i <= num_points; ++i) {
        ld r = r_min + (r_max - r_min) * i / num_points;
        
        // Evaluate basis expansion
        ld psi = evaluate_wavefunction(basis, coefficients, r);
        
        // Asymptotic form: e^(-κr)
        // Normalize so that at large r it matches psi (if psi is already normalized)
        ld asymptotic = std::exp(-kappa * r) / r;
        
        // Ratio to check convergence
        ld ratio = (asymptotic > 1e-10) ? psi / asymptotic : 0.0;
        
        csv << r << "," << psi << "," << asymptotic << "," << ratio << "\n";
    }
    
    csv.close();
    std::cout << "Wavefunction data written to: " << output_csv << "\n";
    
    return 0;
}
