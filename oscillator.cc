/*
╔════════════════════════════════════════════════════════════════════════════════╗
║               oscillator.cc - OPTIMIZER SANITY CHECK DRIVER                    ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/
#include <iostream>
#include <iomanip>
#include <vector>
#include "oscillator.h"
#include "qm/gaussian.h"
#include "qm/solver.h"
#include "qm/jacobi.h"

using namespace qm;

bool is_valid_ho_gaussian(const SpatialWavefunction& psi) {
    ld width = psi.A(0, 0);
    if (width < 0.001 || width > 100.0) return false;
    
    ld shift_sq = psi.s(0,0)*psi.s(0,0) + psi.s(0,1)*psi.s(0,1) + psi.s(0,2)*psi.s(0,2);
    ld physical_shift_fm = std::sqrt(shift_sq) / (2.0 * width);
    
    if (physical_shift_fm > 5.0) return false; 

    return true;
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "   HARMONIC OSCILLATOR OPTIMIZER SANITY CHECK       \n";
    std::cout << "====================================================\n";

    // 2-Particle system (Proton and Neutron)
    Jacobian jac(rvec{938.27, 938.27});
    ld ho_mass = 938.27 / 2.0; 
    
    ld hbar_omega = 197.3269804 * std::sqrt(HO_K / ho_mass);
    ld exact_E0 = 1.5 * hbar_omega;
    ld exact_E1 = 3.5 * hbar_omega; 
    
    std::cout << "Theoretical Ground State (E0):  " << exact_E0 << " MeV\n";
    std::cout << "Theoretical 1st Excited (E1):   " << exact_E1 << " MeV\n";
    std::cout << "Theoretical Target Sum:         " << exact_E0 + exact_E1 << " MeV\n\n";

    std::vector<SpatialWavefunction> basis;
    srand(42); 

    for(int i = 0; i < 3; ++i) {
        Gaussian g;
        g.randomize(jac, 5.0, 0.5); 
        
        SpatialWavefunction psi(1); // Parity +1
        psi.A = zeros<ld>(1, 1);    
        psi.s = zeros<ld>(1, 3);
        psi.set_from_gaussian(g);

        // EXPLICIT PN RULE: Force initial shift to 0.0 so it is a perfect S-wave
        psi.s = zeros<ld>(1, 3);

        basis.push_back(psi);
    }

    std::vector<ld> E;
    ld start_sum = evaluate_ho_sum(basis, jac, E);
    std::cout << "Starting Energies -> E0: " << E[0] << ", E1: " << E[1] << ", Sum: " << start_sum << "\n\n";
    std::cout << "Running Single-State Sweeps to minimize E0 + E1...\n\n";

    for (int sweep = 0; sweep < 25; ++sweep) {
        
        for (size_t k = 0; k < basis.size(); ++k) {
            
            // EXPLICIT PN RULE: Lock the shift parameter for Nelder-Mead!
            bool opt_shift = false; 
            
            rvec p0 = pack_wavefunction(basis[k], opt_shift);

            auto objective = [&](const rvec& p_test) -> ld {
                std::vector<SpatialWavefunction> test_basis = basis;
                unpack_wavefunction(test_basis[k], p_test, opt_shift);
                
                if (!is_valid_ho_gaussian(test_basis[k])) return 999999.0;
                
                std::vector<ld> test_E;
                return evaluate_ho_sum(test_basis, jac, test_E);
            };

            rvec p_best = nelder_mead(p0, objective, 200);
            unpack_wavefunction(basis[k], p_best, opt_shift);
        }
        
        ld current_sum = evaluate_ho_sum(basis, jac, E);
        std::cout << "Sweep " << std::setw(2) << sweep 
                  << " | E0: " << std::setw(8) << E[0] 
                  << " | E1: " << std::setw(8) << E[1] 
                  << " | Sum: " << current_sum << "\n";
    }

    std::cout << "\n====================================================\n";
    std::cout << "               FINAL WAVEFUNCTIONS                  \n";
    std::cout << "====================================================\n";
    for(size_t k = 0; k < basis.size(); ++k) {
        ld width = basis[k].A(0,0);
        ld shift_sq = basis[k].s(0,0)*basis[k].s(0,0) + basis[k].s(0,1)*basis[k].s(0,1) + basis[k].s(0,2)*basis[k].s(0,2);
        ld phys_shift = std::sqrt(shift_sq) / (2.0 * width);
        
        std::cout << "State " << k << " | Width A: " << std::fixed << std::setprecision(4) << width 
                  << " fm^-2 | Phys Shift: " << phys_shift << " fm\n";
    }

    return 0;
}