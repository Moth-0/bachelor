/*
╔════════════════════════════════════════════════════════════════════════════════╗
║              nuc.cc - SINGLE NUCLEON SVM GROUND STATE FINDER                   ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Main driver for the Stochastic Variational Method (SVM) to find the         ║
║   self-energy of a single nucleon (proton or neutron) with pion coupling.     ║
║                                                                                ║
║ PHYSICS:                                                                       ║
║   Solves the 1-body self-energy problem: a single nucleon constantly           ║
║   emitting and reabsorbing virtual pions. The result E_self is then used      ║
║   to compute deuteron binding energy as: Binding = 2*E_self - E_deut         ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <string>

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/solver.h"
#include "nucleus.h"
#include "qm/serialization.h"
#include "SVM.h"

using namespace qm;

// Run SVM for single nucleon
template <typename BasisStateType>
ld run_nucleon_svm(const std::string& nucleon_type, const std::vector<bool>& relativistic,
                   ld b_range, ld b_form, ld S) {
    // Physical Constants
    ld m_nucleon = (nucleon_type == "p") ? 938.27 : 939.565;
    ld m_pi0 = 134.97;
    ld m_pic = 139.57;

    // Jacobians: 1-particle (bare) and 2-particle (nucleon + pion)
    Jacobian jac_bare({m_nucleon});
    Jacobian jac_dressed_0({m_nucleon, m_pi0});
    Jacobian jac_dressed_c({m_nucleon, m_pic});

    // Create appropriate channel templates
    std::vector<BasisState> channel_templates =
        (nucleon_type == "p")
            ? create_proton_templates(jac_bare, jac_dressed_0, jac_dressed_c)
            : create_neutron_templates(jac_bare, jac_dressed_0, jac_dressed_c);

    std::vector<BasisState> basis;
    rvec convergence_energies;

    // Initialize basis with bare nucleon
    std::cout << "--- Initializing Bare " << (nucleon_type == "p" ? "Proton" : "Neutron") << " ---\n";
    basis.push_back(channel_templates[0]);  // Bare nucleon

    // Run SVM phases
    std::cout << "--- Phase 1: Competitive Growth ---\n";
    int num_cycles = 3;

    for (int cycle = 1; cycle < num_cycles + 1; ++cycle) {
        std::cout << " - Cycle " << cycle << " - \n";

        // Competitive search for pion channels
        competitive_search(basis, channel_templates, 1000, b_range, b_form, S, relativistic);
        ld E_now = evaluate_basis_energy(basis, b_form, S, relativistic);
        convergence_energies.push_back(E_now);

        std::cout << "\n-------------------------------------------------------\n";
    }

    // Global sweep optimization
    std::cout << "\n--- Phase 2: Sweep Optimization ---\n";
    sweep_optimize_basis(basis, b_form, S, relativistic, convergence_energies, 10, 1e-4);

    // Evaluate final energy and observables
    ld E_self = evaluate_basis_energy(basis, b_form, S, relativistic);

    std::cout << "\n=== NUCLEON SELF-ENERGY ===\n";
    std::cout << "Nucleon:     " << (nucleon_type == "p" ? "Proton" : "Neutron") << "\n";
    std::cout << "Rest Mass:   " << m_nucleon << " MeV\n";
    std::cout << "E_self:      " << E_self << " MeV\n";
    std::cout << "Binding:     " << (m_nucleon - E_self) << " MeV\n";
    std::cout << "Basis Size:  " << basis.size() << "\n";

    return E_self;
}

int main(int argc, char* argv[]) {
    // Default values
    std::string nucleon_type = "p";  // proton
    ld b_range = 100.0;
    ld b_form = 1.4;
    ld S = 30.0;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--nucleon") && i + 1 < argc) {
            nucleon_type = argv[++i];
        } else if ((arg == "-b_range") && i + 1 < argc) {
            b_range = std::stold(argv[++i]);
        } else if ((arg == "-b_form") && i + 1 < argc) {
            b_form = std::stold(argv[++i]);
        } else if ((arg == "-S") && i + 1 < argc) {
            S = std::stold(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: ./nuc [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -n, --nucleon <p|n>  Nucleon type: proton (p) or neutron (n) [default: p]\n";
            std::cout << "  -b_range <value>     Search space for Gaussian width (default: 100.0 fm)\n";
            std::cout << "  -b_form <value>      Pion interaction range (default: 1.4 fm)\n";
            std::cout << "  -S <value>           Pion coupling strength (default: 30.0 MeV)\n";
            std::cout << "  -h, --help           Show this help message\n";
            return 0;
        }
    }

    // Validate nucleon type
    if (nucleon_type != "p" && nucleon_type != "n") {
        std::cerr << "Error: nucleon type must be 'p' (proton) or 'n' (neutron)\n";
        return 1;
    }

    // Enable OpenMP
    omp_set_nested(0);

    std::cout << "========================================\n";
    std::cout << "  SINGLE NUCLEON SELF-ENERGY (SVM)\n";
    std::cout << "========================================\n";
    std::cout << "Nucleon:  " << (nucleon_type == "p" ? "Proton" : "Neutron") << "\n";
    std::cout << "Parameters:\n";
    std::cout << "  b_range = " << b_range << " fm\n";
    std::cout << "  b_form  = " << b_form << " fm\n";
    std::cout << "  S       = " << S << " MeV\n";
    std::cout << "========================================\n\n";

    // Classical only for nucleon system
    std::vector<bool> relativistic = {false};

    ld E_self = run_nucleon_svm<BasisState>(nucleon_type, relativistic, b_range, b_form, S);

    std::cout << "\n========================================\n";
    std::cout << "Final E_self = " << std::fixed << std::setprecision(6) << E_self << " MeV\n";
    std::cout << "========================================\n";

    return 0;
}