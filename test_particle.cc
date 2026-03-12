// test_particle.cc -- Unit test for spin-isospin algebra in particle.h
#include <iostream>
#include <iomanip>
#include <vector>
#include "qm/particle.h"

using namespace qm;

void print_vertex(const Nucleon& nuc, const Pion& pi) {
    std::cout << "Reaction: " << std::setw(7) << nuc.name 
              << " |sz=" << std::showpos << nuc.sz << "> + " 
              << std::setw(3) << pi.name << "\n";
              
    auto terms = apply_vertex(nuc, pi);
    
    if (terms.empty()) {
        std::cout << "  -> [FORBIDDEN BY ISOSPIN]\n\n";
        return;
    }
    
    for (const auto& t : terms) {
        std::string spin_op;
        long double val = 0.0;
        
        // Identify which spherical component this term represents
        if (std::abs(t.coeff[0]) > 1e-10) { spin_op = "sigma_z"; val = t.coeff[0].real(); }
        if (std::abs(t.coeff[1]) > 1e-10) { spin_op = "sigma_-"; val = t.coeff[1].real(); }
        if (std::abs(t.coeff[2]) > 1e-10) { spin_op = "sigma_+"; val = t.coeff[2].real(); }

        std::cout << "  -> Output Bra: <tz=" << std::showpos << t.bra_tz 
                  << ", sz=" << std::showpos << t.bra_sz << "|  "
                  << "Operator: " << std::setw(7) << std::noshowpos << spin_op 
                  << "  | Coeff = " << std::showpos << val << "\n";
    }
    std::cout << std::noshowpos << "\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " TESTING NUCLEON-PION VERTEX: W = (tau.pi)(sigma.r)     \n";
    std::cout << "========================================================\n\n";

    Nucleon p_up = Nucleon::Proton(+0.5);
    Nucleon p_dn = Nucleon::Proton(-0.5);
    Nucleon n_up = Nucleon::Neutron(+0.5);
    Nucleon n_dn = Nucleon::Neutron(-0.5);

    Pion pi0 = Pion::PiZero();
    Pion pip = Pion::PiPlus();
    Pion pim = Pion::PiMinus();

    std::cout << "--- Neutral Pion Emission (tau_z) ---\n";
    print_vertex(p_up, pi0); // Expect coefs: +1 (z), +1 (-)
    print_vertex(p_dn, pi0); // Expect coefs: -1 (z), +1 (+)
    print_vertex(n_up, pi0); // Expect coefs: -1 (z), -1 (-)

    std::cout << "--- Charged Pion Emission (tau_+, tau_-) ---\n";
    print_vertex(p_up, pip); // Proton emits pi+, becomes neutron. Expect sqrt(2)
    print_vertex(p_dn, pip);
    print_vertex(n_up, pim); // Neutron emits pi-, becomes proton. Expect sqrt(2)
    print_vertex(n_dn, pim);

    std::cout << "--- Forbidden Transitions ---\n";
    print_vertex(n_up, pip); // Neutron cannot emit pi+
    print_vertex(p_up, pim); // Proton cannot emit pi-

    return 0;
}