#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include "qm/particle.h" // Assuming the header is named particle.h

using namespace qm;
using namespace std;

// Helper function to compare floating point values
bool almost_equal(long double a, long double b, long double epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
}

void test_spin_operators() {
    cout << "--- Testing Spin (Sigma) Operators ---" << endl;
    
    // 1. Test Proton Spin Up
    Nucleon p_up = Nucleon::Proton(+0.5);
    
    OpResult res_z = p_up.apply_sigma_z();
    assert(res_z.valid && almost_equal(res_z.coef, 1.0)); // sigma_z |up> = +1 |up>
    
    OpResult res_plus = p_up.apply_sigma_plus();
    assert(!res_plus.valid); // Cannot raise spin up
    
    OpResult res_minus = p_up.apply_sigma_minus();
    assert(res_minus.valid && almost_equal(res_minus.coef, 1.0)); 
    assert(almost_equal(res_minus.new_sz, -0.5)); // sigma_- |up> = 1 |down>
    
    // 2. Test Proton Spin Down
    Nucleon p_down = Nucleon::Proton(-0.5);
    
    OpResult res_z_down = p_down.apply_sigma_z();
    assert(res_z_down.valid && almost_equal(res_z_down.coef, -1.0)); // sigma_z |down> = -1 |down>
    
    OpResult res_plus_down = p_down.apply_sigma_plus();
    assert(res_plus_down.valid && almost_equal(res_plus_down.coef, 1.0)); 
    assert(almost_equal(res_plus_down.new_sz, +0.5)); // sigma_+ |down> = 1 |up>
    
    OpResult res_minus_down = p_down.apply_sigma_minus();
    assert(!res_minus_down.valid); // Cannot lower spin down

    cout << "[OK] Spin operators work correctly.\n\n";
}

void test_isospin_operators() {
    cout << "--- Testing Isospin (Tau) Operators ---" << endl;

    Nucleon p = Nucleon::Proton(); // tz = +0.5
    Nucleon n = Nucleon::Neutron(); // tz = -0.5

    // 1. Test Proton Isospin
    OpResult p_z = p.apply_tau_z();
    assert(p_z.valid && almost_equal(p_z.coef, 1.0)); // tau_z |p> = +1 |p>

    OpResult p_plus = p.apply_tau_plus();
    assert(!p_plus.valid); // Cannot raise proton isospin

    OpResult p_minus = p.apply_tau_minus();
    assert(p_minus.valid && almost_equal(p_minus.coef, 1.0));
    assert(almost_equal(p_minus.new_tz, -0.5)); // tau_- |p> = 1 |n>

    // 2. Test Neutron Isospin
    OpResult n_z = n.apply_tau_z();
    assert(n_z.valid && almost_equal(n_z.coef, -1.0)); // tau_z |n> = -1 |n>

    OpResult n_minus = n.apply_tau_minus();
    assert(!n_minus.valid); // Cannot lower neutron isospin

    OpResult n_plus = n.apply_tau_plus();
    assert(n_plus.valid && almost_equal(n_plus.coef, 1.0));
    assert(almost_equal(n_plus.new_tz, +0.5)); // tau_+ |n> = 1 |p>

    cout << "[OK] Isospin operators work correctly.\n\n";
}

void test_pion_emission_combinations() {
    cout << "--- Testing Pion Emission (tau . pi) ---" << endl;

    Nucleon p = Nucleon::Proton(-0.5);
    Nucleon n = Nucleon::Neutron(+0.5);
    
    Pion pi0 = Pion::PiZero();
    Pion pi_plus = Pion::PiPlus();
    Pion pi_minus = Pion::PiMinus();

    // 1. Proton emits Pi0
    VertexResult p_pi0 = apply_pion_emission(p, pi0);
    assert(p_pi0.allowed);
    assert(almost_equal(p_pi0.coefficient, 1.0)); // tau_z coupling
    assert(p_pi0.resulting_nucleon.name == "proton");
    cout << "  p -> p + pi0 \t| coef: " << p_pi0.coefficient << endl;

    // 2. Neutron emits Pi0
    VertexResult n_pi0 = apply_pion_emission(n, pi0);
    assert(n_pi0.allowed);
    assert(almost_equal(n_pi0.coefficient, -1.0)); // tau_z coupling
    assert(n_pi0.resulting_nucleon.name == "neutron");
    cout << "  n -> n + pi0 \t| coef: " << n_pi0.coefficient << endl;

    // 3. Proton emits Pi+
    VertexResult p_pi_plus = apply_pion_emission(p, pi_plus);
    assert(p_pi_plus.allowed);
    assert(almost_equal(p_pi_plus.coefficient, std::sqrt(2.0L))); // tau_- coupling * sqrt(2)
    assert(p_pi_plus.resulting_nucleon.name == "neutron");
    cout << "  p -> n + pi+ \t| coef: " << p_pi_plus.coefficient << " (sqrt 2)" << endl;

    // 4. Neutron emits Pi+ (Should be forbidden)
    VertexResult n_pi_plus = apply_pion_emission(n, pi_plus);
    assert(!n_pi_plus.allowed);
    cout << "  n -> x + pi+ \t| FORBIDDEN (correct)" << endl;

    // 5. Proton emits Pi- (Should be forbidden)
    VertexResult p_pi_minus = apply_pion_emission(p, pi_minus);
    assert(!p_pi_minus.allowed);
    cout << "  p -> x + pi- \t| FORBIDDEN (correct)" << endl;

    // 6. Neutron emits Pi-
    VertexResult n_pi_minus = apply_pion_emission(n, pi_minus);
    assert(n_pi_minus.allowed);
    assert(almost_equal(n_pi_minus.coefficient, std::sqrt(2.0L))); // tau_+ coupling * sqrt(2)
    assert(n_pi_minus.resulting_nucleon.name == "proton");
    cout << "  n -> p + pi- \t| coef: " << n_pi_minus.coefficient << " (sqrt 2)" << endl;

    cout << "\n[OK] Pion emission vertex logic works correctly.\n\n";
}

int main() {
    cout << "========================================" << endl;
    cout << " Starting Quantum Numbers Algebra Tests" << endl;
    cout << "========================================" << endl << endl;

    test_spin_operators();
    test_isospin_operators();
    test_pion_emission_combinations();

    cout << "========================================" << endl;
    cout << " ALL TESTS PASSED SUCCESSFULLY!" << endl;
    cout << "========================================" << endl;

    return 0;
}