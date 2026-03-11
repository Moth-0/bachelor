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

// --- Previous Test Functions Omitted for Brevity (Assume they are here) ---

// Calculates the W coefficient for a 2-nucleon system emitting a pion: 
// < n1_out, n2_out, pi | O | n1_in, n2_in >
double calculate_W(Nucleon n1_in, Nucleon n2_in, Nucleon n1_out, Nucleon n2_out, Pion pi) {
    double W_total = 0.0;

    // Path A: Nucleon 1 emits the pion, Nucleon 2 is a spectator
    VertexResult v1 = apply_pion_emission(n1_in, pi);
    if (v1.allowed && v1.resulting_nucleon.name == n1_out.name && n2_in.name == n2_out.name) {
        W_total += v1.coefficient;
    }

    // Path B: Nucleon 2 emits the pion, Nucleon 1 is a spectator
    VertexResult v2 = apply_pion_emission(n2_in, pi);
    if (v2.allowed && v2.resulting_nucleon.name == n2_out.name && n1_in.name == n1_out.name) {
        W_total += v2.coefficient;
    }

    return W_total;
}

int main() {
    cout << "--- Calculating Interaction W Coefficients ---" << endl;

    // Define our initial state particles
    Nucleon p = Nucleon::Proton(+0.5);
    Nucleon n = Nucleon::Neutron(+0.5);

    // Define the available pions
    Pion pi0 = Pion::PiZero();
    Pion pi_plus = Pion::PiPlus();
    Pion pi_minus = Pion::PiMinus();

    // 1. Calculate W for <pn | pn pi0>
    // Proton emits pi0 (coef +1) + Neutron emits pi0 (coef -1)
    double W_pn_pi0 = calculate_W(p, n, p, n, pi0);
    cout << "<p n | O | p n pi0>   W = " << W_pn_pi0 << " (Cancellation!)" << endl;

    // 2. Calculate W for <pp | pp pi0>
    // Proton 1 emits pi0 (+1) + Proton 2 emits pi0 (+1)
    double W_pp_pi0 = calculate_W(p, p, p, p, pi0);
    cout << "<p p | O | p p pi0>   W = " << W_pp_pi0 << endl;

    // 3. Calculate W for <nn | p n pi-> (Neutron 1 turns into a proton, emits pi-)
    double W_nn_pn_pim = calculate_W(n, n, p, n, pi_minus);
    cout << "<n n | O | p n pi->   W = " << W_nn_pn_pim << " (sqrt 2)" << endl;

    // 4. Calculate W for <pn | nn pi+> (Proton turns into neutron, emits pi+)
    double W_pn_nn_pip = calculate_W(p, n, n, n, pi_plus);
    cout << "<p n | O | n n pi+>   W = " << W_pn_nn_pip << " (sqrt 2)" << endl;

    return 0;
}