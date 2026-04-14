import subprocess
import re
import sys

# --- TARGET PHYSICS ---
TARGET_E = -2.224  # MeV
TARGET_R = 2.128   # fm
TOLERANCE_E = 0.005 # Stop inner search when E is within 5 keV
TOLERANCE_R = 0.005 # Stop outer search when R is within 0.005 fm

def run_deu(b_form, S, b_range=3.0):
    """Runs the C++ executable and extracts Energy and Radius."""
    cmd = ['./deu', '-b_form', str(b_form), '-S', str(S), '-b_range', str(b_range)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return 999999.0, 0.0

    match = re.search(r'E:\s*([-\d.]+)\s*MeV.*?R:\s*([-\d.]+)\s*fm', result.stdout)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 999999.0, 0.0

def binary_search_S(b_form, s_guess, window=20.0, max_iters=15):
    """INNER LOOP: Binary search to find S that hits TARGET_E for a given b_form."""
    s_min = max(1.0, s_guess - window)
    s_max = s_guess + window
    
    best_E, best_R, best_S = 999999.0, 0.0, s_guess

    for i in range(max_iters):
        s_mid = (s_min + s_max) / 2.0
        print(f"\r    [Inner] Iter {i+1}/{max_iters} | Testing S = {s_mid:.3f} ...", end="", flush=True)
        
        E, R = run_deu(b_form, s_mid)
        
        if abs(E - TARGET_E) < abs(best_E - TARGET_E):
            best_E, best_R, best_S = E, R, s_mid

        if abs(E - TARGET_E) <= TOLERANCE_E:
            break
            
        if E < TARGET_E:
            s_max = s_mid # Too bound, lower S
        else:
            s_min = s_mid # Not bound enough, raise S

    print(f"\r    [Inner] -> E matched! S = {best_S:.3f} gives E = {best_E:.4f} MeV, R = {best_R:.4f} fm")
    return best_S, best_E, best_R

def binary_search_b_form(b_min=0.5, b_max=2.5, max_outer_iters=10):
    """OUTER LOOP: Binary search to find b_form that hits TARGET_R."""
    print("=========================================================")
    print("        DEUTERON NESTED BINARY SEARCH TUNING             ")
    print(f" Targets: E = {TARGET_E} MeV, R = {TARGET_R} fm          ")
    print("=========================================================\n")

    best_b, best_S, best_E, best_R = 0.0, 0.0, 999999.0, 0.0
    
    # Initial guess for S to kickstart the inner loop
    current_s_guess = 40.0 

    for i in range(max_outer_iters):
        b_mid = (b_min + b_max) / 2.0
        print(f"--- [Outer Iter {i+1}/{max_outer_iters}] Testing b_form = {b_mid:.4f} fm ---")
        
        # Run the inner loop to force E = -2.224
        S_opt, E_opt, R_opt = binary_search_S(b_mid, current_s_guess)
        
        # Save the best overall result
        if abs(R_opt - TARGET_R) < abs(best_R - TARGET_R):
            best_b, best_S, best_E, best_R = b_mid, S_opt, E_opt, R_opt

        # Stop if we hit the radius target
        if abs(R_opt - TARGET_R) <= TOLERANCE_R:
            print(f"\n>>> TARGET RADIUS REACHED WITHIN TOLERANCE! <<<")
            break
            
        # Outer Binary Search Logic
        if R_opt < TARGET_R:
            # Radius is too small -> Increase interaction range (b_form)
            b_min = b_mid
        else:
            # Radius is too large -> Decrease interaction range (b_form)
            b_max = b_mid
            
        # Update our S guess for the next iteration to speed up the inner loop
        current_s_guess = S_opt

    print("\n=========================================================")
    print("                 FINAL TUNED PARAMETERS                  ")
    print("=========================================================")
    print(f" Interaction Range (b_form) = {best_b:.4f} fm")
    print(f" Coupling Strength (S)      = {best_S:.4f} MeV")
    print("-" * 57)
    print(f" Resulting Energy           = {best_E:.5f} MeV (Target: {TARGET_E})")
    print(f" Resulting Radius           = {best_R:.5f} fm  (Target: {TARGET_R})")
    print("=========================================================")

if __name__ == "__main__":
    # You can adjust the initial b_form bracket here if you know roughly where it lives
    binary_search_b_form(b_min=0.5, b_max=2.5)