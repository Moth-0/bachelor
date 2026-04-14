import subprocess
import re
import sys

# --- TARGET PHYSICS ---
TARGET_E = -2.224  # MeV
TARGET_R = 2.128   # fm
TOLERANCE_E = 0.001 # Stop binary search when within 5 keV

def run_deu(b_form, S, b_range=3.0):
    """Runs the C++ executable and extracts Energy and Radius."""
    cmd = ['./deu', '-b_form', str(b_form), '-S', str(S), '-b_range', str(b_range)]
    
    # Run the process and capture the output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running deu with b_form={b_form}, S={S}")
        return 999999.0, 0.0

    # Regex to find the final result line. 
    # Adjust this regex if your specific printout looks slightly different!
    # It looks for "E: -2.410 MeV" and "R: 2.465 fm"
    match = re.search(r'E:\s*([-\d.]+)\s*MeV.*?R:\s*([-\d.]+)\s*fm', result.stdout)
    
    if match:
        energy = float(match.group(1))
        radius = float(match.group(2))
        return energy, radius
    else:
        # If it couldn't parse the output, return garbage values
        return 999999.0, 0.0

def binary_search_S(b_form, s_min=10.0, s_max=80.0, max_iters=15):
    """Uses binary search to find the S that hits TARGET_E for a given b_form."""
    print(f"\n--- Tuning S for b_form = {b_form:.2f} fm ---")
    
    best_E = 999999.0
    best_R = 0.0
    best_S = s_min

    for i in range(max_iters):
        s_mid = (s_min + s_max) / 2.0
        
        # We print \r to overwrite the line in the terminal so it looks clean
        print(f"\r  Iter {i+1}/{max_iters} | Testing S = {s_mid:.3f} ...", end="", flush=True)
        
        E, R = run_deu(b_form, s_mid)
        
        # Save best results
        if abs(E - TARGET_E) < abs(best_E - TARGET_E):
            best_E = E
            best_R = R
            best_S = s_mid

        # Stop if we are within the tolerance
        if abs(E - TARGET_E) <= TOLERANCE_E:
            break
            
        # Binary Search Logic:
        # If E is LESS than target (e.g. -3.0 < -2.2), we are TOO BOUND -> Decrease S
        if E < TARGET_E:
            s_max = s_mid
        # If E is GREATER than target (e.g. -1.0 > -2.2), we are NOT BOUND ENOUGH -> Increase S
        else:
            s_min = s_mid

    print(f"\r  -> Found! S = {best_S:.3f} gives E = {best_E:.4f} MeV, R = {best_R:.4f} fm")
    return best_S, best_E, best_R

def main():
    print("=========================================================")
    print("           DEUTERON PARAMETER TUNING SCRIPT              ")
    print(f" Targets: E = {TARGET_E} MeV, R = {TARGET_R} fm          ")
    print("=========================================================")

    # Scan b_form from 1.0 fm to 2.2 fm in steps of 0.2
    b_form_values = [0.6, 0.8, 1.0, 1.2, 1.4]
    
    results = []

    for b_form in b_form_values:
        # For each b_form, binary search to find the correct S
        S_opt, E_opt, R_opt = binary_search_S(b_form)
        results.append((b_form, S_opt, E_opt, R_opt))

    print("\n=========================================================")
    print("                     FINAL SUMMARY                       ")
    print("=========================================================")
    print(f"{'b_form (fm)':<12} | {'Optimal S':<12} | {'Energy (MeV)':<14} | {'Radius (fm)':<12}")
    print("-" * 57)
    
    best_overall_b = 0
    best_overall_S = 0
    smallest_R_error = 9999.0

    for res in results:
        b, s, e, r = res
        print(f"{b:<12.2f} | {s:<12.3f} | {e:<14.4f} | {r:<12.4f}")
        
        # Track which one got closest to the target radius
        r_error = abs(r - TARGET_R)
        if r_error < smallest_R_error:
            smallest_R_error = r_error
            best_overall_b = b
            best_overall_S = s

    print("=========================================================")
    print(f"BEST MATCH FOR RADIUS: b_form = {best_overall_b:.2f} fm, S = {best_overall_S:.3f}")
    print("=========================================================")

if __name__ == "__main__":
    main()