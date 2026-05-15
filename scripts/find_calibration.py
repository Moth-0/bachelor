#!/usr/bin/env python3
"""
find_calibration.py - Find optimal b_range and b_form that make energy and radius targets overlap at same S

Algorithm:
1. Try initial b_range and b_form
2. Run S sweep to find:
   - S_energy: where energy crosses -2.224 MeV
   - S_radius: where radius crosses 2.128 fm
3. If S_energy ≈ S_radius within tolerance, CONVERGED
4. If S_energy < S_radius: increase b_range (makes radius smaller)
5. If S_energy > S_radius: decrease b_range (makes radius larger)
6. Adjust b_form if needed to refine further
7. Repeat until converged

Usage:
  python3 scripts/find_calibration.py \\
    --b_range_init 2.6 --b_form_init 1.2 \\
    --S_min 34.0 --S_max 42.0 --S_steps 12 \\
    --tolerance 0.5 --max_iterations 20
"""

import sys
import csv
import os
import subprocess
import argparse
import json
from pathlib import Path

class CalibrationFinder:
    """Iteratively find b_range and b_form that align energy and radius targets at same S"""
    
    def __init__(self, b_range_init, b_form_init, S_min, S_max, S_steps,
                 energy_target, radius_target, tolerance, max_iterations, jobs=8):
        self.b_range = b_range_init
        self.b_form = b_form_init
        self.S_min = S_min
        self.S_max = S_max
        self.S_steps = S_steps
        self.energy_target = energy_target
        self.radius_target = radius_target
        self.tolerance = tolerance  # MeV (how close S_energy and S_radius must be)
        self.max_iterations = max_iterations
        self.jobs = jobs
        
        self.results_dir = "results/find_calibration"
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        self.history = []
    
    def run_sweep_S(self, b_range, b_form):
        """Run S sweep and return (S_values, energies, radii)"""
        import time
        start_time = time.time()
        
        cmd = [
            "python3", "scripts/sweep.py",
            "--scan", "S",
            "--b_range", str(b_range),
            "--b_form", str(b_form),
            "--S_min", str(self.S_min),
            "--S_max", str(self.S_max),
            "--S_steps", str(self.S_steps),
            "--jobs", str(self.jobs),
            "--basis_size_steps", "5"  # Match Makefile to keep runtime ~10 min
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            
            if result.returncode != 0:
                print(f"  ERROR: sweep.py failed")
                print(f"  stdout: {result.stdout[-500:]}")
                print(f"  stderr: {result.stderr[-500:]}")
                return None, None, None
            
            # Read aggregated results
            csv_path = "results/energy_sweep_S/aggregated.csv"
            if not os.path.exists(csv_path):
                print(f"  ERROR: {csv_path} not found")
                return None, None, None
            
            S_values = []
            energies = []
            radii = []
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    S_values.append(float(row.get("S", 0)))
                    energies.append(float(row.get("energy_mev", 0)))
                    radii.append(float(row.get("radius_fm", 0)))
            
            elapsed = time.time() - start_time
            print(f"  Sweep completed in {elapsed:.1f} seconds")
            return S_values, energies, radii
        
        except Exception as e:
            print(f"  ERROR: {e}")
            return None, None, None
    
    def find_crossing(self, S_values, values, target):
        """Find S where value crosses target (linear interpolation)"""
        for i in range(len(values) - 1):
            v1, v2 = values[i], values[i + 1]
            s1, s2 = S_values[i], S_values[i + 1]
            
            # Check if target is between v1 and v2
            if (v1 - target) * (v2 - target) < 0:
                # Linear interpolation
                S_cross = s1 + (s2 - s1) * (target - v1) / (v2 - v1)
                return S_cross, abs(v1 - target), abs(v2 - target)
        
        # No crossing found
        return None, None, None
    
    def run(self):
        """Execute the calibration finder with 2D (b_range, b_form) optimization"""
        print(f"Starting calibration finder (2D mode)")
        print(f"  Initial: b_range={self.b_range:.4f}, b_form={self.b_form:.4f}")
        print(f"  Targets: energy={self.energy_target} MeV, radius={self.radius_target} fm")
        print(f"  S sweep: {self.S_min} to {self.S_max} in {self.S_steps} steps")
        print(f"  Tolerance: {self.tolerance} MeV\\n")
        
        best_delta_S = float('inf')
        direction = 1  # 1 = increasing b_range, -1 = decreasing
        reversal_count = 0  # Track consecutive reversals
        last_S_energy_pos = None  # Track which side S_energy was on
        
        # Bounds for 2D search
        B_RANGE_MIN, B_RANGE_MAX = 2.0, 5.0
        B_FORM_MIN, B_FORM_MAX = 1.0, 2.0
        
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            print(f"  Parameters: b_range={self.b_range:.4f}, b_form={self.b_form:.4f}")
            
            # Run S sweep
            S_values, energies, radii = self.run_sweep_S(self.b_range, self.b_form)
            
            if S_values is None:
                print("ERROR: S sweep failed")
                break
            
            # Find crossings
            S_energy, _, _ = self.find_crossing(S_values, energies, self.energy_target)
            S_radius, _, _ = self.find_crossing(S_values, radii, self.radius_target)
            
            if S_energy is None or S_radius is None:
                print(f"  WARNING: Target not found in sweep range")
                print(f"    Energy range: {min(energies):.5f} to {max(energies):.5f} MeV")
                print(f"    Radius range: {min(radii):.5f} to {max(radii):.5f} fm")
                
                # Try to adjust S range
                if energies[-1] > self.energy_target:  # Not bound enough
                    print(f"  → Need to increase coupling (not enough binding)")
                    # Adjust b_range or b_form
                    self.b_range *= 1.1
                else:
                    print(f"  → Need to decrease coupling (too much binding)")
                    self.b_range *= 0.9
                
                continue
            
            print(f"  Result: S_energy={S_energy:.4f}, S_radius={S_radius:.4f}")
            delta_S = abs(S_energy - S_radius)
            print(f"  Difference: ΔS = {delta_S:.4f} MeV")
            
            # Backtracking: if getting worse, reverse direction
            if delta_S < best_delta_S:
                best_delta_S = delta_S
                reversal_count = 0  # Reset reversal counter on improvement
                print(f"  ✓ Improved (best so far)")
            else:
                print(f"  ✗ Worse than best (ΔS={best_delta_S:.4f}), reversing direction")
                reversal_count += 1
                direction *= -1  # Reverse direction
                
                # If reversing too many times, also adjust b_form (2D search)
                if reversal_count >= 2:
                    print(f"  ⚠ Multiple reversals ({reversal_count}) - adjusting b_form")
                    
                    # Determine b_form adjustment direction based on crossing position
                    if S_energy < S_radius:
                        # Radius found at higher S (curve too loose) → increase b_form (make it tighter)
                        b_form_delta = 0.05
                        self.b_form += b_form_delta
                        print(f"    Adjust b_form: +{b_form_delta:.4f} (radius curve too loose)")
                    else:
                        # Radius found at lower S (curve too stiff) → decrease b_form (make it looser)
                        b_form_delta = -0.05
                        self.b_form += b_form_delta
                        print(f"    Adjust b_form: {b_form_delta:.4f} (radius curve too stiff)")
                    
                    # Clamp b_form to bounds
                    self.b_form = max(B_FORM_MIN, min(B_FORM_MAX, self.b_form))
                    reversal_count = 0  # Reset reversal counter after b_form adjustment
            
            self.history.append({
                "iteration": iteration,
                "b_range": self.b_range,
                "b_form": self.b_form,
                "S_energy": S_energy,
                "S_radius": S_radius,
                "S_diff": delta_S
            })
            
            # Check convergence
            if delta_S < self.tolerance:
                print(f"\n✓✓✓ CONVERGED! Energy and radius targets overlap at S={S_energy:.4f}")
                
                # Generate plot for this run
                print(f"  Generating plot...")
                subprocess.run([
                    "python3", "scripts/plot_results.py", "energy_sweep_S"
                ], capture_output=True)
                
                # Save final calibration
                final_result = {
                    "b_range": self.b_range,
                    "b_form": self.b_form,
                    "S": S_energy,
                    "S_energy": S_energy,
                    "S_radius": S_radius,
                    "iteration": iteration,
                    "convergence_history": self.history
                }
                
                result_path = os.path.join(self.results_dir, "calibration.json")
                with open(result_path, 'w') as f:
                    json.dump(final_result, f, indent=2)
                print(f"  Saved calibration to: {result_path}")
                
                # Copy sweep results
                src = "results/energy_sweep_S/aggregated.csv"
                dst = os.path.join(self.results_dir, f"sweep_S_final_b_range_{self.b_range:.3f}_b_form_{self.b_form:.3f}.csv")
                if os.path.exists(src):
                    import shutil
                    shutil.copy(src, dst)
                    print(f"  Saved sweep results to: {dst}")
                
                return final_result
            
            # Adjust parameters with backtracking direction and 2D bounds
            if S_energy < S_radius:
                delta_S_gap = S_radius - S_energy
                adjustment = min(0.1, delta_S_gap / 10.0)
                self.b_range += adjustment * direction
                print(f"  → Adjust b_range: {adjustment * direction:+.4f} (S_energy < S_radius)")
            else:
                delta_S_gap = S_energy - S_radius
                adjustment = min(0.1, delta_S_gap / 10.0)
                self.b_range -= adjustment * direction
                print(f"  → Adjust b_range: {-adjustment * direction:+.4f} (S_energy > S_radius)")
            
            # Clamp b_range to bounds
            self.b_range = max(B_RANGE_MIN, min(B_RANGE_MAX, self.b_range))
            
            print()
        
        print(f"Max iterations reached without full convergence")
        if self.history:
            print(f"Best attempt: ΔS = {min(h['S_diff'] for h in self.history):.4f}")
        else:
            print(f"No successful iterations completed")
        
        return None
    
    def save_history(self):
        """Save iteration history to CSV"""
        csv_path = os.path.join(self.results_dir, "finder_history.csv")
        
        if self.history:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["iteration", "b_range", "b_form", "S_energy", "S_radius", "S_diff"])
                writer.writeheader()
                writer.writerows(self.history)
            
            print(f"\nHistory saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Find calibrated b_range and b_form")
    parser.add_argument("--b_range_init", type=float, default=2.6, help="Initial b_range")
    parser.add_argument("--b_form_init", type=float, default=1.2, help="Initial b_form")
    parser.add_argument("--S_min", type=float, default=34.0, help="Minimum S for sweep")
    parser.add_argument("--S_max", type=float, default=42.0, help="Maximum S for sweep")
    parser.add_argument("--S_steps", type=int, default=12, help="Number of S steps")
    parser.add_argument("--energy_target", type=float, default=-2.224, help="Target energy (MeV)")
    parser.add_argument("--radius_target", type=float, default=2.128, help="Target radius (fm)")
    parser.add_argument("--tolerance", type=float, default=0.5, help="S overlap tolerance (MeV)")
    parser.add_argument("--max_iterations", type=int, default=20, help="Maximum iterations")
    parser.add_argument("--jobs", type=int, default=8, help="Number of parallel jobs for S sweep")
    
    args = parser.parse_args()
    
    finder = CalibrationFinder(
        b_range_init=args.b_range_init,
        b_form_init=args.b_form_init,
        S_min=args.S_min,
        S_max=args.S_max,
        S_steps=args.S_steps,
        energy_target=args.energy_target,
        radius_target=args.radius_target,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        jobs=args.jobs
    )
    
    result = finder.run()
    finder.save_history()
    
    if result:
        print(f"\n========================================")
        print(f"CALIBRATION FOUND:")
        print(f"  b_range = {result['b_range']:.4f}")
        print(f"  b_form = {result['b_form']:.4f}")
        print(f"  S = {result['S']:.4f}")
        print(f"========================================")
        return 0
    else:
        print(f"\nCalibration not found within max iterations")
        return 1

if __name__ == "__main__":
    sys.exit(main())
