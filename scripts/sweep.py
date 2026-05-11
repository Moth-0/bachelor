#!/usr/bin/env python3
"""
sweep.py - Orchestrates parameter sweeps for the deuteron SVM solver

Runs multiple deu instances with varying parameters, collects results into
organized CSV files by scan type, with full metadata tracking for reproducibility.

Usage:
  python3 scripts/sweep.py --scan b_form \\
    --b_range 2.5 --S 38.4 \\
    --b_form_min 1.0 --b_form_max 2.5 --b_form_steps 10 \\
    [--jobs 4]

  python3 scripts/sweep.py --scan S \\
    --b_range 2.5 --b_form 1.5 \\
    --S_min 20 --S_max 150 --S_steps 15 \\
    [--jobs 4]

  python3 scripts/sweep.py --scan basis_size \\
    --b_range 2.5 --b_form 1.5 --S 38.4 \\
    --basis_size_steps 5 \\
    [--pn-rel] [--pi-rel] [--jobs 4]

  # basis_size_steps controls convergence: generates {0.0}, {0.1,0.0}, {0.5,0.1,0.0}, etc.
"""

import argparse
import subprocess
import os
import sys
import csv
import json
import glob
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import hashlib

class ParameterSweep:
    """Manages a single parameter sweep experiment"""
    
    def __init__(self, scan_type, fixed_params, sweep_params, num_jobs=1):
        self.scan_type = scan_type
        self.fixed_params = fixed_params
        self.sweep_params = sweep_params
        self.num_jobs = num_jobs
        self.results_dir = f"results/energy_sweep_{scan_type}"
        self.run_dir = self.results_dir
        self.log_file = os.path.join(self.results_dir, "sweep.log")
        
    def generate_parameter_combinations(self):
        """Generate all parameter combinations for the sweep"""
        combinations = []
        
        # Base strengths for box confinement
        base_strengths = [0.0, 0.1, 1.0, 0.5, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]
        
        # Get box strengths configuration (only for non-basis_size scans)
        box_strengths = None
        if self.scan_type in ["b_form", "S"]:
            num_boxes = int(self.sweep_params.get("basis_size_steps", 1))
            box_strengths = sorted(base_strengths[:num_boxes], reverse=True)
        
        if self.scan_type == "b_form":
            b_forms = np.linspace(
                self.sweep_params["b_form_min"],
                self.sweep_params["b_form_max"],
                self.sweep_params["b_form_steps"]
            )
            for b_form in b_forms:
                params = {
                    "b_range": self.fixed_params["b_range"],
                    "b_form": b_form,
                    "S": self.fixed_params["S"]
                }
                if box_strengths is not None:
                    params["box_strengths"] = box_strengths
                combinations.append(params)
                
        elif self.scan_type == "S":
            S_values = np.linspace(
                self.sweep_params["S_min"],
                self.sweep_params["S_max"],
                self.sweep_params["S_steps"]
            )
            for S in S_values:
                params = {
                    "b_range": self.fixed_params["b_range"],
                    "b_form": self.fixed_params["b_form"],
                    "S": S
                }
                if box_strengths is not None:
                    params["box_strengths"] = box_strengths
                combinations.append(params)
                
        elif self.scan_type == "basis_size":
            # Generate box strength configurations with increasing counts
            # Step 1: [0.0], Step 2: [0.1, 0.0], Step 3: [0.5, 0.1, 0.0], etc.
            num_steps = int(self.sweep_params.get("basis_size_steps", 5))
            
            for step in range(7, num_steps + 1):
                # Take 'step' number of base strengths (sorted for increasing basis)
                step_box_strengths = sorted(base_strengths[:step], reverse=True)
                params = dict(self.fixed_params)  # Copy all fixed params
                params.update({
                    "box_strengths": step_box_strengths,  # Pass as list
                    "step": step  # For identification
                })
                combinations.append(params)
        
        return combinations
    
    def run_deu(self, params):
        """Execute a single deu run with given parameters"""
        # Generate a hash-based run ID
        param_str = "_".join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                             for k, v in sorted(params.items()) if k not in ["step", "box_strengths"]])
        # Include box_strengths in hash if present
        if "box_strengths" in params:
            box_str = ",".join([f"{x:.1f}" for x in params["box_strengths"]])
            param_str += f"_box={box_str}"
        if "step" in params:
            param_str += f"_step{params['step']}"
        
        run_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        run_id = f"run_{run_hash}"
        
        # Create output CSV path
        csv_path = os.path.join(self.run_dir, f"{run_id}.csv")
        
        # Build deu command
        cmd = ["./deu"]
        cmd.extend(["-b_range", str(params["b_range"])])
        cmd.extend(["-b_form", str(params["b_form"])])
        cmd.extend(["-S", str(params["S"])])
        cmd.extend(["--output-csv", csv_path])
        
        # Pass box strengths if present
        if "box_strengths" in params:
            box_strengths_str = ",".join([str(x) for x in params["box_strengths"]])
            cmd.extend(["-box-strengths", box_strengths_str])
        
        # Pass relativistic flags if present (default: pn_rel=false, pi_rel=true)
        if "pn_rel" in params:
            cmd.extend(["--pn-rel", "true" if params["pn_rel"] else "false"])
        if "pi_rel" in params:
            cmd.extend(["--pi-rel", "true" if params["pi_rel"] else "false"])
        
        # Execute deu
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=36000  # 10 hour timeout per run
            )
            
            if result.returncode != 0:
                return {
                    "status": "FAILED",
                    "params": params,
                    "run_id": run_id,
                    "csv_path": csv_path,
                    "error": result.stderr
                }
            
            return {
                "status": "SUCCESS",
                "params": params,
                "run_id": run_id,
                "csv_path": csv_path
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "TIMEOUT",
                "params": params,
                "run_id": run_id,
                "csv_path": csv_path
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "params": params,
                "run_id": run_id,
                "csv_path": csv_path,
                "error": str(e)
            }
    
    def extract_final_row(self, csv_path):
        """Extract the FINAL row from a run CSV"""
        try:
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            
            # Skip metadata lines and header
            data_lines = [l for l in lines if not l.startswith("#")]
            
            if len(data_lines) < 2:  # header + at least one data row
                return None
            
            # Last data row should be FINAL (iteration == -1)
            reader = csv.DictReader(data_lines)
            rows = list(reader)
            
            if rows:
                final_row = rows[-1]
                if float(final_row.get("iteration", 0)) == -1:
                    return final_row
                # Otherwise return last row
                return rows[-1]
            
            return None
        except Exception as e:
            print(f"Error extracting final row from {csv_path}: {e}", file=sys.stderr)
            return None
    
    def aggregate_results(self, run_results):
        """Aggregate all run results into a single CSV"""
        aggregated_path = os.path.join(self.results_dir, "aggregated.csv")
        
        aggregated_rows = []
        
        for result in run_results:
            if result["status"] != "SUCCESS":
                print(f"Skipping {result['run_id']}: {result['status']}", file=sys.stderr)
                continue
            
            csv_path = result["csv_path"]
            if not os.path.exists(csv_path):
                print(f"Warning: CSV not found for {result['run_id']}: {csv_path}", file=sys.stderr)
                continue
            
            final_row = self.extract_final_row(csv_path)
            if final_row:
                # Add parameter columns
                row = dict(result["params"])
                row.update(final_row)
                aggregated_rows.append(row)
        
        # Sort by scan parameter for readability
        if self.scan_type == "b_form":
            aggregated_rows.sort(key=lambda r: float(r.get("b_form", 0)))
        elif self.scan_type == "S":
            aggregated_rows.sort(key=lambda r: float(r.get("S", 0)))
        elif self.scan_type == "basis_size":
            aggregated_rows.sort(key=lambda r: int(r.get("max_basis_size", 0)))
        
        # Write aggregated CSV
        if aggregated_rows:
            headers = list(aggregated_rows[0].keys())
            with open(aggregated_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(aggregated_rows)
            
            print(f"\nAggregated results written to: {aggregated_path}")
            print(f"  Rows: {len(aggregated_rows)}")
            print(f"  Headers: {', '.join(headers[:5])}...")
            
            # Clean up individual run_*.csv files
            run_files = glob.glob(os.path.join(self.results_dir, "run_*.csv"))
            for run_file in run_files:
                try:
                    os.remove(run_file)
                except Exception as e:
                    print(f"Warning: Could not delete {run_file}: {e}", file=sys.stderr)
            if run_files:
                print(f"  Cleaned up {len(run_files)} individual run files")
            
            return aggregated_path
        else:
            print(f"ERROR: No results to aggregate!", file=sys.stderr)
            return None
    
    def print_summary(self, run_results):
        """Print summary of sweep results"""
        print("\n" + "="*80)
        print(f"SWEEP SUMMARY: {self.scan_type}")
        print("="*80)
        
        success = sum(1 for r in run_results if r["status"] == "SUCCESS")
        failed = sum(1 for r in run_results if r["status"] != "SUCCESS")
        
        print(f"Total runs: {len(run_results)}")
        print(f"Successful: {success}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed runs:")
            for r in run_results:
                if r["status"] != "SUCCESS":
                    print(f"  {r['run_id']}: {r['status']}")
                    if "error" in r:
                        print(f"    Error: {r['error'][:100]}")
        
        print(f"\nResults directory: {self.results_dir}")
        print("="*80 + "\n")
    
    def run(self):
        """Execute the complete sweep"""
        # Setup
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Starting parameter sweep: {self.scan_type}")
        print(f"  Results directory: {self.results_dir}")
        print(f"  Parallel jobs: {self.num_jobs}")
        
        # Generate combinations
        combinations = self.generate_parameter_combinations()
        print(f"  Parameter combinations: {len(combinations)}")
        
        # Run in parallel
        print(f"\nExecuting runs...")
        if self.num_jobs == 1:
            run_results = [self.run_deu(params) for params in combinations]
        else:
            with Pool(self.num_jobs) as pool:
                run_results = pool.map(self.run_deu, combinations)
        
        # Aggregate
        print(f"\nAggregating results...")
        agg_path = self.aggregate_results(run_results)
        
        # Summary
        self.print_summary(run_results)
        
        # Save metadata
        metadata = {
            "scan_type": self.scan_type,
            "timestamp": datetime.now().isoformat(),
            "fixed_params": self.fixed_params,
            "sweep_params": self.sweep_params,
            "num_jobs": self.num_jobs,
            "total_runs": len(run_results),
            "successful_runs": sum(1 for r in run_results if r["status"] == "SUCCESS"),
            "aggregated_csv": agg_path
        }
        
        metadata_path = os.path.join(self.results_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return agg_path

def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep orchestrator for deuteron SVM solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--scan", required=True, 
                       choices=["b_form", "S", "basis_size"],
                       help="Parameter to sweep")
    
    # Fixed parameters
    parser.add_argument("--b_range", type=float, required=True,
                       help="Fixed Gaussian width search space (fm)")
    parser.add_argument("--b_form", type=float, 
                       help="Fixed pion form factor range (fm) - required unless scanning b_form")
    parser.add_argument("--S", type=float,
                       help="Fixed pion coupling strength (MeV) - required unless scanning S")
    
    # Sweep ranges
    parser.add_argument("--b_form_min", type=float,
                       help="Minimum b_form value")
    parser.add_argument("--b_form_max", type=float,
                       help="Maximum b_form value")
    parser.add_argument("--b_form_steps", type=int, default=10,
                       help="Number of b_form steps (default: 10)")
    
    parser.add_argument("--S_min", type=float,
                       help="Minimum S value")
    parser.add_argument("--S_max", type=float,
                       help="Maximum S value")
    parser.add_argument("--S_steps", type=int, default=10,
                       help="Number of S steps (default: 10)")
    
    parser.add_argument("--basis_size_steps", type=int, default=1,
                       help="For basis_size scan: convergence steps. For b_form/S: number of box strengths to use (default: 1)")
    parser.add_argument("--pn-rel", action="store_true", default=False,
                       help="Use relativistic PN channel (default: classical)")
    parser.add_argument("--pi-rel", action="store_true", default=False,
                       help="Use relativistic pion channel (default: classical)")
    
    parser.add_argument("--jobs", type=int, default=1,
                       help="Number of parallel jobs (default: 1)")
    
    args = parser.parse_args()
    
    # Validate fixed parameters
    if args.scan == "b_form":
        if not args.b_form_min or not args.b_form_max:
            parser.error("--b_form_min and --b_form_max required for b_form sweep")
        if not args.S:
            parser.error("--S required for b_form sweep")
    elif args.scan == "S":
        if not args.S_min or not args.S_max:
            parser.error("--S_min and --S_max required for S sweep")
        if not args.b_form:
            parser.error("--b_form required for S sweep")
    elif args.scan == "basis_size":
        if not args.b_form or not args.S:
            parser.error("--b_form and --S required for basis_size sweep")
    
    # Prepare parameters
    fixed_params = {
        "b_range": args.b_range,
    }
    
    sweep_params = {}
    
    if args.scan == "b_form":
        fixed_params["S"] = args.S
        sweep_params = {
            "b_form_min": args.b_form_min,
            "b_form_max": args.b_form_max,
            "b_form_steps": args.b_form_steps,
            "basis_size_steps": args.basis_size_steps  # Box strengths for this scan
        }
    elif args.scan == "S":
        fixed_params["b_form"] = args.b_form
        sweep_params = {
            "S_min": args.S_min,
            "S_max": args.S_max,
            "S_steps": args.S_steps,
            "basis_size_steps": args.basis_size_steps  # Box strengths for this scan
        }
    elif args.scan == "basis_size":
        fixed_params["b_form"] = args.b_form
        fixed_params["S"] = args.S
        if args.pn_rel:
            fixed_params["pn_rel"] = True
        if args.pi_rel:
            fixed_params["pi_rel"] = True
        sweep_params = {
            "basis_size_steps": args.basis_size_steps
        }
    
    # Run sweep
    sweep = ParameterSweep(args.scan, fixed_params, sweep_params, args.jobs)
    agg_path = sweep.run()
    
    return 0 if agg_path else 1

if __name__ == "__main__":
    sys.exit(main())
