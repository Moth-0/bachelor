MAKEFLAGS += -r
# Variables
CXXFLAGS = -Wall -Werror -O3 -std=c++23 -fopenmp
CXX = c++
LDFLAGS += -fopenmp
LDLIBS = -lstdc++ -lm
RM = rm -f

all : deu

HEADERS = qm/matrix.h qm/gaussian.h qm/operators.h qm/jacobi.h qm/solver.h qm/serialization.h qm/csv_writer.h deuterium.h SVM.h

% : %.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o : %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ALL CONFIGURATIONS: Run all 4 kinematic configs with timestamped results
# Default parameters (override with: make all-configs B_RANGE=2.0 B_FORM=1.5 S=40.0)
B_RANGE ?= 2.44
B_FORM ?= 1.2
S ?= 39.17

all-configs: deu
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	RESULTS_DIR="results/all_configs_$$TIMESTAMP"; \
	mkdir -p $$RESULTS_DIR; \
	echo "========================================"; \
	echo "Running all 4 configurations in parallel..."; \
	echo "Parameters: b_range=$(B_RANGE) b_form=$(B_FORM) S=$(S)"; \
	echo "Timestamp: $$TIMESTAMP"; \
	echo "Results: $$RESULTS_DIR"; \
	echo "========================================"; \
	./deu -b_range $(B_RANGE) -b_form $(B_FORM) -S $(S) --output-csv $$RESULTS_DIR/PN_Cla_Pi_Cla.csv & \
	PID1=$$!; \
	./deu -b_range $(B_RANGE) -b_form $(B_FORM) -S $(S) --pi-rel --output-csv $$RESULTS_DIR/PN_Cla_Pi_Rel.csv & \
	PID2=$$!; \
	./deu -b_range $(B_RANGE) -b_form $(B_FORM) -S $(S) --pn-rel --output-csv $$RESULTS_DIR/PN_Rel_Pi_Cla.csv & \
	PID3=$$!; \
	./deu -b_range $(B_RANGE) -b_form $(B_FORM) -S $(S) --pn-rel --pi-rel --output-csv $$RESULTS_DIR/PN_Rel_Pi_Rel.csv & \
	PID4=$$!; \
	wait $$PID1 $$PID2 $$PID3 $$PID4; \
	echo "✓ Config 1/4: PN_Cla Pi_Cla"; \
	echo "✓ Config 2/4: PN_Cla Pi_Rel"; \
	echo "✓ Config 3/4: PN_Rel Pi_Cla"; \
	echo "✓ Config 4/4: PN_Rel Pi_Rel"; \
	echo "========================================"; \
	echo "All configurations complete!"; \
	echo "Results saved to: $$RESULTS_DIR"; \
	echo "========================================"


# Parameter Sweeps 
sweep_S : deu scripts/sweep.py scripts/plot_results.py Makefile
	python3 scripts/sweep.py --scan S --b_range $(B_RANGE) --b_form $(B_FORM) \
	--jobs 6 --basis_size_steps 6 \
	--S_min 34.0 --S_max 42.0 --S_steps 12
	python3 scripts/plot_results.py energy_sweep_S

sweep_b : deu scripts/sweep.py scripts/plot_results.py Makefile
	python3 scripts/sweep.py --scan b_form --b_range $(B_RANGE) --S $(S) --jobs 3 \
	--b_form_min 0.8 --b_form_max 1.4 --b_form_steps 6
	python3 scripts/plot_results.py energy_sweep_b_form
	python3 scripts/sweep.py --scan b_range --b_form $(B_FORM) --S $(S) --jobs 3 \
	--b_range_min 1.6 --b_range_max 2.8 --b_range_steps 6
	python3 scripts/plot_results.py energy_sweep_b_range

sweep_size : deu scripts/sweep.py scripts/plot_results.py Makefile
	python3 scripts/sweep.py --scan basis_size --b_range $(B_RANGE) --b_form $(B_FORM) --S $(S) \
	--jobs 4 --basis_size_steps 8
	python3 scripts/plot_results.py energy_sweep_basis_size

clean :
	$(RM) *.o *.log *.dat *.gpi *.out *.err

.PHONY: all clean all-configs sweep_S sweep_b sweep_size