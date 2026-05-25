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
B_RANGE ?= 2.24
B_FORM ?= 1.4
S ?= 31.29

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
	( ./deu -b_range $(B_RANGE) -b_form $(B_FORM) -S $(S) --output-csv $$RESULTS_DIR/PN_Cla_Pi_Cla.csv > $$RESULTS_DIR/PN_Cla_Pi_Cla.log 2>&1; echo "✓ Config 1/4 (PN_Cla Pi_Cla) finished!" ) & \
	PID1=$$!; \
	( ./deu -b_range $(B_RANGE) -b_form $(B_FORM) -S $(S) --pi-rel --output-csv $$RESULTS_DIR/PN_Cla_Pi_Rel.csv > $$RESULTS_DIR/PN_Cla_Pi_Rel.log 2>&1; echo "✓ Config 2/4 (PN_Cla Pi_Rel) finished!" ) & \
	PID2=$$!; \
	( ./deu -b_range $(B_RANGE) -b_form $(B_FORM) -S $(S) --pn-rel --output-csv $$RESULTS_DIR/PN_Rel_Pi_Cla.csv > $$RESULTS_DIR/PN_Rel_Pi_Cla.log 2>&1; echo "✓ Config 3/4 (PN_Rel Pi_Cla) finished!" ) & \
	PID3=$$!; \
	( ./deu -b_range $(B_RANGE) -b_form $(B_FORM) -S $(S) --pn-rel --pi-rel --output-csv $$RESULTS_DIR/PN_Rel_Pi_Rel.csv > $$RESULTS_DIR/PN_Rel_Pi_Rel.log 2>&1; echo "✓ Config 4/4 (PN_Rel Pi_Rel) finished!" ) & \
	PID4=$$!; \
	wait $$PID1 $$PID2 $$PID3 $$PID4; \
	echo "========================================"; \
	echo "All configurations complete!"; \
	echo "Results and logs saved to: $$RESULTS_DIR"; \
	echo "========================================"
	

# Plot wavefunction: Generate basis_final.txt, then analyze asymptotic behavior
plot_wavefunction: deu plot_wavefunction.o scripts/plot_wavefunction.py
	./deu -b_range $(B_RANGE) -b_form $(B_FORM) -S $(S)
	./plot_wavefunction basis_final.txt wavefunction.csv
	python3 scripts/plot_wavefunction.py wavefunction.csv wavefunction_plot.png
	@echo "✓ Wavefunction plot generated: wavefunction_plot.png"


# Parameter Sweeps 
sweep_S : deu scripts/sweep_S.py Makefile
	python3 scripts/sweep_S.py --b_range $(B_RANGE) --b_form $(B_FORM) \
	--S_min 60.0 --S_max 62.0 --S_steps 10 --jobs 10

sweep_b_range : deu scripts/sweep_b_range.py Makefile
	python3 scripts/sweep_b_range.py --S $(S) --b_form $(B_FORM) \
	--b_range_min 2.0 --b_range_max 6.0 --b_range_steps 10 --jobs 10

sweep_b_form : deu scripts/sweep_b_form.py Makefile
	python3 scripts/sweep_b_form.py --S $(S) --b_range $(B_RANGE) \
	--b_form_min 0.8 --b_form_max 1.8 --b_form_steps 10 --jobs 10

sweep_size : deu scripts/sweep_basis_size.py Makefile
	python3 scripts/sweep_basis_size.py --b_range $(B_RANGE) --b_form $(B_FORM) --S $(S) \
	--basis_size_steps 6 --jobs 6

contour_b_range : deu scripts/contour_plot_b_range.py Makefile
	python3 scripts/contour_plot_b_range.py --b_range_min 2.0 --b_range_max 2.5 --b_range_steps 10 \
	--S_init_anchor 35.0 --S_window 2.0 \
	--b_form $(B_FORM) --S_steps 10 --jobs 10

contour_b_form : deu scripts/contour_plot_b_form.py Makefile
	python3 scripts/contour_plot_b_form.py --b_form_min 1.0 --b_form_max 1.6 --b_form_steps 6 \
	--S_init_anchor 40.0 --S_window 10.0 \
	--b_range $(B_RANGE) --S_steps 10 --jobs 10

clean :
	$(RM) *.o *.log *.dat *.gpi *.out *.err

.PHONY: all clean all-configs sweep_S sweep_b_range sweep_b_form sweep_size plot_wavefunction contour_b_range contour_b_form