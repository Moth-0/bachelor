MAKEFLAGS += -r
# Variables
CXXFLAGS = -Wall -Werror -O3 -std=c++23 -fopenmp
CXX = c++
LDFLAGS += -fopenmp
LDLIBS = -lstdc++ -lm
RM = rm -f

all : main

HEADERS = qm/matrix.h qm/gaussian.h qm/hamiltonian.h qm/particle.h qm/jacobi.h qm/solver.h

%.png : %.dat Makefile
	gnuplot -e " \
	set terminal pngcairo size 900,700 enhanced font 'Arial,12'; \
	set output '$@'; \
	set title 'Deuteron E_0(S,b)'; \
	set xlabel 'S (MeV)'; \
	set ylabel 'b (fm)'; \
	set cblabel 'E_0 (MeV)'; \
	set cbrange [-5.0:0.0]; \
	set palette defined ( \
		-5 'dark-blue', \
		-2.2   'white',     \
		0       'dark-red'); \
	set yrange [0.5:4.0]; \
	set xrange [10:50]; \
	plot '$<' using 1:2:3 with image notitle"

heatmap.dat : 
	./heatmap --K_max 10 --N_trial 10 \
    --S_min 10 --S_max 50 --N_S 50 \
    --b_min 0.5 --b_max 4.0 --N_b 50 \
	--s_max 0.1 \
    --output heatmap.dat

% : %.o 
	$(CXX) $(LDFLAGS) $(LDLIBS) -o $@ $^

%.o : %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean :
	$(RM) *.o *.log *.dat *.gpi