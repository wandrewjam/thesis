### ====================================================================
### Makefile for the jump-velocity plots
### Current target list:
###	all
###	clean
###	avg-vel_test

TXT_FILES = $(wildcard par-files/*.txt)
PNG_FILES = $(patsubst par-files/%.txt, plots/%.png, $(TXT_FILES))
MP4_FILES = $(patsubst par-files/%.txt, plots/%.mp4, $(TXT_FILES))
NPZ_FILES = $(patsubst par-files/%.txt, npz-files/%.npz, $(TXT_FILES))

EXP_FILES = $(wildcard par-files/experiments/*.txt)
SIM_FILES = $(patsubst par-files/experiments/%.txt, dat-files/simulations/%-sim.dat, $(EXP_FILES))
EST_FILES = $(patsubst par-files/experiments/%.txt, dat-files/ml-estimates/%-est.dat, $(EXP_FILES))
BOO_FILES = $(patsubst par-files/experiments/%.txt, dat-files/bootstrap/%-boot.dat, $(EXP_FILES))

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
	TRIALS = 4
	PROC = 2
endif
ifeq ($(UNAME), Linux)
	TRIALS = 100
	PROC = 8
endif

PYTHON = python

ANI_SRC = src/animate.py
PNG_SRC = src/avg-vel.py
NPZ_SRC = src/jv.py
SIM_SRC = src/simulate.py
EST_SRC = src/mle.py
BOO_SRC = src/bootstrap.py

ANI_EXE = $(PYTHON) $(ANI_SRC)
PNG_EXE = $(PYTHON) $(PNG_SRC)
NPZ_EXE = $(PYTHON) $(NPZ_SRC)
SIM_EXE = $(PYTHON) $(SIM_SRC)
EST_EXE = $(PYTHON) $(EST_SRC)
BOO_EXE = $(PYTHON) $(BOO_SRC)

.PHONY : all
all : exact simulate

.PHONY : clean
clean : exact-clean

.PHONY : distclean
distclean : exact-clean simulate-clean

.PHONY : exact
exact-all : exact-animations exact-plots

.PHONY : exact-animations
exact-animations: $(MP4_FILES)

.PHONY : exact-clean
exact-clean :
	rm *~
	rm $(PNG_FILES)
	rm $(MP4_FILES)

.PHONY : exact-data
exact-data : $(NPZ_FILES)

.PHONY : exact-plots
exact-plots : $(PNG_FILES)

.PHONY : install
install :
	@echo There is nothing to install for this package

.PHONY : simulate
simulate : simulate-bootstrap simulate-estimate

.PHONY : simulate-bootstrap
simulate-bootstrap : $(BOO_FILES)

.PHONY : simulate-clean
simulate-clean :
	rm $(SIM_FILES)
	rm $(EST_FILES)
	rm $(BOO_FILES)

.PHONY : simulate-data
simulate-data : $(SIM_FILES)

.PHONY : simulate-estimate
simulate-estimate : $(EST_FILES)

.PHONY : variables
variables :
	@echo TXT_FILES: $(TXT_FILES)
	@echo PNG_FILES: $(PNG_FILES)
	@echo MP4_FILES: $(MP4_FILES)
	@echo NPZ_FILES: $(NPZ_FILES)
	@echo EXP_FILES: $(EXP_FILES)
	@echo SIM_FILES: $(SIM_FILES)
	@echo EST_FILES: $(EST_FILES)
	@echo BOO_FILES: $(BOO_FILES)
	@echo TRIALS: $(TRIALS)
	@echo PROC: $(PROC)

###########################################
# Defines rules for specific targets

# Exact numerical simulations
plots/%.mp4 : npz-files/%.npz $(ANI_SRC)
	$(ANI_EXE) $*

plots/%.png : npz-files/%.npz $(PNG_SRC)
	$(PNG_EXE) $*

npz-files/%.npz : par-files/%.txt $(NPZ_SRC)
	$(NPZ_EXE) $*

# Statistical simulations
dat-files/simulations/%-sim.dat : par-files/experiments/%.txt $(SIM_SRC)
	$(SIM_EXE) $*

dat-files/ml-estimates/%-est.dat : dat-files/simulations/%-sim.dat $(EST_SRC)
	$(EST_EXE) $*

dat-files/bootstrap/%-boot.dat : dat-files/simulations/%-sim.dat $(BOO_SRC)
	$(BOO_EXE) $* $(TRIALS) $(PROC)
