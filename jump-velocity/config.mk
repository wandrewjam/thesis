# Define files variables
TXT_FILES = $(wildcard par-files/*.txt)
PNG_FILES = $(patsubst par-files/%.txt, plots/%.png, $(TXT_FILES))
MP4_FILES = $(patsubst par-files/%.txt, plots/%.mp4, $(TXT_FILES))
NPZ_FILES = $(patsubst par-files/%.txt, npz-files/%.npz, $(TXT_FILES))
DST_FILES = $(patsubst par-files/%.txt, dat-files/distributions/%.dat, $(TXT_FILES))

EXP_FILES = $(wildcard par-files/experiments/*.txt)
SIM_FILES = $(patsubst par-files/experiments/%.txt, dat-files/simulations/%-sim.dat, $(EXP_FILES))
EST_FILES = $(patsubst par-files/experiments/%.txt, dat-files/ml-estimates/%-est.dat, $(EXP_FILES))
BOO_FILES = $(patsubst par-files/experiments/%.txt, dat-files/bootstrap/%-boot.dat, $(EXP_FILES))

# Define the bootstrap trials and processes based on the system
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
	TRIALS = 4
	PROC = 2
endif
ifeq ($(UNAME), Linux)
	TRIALS = 64
	PROC = 8
endif

# Define the excecutables
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
