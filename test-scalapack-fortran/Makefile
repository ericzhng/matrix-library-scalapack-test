# define compiler and compiler flag variables
CC		= mpicc
CXX		= mpic++
CFLAGS = -O3 -Wall
CPPFLAGS = -O3 -Wall -std=c++11

FC		= mpif90
FCFLAGS = -O3 -Wall -Wno-conversion -Wno-unused-label -Wno-unused-dummy-argument
LDFLAGS = -lm -lrt

LAPACK_DIR = $(HOME)/matrix-library/lapack/3.8.0
SCALAPACK_DIR = $(HOME)/matrix-library/scalapack/2.0.0
#  BLAS, LAPACK (and possibly other) libraries needed for linking test programs
LIBS  = -L$(SCALAPACK_DIR)/lib64 -lscalapack  \
        -L$(LAPACK_DIR)/lib64 -llapack -lblas

EXTRA_INC = -I./src/common
EXTRA_LIB = ./src/common/pdscaexinfo.f
LDINC = 
EXTCOMMANDS = --mca btl_vader_single_copy_mechanism none 
 
# define source folder (src or .) and its content
SRC_DIR	= src

# TESTNAME = test-gbmv
# SRCS = $(SRC_DIR)/$(TESTNAME)
SRCS := $(shell find $(SRC_DIR) -maxdepth 1 -name '*.f')

BUILD_DIR ?= exec
SRCOBJS := $(SRCS:$(SRC_DIR)/%.f=$(BUILD_DIR)/%.o)
TARGETS := $(patsubst %.o,%,$(SRCOBJS))
OUTFILES := $(patsubst %.o,%.txt,$(SRCOBJS))

default: $(TARGETS)
	@echo Make Complete

# pattern rules to compile objectives
$(TARGETS): $(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	$(FC) $(FCFLAGS) $(LDINC) $(EXTRA_INC) -o $@ $< $(EXTRA_LIB) $(LIBS) $(LDFLAGS)
	@rm $<
	cd $(dir $@) && mpirun -np 4 ./$(notdir $@) > ../$(notdir $@.txt)
	@echo

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.f
	@echo '====================================='
	@echo 'Compile and execute: ' $<
	$(FC) $(FCFLAGS) -c $< -o $@ $(LDINC) $(EXTRA_INC)

clean:
	rm -rf *.o *.txt $(TARGETS)
