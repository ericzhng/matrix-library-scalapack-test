
TESTOBJS = main.cpp

FC       = mpif90
FC_FLAGS   = -O3

CC       = mpic++
CC_FLAGS   = -Wall -O3

TARGETS = exec/main
SRC = src/main.cpp

LAPACK_DIR = $(HOME)/matrix-library/lapack/3.8.0
SCALAPACK_DIR = $(HOME)/matrix-library/scalapack/2.0.0
#  BLAS, LAPACK (and possibly other) libraries needed for linking test programs
LIBS  = -L$(SCALAPACK_DIR)/lib64 -lscalapack  \
        -L$(LAPACK_DIR)/lib64 -llapack -lblas -lgfortran

all : $(TARGETS)
	@echo 
	@echo Program Compiled Successfully!
	@echo 
	mpirun -np 4 ./exec/main matrix.txt 8 8 2 2

$(TARGETS): $(SRC)
	mkdir -p exec
	$(CC) $(CC_FLAGS) -o $@ $(SRC) $(LIBS)

clean :
	rm -rf *.o $(TARGETS)
