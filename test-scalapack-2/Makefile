
TESTOBJS = main.cpp

FC       = mpif90
FC_FLAGS   = -O3

CC       = mpic++
CC_FLAGS   = -Wall -O3

TARGETS = main

#  BLAS, LAPACK (and possibly other) libraries needed for linking test programs
LIBS  = -L$(SCALAPACK_DIR)/lib $(SCALAPACK_LIBRARIES)  \
        -L$(LAPACK_DIR)/lib $(LAPACK_LIBRARIES)  \
        -L$(BLAS_DIR)/lib $(BLAS_LIBRARIES)

all : $(TARGETS)
	@echo Make Complete

$(TARGETS): $(TESTOBJS)
	$(CC) $(CC_FLAGS) -o main $(TESTOBJS) $(LIBS)

run :
	mpirun -np 4 ./main matrix.txt 8 8 2 2

clean :
	rm -rf *.out *~ *.o $(TARGETS)
