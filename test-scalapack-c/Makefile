
TESTOBJS = main.cpp

CC       = mpicc
CC_FLAGS = -Wall -O3 -Wno-implicit-function-declaration

LAPACK_DIR = $(HOME)/matrix-library/lapack/3.8.0
SCALAPACK_DIR = $(HOME)/matrix-library/scalapack/2.0.0
#  BLAS, LAPACK (and possibly other) libraries needed for linking test programs
LIBS  = -L$(SCALAPACK_DIR)/lib64 -lscalapack  \
        -L$(LAPACK_DIR)/lib64 -llapack -lblas -lrt -lm -lgfortran

all : exec/test_pdgemv exec/test_pdgesv
	@echo 
	@echo Program Compiled Successfully!
	@echo 

exec/test_pdgemv: src/test_pdgemv.c
	$(CC) $(CC_FLAGS) -o $@ $< $(LIBS)

exec/test_pdgesv: src/test_pdgesv.c
	$(CC) $(CC_FLAGS) -o $@ $< $(LIBS)

run1:
	mpirun -np 4 exec/test_pdgemv

run2:
	mpirun -np 4 exec/test_pdgesv

clean :
	rm -rf exec/test_pdgemv exec/test_pdgesv
