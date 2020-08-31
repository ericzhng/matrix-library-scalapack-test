#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include <sys/time.h>


static int max(int a, int b)
{
	if (a > b)
		return (a);
	else
		return (b);
}


#ifdef F77_WITH_NO_UNDERSCORE
#define numroc_ numroc
#define descinit_ descinit
#define pdlamch_ pdlamch
#define pdlange_ pdlange
#define pdlacpy_ pdlacpy
#define pdgesv_ pdgesv
#define pdgemm_ pdgemm
#define indxg2p_ indxg2p
#endif


extern void Cblacs_pinfo(int *mypnum, int *nprocs);
extern void Cblacs_get(int context, int request, int *value);
extern int Cblacs_gridinit(int *context, char *order, int np_row, int np_col);
extern void Cblacs_gridinfo(int context, int *np_row, int *np_col, int *my_row, int *my_col);
extern void Cblacs_gridexit(int context);
extern void Cblacs_exit(int error_code);

extern int numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern void descinit_(int *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc,
					  int *ictxt, int *lld, int *info);
extern double pdlamch_(int *ictxt, char *cmach);
extern double pdlange_(char *norm, int *m, int *n, double *A, int *ia, int *ja, int *desca, double *work);

extern void pdlacpy_(char *uplo, int *m, int *n, double *a, int *ia, int *ja, int *desca,
					 double *b, int *ib, int *jb, int *descb);
extern void pdgesv_(int *n, int *nrhs, double *A, int *ia, int *ja, int *desca, int *ipiv,
					double *B, int *ib, int *jb, int *descb, int *info);
extern void pdgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K, double *ALPHA,
					double *A, int *IA, int *JA, int *DESCA, double *B, int *IB, int *JB, int *DESCB,
					double *BETA, double *C, int *IC, int *JC, int *DESCC);
extern int indxg2p_(int *indxglob, int *nb, int *iproc, int *isrcproc, int *nprocs);


int main(int argc, char **argv)
{
	int iam, nprocs;
	int myrank_mpi, nprocs_mpi;
	int ictxt, nprow, npcol, myrow, mycol;
	int np, nq, n, nb, nqrhs, nrhs;
	int i, j, k, info, itemp, seed;
	int descA[9], descB[9];
	double *A, *Acpy, *B, *X, *R, eps;
	double AnormF, XnormF, RnormF, BnormF, residF;
	double AnormI, XnormI, RnormI, BnormI, residI;
	double Anorm1, Xnorm1, Rnorm1, Bnorm1, resid1;
	int *ippiv;
	int izero = 0, ione = 1;
	double mone = (-1.0e0), pone = (1.0e0);
	int iarow, mp0, iacol, np0;

	double SYSt1, SYSt2, SYSelapsed;
	double MPIt1, MPIt2, MPIelapsed;
	struct timeval tp;

	/* mpi initialization */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

	n = 16;
	nrhs = 1;
	nprow = 2;
	npcol = 2;
	nb = 4;

	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-n") == 0)
		{
			n = atoi(argv[i + 1]);
			i++;
		}
		if (strcmp(argv[i], "-nrhs") == 0)
		{
			nrhs = atoi(argv[i + 1]);
			i++;
		}
		if (strcmp(argv[i], "-p") == 0)
		{
			nprow = atoi(argv[i + 1]);
			i++;
		}
		if (strcmp(argv[i], "-q") == 0)
		{
			npcol = atoi(argv[i + 1]);
			i++;
		}
		if (strcmp(argv[i], "-nb") == 0)
		{
			nb = atoi(argv[i + 1]);
			i++;
		}
	}
	/**/
	if (nb > n)
		nb = n;
	if (nprow * npcol > nprocs_mpi)
	{
		if (myrank_mpi == 0)
		{
			printf(" **** ERROR : we do not have enough processes available to make a p-by-q process grid ***\n");
			printf(" **** Bye-bye                                                                         ***\n");
			MPI_Finalize();
			exit(1);
		}
	}

	/**/
	Cblacs_pinfo(&iam, &nprocs);
	Cblacs_get(-1, 0, &ictxt);
	Cblacs_gridinit(&ictxt, "Row", nprow, npcol);
	Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
	/**/

	if (iam == 0)
	{
		printf("\tn = %d\t  nrhs = %d\t  (%d,%d)\t  %dx%d \n", n, nrhs, nprow, npcol, nb, nb);
		printf("Hello World, I am proc %d over %d for MPI, proc %d over %d for BLACS in position (%d,%d) in the process grid\n",
			   myrank_mpi, nprocs_mpi, iam, nprocs, myrow, mycol);
	}

	/*
	*
	*     Work only the process in the process grid
	*
	*/
	if ((myrow < nprow) & (mycol < npcol))
	{
		/*
		*
		*     Compute the size of the local matrices (thanks to numroc)
		*
		*/
		np = numroc_(&n, &nb, &myrow, &izero, &nprow);
		nq = numroc_(&n, &nb, &mycol, &izero, &npcol);
		nqrhs = numroc_(&nrhs, &nb, &mycol, &izero, &npcol);
		/*
		*
		*     Allocate and fill the matrices A and B
		*
		*/
		seed = iam * n * (n + nrhs);
		srand(seed);
		/**/
		A = (double *)calloc(np * nq, sizeof(double));
		if (A == NULL)
		{
			printf("error of memory allocation A on proc %dx%d\n", myrow, mycol);
			exit(0);
		}
		/**/
		Acpy = (double *)calloc(np * nq, sizeof(double));
		if (Acpy == NULL)
		{
			printf("error of memory allocation Acpy on proc %dx%d\n", myrow, mycol);
			exit(0);
		}
		/**/
		B = (double *)calloc(np * nqrhs, sizeof(double));
		if (B == NULL)
		{
			printf("error of memory allocation B on proc %dx%d\n", myrow, mycol);
			exit(0);
		}
		/**/
		X = (double *)calloc(np * nqrhs, sizeof(double));
		if (X == NULL)
		{
			printf("error of memory allocation X on proc %dx%d\n", myrow, mycol);
			exit(0);
		}
		/**/
		R = (double *)calloc(np * nqrhs, sizeof(double));
		if (R == NULL)
		{
			printf("error of memory allocation R on proc %dx%d\n", myrow, mycol);
			exit(0);
		}
		/**/
		ippiv = (int *)calloc(np + nb, sizeof(int));
		if (ippiv == NULL)
		{
			printf("error of memory allocation IPIV on proc %dx%d\n", myrow, mycol);
			exit(0);
		}
		/**/
		k = 0;
		for (i = 0; i < np; i++)
		{
			for (j = 0; j < nq; j++)
			{
				A[k] = ((double)rand()) / ((double)RAND_MAX) - 0.5;
				k++;
			}
		}
		k = 0;
		for (i = 0; i < np; i++)
		{
			for (j = 0; j < nqrhs; j++)
			{
				B[k] = ((double)rand()) / ((double)RAND_MAX) - 0.5;
				k++;
			}
		}

		// Initialize the array descriptor for the matrix A and B

		itemp = max(1, np);
		descinit_(descA, &n, &n, &nb, &nb, &izero, &izero, &ictxt, &itemp, &info);
		descinit_(descB, &n, &nrhs, &nb, &nb, &izero, &izero, &ictxt, &itemp, &info);

		// Make a copy of A and the rhs for checking purposes

		pdlacpy_("All", &n, &n, A, &ione, &ione, descA, Acpy, &ione, &ione, descA);
		pdlacpy_("All", &n, &nrhs, B, &ione, &ione, descB, X, &ione, &ione, descB);
		
		/*
		**********************************************************************
		*     Call ScaLAPACK PDGESV routine
		**********************************************************************
		*/
		/**/
		gettimeofday(&tp, NULL);
		SYSt1 = (double)tp.tv_sec + (1.e-6) * tp.tv_usec;
		MPIt1 = MPI_Wtime();
		
		pdgesv_(&n, &nrhs, A, &ione, &ione, descA, ippiv, X, &ione, &ione, descB, &info);
		
		MPIt2 = MPI_Wtime();

		double MPIelapsed = MPIt2 - MPIt1;
		gettimeofday(&tp, NULL);
		SYSt2 = (double)tp.tv_sec + (1.e-6) * tp.tv_usec;

		double SYSelapsed = SYSt2 - SYSt1;
		
		if (iam == 0)
		{
			printf("time elapsed: mpi: %f, sys: %f\n", MPIelapsed, SYSelapsed);
		}

		/*
		*     Compute residual ||A * X  - B|| / ( ||X|| * ||A|| * eps * N )
		*     Froebenius norm
		*/
		pdlacpy_("All", &n, &nrhs, B, &ione, &ione, descB, R, &ione, &ione, descB);

		eps = pdlamch_(&ictxt, "Epsilon");

		pdgemm_("N", "N", &n, &nrhs, &n, &pone, Acpy, &ione, &ione, descA, X, &ione, &ione, descB,
				&mone, R, &ione, &ione, descB);

		double workt = 0.0;
		double *work = &workt;

		AnormF = pdlange_("F", &n, &n, A, &ione, &ione, descA, work);
		BnormF = pdlange_("F", &n, &nrhs, B, &ione, &ione, descB, work);
		XnormF = pdlange_("F", &n, &nrhs, X, &ione, &ione, descB, work);
		RnormF = pdlange_("F", &n, &nrhs, R, &ione, &ione, descB, work);
		residF = RnormF / (AnormF * XnormF * eps * ((double)n));
		/**/

		if (iam == 0)
		{
			printf("\t%5d\t%e\t%e\t%e\t%e\t%e\t %% nrm_frob\n", n, AnormF, BnormF, XnormF, RnormF, residF);
		}

		/*
		*     Compute residual ||A * X  - B|| / ( ||X|| * ||A|| * eps * N )
		*     Infinite norm
		*
		*     Compute the size of the workspace, we want Mp0 such that:
		*          IROFFA = MOD( IA-1, MB_A )
		*          IAROW = INDXG2P( IA, MB_A, MYROW, RSRC_A, NPROW )
		*          Mp0 = NUMROC( M+IROFFA, MB_A, MYROW, IAROW, NPROW )
		*     But not that here IA-1 = 0 so IROFFA = 0     
		*/

		iarow = indxg2p_(&ione, &nb, &myrow, &izero, &nprow);
		mp0 = numroc_(&n, &nb, &myrow, &iarow, &nprow);
		work = (double *)calloc(mp0, sizeof(double));
		if (work == NULL)
		{
			printf("error of memory allocation\n");
			exit(0);
		}
		/**/


		pdlacpy_("All", &n, &nrhs, B, &ione, &ione, descB, R, &ione, &ione, descB);
		eps = pdlamch_(&ictxt, "Epsilon");

		pdgemm_("N", "N", &n, &nrhs, &n, &pone, Acpy, &ione, &ione, descA, X, &ione, &ione, descB,
				&mone, R, &ione, &ione, descB);
		
		AnormI = pdlange_("I", &n, &n, A, &ione, &ione, descA, work);
		BnormI = pdlange_("I", &n, &nrhs, B, &ione, &ione, descB, work);
		XnormI = pdlange_("I", &n, &nrhs, X, &ione, &ione, descB, work);
		RnormI = pdlange_("I", &n, &nrhs, R, &ione, &ione, descB, work);
		residI = RnormI / (AnormI * XnormI * eps * ((double)n));
		/**/
		free(work);
		/**/
		if (iam == 0)
		{
			printf("\t%5d\t%e\t%e\t%e\t%e\t%e\t %% nrm_inf\n", n, AnormI, BnormI, XnormI, RnormI, residI);
		}

/*
*     Compute residual ||A * X  - B|| / ( ||X|| * ||A|| * eps * N )
*     in 1-norm
*
*     Compute the size of the workspace, we want Mp0 such that:
*          ICOFFA = MOD( JA-1, NB_A )
*          IACOL = INDXG2P( JA, NB_A, MYCOL, CSRC_A, NPCOL ),
*          Nq0 = NUMROC( N+ICOFFA, NB_A, MYCOL, IACOL, NPCOL ),
*     But not that here JA-1 = 0 so ICOFFA = 0     
*/

		iacol = indxg2p_(&ione, &nb, &mycol, &izero, &npcol);
		np0 = numroc_(&n, &nb, &mycol, &iacol, &npcol);
		work = (double *)calloc(np0, sizeof(double));
		if (work == NULL)
		{
			printf("error of memory allocation\n");
			exit(0);
		}
		/**/

		pdlacpy_("All", &n, &nrhs, B, &ione, &ione, descB, R, &ione, &ione, descB);
		eps = pdlamch_(&ictxt, "Epsilon");
		pdgemm_("N", "N", &n, &nrhs, &n, &pone, Acpy, &ione, &ione, descA, X, &ione, &ione, descB,
				&mone, R, &ione, &ione, descB);
		Anorm1 = pdlange_("1", &n, &n, A, &ione, &ione, descA, work);
		Bnorm1 = pdlange_("1", &n, &nrhs, B, &ione, &ione, descB, work);
		Xnorm1 = pdlange_("1", &n, &nrhs, X, &ione, &ione, descB, work);
		Rnorm1 = pdlange_("1", &n, &nrhs, R, &ione, &ione, descB, work);
		resid1 = Rnorm1 / (Anorm1 * Xnorm1 * eps * ((double)n));
		/**/
		free(work);
		/**/
		if (iam == 0)
		{
			printf("\t%5d\t%e\t%e\t%e\t%e\t%e\t %% nrm1\n", n, Anorm1, Bnorm1, Xnorm1, Rnorm1, resid1);
		}

		/**/
		free(A);
		free(Acpy);
		free(B);
		free(X);
		free(ippiv);
	}

	Cblacs_gridexit(0);
	MPI_Finalize();
	
	exit(0);
}
