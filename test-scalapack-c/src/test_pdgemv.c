#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#define AA(i, j) AA[(i)*M + (j)]

int main(int argc, char **argv)
{
   int i = 0, j = 0;
   int myrank_mpi, nprocs_mpi;

   /************  MPI ***************************/
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

   /************  BLACS ***************************/
   int ictxt, nprow, npcol, myrow, mycol, nb;
   int info;

   int ZERO = 0, ONE = 1;
   nprow = 2;
   npcol = 2;
   nb = 2;

   Cblacs_pinfo(&myrank_mpi, &nprocs_mpi);
   Cblacs_get(-1, 0, &ictxt);
   Cblacs_gridinit(&ictxt, "Row", nprow, npcol);
   Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
   
   int M = 5;
   double *AA = (double *)malloc(M * M * sizeof(double));
   for (i = 0; i < M; i++)
      for (j = 0; j < M; j++)
         AA[i * M + j] = (i + M * j + 1);
  
   /* Parse command line arguments */
   if (myrank_mpi == 0) {
      /* Print matrix */
      printf("Process=%d -- Matrix AA:\n", myrank_mpi);
      for (i = 0; i < M; i++){
         for (j = 0; j < M; j++)
               printf("%5.1f  ", AA[i * M + j]);
         printf("\n");
      }
      printf("\n");
   }

   double *X = (double *)malloc(M * sizeof(double));
   X[0] = 1;
   X[1] = 0;
   X[2] = 0;
   X[3] = 1;
   X[4] = 1;

   int descA[9], descx[9], descy[9];

   int mA = numroc_(&M, &nb, &myrow, &ZERO, &nprow);
   int nA = numroc_(&M, &nb, &mycol, &ZERO, &npcol);
   
   int nx = numroc_(&M, &nb, &myrow, &ZERO, &nprow);
   int ny = numroc_(&M, &nb, &myrow, &ZERO, &nprow);

   //Cblacs_barrier(ictxt, "All");
   //printf("rank: %d - A divided into (%d, %d)\n", myrank_mpi, mA, nA);
   
   //printf("rank: %d - vector x divided into (%d, 1)\n", myrank_mpi, nx);
   //printf("rank: %d - vector y divided into (%d, 1)\n", myrank_mpi, ny);
   Cblacs_barrier(ictxt, "All");

   descinit_(descA, &M, &M, &nb, &nb, &ZERO, &ZERO, &ictxt, &mA, &info);
   descinit_(descx, &M, &ONE, &nb, &ONE, &ZERO, &ZERO, &ictxt, &nx, &info);
   descinit_(descy, &M, &ONE, &nb, &ONE, &ZERO, &ZERO, &ictxt, &ny, &info);

   double *x = (double *)malloc(nx * sizeof(double));
   double *y = (double *)calloc(ny, sizeof(double));
   double *A = (double *)malloc(mA * nA * sizeof(double));

   Cblacs_barrier(ictxt, "All");

   int sat, sut;
   for (i = 0; i < mA; i++)
      for (j = 0; j < nA; j++)
      {
         sat = (myrow * 3) + i;
         sut = (mycol * 3) + j;
         A[j * mA + i] = AA[sat * M + sut];
         // A[i * nA + j] = 1.0;
      }

   /* Print matrix */
   Cblacs_barrier(ictxt, "All");
   printf(" == processor %d (row=%d, col=%d) -- matrix A (mA=%d, nA=%d):\n", myrank_mpi, myrow, mycol, mA, nA);
   for (j = 0; j < nA; j++){
      for (i = 0; i < mA; i++){
         printf("%5.1f  ", A[j * mA + i]); 
      }
      printf("\n");
   }
   printf("\n");
   Cblacs_barrier(ictxt, "All");


   for (i = 0; i < nx; i++)
   {
      sut = (myrow * 3) + i;
      x[i] = X[sut];
   }


   double alpha = 1.0;
   double beta = 0.0;
   pdgemv_("N", &M, &M, &alpha, A, &ONE, &ONE, descA, x, &ONE, &ONE, descx, &ONE, &beta, y, &ONE, &ONE, descy, &ONE);
   Cblacs_barrier(ictxt, "A");
   

   for (i = 0; i < ny; i++)
      printf("y[%d] = %.2f (for rank %d)\n", i+1, y[i], myrank_mpi);
   

   Cblacs_gridexit(0);
   MPI_Finalize();
   return 0;
}
