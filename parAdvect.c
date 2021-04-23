// parallel 2D advection solver module
// written for COMP4300/8300 Assignment 1 
// v1.0 25 Feb 

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

#include "serAdvect.h"

#define HALO_TAG 100

int M_loc, N_loc; // local advection field size (excluding halo) 
int M0, N0;       // local field element (0,0) is global element (M0,N0)
static int P0, Q0; // 2D process id (P0, Q0) in P x Q process grid 

static int M, N, P, Q; // local store of problem parameters
static int verbosity;
static int rank, nprocs;       // MPI values
static MPI_Comm comm;
static int comm_mode;

//sets up parallel parameters above
void initParParams(int M_, int N_, int P_, int Q_, int verb, int comm_mode_) {
  M = M_, N = N_; P = P_, Q = Q_;
  comm_mode = comm_mode_;
  verbosity = verb;
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  P0 = rank / Q;
  M0 = (M / P) * P0;
  M_loc = (P0 < P-1)? (M / P): (M - M0);

  Q0 = rank / P;
  N0 = (N / Q) * Q0;
  N_loc = (Q0 < Q-1)? (N / Q): (N - N0);
} //initParParams()


void checkHaloSize(int w) {
  if (w > M_loc || w > N_loc) {
    printf("%d: w=%d too large for %dx%d local field! Exiting...\n",
           rank, w, M_loc, N_loc);
    exit(1);
  }
}


static void updateBoundary(double *u, int ldu) {
  int i, j;

  //top and bottom halo
  //note: we get the left/right neighbour's corner elements from each end
  if (P == 1) {
    for (j = 1; j < N_loc+1; j++) {
      V(u, 0, j) = V(u, M_loc, j);
      V(u, M_loc+1, j) = V(u, 1, j);      
    }
  }
  else {
    int topProc = (rank + Q) % nprocs, botProc = (rank - Q + nprocs) % nprocs;
    /* Send from all odd nodes first then send from all even nodes */
    /* Only works till Q3 */
    if (comm_mode == 0) {
        if (rank % 2 == 0) {
          MPI_Send(&V(u, M_loc, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm);
          MPI_Recv(&V(u, 0, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm,
                   MPI_STATUS_IGNORE);
          MPI_Send(&V(u, 1, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm);
          MPI_Recv(&V(u, M_loc+1, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG,
                   comm, MPI_STATUS_IGNORE);
        } else {
          MPI_Recv(&V(u, 0, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm,
                   MPI_STATUS_IGNORE);
          MPI_Send(&V(u, M_loc, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm);
          MPI_Recv(&V(u, M_loc+1, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG,
                   comm, MPI_STATUS_IGNORE);
          MPI_Send(&V(u, 1, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm);
        }
    }
    else {
      MPI_Request request[4]; int nReq = 0;
      MPI_Isend(&V(u, M0 + M_loc    , N0 + 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm, &request[nReq++]);
      MPI_Irecv(&V(u, M0            , N0 + 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm, &request[nReq++]);
      MPI_Isend(&V(u, M0 + 1        , N0 + 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm, &request[nReq++]);
      MPI_Irecv(&V(u, M0 + M_loc + 1, N0 + 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm, &request[nReq++]);
      MPI_Waitall(nReq, request, MPI_STATUSES_IGNORE);
    }

  }

  // left and right sides of halo
  if (Q == 1) { 
    for (i = 0; i < M_loc+2; i++) {
      V(u, i, 0) = V(u, i, N_loc);
      V(u, i, N_loc+1) = V(u, i, 1);
    }
  }
  else {
    for (i = M0; i < M0 + M_loc + 2; i++) {
      V(u, i, N0) = V(u, i, N0 + N_loc);
      V(u, i, N0 + N_loc + 1) = V(u, i, N0 + 1);
    }
  }
} //updateBoundary()


// evolve advection over r timesteps, with (u,ldu) containing the local field
void parAdvect(int reps, double *u, int ldu) {
  int r; 
  double *v; int ldv = N_loc+2;
  v = calloc(ldv*(M_loc+2), sizeof(double)); assert(v != NULL);
  assert(ldu == N_loc + 2);
  
  for (r = 0; r < reps; r++) {
    updateBoundary(u, ldu);
    updateAdvectField(M_loc, N_loc, &V(u,1,1), ldu, &V(v,1,1), ldv);
    copyField(M_loc, N_loc, &V(v,1,1), ldv, &V(u,1,1), ldu);

    if (verbosity > 2) {
      char s[64]; sprintf(s, "%d reps: u", r+1);
      printAdvectField(rank, s, M_loc+2, N_loc+2, u, ldu);
    }
  }
 
  free(v);
} //parAdvect()


// overlap communication variant
void parAdvectOverlap(int reps, double *u, int ldu) {

} //parAdvectOverlap()


// wide halo variant
void parAdvectWide(int reps, int w, double *u, int ldu) {

} //parAdvectWide()


// extra optimization variant
void parAdvectExtra(int r, double *u, int ldu) {

} //parAdvectExtra()
