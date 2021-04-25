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
static int halo_error = 0;
static int o = 0;

//sets up parallel parameters above
void initParParams(int M_, int N_, int P_, int Q_, int verb, int comm_mode_, int o_) {
  M = M_, N = N_; P = P_, Q = Q_;
  comm_mode = comm_mode_;
  o  = o_;
  verbosity = verb;
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  P0 = rank / Q;
  M0 = (M / P) * P0;
  M_loc = (P0 < P-1)? (M / P): (M - M0);

  Q0 = rank % Q;
  N0 = (N / Q) * Q0;
  N_loc = (Q0 < Q-1)? (N / Q): (N - N0);
} //initParParams()


int checkHaloSize(int w) {
  if (rank == 0) {
    if (w > M_loc || w > N_loc) {
      halo_error = 1;
    }
  }
  MPI_Bcast(&halo_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (halo_error == 1) {
    if (rank == 0) {
      printf("%d: w=%d too large for %dx%d local field! Exiting...\n",
             rank, w, M_loc, N_loc);
    }
    return -1; // Could have used exit(0); here but didn't to gracefully exit from main
  }
  return 0;
}


static void updateBoundary(double *u, int ldu, int w) {
  if (verbosity > 1) {
  printf("%d:omoim %d\n", rank, w);
  }
  int i, j, w_i;
  //top and bottom halo
  //note: we get the left/right neighbour's corner elements from each end
  if (P == 1) {
    if (verbosity > 1) {
      printAdvectField(rank, "Before UB transfer", M_loc+2*w, N_loc+2*w, u, ldu);
    }

    for (w_i = 0; w_i < w; w_i++) {
      for (j = 1; j < N_loc+1; j++) {
        V(u, w_i, j) = V(u, M_loc + w_i, j);
        V(u, w + M_loc + w_i, j) = V(u, w + w_i, j);
      }
    }
    if (verbosity > 1) {
      printAdvectField(rank, "Before UB transfer", M_loc+2*w, N_loc+2*w, u, ldu);
    }
  }
  else {
    int botProc = (rank + Q) % nprocs, topProc = (rank - Q + nprocs) % nprocs; // Think of think as clockwise and anticlockwise
    /* Send from all odd nodes first then send from all even nodes */
    /* Only works till Q3 */
    // printf("%d %d %d\n", rank, topProc, botProc);
    // printf("Rank: %d - P0=%d Q0=%d M0=%d N0=%d M_loc=%d N_loc = %d\n ", rank, P0, Q0, M0, N0, M_loc, N_loc);
    /*
    if (comm_mode == 0) {
        if (rank % 2 == 0) {
          MPI_Send(&V(u, M_loc, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm);
          MPI_Recv(&V(u, 0, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm,
                   MPI_STATUS_IGNORE);
          MPI_Send(&V(u, 1, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm);
          MPI_Recv(&V(u, M_loc+1, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG,
                   comm, MPI_STATUS_IGNORE);
        } else {
          MPI_Recv(&V(u, 0, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm,
                   MPI_STATUS_IGNORE);
          MPI_Send(&V(u, M_loc, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm);
          MPI_Recv(&V(u, M_loc+1, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG,
                   comm, MPI_STATUS_IGNORE);
          MPI_Send(&V(u, 1, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm);
        }
    }
    else {
    */
    if (verbosity > 1) {
      printAdvectField(rank, "Before UB transfer", M_loc+2*w, N_loc+2*w, u, ldu);
    }

    MPI_Datatype s_rowtype; int nReq = 0;
    MPI_Request request[4];
    MPI_Type_vector(w, N_loc, N_loc + 2*w, MPI_DOUBLE, &s_rowtype); // Number of rows being transmitted are w with N_loc elements
    MPI_Type_commit(&s_rowtype);
    MPI_Isend(&V(u, M_loc    , w), 1, s_rowtype, botProc, HALO_TAG, comm, &request[nReq++]);
    MPI_Irecv(&V(u, 0        , w), 1, s_rowtype, topProc, HALO_TAG, comm, &request[nReq++]);
    MPI_Isend(&V(u, w        , w), 1, s_rowtype, topProc, HALO_TAG, comm, &request[nReq++]);
    MPI_Irecv(&V(u, w + M_loc, w), 1, s_rowtype, botProc, HALO_TAG, comm, &request[nReq++]);
    MPI_Waitall(nReq, request, MPI_STATUSES_IGNORE);
    // }
    if (verbosity > 1) {
      printAdvectField(rank, "After UB transfer", M_loc+2*w, N_loc+2*w, u, ldu);
    }

  }
  // left and right sides of halo
    // char s[64];
  if (Q == 1) {

    for (w_i = 0; w_i < w; w_i++) {
      for (i = 0; i < M_loc+2; i++) {
        V(u, i, w_i) = V(u, i, N_loc + w_i);
        V(u, i, w + N_loc + w_i) = V(u, i, w + w_i);
      }
    }
  }
  else {
    int rightProc = (Q0 < Q - 1) ? (rank + 1) % nprocs : (rank - Q + 1 + nprocs) % nprocs;
    int leftProc =  (Q0 > 0) ? (rank - 1 + nprocs) % nprocs : (rank + Q - 1);
    // printf("LR %d %d %d\n", rank, leftProc, rightProc);
    MPI_Datatype s_coltype; int nReq = 0;
    MPI_Request request[4];
    MPI_Type_vector(M_loc + 2*w, w, N_loc + 2*w, MPI_DOUBLE, &s_coltype); // Count of each element i
    MPI_Type_commit(&s_coltype);
    if (verbosity > 1) {
      printAdvectField(rank, "Before LR transfer", M_loc+2*w, N_loc+2*w, u, ldu);
    }

    MPI_Isend(&V(u, 0, N_loc    ), 1, s_coltype, rightProc, HALO_TAG, comm, &request[nReq++]); // w + N_loc - w
    MPI_Irecv(&V(u, 0, 0        ), 1, s_coltype, leftProc, HALO_TAG, comm, &request[nReq++]);
    MPI_Isend(&V(u, 0, w        ), 1, s_coltype, leftProc, HALO_TAG, comm, &request[nReq++]);
    MPI_Irecv(&V(u, 0, w + N_loc), 1, s_coltype, rightProc, HALO_TAG, comm, &request[nReq++]);
    MPI_Waitall(nReq, request, MPI_STATUSES_IGNORE);
    if (verbosity > 1) {
      printAdvectField(rank, "After LR transfer", M_loc+2*w, N_loc+2*w, u, ldu);
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
    // static void updateBoundary(double *u, int ldu, int w, int o, MPI_Request* request) {
    updateBoundary(u, ldu, 1);
    updateAdvectField(M_loc, N_loc, &V(u,1,1), ldu, &V(v,1,1), ldv);
    copyField(M_loc, N_loc, &V(v,1,1), ldv, &V(u,1,1), ldu);

    if (verbosity > 2) {
      char s[64]; sprintf(s, "%d reps: u", r+1);
      printAdvectField(rank, s, M_loc+2, N_loc+2, u, ldu);
    }
  }
 
  free(v);
} //parAdvect()


static void updateBoundaryO(double *u, int ldu, MPI_Request* request) {
  int i, j, nReq = 0;
  //top and bottom halo
  //note: we get the left/right neighbour's corner elements from each end
  // printf("%d: %d %d\n", rank, P, Q);
  if (P == 1) {
    for (j = 1; j < N_loc+1; j++) {
      V(u, 0, j) = V(u, M_loc, j);
      V(u, M_loc+1, j) = V(u, 1, j);
    }
  }
  else {
    int botProc = (rank + Q) % nprocs, topProc = (rank - Q + nprocs) % nprocs; // Think of think as clockwise and anticlockwise

    MPI_Isend(&V(u, M_loc    , 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm, &request[nReq++]);
    MPI_Irecv(&V(u, 0        , 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm, &request[nReq++]);
    MPI_Isend(&V(u, 1        , 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm, &request[nReq++]);
    MPI_Irecv(&V(u, M_loc + 1, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm, &request[nReq++]);

    MPI_Datatype s_corner_type;
    MPI_Type_vector(2, 1, N_loc + 1, MPI_DOUBLE, &s_corner_type);
    MPI_Type_commit(&s_corner_type);
    MPI_Request corner_elem[4]; int n_cor = 0; // Wait flag only for the corner elements
    MPI_Isend(&V(u, M_loc    , 1), 1, s_corner_type, botProc, HALO_TAG, comm, &corner_elem[n_cor++]);
    MPI_Irecv(&V(u, 0        , 1), 1, s_corner_type, topProc, HALO_TAG, comm, &corner_elem[n_cor++]);
    MPI_Isend(&V(u, 1        , 1), 1, s_corner_type, topProc, HALO_TAG, comm, &corner_elem[n_cor++]);
    MPI_Irecv(&V(u, M_loc + 1, 1), 1, s_corner_type, botProc, HALO_TAG, comm, &corner_elem[n_cor++]);
    MPI_Waitall(n_cor, corner_elem, MPI_STATUS_IGNORE);
  }
  // left and right sides of halo
  if (Q == 1) {
    for (i = 0; i < M_loc+2; i++) {
      V(u, i, 0) = V(u, i, N_loc);
      V(u, i, N_loc+1) = V(u, i, 1);
    }
  }
  else {
    int rightProc = (Q0 < Q - 1) ? (rank + 1) % nprocs : (rank - Q + 1 + nprocs) % nprocs;
    int leftProc =  (Q0 > 0) ? (rank - 1 + nprocs) % nprocs : (rank + Q - 1);
    // printf("LR %d %d %d\n", rank, leftProc, rightProc);

    MPI_Datatype s_coltype;
    MPI_Type_vector(M_loc + 2, 1, N_loc + 2, MPI_DOUBLE, &s_coltype); // Count of each element i
    MPI_Type_commit(&s_coltype);

    /* if (verbosity > 2) { */
    /*   char s[64]; printf("%d: Before transfer", rank); */
    /*   printAdvectField(rank, s, M_loc+2, N_loc+2, u, ldu); */
    /* } */

    MPI_Isend(&V(u, 0, N_loc        ), 1, s_coltype, rightProc, HALO_TAG, comm, &request[nReq++]); // w + N_loc - w
    MPI_Irecv(&V(u, 0, 0            ), 1, s_coltype, leftProc, HALO_TAG, comm, &request[nReq++]);
    MPI_Isend(&V(u, 0, 1            ), 1, s_coltype, leftProc, HALO_TAG, comm, &request[nReq++]);
    MPI_Irecv(&V(u, 0, N_loc + 1    ), 1, s_coltype, rightProc, HALO_TAG, comm, &request[nReq++]);

    /* if (verbosity > 2) { */
    /*   char s[64]; printf("%d: After transfer", rank); */
    /*   printAdvectField(rank, s, M_loc+2, N_loc+2, u, ldu); */
    /* } */
  }

} //updateBoundaryO()


// overlap communication variant
void parAdvectOverlap(int reps, double *u, int ldu) {
  int r;
  int n_requests = ((P == 1) ? 0 : 4) + ((Q == 1) ? 0 : 4);
  double *v; int ldv = N_loc+2;
  v = calloc(ldv*(M_loc+2), sizeof(double)); assert(v != NULL);
  assert(ldu == N_loc + 2);
  MPI_Request request[n_requests];
  for (r = 0; r < reps; r++) {
    // static void updateBoundary(double *u, int ldu, int w, int o, MPI_Request* request) {
    updateBoundaryO(u, ldu, request);

    // updateAdvectField(M_loc, N_loc, &V(u,1,1), ldu, &V(v,1,1), ldv);
    // Send inner halo
    updateAdvectField(M_loc - 1, N_loc - 1, &V(u,2,2), ldu, &V(v,2,2), ldv); // Start from (2,2) and end earlier in (M_loc - 1, N_loc -1)
    if (verbosity > 1) {
      char s[64]; sprintf(s, "%d reps: u", r+1);
      printAdvectField(rank, s, M_loc+2, N_loc+2, u, ldu);
    }    MPI_Waitall(n_requests, request, MPI_STATUSES_IGNORE);
    // Update edges manually
    updateAdvectField(1    , N_loc, &V(u,1,1)        , ldu, &V(v,1,1)          , ldv);
    updateAdvectField(1    , N_loc, &V(u,M_loc,1)    , ldu, &V(v,M_loc,1)      , ldv);
    updateAdvectField(M_loc, 1    , &V(u,1,1)        , ldu, &V(v,1,1)          , ldv);
    updateAdvectField(M_loc, 1    , &V(u,1,N_loc)    , ldu, &V(v,1,N_loc)      , ldv);
    copyField(M_loc, N_loc, &V(v,1,1), ldv, &V(u,1,1), ldu);
    if (verbosity > 2) {
      char s[64]; sprintf(s, "%d reps: u", r+1);
      printAdvectField(rank, s, M_loc+2, N_loc+2, u, ldu);
    }
  }

  free(v);
} //parAdvectOverlap()


// wide halo variant
void parAdvectWide(int reps, int w, double *u, int ldu) {
  int r, w_i;
  double *v; int ldv = N_loc+2*w;
  if (verbosity > 1) {
  printf("%d:omoi %d\n", rank, w);
  }
  v = calloc(ldv*(M_loc+2*w), sizeof(double)); assert(v != NULL);
  assert(ldu == N_loc + 2*w);
   // reps = r * w
  for (r = 0; r < reps / w; r++) {
    updateBoundary(u, ldu, w);
    // The sequential part :/
    for (w_i = 1; w_i <= w; w_i++) {
      // perform w updates to the local field of sizes (m+2*w-2)x(n+2*w-2),(m+2*w-4)x(n+2*w-4), ...,(m)x(n)
      // May not be correct as of nwo
      int UR_size = M_loc + (2 * w) - (2 * w_i);
      int UC_size = N_loc + (2 * w) - (2 * w_i);
      updateAdvectField(UR_size, UC_size, &V(u,w_i,w_i), ldu, &V(v,w_i,w_i), ldv);
      copyField(UR_size, UC_size, &V(v,w_i,w_i), ldv, &V(u,w_i,w_i), ldu); // TODO correct copying
    }

  }
  int reps_left = reps % w;
  if (reps_left > 0) {
    updateBoundary(u, ldu, w);
    for (w_i = 1; w_i <= reps_left; w_i++) {
      int UR_size = M_loc + (2 * w) - (2 * w_i);
      int UC_size = N_loc + (2 * w) - (2 * w_i);
      updateAdvectField(UR_size, UC_size, &V(u,w_i,w_i), ldu, &V(v,w_i,w_i), ldv);
      copyField(UR_size, UC_size, &V(v,w_i,w_i), ldv, &V(u,w_i,w_i), ldu);
    }
  }


  if (verbosity > 2) {
    char s[64]; sprintf(s, "%d reps: u", r+1);
    printAdvectField(rank, s, M_loc+2*w, N_loc+2*w, u, ldu);
  }

  free(v);
} //parAdvectWide()


// extra optimization variant (The same as parAdvectOverlap and working with Q > 1)
void parAdvectExtra(int reps, double *u, int ldu) {
  int r;
  int n_requests = ((P == 1) ? 0 : 4) + ((Q == 1) ? 0 : 4);
  double *v; int ldv = N_loc+2;
  v = calloc(ldv*(M_loc+2), sizeof(double)); assert(v != NULL);
  assert(ldu == N_loc + 2);
  MPI_Request request[n_requests];
  for (r = 0; r < reps; r++) {
    // static void updateBoundary(double *u, int ldu, int w, int o, MPI_Request* request) {
    updateBoundaryO(u, ldu, request);

    // updateAdvectField(M_loc, N_loc, &V(u,1,1), ldu, &V(v,1,1), ldv);
    // Send inner halo
    updateAdvectField(M_loc - 1, N_loc - 1, &V(u,2,2), ldu, &V(v,2,2), ldv); // Start from (2,2) and end earlier in (M_loc - 1, N_loc -1)
    if (verbosity > 1) {
      char s[64]; sprintf(s, "%d reps: u", r+1);
      printAdvectField(rank, s, M_loc+2, N_loc+2, u, ldu);
    }    MPI_Waitall(n_requests, request, MPI_STATUSES_IGNORE);
    // Update edges manually
    updateAdvectField(1    , N_loc, &V(u,1,1)        , ldu, &V(v,1,1)          , ldv);
    updateAdvectField(1    , N_loc, &V(u,M_loc,1)    , ldu, &V(v,M_loc,1)      , ldv);
    updateAdvectField(M_loc, 1    , &V(u,1,1)        , ldu, &V(v,1,1)          , ldv);
    updateAdvectField(M_loc, 1    , &V(u,1,N_loc)    , ldu, &V(v,1,N_loc)      , ldv);
    copyField(M_loc, N_loc, &V(v,1,1), ldv, &V(u,1,1), ldu);
    if (verbosity > 2) {
      char s[64]; sprintf(s, "%d reps: u", r+1);
      printAdvectField(rank, s, M_loc+2, N_loc+2, u, ldu);
    }
  }

  free(v);
} //parAdvectExtra()
