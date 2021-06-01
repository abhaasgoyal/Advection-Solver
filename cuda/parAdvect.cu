// CUDA parallel 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2021
// v1.0 15 Apr

#include "serAdvect.h" // advection parameters
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static int M, N, Gx, Gy, Bx, By; // local store of problem parameters
static int verbosity;

struct block_ranges {
  int i_start, i_end, j_start, j_end;
};

// sets up parameters above
void initParParams(int M_, int N_, int Gx_, int Gy_, int Bx_, int By_,
                   int verb) {
  M = M_, N = N_;
  Gx = Gx_;
  Gy = Gy_;
  Bx = Bx_;
  By = By_;
  verbosity = verb;
} // initParParams()

__host__ __device__ static void N2Coeff(double v, double *cm1, double *c0,
                                        double *cp1) {
  double v2 = v / 2.0;
  *cm1 = v2 * (v + 1.0);
  *c0 = 1.0 - v * v;
  *cp1 = v2 * (v - 1.0);
}

__global__ void updateBoundaryNSP(int N, int M, double *u, int ldu) {

  // Parallelize in j-axis
  int j_td = (blockIdx.y * blockDim.y) + threadIdx.y;
  int tot_tdy = blockDim.y * gridDim.y;
  int n_tdy = N / tot_tdy;
  int j_start = j_td * n_tdy;
  int j_end = j_td < tot_tdy - 1 ? (j_td + 1) * n_tdy : N;

  for (int j = j_start + 1; j < j_end + 1; j++) { // top and bottom halo
    V(u, 0, j) = V(u, M, j);
    V(u, M + 1, j) = V(u, 1, j);
  }
}

__global__ void updateBoundaryEWP(int M, int N, double *u, int ldu) {
  // Parallelize only in i-axis
  int i_td = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tot_tdx = blockDim.x * gridDim.x;
  int n_tdx = M / tot_tdx;
  int i_start = i_td * n_tdx;
  int i_end = i_td < tot_tdx - 1 ? (i_td + 1) * n_tdx : M + 2;
  for (int i = i_start; i < i_end; i++) { // left and right sides of halo
    V(u, i, 0) = V(u, i, N);
    V(u, i, N + 1) = V(u, i, 1);
  }
}

__global__ void updateAdvectFieldKP(int M, int N, double *u, int ldu, double *v,
                                    int ldv, double Ux, double Uy) {
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  N2Coeff(Ux, &cim1, &ci0, &cip1);
  N2Coeff(Uy, &cjm1, &cj0, &cjp1);

  // Global value of thread
  int i_td = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j_td = (blockIdx.y * blockDim.y) + threadIdx.y;

  // Divide the blocks in (n_tdx * n_tdy) parts
  int tot_tdx = blockDim.x * gridDim.x;
  int tot_tdy = blockDim.y * gridDim.y;
  int n_tdx = M / tot_tdx;
  int n_tdy = N / tot_tdy;

  // Setting up start and end i and j index for each region
  // assigned to a thread
  int i_start = i_td * n_tdx;
  int i_end = i_td < tot_tdx - 1 ? (i_td + 1) * n_tdx : M;
  int j_start = j_td * n_tdy;
  int j_end = j_td < tot_tdy - 1 ? (j_td + 1) * n_tdy : N;

  // The for-loop runs in parallel among the 2-D threads
  for (int i = i_start; i < i_end; i++)
    for (int j = j_start; j < j_end; j++)
      V(v, i, j) = cim1 * (cjm1 * V(u, i - 1, j - 1) + cj0 * V(u, i - 1, j) +
                           cjp1 * V(u, i - 1, j + 1)) +
                   ci0 * (cjm1 * V(u, i, j - 1) + cj0 * V(u, i, j) +
                          cjp1 * V(u, i, j + 1)) +
                   cip1 * (cjm1 * V(u, i + 1, j - 1) + cj0 * V(u, i + 1, j) +
                           cjp1 * V(u, i + 1, j + 1));
}

__global__ void updateAdvectFieldOPN(int M, int N, double *u, int ldu,
                                     double *v, int ldv, double Ux, double Uy) {
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  N2Coeff(Ux, &cim1, &ci0, &cip1);
  N2Coeff(Uy, &cjm1, &cj0, &cjp1);

  // Initializing parameters
  int Bx = blockDim.x, By = blockDim.y, tdx = threadIdx.x, tdy = threadIdx.y, Gx = gridDim.x, Gy = gridDim.y;

  // Setting up aData and bData in shared memory
  // (Bx + 2) since we load 2 extra rows initially
  // in shared memory for updating through top and bottom
  extern __shared__ double s[];
  double *aData = s;
  double *bData = (double *)&aData[(Bx + 2) * By];

  // Allow padding for uneven distribution of tiles at the right
  // and bottom end
  int tot_tdx = Bx * Gx;
  int tot_tdy = By * Gy;
  int n_tdx = (M + tot_tdx - 1) / tot_tdx;
  int n_tdy = (N + tot_tdy - 1) / tot_tdy;

  // i and j tiled-block steps
  for (int i = 0; i < n_tdx; i++) {
    for (int j = 0; j < n_tdy; j++) {
      // Determining the position to load for the current tile
      int tp_i = i * Bx * Gx + blockIdx.x * Bx + tdx;
      int tp_j = j * By * Gy + blockIdx.y * By + tdy;

      // If the thread overbounds ignore that thread and wait for rest
      // of the operation
      if (tp_i < M && tp_j < N) {
        // Load 3 contiguous memory elements in shared memory
        V_(aData, By, 1 + tdx, tdy) = cjm1 * V(u, tp_i, tp_j - 1) +
                                      cj0 * V(u, tp_i, tp_j) +
                                      cjp1 * V(u, tp_i, tp_j + 1);

        // Also Load extra topmost and bottomest rows for the next step
        if (tdx == 0) {
          V_(aData, By, 0, tdy) = cjm1 * V(u, tp_i - 1, tp_j - 1) +
                                  cj0 * V(u, tp_i - 1, tp_j) +
                                  cjp1 * V(u, tp_i - 1, tp_j + 1);
        }


        if (tdx == Bx - 1 || tp_i == M - 1) {
          V_(aData, By, tdx + 2, tdy) = cjm1 * V(u, tp_i + 1, tp_j - 1) +
                                        cj0 * V(u, tp_i + 1, tp_j) +
                                        cjp1 * V(u, tp_i + 1, tp_j + 1);
        }

        __syncthreads();

        // Doing remaining strided operation in shared (scratch memory)
        // itself hence no access from memory needed
        V_(bData, By, tdx, tdy) = cim1 * V_(aData, By, tdx, tdy) +
                                  ci0 * V_(aData, By, tdx + 1, tdy) +
                                  cip1 * V_(aData, By, tdx + 2, tdy);

        // Putting back the final result in a new global memory
        // array but in the same location that it was retrieved
        // from in the first place
        V(v, tp_i, tp_j) = V_(bData, By, tdx, tdy);
      }
      __syncthreads();
    }
  }
}

__global__ void copyFieldKP(int M, int N, double *u, int ldu, double *v,
                            int ldv) {
  // No need for shared memory since no data reuse :/
  int i_td = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j_td = (blockIdx.y * blockDim.y) + threadIdx.y;
  int tot_tdx = blockDim.x * gridDim.x;
  int tot_tdy = blockDim.y * gridDim.y;
  int n_tdx = M / tot_tdx;
  int n_tdy = N / tot_tdy;
  int i_start = i_td * n_tdx;
  int i_end = i_td < tot_tdx - 1 ? (i_td + 1) * n_tdx : M;
  int j_start = j_td * n_tdy;
  int j_end = j_td < tot_tdy - 1 ? (j_td + 1) * n_tdy : N;

  for (int i = i_start; i < i_end; i++)
    for (int j = j_start; j < j_end; j++)
      V(v, i, j) = V(u, i, j);
}

// evolve advection over reps timesteps, with (u,ldu) containing the field
// parallel (2D decomposition) variant
void cuda2DAdvect(int reps, double *u, int ldu) {
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  int ldv = N + 2;
  double *v, *temp_u;
  HANDLE_ERROR(cudaMalloc(&v, ldv * (M + 2) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&temp_u, ldv * (M + 2) * sizeof(double)));
  dim3 dimG(Gx, Gy);
  dim3 dimB(Bx, By);
  HANDLE_ERROR(cudaMemcpy(temp_u, u, ldv * (M + 2) * sizeof(double),
                          cudaMemcpyHostToDevice));
  for (int r = 0; r < reps; r++) {
    updateBoundaryNSP<<<dimG, dimB>>>(N, M, temp_u, ldu);
    updateBoundaryEWP<<<dimG, dimB>>>(
        M, N, temp_u, ldu); // <<<1,1>>> is also cool cuz stridestuff :/?
    updateAdvectFieldKP<<<dimG, dimB>>>(M, N, &V_(temp_u, ldu, 1, 1), ldu,
                                        &V(v, 1, 1), ldv, Ux, Uy);
    copyFieldKP<<<dimG, dimB>>>(M, N, &V(v, 1, 1), ldv, &V_(temp_u, ldu, 1, 1),
                                ldu);
  } // for(r...)
  HANDLE_ERROR(cudaMemcpy(u, temp_u, ldv * (M + 2) * sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(v));
} // cuda2DAdvect()

// ... optimized parallel variant
void cudaOptAdvect(int reps, double *u, int ldu, int w) {
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  int ldv = N + 2;
  double *v, *temp_u;
  HANDLE_ERROR(cudaMalloc(&v, ldv * (M + 2) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&temp_u, ldv * (M + 2) * sizeof(double)));
  dim3 dimG(Gx, Gy);
  dim3 dimB(Bx, By);
  HANDLE_ERROR(cudaMemcpy(temp_u, u, ldv * (M + 2) * sizeof(double),
                          cudaMemcpyHostToDevice));
  for (int r = 0; r < reps; r++) {
    if (r % 2 == 0) {
    updateBoundaryNSP<<<dimG, dimB>>>(N, M, temp_u, ldu);
    updateBoundaryEWP<<<dimG, dimB>>>(M, N, temp_u, ldu);
    updateAdvectFieldOPN<<<dimG, dimB, (2 * Bx + 2) * By * sizeof(double)>>>(
        M, N, &V_(temp_u, ldu, 1, 1), ldu, &V(v, 1, 1), ldv, Ux, Uy);
    } else {
    updateBoundaryNSP<<<dimG, dimB>>>(N, M, v, ldu);
    updateBoundaryEWP<<<dimG, dimB>>>(M, N, v, ldu);
    updateAdvectFieldOPN<<<dimG, dimB, (2 * Bx + 2) * By * sizeof(double)>>>(
        M, N, &V(v, 1, 1), ldu, &V_(temp_u, ldu, 1, 1), ldv, Ux, Uy);
    }
  } // for(r...)
  if (reps % 2 == 1) {
    copyFieldKP<<<dimG, dimB>>>(M, N, &V(v, 1, 1), ldv, &V_(temp_u, ldu, 1, 1), ldu);

  }
  HANDLE_ERROR(cudaMemcpy(u, temp_u, ldv * (M + 2) * sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(v));

  // Added to remove warning on compilation
  if (verbosity > 1)
    ;
} // cudaOptAdvect()
