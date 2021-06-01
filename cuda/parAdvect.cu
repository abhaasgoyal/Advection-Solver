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
  /*
  printf(":%d :%d i_start=%d i_end=%d j_start=%d j_end=%d\n",
         i_td, j_td, i_start, i_end, j_start, j_end);
         */

  for (int i = i_start; i < i_end; i++)
    for (int j = j_start; j < j_end; j++)
      V(v, i, j) = cim1 * (cjm1 * V(u, i - 1, j - 1) + cj0 * V(u, i - 1, j) +
                           cjp1 * V(u, i - 1, j + 1)) +
                   ci0 * (cjm1 * V(u, i, j - 1) + cj0 * V(u, i, j) +
                          cjp1 * V(u, i, j + 1)) +
                   cip1 * (cjm1 * V(u, i + 1, j - 1) + cj0 * V(u, i + 1, j) +
                           cjp1 * V(u, i + 1, j + 1));
}

__global__ void updateAdvectFieldOP(int M, int N, double *u, int ldu, double *v,
                                    int ldv, double Ux, double Uy) {
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  N2Coeff(Ux, &cim1, &ci0, &cip1);
  N2Coeff(Uy, &cjm1, &cj0, &cjp1);

  /* Define a __shared__ variable for the whole block (temp_u)
   * 1. Copy from global to shared
   * 2. Store (cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) in
   * V(temp_u,i,j)
   * 3. Do cim1 * V(temp_u, i-1,j)  + ci0 * V(temp_u, i, j) + cip1 * V(temp_u,
   * i+1, j) and store in either temp_u (sync nope) or another shared variable?
   * 4. Copy temp_u from Shared to global
   *
   * Self Imposed Rules :)
   * ---------
   * Shared Block size should be <= 32 x 32
   * i.e -> M / Gx   and N / Gy both should be <=32
   *
   * Also Bx and By should fit in the whole block (Don't put any of them on hold)
   *
   * TODO
   * Load into local memory, then transfer to shared memory, do operations with shared
   * transfer back to local
   */

  int i_td = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j_td = (blockIdx.y * blockDim.y) + threadIdx.y;
  int tot_tdx = blockDim.x * gridDim.x;
  int tot_tdy = blockDim.y * gridDim.y;
  int n_tdx = M / tot_tdx;
  int n_tdy = N / tot_tdy;
  int i_start = i_td * n_tdx;
  int i_end = i_td < tot_tdx - 1 ? i_start + n_tdx : M;
  int j_start = j_td * n_tdy;
  int j_end = j_td < tot_tdy - 1 ? j_start + n_tdy : N;
  int it_start = threadIdx.x * n_tdx;
  int jt_start = threadIdx.y * n_tdy;
  
  const int std_i_range = (i_end - i_start) ;
  const int std_j_range = (j_end - j_start) ;
  // printf(":%d :%d i_start=%d i_end=%d j_start=%d j_end=%d it_start=%d
  // jt_start=%d\n",
  //        i_td, j_td, i_start, i_end, j_start, j_end, it_start, jt_start);

  __shared__ double aData[34][32], bData[32][32];

  for (int i = 0; i < std_i_range + 2; i++) {
    for (int j = 0; j < std_j_range; j++) {
      aData[i + it_start][j + jt_start] =
          cjm1 * V(u, i + i_start - 1, j + j_start - 1) +
          cj0 * V(u, i + i_start - 1, j + j_start) +
          cjp1 * V(u, i + i_start - 1, j + j_start + 1);
    }
  }
  __syncthreads();

  for (int i = 1; i <= std_i_range; i++) {
    for (int j = 0; j < std_j_range; j++) {
      bData[i + it_start - 1][j + jt_start] =
          cim1 * aData[i + it_start - 1][j + jt_start] +
          ci0 * aData[i + it_start][j + jt_start] +
          cip1 * aData[i + it_start + 1][j + jt_start];
    }
  }
  __syncthreads();

  for (int i = 0; i < std_i_range; i++) {
    for (int j = 0; j < std_j_range; j++) {
      V(v, i_start + i, j_start + j) = bData[i + it_start][j + jt_start];
    }
  }
  __syncthreads();

  /*
  for (int i=i_start; i < i_end; i++)
    for (int j=j_start; j < j_end; j++)
      V(v,i,j) =
        cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
        ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
        cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));
  */
}


__global__ void updateAdvectFieldOPN(int M, int N, double *u, int ldu, double *v,
                                    int ldv, double Ux, double Uy) {
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  N2Coeff(Ux, &cim1, &ci0, &cip1);
  N2Coeff(Uy, &cjm1, &cj0, &cjp1);

  /* Best tiling
   * Self Imposed Rules :)
   * ---------
   * Divisible
   */

  __shared__ double aData[34][32], bData[32][32];
  int bdx = blockDim.x, bdy = blockDim.y, tdx = threadIdx.x, tdy = threadIdx.y;
  int tot_tdx = blockDim.x * gridDim.x;
  int tot_tdy = blockDim.y * gridDim.y;
  int n_tdx = M / tot_tdx;
  int n_tdy = N / tot_tdy;
  int nx_bsize = M / gridDim.x;
  int ny_bsize = N / gridDim.y;
  for (int i = 0; i < n_tdx; i++) {
      for (int j = 0; j < n_tdy; j++) {
          // printf("\n");
          int tp_i = i * bdx + blockIdx.x * nx_bsize + tdx;
          int tp_j = j * bdy + blockIdx.y * ny_bsize + tdy;
          // printf("%d %d :%d%d %d %d\n", i, j, tdx, tdy, tp_i, tp_j);

          aData[1 + tdx][tdy] =
              cjm1 * V(u, tp_i, tp_j - 1)
              + cj0 * V(u, tp_i , tp_j)
              + cjp1 * V(u, tp_i, tp_j + 1);

          if (tdx == 0) {
              aData[0][tdy]
                  = cjm1 * V(u, tp_i -1, tp_j - 1)
                  + cj0 * V(u, tp_i - 1 , tp_j)
                  + cjp1 * V(u, tp_i -1, tp_j + 1);
          }

          if (tdx == bdx - 1) {
              aData[bdx + 1][tdy]
                  = cjm1 * V(u, tp_i + 1, tp_j - 1)
                  + cj0 * V(u, tp_i + 1, tp_j)
                  + cjp1 * V(u, tp_i + 1, tp_j + 1);
          }

          __syncthreads();

          bData[tdx][tdy] =
              cim1 * aData[tdx][tdy] +
              ci0 * aData[tdx + 1][tdy] +
              cip1 * aData[tdx + 2][tdy];

          __syncthreads();

          V(v, tp_i,tp_j) = bData[tdx][tdy];

          __syncthreads();
      }
  }

  /*
  int i_td = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j_td = (blockIdx.y * blockDim.y) + threadIdx.y;
  int tot_tdx = blockDim.x * gridDim.x;
  int tot_tdy = blockDim.y * gridDim.y;
  int n_tdx = M / tot_tdx;
  int n_tdy = N / tot_tdy;
  int i_start = i_td * n_tdx;
  int i_end = i_td < tot_tdx - 1 ? i_start + n_tdx : M;
  int j_start = j_td * n_tdy;
  int j_end = j_td < tot_tdy - 1 ? j_start + n_tdy : N;

  for (int i=i_start; i < i_end; i++)
    for (int j=j_start; j < j_end; j++)
      V(v,i,j) =
        cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
        ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
        cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));
        */

  /*
  for (int i=i_start; i < i_end; i++)
    for (int j=j_start; j < j_end; j++)
      V(v,i,j) =
        cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
        ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
        cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));
  */
}

__global__ void copyFieldKP(int M, int N, double *u, int ldu, double *v,
                            int ldv) {
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
  bool isEvenlyDiv = (M % Bx == 0) && (M % Gx == 0) && (N % By ==0) && (N % Gy ==0);
  for (int r = 0; r < reps; r++) {
    updateBoundaryNSP<<<dimG, dimB>>>(N, M, temp_u, ldu);
    updateBoundaryEWP<<<dimG, dimB>>>(
        M, N, temp_u, ldu); // <<<1,1>>> is also cool cuz stridestuff :/?
    if (isEvenlyDiv) {
        updateAdvectFieldOPN<<<dimG, dimB>>>(M, N, &V_(temp_u, ldu, 1, 1), ldu,
                                        &V(v, 1, 1), ldv, Ux, Uy);
    }
    else {
        updateAdvectFieldKP<<<dimG, dimB>>>(M, N, &V_(temp_u, ldu, 1, 1), ldu,
                                        &V(v, 1, 1), ldv, Ux, Uy);

    }
    copyFieldKP<<<dimG, dimB>>>(M, N, &V(v, 1, 1), ldv, &V_(temp_u, ldu, 1, 1),
                                ldu);

  } // for(r...)
  HANDLE_ERROR(cudaMemcpy(u, temp_u, ldv * (M + 2) * sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(v));
} // cudaOptAdvect()
