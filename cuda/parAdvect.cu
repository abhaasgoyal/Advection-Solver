// CUDA parallel 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2021
// v1.0 15 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "serAdvect.h" // advection parameters

static int M, N, Gx, Gy, Bx, By; // local store of problem parameters
static int verbosity;

struct block_ranges {
    int i_start, i_end, j_start, j_end;
};

//sets up parameters above
void initParParams(int M_, int N_, int Gx_, int Gy_, int Bx_, int By_, 
		   int verb) {
  M = M_, N = N_; Gx = Gx_; Gy = Gy_;  Bx = Bx_; By = By_; 
  verbosity = verb;
} //initParParams()


__host__ __device__
static void N2Coeff(double v, double *cm1, double *c0, double *cp1) {
  double v2 = v/2.0;
  *cm1 = v2*(v+1.0);
  *c0  = 1.0 - v*v;
  *cp1 = v2*(v-1.0);
}

__global__
void updateBoundaryNSP(int N, int M, double *u, int ldu) {

  int j_td = (blockIdx.y * blockDim.y) + threadIdx.y;
  int tot_tdy = blockDim.y * gridDim.y;
  int n_tdy = N / tot_tdy;
  int j_start = j_td * n_tdy;
  int j_end = j_td < tot_tdy - 1 ? (j_td + 1) * n_tdy : N;

  for (int j=j_start + 1; j < j_end + 1; j++) { //top and bottom halo
    V(u, 0, j)   = V(u, M, j);
    V(u, M+1, j) = V(u, 1, j);
  }
}

__global__
void updateBoundaryEWP(int M, int N, double *u, int ldu) {
  int i_td = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tot_tdx = blockDim.x * gridDim.x;
  int n_tdx = M / tot_tdx;
  int i_start = i_td * n_tdx;
  int i_end = i_td < tot_tdx - 1 ? (i_td + 1) * n_tdx : M + 2;
  for (int i=i_start; i < i_end; i++) { //left and right sides of halo
    V(u, i, 0) = V(u, i, N);
    V(u, i, N+1) = V(u, i, 1);
  }
}

__global__
void updateAdvectFieldKP(int M, int N, double *u, int ldu, double *v, int ldv,
			double Ux, double Uy) {
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

  for (int i=i_start; i < i_end; i++)
    for (int j=j_start; j < j_end; j++)
      V(v,i,j) =
        cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
        ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
        cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));
}


__global__
void copyFieldKP(int M, int N, double *u, int ldu, double *v, int ldv) {
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

  for (int i=i_start; i < i_end; i++)
    for (int j=j_start; j < j_end; j++)
      V(v,i,j) = V(u,i,j);
}


// evolve advection over reps timesteps, with (u,ldu) containing the field
// parallel (2D decomposition) variant
void cuda2DAdvect(int reps, double *u, int ldu) {
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  int ldv = N+2; double *v;
  HANDLE_ERROR( cudaMalloc(&v, ldv*(M+2)*sizeof(double)) );
  dim3 dimG(Gx, Gy);
  dim3 dimB(Bx, By);
  for (int r = 0; r < reps; r++) {
    updateBoundaryNSP <<<dimG,dimB>>> (N, M, u, ldu);
    updateBoundaryEWP <<<dimG,dimB>>> (M, N, u, ldu); // <<<1,1>>> is also cool cuz stridestuff :/?
    updateAdvectFieldKP <<<dimG,dimB>>> (M, N, &V(u,1,1), ldu, &V(v,1,1), ldv,
				  Ux, Uy);
    copyFieldKP <<<dimG,dimB>>> (M, N, &V(v,1,1), ldv, &V(u,1,1), ldu);
  } //for(r...)
  HANDLE_ERROR( cudaFree(v) );
} //cuda2DAdvect()



// ... optimized parallel variant
void cudaOptAdvect(int reps, double *u, int ldu, int w) {

} //cudaOptAdvect()
