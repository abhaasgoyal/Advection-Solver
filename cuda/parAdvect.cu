// CUDA parallel 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2021
// v1.0 15 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "serAdvect.h" // advection parameters

static int M, N, Gx, Gy, Bx, By; // local store of problem parameters
static int verbosity;

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
  for (int j=1; j < N+1; j++) { //top and bottom halo
    V(u, 0, j)   = V(u, M, j);
    V(u, M+1, j) = V(u, 1, j);
  }
}

__global__
void updateBoundaryEWP(int M, int N, double *u, int ldu) {
  for (int i=0; i < M+2; i++) { //left and right sides of halo
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

  for (int i=0; i < M; i++)
    for (int j=0; j < N; j++)
      V(v,i,j) =
        cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
        ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
        cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));
}


__global__
void copyFieldKP(int M, int N, double *u, int ldu, double *v, int ldv) {
  for (int i=0; i < M; i++)
    for (int j=0; j < N; j++)
      V(v,i,j) = V(u,i,j);
}


// evolve advection over reps timesteps, with (u,ldu) containing the field
// parallel (2D decomposition) variant
void cuda2DAdvect(int reps, double *u, int ldu) {
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  int ldv = N+2; double *v;
  HANDLE_ERROR( cudaMalloc(&v, ldv*(M+2)*sizeof(double)) );
  for (int r = 0; r < reps; r++) {
    updateBoundaryNSP <<<1,1>>> (N, M, u, ldu);
    updateBoundaryEWP <<<1,1>>> (M, N, u, ldu);
    updateAdvectFieldKP <<<1,1>>> (M, N, &V(u,1,1), ldu, &V(v,1,1), ldv,
				  Ux, Uy);
    copyFieldKP <<<1,1>>> (M, N, &V(v,1,1), ldv, &V(u,1,1), ldu);
  } //for(r...)
  HANDLE_ERROR( cudaFree(v) );
} //cuda2DAdvect()



// ... optimized parallel variant
void cudaOptAdvect(int reps, double *u, int ldu, int w) {

} //cudaOptAdvect()
