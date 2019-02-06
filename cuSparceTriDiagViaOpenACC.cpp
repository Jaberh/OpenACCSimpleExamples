#include "cusparse.h"
#include "openacc.h"
#include "stdlib.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "math.h"
#include <iostream>
#define PRINT 0
/*
 *           CUISPARSE TRIDIAGONAL SOLVER WITHOUT PIVOTING
 * The routine does not perform any pivoting and uses a combination of the
   Cyclic Reduction (CR) and the Parallel Cyclic Reduction (PCR) algorithms
   to find the solution. It achieves better performance when m is a power of 2
 *
 * */

// This code solves the 1D Poisson's equation using cuSPARSE tridiagonal solver library
// The source is calculated by plugging in the sin(omega*pi*x) into the poisson's equation,
// The solution is presented on [0:1.0] domain
// the output is the error

#define OMEGA 1.0

using namespace std;

void performCyclicReduction(int size, double *dl, double *d, double *du,
                            double *rhs) {

#pragma acc host_data use_device(rhs, du, dl, d)
  {
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    cusparseDgtsv_nopivot(handle, size, 1, dl, d, du, rhs, size);
    cusparseDestroy(handle);
  }
}
int main(int arg, char *args[]) {
  int size;

  std::cout << " Enter the number of points " << endl;

  size = atoi(args[1]);

  double *r = NULL;
  r = new double[size];


  double pi = atan(1.0) * 4.0;
  double delx = 1.0 / (size - 1.0);

  for (int i = 0; i < size; i++) {
    r[i] = -OMEGA * OMEGA * pi * pi * sin(OMEGA * i * delx * pi) * delx * delx;
  }

#if (PRINT == 1)
  for (int i = 0; i < size; i++) {
    cout << "exact " << sin(OMEGA * i * delx * pi) << endl;
  }

  for (int i = 0; i < size; i++) {
    cout << "r =" << r[i] << endl;
  }
#endif

  double *d = new double[size];
  double *dl = new double[size];
  double *du = new double[size];

  for (int i = 0; i < size; i++) {
    d[i] = -2.0;
  }

  dl[size - 1] = 0.0;

  for (int i = 1; i < size; i++) {
    dl[i] = 1.0;
    du[i - 1] = 1.0;
  }

  // for Dirichlet BC
  du[0] = 0.0;
  dl[size - 1] = 0.0;

#if (PRINT == 1)
  for (int i = 0; i < size; i++) {
    cout << dl[i] << " " << d[i] << " " << du[i] << endl;
  }
#endif

#pragma acc data copy(dl[0 : size], d[0 : size], du[0 : size], r[0 : size])
  { performCyclicReduction(size, dl, d, du, r); }

  //#pragma acc data update self(r[0:size])

  double err0 = 0.0;
  double err1 = 0.0;
  for (int i = 0; i < size; i++) {
    // printf("r %lf %lf %lf %lf\n ",dl[i], d[i],du[i], r[i]);
    err1 = fabs(r[i] - sin(OMEGA * i * delx * pi));
    if (err0 < err1) {
      err0 = err1;
    }
  }

  cout << " l infinity of Error = " << err0 << endl;

  delete [] d;
  delete [] dl;
  delete [] du; 
  delete [] r; 
  
#if (PRINT == 1)
  for (int i = 0; i < size; i++) {
    // printf("r %lf %lf %lf %lf\n ",dl[i], d[i],du[i], r[i]);
    cout << r[i] << endl;
  }
#endif
}
