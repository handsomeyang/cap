// CUDA implementation of the Affinity Propagation clustering algorithm. See
// BJ Frey and D Dueck, Science 315, 972-976, Feb 16, 2007, for a
// description of the algorithm.
//
// Copyright (c) 2017. Jun Yang and Ge Li. All Rights Reserved. Permission to
// use, copy, modify, and distribute this software and its documentation for
// educational, research, and not-for-profit purposes, without fee and without
// a signed licensing agreement, is hereby granted, provided that the above
// copyright notice, this paragraph and the following two paragraphs appear in
// all copies, modifications, and distributions. Contact the authors at
// handsomeyang@gmail.com for commercial licensing opportunities.
//
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE TO ANY PARTY FOR DIRECT,
// INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST
// PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
// EVEN IF THE COPYRIGHT HOLDERS HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH
// DAMAGE.
//
// THE COPYRIGHT HOLDERS SPECIFICALLY DISCLAIM ANY WARRANTIES, INCLUDING, BUT
// NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY,
// PROVIDED HEREUNDER IS PROVIDED "AS IS". THE COPYRIGHT HOLDERS HAVE NO
// OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
// MODIFICATIONS.

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
/*#include <value.h>*/

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

#ifndef MINDOUBLE
#define MINDOUBLE 2.2250e-308
#endif
#ifndef MAXDOUBLE
#define MAXDOUBLE 1.7976e308
#endif

unsigned long *h_i, *h_k, *h_t;

int cmp(const void *a, const void *b)
{
    unsigned long ia = *(unsigned long *)a;
    unsigned long ib = *(unsigned long *)b;
    return ia - ib;
}

int cmp_i(const void *a, const void *b)
{
    unsigned long ia = *(unsigned long *)a;
    unsigned long ib = *(unsigned long *)b;
    return h_i[ia] - h_i[ib];
}

int cmp_k(const void *a, const void *b)
{
    unsigned long ia = *(unsigned long *)a;
    unsigned long ib = *(unsigned long *)b;
    return h_k[ia] - h_k[ib];
}

int cmp_t(const void *a, const void *b)
{
    unsigned long ia = *(unsigned long *)a;
    unsigned long ib = *(unsigned long *)b;
    return h_t[ia] - h_t[ib];
}

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ double atomicMax(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void calc_res_1(double *mx1, double *mx2, int N)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (j<N)
    {
        mx1[j]=-MAXDOUBLE;
        mx2[j]=-MAXDOUBLE;
    }
}

__global__ void calc_res_2(unsigned long *i, double *a, double *s, double *mx1, double *mx2, int M)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x, ij;
    double tmp, minv, old, aj, sj;

    if (j<M)
    {
        ij=i[j];
        aj=a[j];
        sj=s[j];

        tmp=aj+sj;

        old = atomicMax(mx1+ij, tmp);
        minv = fmin(old, tmp);
        atomicMax(mx2+ij, minv);
    }
}

__global__ void calc_res_3(unsigned long *i, double *a, double *r, double *s, double *mx1, double *mx2, double df, int M)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x, ij;
    double tmp, aj, rj, sj, mx1i, mx2i, mxv;

    if (j<M)
    {
        ij=i[j];
        aj=a[j];
        rj=r[j];
        sj=s[j];
        mx1i=mx1[ij];
        mx2i=mx2[ij];

        mxv = (aj+sj==mx1i)?mx2i:mx1i;
        r[j]=df*rj+(1-df)*(sj-mxv);
    }
}

__global__ void calc_res_3(unsigned long *i, double *a, double *r, double *s, double *mx1, double *mx2, double df, int M, unsigned long *tr, double *r_is)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x, ij, trj;
    double tmp, aj, rj, sj, mx1i, mx2i, mxv;

    if (j<M)
    {
        ij=i[j];
        aj=a[j];
        rj=r[j];
        sj=s[j];
        mx1i=mx1[ij];
        mx2i=mx2[ij];

        mxv = (aj+sj==mx1i)?mx2i:mx1i;
        tmp = df*rj+(1-df)*(sj-mxv);
        r[j]=tmp;

        trj= tr[j];
        r_is[trj]=tmp;
    }
}

__global__ void calc_ava_1(double *srp, int N)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (j<N)
        srp[j]=0.0;
}

__global__ void calc_ava_2(unsigned long *k, double *r, double *srp, int M, int N)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x, kk;
    double rj, tmp;

    if (j<M)
    {
        kk=k[j];
        rj=r[j];

        if ((j<M-N&&rj>0.0) || j>=M-N)
            atomicAdd(srp+kk, rj);
    }
}

__global__ void calc_ava_3(unsigned long *k, double *a, double *r, double *srp, double df, int M, int N)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x, kk;
    double tmp, rj, aj, srpk;

    if(j<M)
    {
        kk=k[j];
        rj=r[j];
        aj=a[j];
        srpk=srp[kk];

        tmp = ((j<M-N&&rj>0.0) || j>=M-N)?srpk-rj:srpk;
        tmp = (j<M-N&&tmp>=0.0)?0:tmp;
        a[j]=df*aj+(1-df)*tmp;
    }
}

__global__ void update_dec(unsigned long *decsum, unsigned long *dec, double *a, double *r, int M, int N)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long  dsj, dj, djv;
    double ax, rx;

    if (j<N)
    {
        dsj=decsum[j];
        dj=dec[j];
        ax=a[M-N+j];
        rx=r[M-N+j];

        decsum[j]=dsj-dj;
        djv = (ax+rx>0.0)?1:0;
        dec[j]=djv;
        decsum[j]=decsum[j]+djv;
    }
}

__global__ void consolidate_dec_1(double *mx1, int N)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (j<N)
        mx1[j] = -MAXDOUBLE;
}

__global__ void consolidate_dec_2(unsigned long *i, unsigned long *k, unsigned long *idx, double *s, double *srp, int M)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x, ij, kj;
    double sj, srpj;

    if (j<M)
    {
        ij = i[j];
        kj = k[j];
        sj = s[j];
        srpj = srp[kj];

        if(idx[ij]==idx[kj])
            atomicAdd(srp+kj, sj);
    }
}

__global__ void consolidate_dec_3(unsigned long *idx, double *srp, double *mx1, int N)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x, idxj;
    double mx1v, srpj;

    if (j<N)
    {
        srpj = srp[j];
        idxj = idx[j];

        atomicMax(mx1+idxj, srpj);
    }
}

__global__ void consolidate_dec_4(unsigned long *idx, unsigned long *dec, double *srp, double *mx1, int N)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long djv;
    double mx1v, srpj;

    if (j<N)
    {
        srpj = srp[j];
        mx1v = mx1[idx[j]];

        djv = (srpj==mx1v) ? 1 : 0;
        dec[j] = djv;
    }
}

int main(int argc, char** argv)
{
    int flag, dn, it, conv, decit, maxits, convits, restart;
    unsigned long i1, i2, j, *h_i_is, *h_tr, *h_k_is, *h_b, *d_i, *d_k, *d_k_is, *d_tr, m, n, l, *h_dec, *h_decsum, *h_idx, **d_dec, *d_decsum, K;
    double tmp, *h_s, *h_s_is, *h_a, *h_r, *d_s, *d_a, *d_r, *d_r_is, *h_mx1, *h_mx2, *h_srp, *d_mx1, *d_mx2, *d_srp, netsim, dpsim, expref, lam, run_time = 0;
  FILE *f;
  int threadsPerBlock = 256;
  int blocksPerGrid;
  clock_t time1, time2; 

  /* Usage */
  if((argc!=4)&&(argc!=7)){
    printf("\nUsage:\n\n");
    printf("cap <Similarity file> <Preference file | Preference-value | median> <Output Folder>\n");
    printf("          [ <maxits> <convits> <dampfact> ]\n\n");

    return EXIT_FAILURE;
  }

  if (MINDOUBLE==0.0) {
	  printf("There are numerical precision problems on this architecture.  Please recompile after adjusting MIN_DOUBLE and MAX_DOUBLE\n\n");
  }

  /* Parse command line */
  if(argc==4){ lam=0.9; maxits=2000; convits=200;
  } else {
    flag=sscanf(argv[4],"%d",&maxits);
    if(flag==1){
      flag=sscanf(argv[5],"%d",&convits);
      if(flag==1){
	flag=sscanf(argv[6],"%lf",&lam);
	if(flag==0){
	  printf("\n\n*** Error in <damping factor> argument\n\n");
	  return EXIT_FAILURE;
	}
      } else {
	printf("\n\n*** Error in <convergence iterations> argument\n\n");
	return EXIT_FAILURE;
      }
    } else {
      printf("\n\n*** Error in <maximum iterations> argument\n\n");
      return EXIT_FAILURE;
    }
  }
  if(maxits<1){
    printf("\n\n*** Error: maximum number of iterations must be at least 1\n\n");
    return EXIT_FAILURE;
  }
  if(convits<1){
    printf("\n\n*** Error: number of iterations to test convergence must be at least 1\n\n");
    return EXIT_FAILURE;
  }
  if((lam<0.5)||(lam>=1)){
    printf("\n\n*** Error: damping factor must be between 0.5 and 1\n\n");
    return EXIT_FAILURE;
  }
  printf("\nmaxits=%d, convits=%d, dampfact=%lf\n\n",maxits,convits,lam);

  int dev = findCudaDevice(argc, (const char **) argv);

  if (dev == -1)
  {
      printf("\n*** Error: no CUDA GPU found\n\n");
      return EXIT_FAILURE;
  }

  /* Find out how many data points and similarities there are */
  f=fopen(argv[1],"r");
  if(f==NULL){
    printf("\n\n*** Error opening similarities file\n\n");
    return EXIT_FAILURE;
  }
  m=0; n=0;
  flag=fscanf(f,"%lu %lu %lf",&i1,&i2,&tmp);
  while(flag!=EOF){
    if(i1>n) n=i1;
    if(i2>n) n=i2;
    m=m+1;
    flag=fscanf(f,"%lu %lu %lf",&i1,&i2,&tmp);
  }
  fclose(f);

  /* Allocate memory for similarities, preferences, messages, etc */
  h_i=(unsigned long *)calloc(m+n,sizeof(unsigned long));
  h_k=(unsigned long *)calloc(m+n,sizeof(unsigned long));
  h_s=(double *)calloc(m+n,sizeof(double));
  h_t=(unsigned long *)calloc(m+n,sizeof(unsigned long));
  h_tr=(unsigned long *)calloc(m+n,sizeof(unsigned long));
  h_i_is=(unsigned long *)calloc(m+n,sizeof(unsigned long));
  h_k_is=(unsigned long *)calloc(m+n,sizeof(unsigned long));
  h_s_is=(double *)calloc(m+n,sizeof(double));
  h_a=(double *)calloc(m+n,sizeof(double));
  h_r=(double *)calloc(m+n,sizeof(double));
  h_mx1=(double *)calloc(n,sizeof(double));
  h_mx2=(double *)calloc(n,sizeof(double));
  h_srp=(double *)calloc(n,sizeof(double));
  h_dec=(unsigned long *)calloc(n,sizeof(unsigned long));
  h_decsum=(unsigned long *)calloc(n,sizeof(unsigned long));
  h_idx=(unsigned long *)calloc(n,sizeof(unsigned long));

  /* Read similarities and preferences */
  f=fopen(argv[1],"r");
  for(j=0;j<m;j++){
      fscanf(f,"%lu %lu %lf",&(h_i[j]),&(h_k[j]),&(h_s[j]));
      h_i[j]--; h_k[j]--;
  }
  fclose(f);
  f=fopen(argv[2],"r");
  if(f==NULL){
    printf("\n\n*** Error opening preferences file\n\n");
    return EXIT_FAILURE;
  }
  for(j=0;j<n;j++){
      h_i[m+j]=j; h_k[m+j]=j;
      flag=fscanf(f,"%lf",&(h_s[m+j]));
  }
  fclose(f);
  if(flag==EOF){
    printf("\n*** Error: Number of entries in the preferences file is\n");
    printf("    less than number of data points\n\n");
    return EXIT_FAILURE;
  }
  m=m+n;

  for (j=0;j<m;j++)
      h_t[j] = j;
  qsort(h_t, m-n, sizeof(unsigned long), cmp_k);
  for (j=0;j<m;j++)
  {
      h_i_is[j] = h_i[h_t[j]];
      h_k_is[j] = h_k[h_t[j]];
      h_s_is[j] = h_s[h_t[j]];
  }
  memcpy(h_i, h_i_is, m*sizeof(unsigned long));
  memcpy(h_k, h_k_is, m*sizeof(unsigned long));
  memcpy(h_s, h_s_is, m*sizeof(double));
  for (j=0;j<m;j++)
      h_t[j] = j;
  qsort(h_t, m-n, sizeof(unsigned long), cmp_i);
  for (j=0;j<m;j++)
      h_k_is[j] = h_k[h_t[j]];
  for (j=0;j<m;j++)
      h_tr[j] = j;
  qsort(h_tr, m-n, sizeof(unsigned long), cmp_t);

  time1 = clock();

  /* Include a tiny amount of noise in similarities to avoid degeneracies */
  for(j=0;j<m;j++) h_s[j]=h_s[j]+(1e-16*h_s[j]+MINDOUBLE*100)*(rand()/((double)RAND_MAX+1));

   // printf("GPU memory started\n");
  /* allocate device memory and copy host memory to device */
  checkCudaErrors(cudaMalloc((void **) &d_i, m*sizeof(unsigned long)));
  // printf("here1 %f\n",m*sizeof(unsigned long)*1.0/(1024*1024));
  checkCudaErrors(cudaMemcpy(d_i, h_i, m*sizeof(unsigned long),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void **) &d_k, m*sizeof(unsigned long)));
  // printf("here2 %f\n",m*sizeof(unsigned long)*1.0/(1024*1024));
  checkCudaErrors(cudaMemcpy(d_k, h_k, m*sizeof(unsigned long),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void **) &d_k_is, m*sizeof(unsigned long)));
  checkCudaErrors(cudaMemcpy(d_k_is, h_k_is, m*sizeof(unsigned long),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void **) &d_tr, m*sizeof(unsigned long)));
  checkCudaErrors(cudaMemcpy(d_tr, h_tr, m*sizeof(unsigned long),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void **) &d_s, m*sizeof(double)));
  // printf("here3 %f\n",m*sizeof(double)*1.0/(1024*1024));
  checkCudaErrors(cudaMemcpy(d_s, h_s, m*sizeof(double),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void **) &d_a, m*sizeof(double)));
  checkCudaErrors(cudaMemset((void *) d_a, 0, m*sizeof(double)));
  // printf("here4 %f\n",m*sizeof(double)*1.0/(1024*1024));

  checkCudaErrors(cudaMalloc((void **) &d_r, m*sizeof(double)));
  // printf("here5 %f\n",m*sizeof(double)*1.0/(1024*1024));
  checkCudaErrors(cudaMemset((void *) d_r, 0, m*sizeof(double)));

  checkCudaErrors(cudaMalloc((void **) &d_r_is, m*sizeof(double)));

  checkCudaErrors(cudaMalloc((void **) &d_mx1, n*sizeof(double)));
  // printf("here6 %f\n",n*sizeof(double)*1.0/(1024*1024));

  checkCudaErrors(cudaMalloc((void **) &d_mx2, n*sizeof(double)));
  // printf("here7 %f\n",n*sizeof(double)*1.0/(1024*1024));

  checkCudaErrors(cudaMalloc((void **) &d_srp, n*sizeof(double)));
  // printf("here8 %f\n",n*sizeof(double)*1.0/(1024*1024));

  d_dec=(unsigned long **)calloc(convits,sizeof(unsigned long *));
  for(j=0;j<convits;j++)
  { 
      checkCudaErrors(cudaMalloc((void **) &(d_dec[j]), n*sizeof(unsigned long)));
      // printf("here9 %f\n",n*sizeof(unsigned long)*1.0/(1024*1024));
      checkCudaErrors(cudaMemset((void *) d_dec[j], 0, n*sizeof(unsigned long)));
  }

  checkCudaErrors(cudaMalloc((void **) &d_decsum, n*sizeof(unsigned long)));
  // printf("here10 %f\n",n*sizeof(unsigned long)*1.0/(1024*1024));
  checkCudaErrors(cudaMemset((void *) d_decsum, 0, n*sizeof(unsigned long)));
  // printf("GPU memory ended\n");

  dn=0; it=0; decit=convits;
  while(dn==0){
    it++; /* Increase iteration index */
    printf("it=%d n=%d m=%d\n",it,n,m);
    /* Compute responsibilities */
    blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    calc_res_1<<<blocksPerGrid, threadsPerBlock>>>(d_mx1, d_mx2, n);
    getLastCudaError("Kernel execution failed");
    cudaDeviceSynchronize();
    // printf("res_1\n");

    blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;
    calc_res_2<<<blocksPerGrid, threadsPerBlock>>>(d_i, d_a, d_s, d_mx1, d_mx2, m);
    getLastCudaError("Kernel execution failed");
    cudaDeviceSynchronize();
    // printf("res_2\n");

    blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;
    calc_res_3<<<blocksPerGrid, threadsPerBlock>>>(d_i, d_a, d_r, d_s, d_mx1, d_mx2, lam, m, d_tr, d_r_is);
    getLastCudaError("Kernel execution failed");
    cudaDeviceSynchronize();
    // printf("res_3\n");

    checkCudaErrors(cudaMemset((void *) d_srp, 0, n*sizeof(double)));

    blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;
    calc_ava_2<<<blocksPerGrid, threadsPerBlock>>>(d_k_is, d_r_is, d_srp, m, n);
    getLastCudaError("Kernel execution failed");
    cudaDeviceSynchronize();
    // printf("ava_2\n");

    // blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;
    // calc_res_3<<<blocksPerGrid, threadsPerBlock>>>(d_i, d_a, d_r, d_s, d_mx1, d_mx2, lam, m);
    // getLastCudaError("Kernel execution failed");
    // cudaDeviceSynchronize();
    // // printf("res_3\n");

    // checkCudaErrors(cudaMemset((void *) d_srp, 0, n*sizeof(double)));

    // blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;
    // calc_ava_2<<<blocksPerGrid, threadsPerBlock>>>(d_k, d_r, d_srp, m, n);
    // getLastCudaError("Kernel execution failed");
    // cudaDeviceSynchronize();
    // // printf("ava_2\n");

    blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;
    calc_ava_3<<<blocksPerGrid, threadsPerBlock>>>(d_k, d_a, d_r, d_srp, lam, m, n);
    getLastCudaError("Kernel execution failed");
    cudaDeviceSynchronize();
    // printf("ava_3\n");

    /* Identify exemplars and check to see if finished */
    decit++; if(decit>=convits) decit=0;

    blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    update_dec<<<blocksPerGrid, threadsPerBlock>>>(d_decsum, d_dec[decit], d_a, d_r, m, n);
    getLastCudaError("Kernel execution failed");
    cudaDeviceSynchronize();
    // printf("update_dec\n");

    if((it>=convits)||(it>=maxits)){
        checkCudaErrors(cudaMemcpy(h_dec, d_dec[decit], n*sizeof(unsigned long),
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_decsum, d_decsum, n*sizeof(unsigned long),
                                   cudaMemcpyDeviceToHost));

       K=0; for(j=0;j<n;j++) K=K+h_dec[j];
      /* Check convergence */
      conv=1;
      for(j=0;j<n;j++)
          if((h_decsum[j]!=0)&&(h_decsum[j]!=convits))
          {
              conv=0;
              break;
          }
      /* Check to see if done */
      if(((conv==1)&&(K>0))||(it==maxits)) dn=1;
    }
  }
  /* If clusters were identified, find the assignments and output them */
  if(K>0){
      for(j=0;j<m;j++)
          if(h_dec[h_k[j]]==1) h_a[j]=0.0; else h_a[j]=-MAXDOUBLE;
      for(j=0;j<n;j++) h_mx1[j]=-MAXDOUBLE;
      for(j=0;j<m;j++){
          tmp=h_a[j]+h_s[j];
          if(tmp>h_mx1[h_i[j]]){
              h_mx1[h_i[j]]=tmp;
              h_idx[h_i[j]]=h_k[j];
          }
      }
      for(j=0;j<n;j++) if(h_dec[j]) h_idx[j]=j;

      for(j=0;j<n;j++) h_srp[j]=0.0;
      for(j=0;j<m;j++) if(h_idx[h_i[j]]==h_idx[h_k[j]]) h_srp[h_k[j]]=h_srp[h_k[j]]+h_s[j];
      for(j=0;j<n;j++) h_mx1[j]=-MAXDOUBLE;
      for(j=0;j<n;j++) if(h_srp[j]>h_mx1[h_idx[j]]) h_mx1[h_idx[j]]=h_srp[j];
      for(j=0;j<n;j++)
          if(h_srp[j]==h_mx1[h_idx[j]]) h_dec[j]=1; else h_dec[j]=0;

      for(j=0;j<m;j++)
          if(h_dec[h_k[j]]==1) h_a[j]=0.0; else h_a[j]=-MAXDOUBLE;
      for(j=0;j<n;j++) h_mx1[j]=-MAXDOUBLE;
      for(j=0;j<m;j++){
          tmp=h_a[j]+h_s[j];
          if(tmp>h_mx1[h_i[j]]){
              h_mx1[h_i[j]]=tmp;
              h_idx[h_i[j]]=h_k[j];
          }
      }
      for(j=0;j<n;j++) if(h_dec[j]) h_idx[j]=j;

      f=fopen(argv[3],"w");
      for(j=0;j<n;j++) fprintf(f,"%lu\n",h_idx[j]+1);
      fclose(f);
      dpsim=0.0; expref=0.0;
      for(j=0;j<m;j++){
          if(h_idx[h_i[j]]==h_k[j]){
              if(h_i[j]==h_k[j]) expref=expref+h_s[j];
              else dpsim=dpsim+h_s[j];
          }
      }
      netsim=dpsim+expref;
      printf("\nNumber of identified clusters: %d\n",K);
      printf("Fitness (net similarity): %f\n",netsim);
      printf("  Similarities of data points to exemplars: %f\n",dpsim);
      printf("  Preferences of selected exemplars: %f\n",expref);
      printf("Number of iterations: %d\n\n",it);
  } else printf("\nDid not identify any clusters\n");
  if(conv==0){
      printf("\n*** Warning: Algorithm did not converge. Consider increasing\n");
      printf("    maxits to enable more iterations. It may also be necessary\n");
      printf("    to increase damping (increase dampfact).\n\n");
  }

  free(h_i);
  free(h_k);
  free(h_s);
  free(h_t);
  free(h_tr);
  free(h_i_is);
  free(h_k_is);
  free(h_s_is);
  free(h_a);
  free(h_r);
  free(h_mx1);
  free(h_mx2);
  free(h_srp);
  free(h_dec);
  free(h_decsum);
  free(h_idx);

  checkCudaErrors(cudaFree(d_i));
  checkCudaErrors(cudaFree(d_k));
  checkCudaErrors(cudaFree(d_k_is));
  checkCudaErrors(cudaFree(d_tr));
  checkCudaErrors(cudaFree(d_s));
  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_r));
  checkCudaErrors(cudaFree(d_r_is));
  checkCudaErrors(cudaFree(d_mx1));
  checkCudaErrors(cudaFree(d_mx2));
  checkCudaErrors(cudaFree(d_srp));
  for(j=0;j<convits;j++)
      checkCudaErrors(cudaFree(d_dec[j]));
  free(d_dec);
  checkCudaErrors(cudaFree(d_decsum));

  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits   
  cudaDeviceReset();

  time2 = clock(); 
  run_time = (float)(time2-time1)/CLOCKS_PER_SEC;
  printf("run time = %g seconds\n", run_time);
}
