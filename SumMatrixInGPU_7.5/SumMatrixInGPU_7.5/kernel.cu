
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <random>
#include <string.h>

#include "eTimer.h"

//Matrix size
#define N 6*1024

__global__ void addKernel(double *c, const double *a, const double *b, const double alpha, const double beta)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	c[y*N + x] = alpha*a[y*N + x] + beta*b[y*N + x];
}

//TODO liberar la memoria de la GPU y la CPU

int main()
{

	double *A, *B, *C;
	//define weights for matrixes
	double alpha = 0.7;
	double beta = 0.6;

	//init random number generator
	std::default_random_engine generador;
	std::normal_distribution<double> distribucion(0.0, 1.0);

	//allocate space for matrixes
	A = (double*)_aligned_malloc(N*N*sizeof(double), 64);
	B = (double*)_aligned_malloc(N*N*sizeof(double), 64);
	C = (double*)_aligned_malloc(N*N*sizeof(double), 64);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A[i*N + j] = distribucion(generador);
			B[i*N + j] = distribucion(generador);
		}
	}

	eTimer *Tcpu = new eTimer(); // timer for time to compute on CPU
	eTimer *THtD = new eTimer(); //timer for time to transfer data to GPU
	eTimer *Tkernel = new eTimer(); //timer for GPU to compute
	eTimer *TDtH = new eTimer(); //timer to transfer data from GPU

	Tcpu->start();
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			C[i*N + j] = alpha*A[i*N + j] + beta * B[i*N + j];
		}
	}
	Tcpu->stop();
	Tcpu->report("CPU");

	for (int i = 0; i < 5; i++) printf("%lf ", C[i]);
	printf("\n\n");

	memset(C, 0, N*N*sizeof(double));
	for (int i = 0; i < 5; i++)
	{
		printf("%lf ", C[i]);
	}
	printf("\n\n");

	/*---------------------------GPU-------------------------------------------*/
	cudaError_t cudaStatus;

	//It's supposed to be one and only one GPU. And that's the chosen one.
	cudaStatus = cudaSetDevice(0);

	//pointers to GPU memory
	double *dev_A, *dev_B, *dev_C;

	cudaStatus = cudaMalloc((void**)&dev_C, N*N*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_B, N*N*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_A, N*N*sizeof(double));

	THtD->start();
	cudaStatus = cudaMemcpy(dev_A, A, N*N*sizeof(double),cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_B, B, N*N*sizeof(double), cudaMemcpyHostToDevice);
	THtD->stop();
	THtD->report("HostToDevice");

	double AnchoBanda = 2 * N*N*sizeof(double) / THtD->get();
	printf("\nAncho de banda(promedio): %lf GBs\n", AnchoBanda*1.0e-9);
	Tkernel->start();
	dim3 Grid, Block;
	Block.x = 32;
	Block.y = 16;
	Grid.x = N / Block.x;
	Grid.y = N / Block.y;

	addKernel <<< Grid, Block >>>(dev_C, dev_A, dev_B,alpha, beta);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "addKernel lauch failed %s\n", cudaGetErrorString(cudaStatus));
		exit(1);
	}
	cudaStatus = cudaDeviceSynchronize();
	Tkernel->stop();
	Tkernel->report("Kernel");

	TDtH->start();
	cudaStatus = cudaMemcpy(C, dev_C, N*N*sizeof(double), cudaMemcpyDeviceToHost);
	TDtH->stop();
	TDtH->report("DeviceToHost");

	for (int i = 0; i < 5; i++)
	{
		printf("%lf ", C[i]);
	}
	printf("\n\n");

	cudaStatus = cudaDeviceReset();

	delete Tcpu;
	delete THtD;
	delete Tkernel;
	delete TDtH;

	std::getchar();
	return 0;
}
