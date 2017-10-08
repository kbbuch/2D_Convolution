#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


int M=0, N=0, M_final, N_final;
int J=0, K=0;

long pos;

double A[10000][10000], H[10][10];


/*--------------------------------
Reading from the input matrix file 
=> A of size M_final x N_final
=> H of size J x K
---------------------------------*/
void file_read(FILE *file){
	
	int i, j;
	
	if (file == NULL) {
        printf("Error: file is not provided.");
        return;;
    }
	
	char c, c1;
	int space, space1 ,flag=0 ;
	
	for(i = 0; i < 1000000; i++){
		space1 = space;
		space = 0;
		M++;
		for(j = 0; j < 100000; j++){
			
			c=fgetc(file);
			//printf("%d\t", c);
			if(c == 32){
				space++;
			}
			else if (c == '\n'){
				//printf("hello\n");
				break;
			}
		}
		
		if(space==0) break;
	}
		N = space1 + 1;
		M = M-1;	
		printf("size of array A is %d x %d\n",M,N);

		rewind(file);
	M_final = M;
	N_final = N;
	for(i=0;i<M;i++){
		for(j=0;j<N;j++){		
			fscanf(file,"%lf",&A[i][j]);
		}
	}
	pos = ftell(file);
	
	rewind(file);
	M=0;
	
	for(i = 0; i < 1000000; i++){
		if(space==0) M=0;
		space1 = space;
		space = 0;
		M++;
		for(j = 0; j < 100000; j++){
			c1=c;
			c=fgetc(file);
			//printf("%d\t", c);
			
			if(c == 32){
				space++;
			}
			else if(c == '\n'){
				break;
			}
			else if(c == EOF){
				flag = 1;
				break;
			}
		}
		//printf("\n");
		//printf("line = %d , space = %d\n",M, space);
		if(flag == 1){
			break;
		}
	}
	
	K = space1+1;
	if(c1=='\n'){
		J = M-1;
	}
	else{
		J = M;
	}
	
	printf("size of array H is %d x %d\n",J,K);
	fseek(file, pos, SEEK_SET);
	
	for(i=0;i<J;i++){
		for(j=0;j<K;j++){		
			fscanf(file,"%lf",&H[i][j]);
		}
	}
	
	rewind(file);
}

__global__ void convolution(double *d_A, double *d_C, int size_c, int J, int K, int M_final, int N_final)
{ 
    int    tidx = blockIdx.x*blockDim.x+threadIdx.x;
	int    tidy = blockIdx.y*blockDim.y+threadIdx.y; 
	
    int j,k; 
	
	double sum;
	
	__shared__ double shared_H[100];
	
	if(tidy%32<J && tidx%32<K){
		shared_H[(tidy%32)*K+(tidx%32)] = d_A[(M_final*N_final)+(tidy%32)*K+(tidx%32)];
	}
	
	__syncthreads();
	
	if (tidy<(M_final+J-1) && tidx<(N_final+K-1)){
		sum=0;
		for(j=0;j<J;j++){
			for(k=0;k<K;k++){
				if(!((tidy-j)<0 || (tidx-k)<0 || (tidy-j)>=M_final || (tidx-k)>=N_final)){
					sum += d_A[((tidy-j)*N_final)+(tidx-k)]*shared_H[j*K+k];
				}
			}
		}
		d_C[tidy*(N_final+K-1)+tidx]=sum;
	}

} 

int main(int argc, char** argv){

	char *inputfile;
	double *d_A = NULL, *d_C = NULL;
	cudaError_t err = cudaSuccess;
	
	inputfile = argv[1];
	
	int m;
	
	FILE *fp = fopen(inputfile, "r");
	
	file_read(fp);
	
	int size_c = (M_final+J-1)*(N_final+K-1);
	size_t size_A = ((M_final*N_final)+(J*K)) * sizeof(double);
	size_t size_C = (M_final+J-1) * (N_final+K-1) * sizeof(double);
	
	double *h_A, *h_C; 
	
	h_A = (double*) malloc (((M_final*N_final)+(J*K))*sizeof(double));
	h_C = (double*) malloc ((M_final+J-1)*(N_final+K-1)*sizeof(double));
	
	printf("\nh_A: %dx%d\n",M_final,N_final);
	
	for(int i=0;i<M_final;i++){
		for(int j=0;j<N_final;j++){
			h_A[i*N_final+j]=A[i][j];
		}
	}
	
	printf("\nh_H: %dx%d\n",J,K);
	
	for(int i=0;i<J;i++){
		for(int j=0;j<K;j++){
			h_A[(M_final*N_final)+i*K+j]=H[i][j];
		}
	}	
	
	printf("Allocating memory on the CUDA device\n");
	
	err = cudaMalloc((void **)&d_A, size_A);
		
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	
	err = cudaMalloc((void **)&d_C, size_C);
		
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		
	printf("Size of C = %dx%d\n",M_final+J-1,N_final+K-1);
    
	printf("Copy input data from the host memory to the CUDA device\n");

	err = cudaMemcpy(d_A,h_A,size_A,cudaMemcpyHostToDevice);
	
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to Copy device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
		
	dim3 Grid_Size(((N_final+K-2)/32)+1,((M_final+J-2)/32)+1,1);
	dim3 Block_Size(32,32,1);
	
		printf("No. of Blocks Launched = %dx%d\n",((N_final+K-2)/32)+1,((M_final+J-2)/32)+1);
	
	convolution<<<Grid_Size,Block_Size>>>(d_A,d_C,size_c,J,K,M_final,N_final);
	
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize the device (error code: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(h_C,d_C,size_C,cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to Copy the result back (error code: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	printf("Copied!!\n");
	for(m=0;m<size_c;m++){
		printf("C[%d] = %.3lf\n",m,h_C[m]);
	}
	
	cudaFree(d_A);
	cudaFree(d_C);
	
	free(h_A);
	free(h_C);
	
	cudaDeviceReset();
	
	printf("-----------------Done and Dusted----------------\n");

	fclose(fp);
	return 0;	
}