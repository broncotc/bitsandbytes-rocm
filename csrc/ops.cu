// Copyright (c) Facebook, Inc. and its affiliates. 
//   
// This source code is licensed under the MIT license found in the 
// LICENSE file in the root directory of this source tree.


#include <hip/hip_runtime.h>
#include "ops.cuh"
#include "kernels.cuh"
#include <cub/device/device_scan.cuh>
#include <limits>
// #include <BinSearch.h>
#include <AAlloc.h>
#include <BinAlgo.h>
#include <cassert>
// #include <common.h>

using namespace BinSearch;
using std::cout;
using std::endl;

void histogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, int maxidx1, int n)
{
  int threads = 512;
  int num_blocks = n/threads;
  num_blocks = n % threads == 0 ? num_blocks : num_blocks + 1;
  kHistogramScatterAdd2D<<<num_blocks, 512>>>(histogram, index1, index2, src, maxidx1, n);
  CUDA_CHECK_RETURN(hipPeekAtLastError());
}

template <typename T> void estimateQuantiles(T *A, float *code, float offset, int n)
{
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
	CUDA_CHECK_RETURN(hipMemset(code, 0, 256*sizeof(float)));
  kEstimateQuantiles<T><<<num_blocks, 512>>>(A, code, offset, std::numeric_limits<T>::max(), n);
  CUDA_CHECK_RETURN(hipPeekAtLastError());
}

void quantize(float *code, float *A, unsigned char *out, int n)
{
  int num_blocks = n/1024;
  num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
  kQuantize<<<num_blocks, 1024>>>(code, A, out, n);
  CUDA_CHECK_RETURN(hipPeekAtLastError());
}

void dequantize(float *code, unsigned char *A, float *out, int n)
{
  int num_blocks = n/1024;
  num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
  kDequantize<<<num_blocks, 1024>>>(code, A, out, n);
  CUDA_CHECK_RETURN(hipPeekAtLastError());
}

template <typename T, int STOCHASTIC> void quantizeBlockwise(float * code, T *A, float *absmax, unsigned char *out, float *rand, int rand_offset, int blocksize, const int n)
{
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  if(STOCHASTIC == 1)
    assert(blocksize == 4096);

  if(blocksize == 4096)
    kQuantizeBlockwise<T, 4096, 4, STOCHASTIC><<<num_blocks, 1024>>>(code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 2048)
    kQuantizeBlockwise<T, 2048, 4, 0><<<num_blocks, 512>>>(code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 1024)
    kQuantizeBlockwise<T, 1024, 4, 0><<<num_blocks, 256>>>(code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 512)
    kQuantizeBlockwise<T, 512, 2, 0><<<num_blocks, 256>>>(code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 256)
    kQuantizeBlockwise<T, 256, 2, 0><<<num_blocks, 128>>>(code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 128)
    kQuantizeBlockwise<T, 128, 2, 0><<<num_blocks, 64>>>(code, A, absmax, out, rand, rand_offset, n);
  else if(blocksize == 64)
    kQuantizeBlockwise<T, 64, 1, 0><<<num_blocks, 64>>>(code, A, absmax, out, rand, rand_offset, n);


  CUDA_CHECK_RETURN(hipPeekAtLastError());
}

template<typename T> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int blocksize, const int n)
{
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  if(blocksize == 4096)
    kDequantizeBlockwise<T, 4096, 1024, 4><<<num_blocks, 4096/4>>>(code, A, absmax, out, n);
  else if(blocksize == 2048)
    kDequantizeBlockwise<T, 2048, 512, 4><<<num_blocks, 2048/4>>>(code, A, absmax, out, n);
  else if(blocksize == 1024)
    kDequantizeBlockwise<T, 1024, 256, 4><<<num_blocks, 1024/4>>>(code, A, absmax, out, n);
  else if(blocksize == 512)
    kDequantizeBlockwise<T, 512, 256, 2><<<num_blocks, 512/2>>>(code, A, absmax, out, n);
  else if(blocksize == 256)
    kDequantizeBlockwise<T, 256, 128, 2><<<num_blocks, 256/2>>>(code, A, absmax, out, n);
  else if(blocksize == 128)
    kDequantizeBlockwise<T, 128, 64, 2><<<num_blocks, 128/2>>>(code, A, absmax, out, n);
  else if(blocksize == 64)
    kDequantizeBlockwise<T, 64, 64, 1><<<num_blocks, 64/1>>>(code, A, absmax, out, n);

  CUDA_CHECK_RETURN(hipPeekAtLastError());
}

template<typename T, int OPTIMIZER> void optimizer32bit(T* g, T* p,
                float* state1, float* state2, float *unorm, float max_unorm, float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, bool skip_zeros, const int n)
{
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
	switch(OPTIMIZER)
	{
		case ADAM:
      if(max_unorm > 0.0f)
			{
				CUDA_CHECK_RETURN(hipMemset(unorm, 0, 1*sizeof(float)));
        kPreconditionOptimizer32bit2State<T, OPTIMIZER, 4096, 8><<<num_blocks, 512>>>(g, p, state1, state2, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n);
        CUDA_CHECK_RETURN(hipPeekAtLastError());
      }
			kOptimizer32bit2State<T, OPTIMIZER><<<num_blocks, 1024>>>(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n);
      CUDA_CHECK_RETURN(hipPeekAtLastError());
			break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:

      if(max_unorm > 0.0f)
			{
				CUDA_CHECK_RETURN(hipMemset(unorm, 0, 1*sizeof(float)));
				kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8><<<num_blocks, 512>>>(g, p, state1, unorm, beta1, eps, weight_decay, step, lr, gnorm_scale, n);
        CUDA_CHECK_RETURN(hipPeekAtLastError());
			}

			kOptimizer32bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(g, p, state1, unorm, max_unorm, param_norm, beta1, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n);
      CUDA_CHECK_RETURN(hipPeekAtLastError());
			break;
	}
}

template<typename T, int OPTIMIZER> void optimizerStatic8bit(T* p, T* g,
                unsigned char* state1, unsigned char* state2,
                float *unorm, float max_unorm, float param_norm,
                float beta1, float beta2,
                float eps, int step, float lr,
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, int n)
{
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;

  if(max_unorm > 0.0f){ CUDA_CHECK_RETURN(hipMemset(unorm, 0, 1*sizeof(float))); }

	switch(OPTIMIZER)
	{
		case ADAM:
			CUDA_CHECK_RETURN(hipMemset(new_max1, 0, 1*sizeof(float)));
			CUDA_CHECK_RETURN(hipMemset(new_max2, 0, 1*sizeof(float)));
			kPreconditionOptimizerStatic8bit2State<T, OPTIMIZER><<<num_blocks, 256>>>(p, g, state1, state2, unorm, beta1, beta2, eps, step, quantiles1, quantiles2, max1, max2, new_max1, new_max2, gnorm_scale, n);
			CUDA_CHECK_RETURN(hipPeekAtLastError());
			kOptimizerStatic8bit2State<T, OPTIMIZER><<<num_blocks, 1024>>>(p, g, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr,
																														quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n);
			CUDA_CHECK_RETURN(hipPeekAtLastError());
		break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
			CUDA_CHECK_RETURN(hipMemset(new_max1, 0, 1*sizeof(float)));
			kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER><<<num_blocks, 256>>>(p, g, state1, unorm, beta1, eps, step, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n);
			CUDA_CHECK_RETURN(hipPeekAtLastError());
			kOptimizerStatic8bit1State<T, OPTIMIZER><<<num_blocks, 1024>>>(p, g, state1, unorm, max_unorm, param_norm, beta1, eps, step, lr,
																														quantiles1, max1, new_max1, weight_decay, gnorm_scale, n);
			CUDA_CHECK_RETURN(hipPeekAtLastError());
			break;
		default:
			break;
	}
}

#define BLOCKSIZE_2STATE 2048
#define NUM_2STATE 8
#define BLOCKSIZE_1STATE 2048
#define NUM_1STATE 8

template<typename T, int OPTIMIZER> void optimizerStatic8bitBlockwise(T* p, T* g,
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr,
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n)
{

	int num_blocks = 0;
	switch(OPTIMIZER)
	{
		case ADAM:
			num_blocks = n/BLOCKSIZE_2STATE;
			num_blocks = n % BLOCKSIZE_2STATE == 0 ? num_blocks : num_blocks + 1;
			kOptimizerStatic8bit2StateBlockwise<T, OPTIMIZER, BLOCKSIZE_2STATE, NUM_2STATE><<<num_blocks, BLOCKSIZE_2STATE/NUM_2STATE>>>(p, g, state1, state2, beta1, beta2, eps, step, lr,
																														quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, skip_zeros, n);
			CUDA_CHECK_RETURN(hipPeekAtLastError());
		break;
		case MOMENTUM:
		case RMSPROP:
    case ADAGRAD:
			num_blocks = n/BLOCKSIZE_1STATE;
			num_blocks = n % BLOCKSIZE_1STATE == 0 ? num_blocks : num_blocks + 1;
			kOptimizerStatic8bit1StateBlockwise<T, OPTIMIZER, BLOCKSIZE_1STATE, NUM_1STATE><<<num_blocks, BLOCKSIZE_1STATE/NUM_1STATE>>>(p, g, state1, beta1, beta2, eps, step, lr,
																														quantiles1, absmax1, weight_decay, gnorm_scale, skip_zeros, n);
			CUDA_CHECK_RETURN(hipPeekAtLastError());
		break;
	}
}



template<typename T> void percentileClipping(T * g, float *gnorm_vec, int step, const int n)
{
  int num_blocks = n/2048;
  num_blocks = n % 2048 == 0 ? num_blocks : num_blocks + 1;
	CUDA_CHECK_RETURN(hipMemset(&gnorm_vec[step % 100], 0, 1*sizeof(float)));
  kPercentileClipping<T, 2048, 4><<<num_blocks, 512>>>(g, gnorm_vec, step, n);
  CUDA_CHECK_RETURN(hipPeekAtLastError());
}

void gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc)
{
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	hipblasStatus_t status;

			status = hipblasGemmEx(context->m_handle,
					transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N,
					transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
					m, n,	k,
					alpha, A, HIPBLAS_R_8I, lda, B, HIPBLAS_R_8I, ldb, beta,
					C, HIPBLAS_R_32I, ldc,
          HIPBLAS_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != HIPBLAS_STATUS_SUCCESS)
    {
      std::cout << "CUBLAS ERROR: Status " << status << std::endl;
    }

}

void strided_gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc, 
                    long long int strideA, long long int strideB, long long int strideC, int batchCount)
{
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	hipblasStatus_t status;

  //cout << transposeA << transposeB << endl;
  //printf("%i %i %i\n", m,n,k);
  //printf("%i %i %i\n", lda,ldb,ldc);
  //printf("%i %i %i\n", strideA, strideB, strideC);
  //printf("%i\n", batchCount);

			status = hipblasGemmStridedBatchedEx(context->m_handle,
					transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N,
					transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
					m, n,	k,
					alpha, A, HIPBLAS_R_8I, lda, (long long int)strideA, B, HIPBLAS_R_8I, ldb, (long long int)strideB, beta,
					C, HIPBLAS_R_32I, ldc, (long long int)strideC, batchCount,
          HIPBLAS_R_32I, HIPBLAS_GEMM_DEFAULT);

    if (status != HIPBLAS_STATUS_SUCCESS)
    {
      std::cout << "CUBLAS ERROR: Status " << status << std::endl;
    }

}

int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}


template<int ORDER> int get_leading_dim(int dim1, int dim2)
{
	switch(ORDER)
	{
		case ROW:
      return dim2;
			break;
    case COL:
      return dim1;
      break;
    case COL32:
      // 32*row tiles
      return dim1*32;
      break;
    case COL_TURING:
      return 32*roundoff(dim1, 8);
      break;
    case COL_AMPERE:
      // 32*32 tiles
      return 32*roundoff(dim1, 32);
      break;
		default:
			return 0;
			break;
  }
}

template int get_leading_dim<ROW>(int dim1, int dim2);
template int get_leading_dim<COL>(int dim1, int dim2);
template int get_leading_dim<COL32>(int dim1, int dim2);

template <typename T, int SRC, int TARGET, bool transpose, int DTYPE> void transform(cublasLtHandle_t ltHandle, T *A, T *out, int dim1, int dim2)
{
  cout << "" << endl;
  cout << "=============================================" << endl;
  cout << "ERROR: Your GPU does not support Int8 Matmul!" << endl;
  cout << "=============================================" << endl;
  cout << "" << endl;
  assert(false);
}

template void transform<int8_t, ROW, COL, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, ROW, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL32, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, ROW, COL32, false, 32>(cublasLtHandle_t ltHandle, int32_t *A, int32_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_TURING, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_AMPERE, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, COL32, ROW, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, COL32, ROW, false, 32>(cublasLtHandle_t ltHandle, int32_t *A, int32_t *out, int dim1, int dim2);

template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> int igemmlt(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc) 
{
  cout << "" << endl;
  cout << "=============================================" << endl;
  cout << "ERROR: Your GPU does not support Int8 Matmul!" << endl;
  cout << "=============================================" << endl;
  cout << "" << endl;
  assert(false);

	return 0;
}

int fill_up_to_nearest_multiple(int value, int multiple)
{
  return value + (value % multiple == 0 ? 0 : (multiple - (value % multiple)));
}

void dequant_mm_int32_fp16(int *A, float *rowStats, float *colStats, half *out, float* newRowStats, float* newcolStats, half *bias, int numRows, int numCols)
{
  int threads = 512;
  int tileCols = fill_up_to_nearest_multiple(numCols, 32);
  int n = numRows*tileCols;
  int subtile_rows = 128;
  int tilesize = 32*subtile_rows;
  int num_blocks = numRows/subtile_rows;
  num_blocks += (numRows % subtile_rows == 0) ? 0 : 1;
  num_blocks = num_blocks*(tileCols/32);
  assert(threads <= tilesize);

  kdequant_mm_int32_fp16<4, 128, 512><<<num_blocks, threads>>>(A, rowStats, colStats, out, newRowStats, newcolStats, bias, numRows, numCols, tileCols, n);
  CUDA_CHECK_RETURN(hipPeekAtLastError());
}

#define STATS_THREADS 64
#define STATS_ITEMS 4
#define STATS_ROWS 16
void getColRowStats(half * A, float *rowStats, float *colStats, int *nnz_count_row, float nnz_threshold, int rows, int cols)
{
  int tile_cols = STATS_THREADS*STATS_ITEMS;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, STATS_ROWS);
	int row_tiles = (tiledRows/STATS_ROWS);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;

  if(nnz_threshold == 0.0)
    kgetColRowStats<half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0><<<num_blocks, STATS_THREADS>>>(A, rowStats, colStats, nnz_count_row, nnz_threshold, rows, cols, tiledRows, tiledCols);
  else if(nnz_threshold != 0.0)
    kgetColRowStats<half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 1><<<num_blocks, STATS_THREADS>>>(A, rowStats, colStats, nnz_count_row, nnz_threshold, rows, cols, tiledRows, tiledCols);
  CUDA_CHECK_RETURN(hipPeekAtLastError());

}

void doubleRowColQuant(half * A, float *rowStats, float *colStats, char *out_col_normed, char *out_row_normed, int *rowidx, int *colidx, half *val, int *nnz_block_ptr, float threshold, int rows, int cols)
{
  int threads = 64;
  int items_per_thread = 4;
  int tile_cols = threads*items_per_thread;
  int tile_rows = 16;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, tile_rows);
	int row_tiles = (tiledRows/tile_rows);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;


  if(threshold > 0.0f)
    kDoubleRowColQuant<64, 4, 16, 64*4, 1><<<num_blocks, threads>>>(A, rowStats, colStats, out_col_normed, out_row_normed, rowidx, colidx, val, nnz_block_ptr, threshold, rows, cols, tiledCols);
  else
    kDoubleRowColQuant<64, 4, 16, 64*4, 0><<<num_blocks, threads>>>(A, rowStats, colStats, out_col_normed, out_row_normed, rowidx, colidx, val, nnz_block_ptr, threshold, rows, cols, tiledCols);

  CUDA_CHECK_RETURN(hipPeekAtLastError());
}

template <int FORMAT, int TRANSPOSE> void transformRowToFormat(char * A, char *out, int rows, int cols)
{
  int threads = 256;
  int items_per_thread = 8;
  // we load 128 column values per warp
  int tile_cols = 32*items_per_thread;
  int tile_rows = 32;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, tile_rows);
	int row_tiles = (tiledRows/tile_rows);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;

  int outCols = fill_up_to_nearest_multiple(cols, 32);
  int outRows = fill_up_to_nearest_multiple(rows, 32);
  if(FORMAT == COL_TURING)
  {
    if(TRANSPOSE)
      outRows = fill_up_to_nearest_multiple(cols, 8);
    else
      outRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == COL_AMPERE)
  {
    if(TRANSPOSE)
      outRows = fill_up_to_nearest_multiple(cols, 32);
    else
      outRows = fill_up_to_nearest_multiple(rows, 32);
  }
  else
  {
    if(TRANSPOSE)
    {
      outCols = fill_up_to_nearest_multiple(rows, 32);
      outRows = cols;
    }
  }

  kTransformRowToFormat<256, 8, 32, 32*8, TRANSPOSE, FORMAT><<<num_blocks, threads>>>(A, out, rows, cols, tiledCols, outRows, outCols);
  CUDA_CHECK_RETURN(hipPeekAtLastError());
}

void spmm_coo(hipsparseHandle_t handle, int *A_rowidx, int *A_colidx, half *A_vals, int A_nnz, int A_rows, int A_cols, int B_cols, int ldb, half *B, int ldc, half* C, bool transposed_B)
{

  cout << "" << endl;
  cout << "=============================================" << endl;
  cout << "ERROR: Your GPU does not support Int8 Matmul!" << endl;
  cout << "=============================================" << endl;
  cout << "" << endl;
  assert(false);

	return;
}

template <typename T, int BITS> void spmm_coo_very_sparse_naive(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, T *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB)
{

  kspmm_coo_very_sparse_naive<T, 8, BITS><<<nnz_rows, 256>>>(max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz, rowsA, rowsB, colsB);
  CUDA_CHECK_RETURN(hipPeekAtLastError());
}


template <int FORMAT> void extractOutliers(char * A, int *idx, char *out, int idx_size, int rows, int cols)
{
  int threads = 256;
  // we load 128 column values per warp
  int tiledCols = tiledCols = fill_up_to_nearest_multiple(cols, 32);
  int tiledRows = 0;

	int num_blocks = idx_size;

  if(FORMAT == COL_TURING)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == COL_AMPERE)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 32);
	}

  kExtractOutliers<FORMAT><<<num_blocks, threads>>>(A, idx, out, idx_size, rows, cols, tiledRows, tiledCols);
  CUDA_CHECK_RETURN(hipPeekAtLastError());
}

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void extractOutliers<COL_TURING>(char * A, int *idx, char *out, int idx_size, int rows, int cols);
template void extractOutliers<COL_AMPERE>(char * A, int *idx, char *out, int idx_size, int rows, int cols);

template void spmm_coo_very_sparse_naive<half, 16>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, half *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);
template void spmm_coo_very_sparse_naive<signed char, 8>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, signed char *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);

template int igemmlt<COL_TURING, 32, 0>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 0>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 1>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 32, 0>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 0>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 1>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);

template void transformRowToFormat<COL32, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL32, 1>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_TURING, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_TURING, 1>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_AMPERE, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_AMPERE, 1>(char * A, char *out, int rows, int cols);

template void estimateQuantiles(half *A, float *code, float offset, int n);
template void estimateQuantiles(float *A, float *code, float offset, int n);

template void quantizeBlockwise<half, 0>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<half, 1>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 1>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void dequantizeBlockwise<half>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n);
template void dequantizeBlockwise<float>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);

#define MAKE_optimizer32bit(name, gtype) \
template void optimizer32bit<gtype, name>(gtype* g, gtype* p, \
                float* state1, float* state2, float* unorm, float max_unorm, float param_norm, \
                const float beta1, const float beta2, const float eps, const float weight_decay, \
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n);

MAKE_optimizer32bit(ADAM, half)
MAKE_optimizer32bit(ADAM, float)
MAKE_optimizer32bit(MOMENTUM, half)
MAKE_optimizer32bit(MOMENTUM, float)
MAKE_optimizer32bit(RMSPROP, half)
MAKE_optimizer32bit(RMSPROP, float)
MAKE_optimizer32bit(ADAGRAD, half)
MAKE_optimizer32bit(ADAGRAD, float)

#define MAKE_optimizerStatic8bit(name, gtype) \
template void optimizerStatic8bit<gtype, name>(gtype* p, gtype* g, unsigned char* state1, unsigned char* state2, \
                float *unorm, float max_unorm, float param_norm, \
                float beta1, float beta2, \
                float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                float weight_decay, \
                const float gnorm_scale, int n); \

MAKE_optimizerStatic8bit(ADAM, half)
MAKE_optimizerStatic8bit(ADAM, float)
MAKE_optimizerStatic8bit(MOMENTUM, half)
MAKE_optimizerStatic8bit(MOMENTUM, float)
MAKE_optimizerStatic8bit(RMSPROP, half)
MAKE_optimizerStatic8bit(RMSPROP, float)

#define MAKE_optimizerStatic8bitBlockwise(gtype, optim_name) \
template void optimizerStatic8bitBlockwise<gtype, optim_name>(gtype* p, gtype* g, \
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n); \

MAKE_optimizerStatic8bitBlockwise(half, ADAM);
MAKE_optimizerStatic8bitBlockwise(float, ADAM);
MAKE_optimizerStatic8bitBlockwise(half, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(float, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(half, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(float, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(half, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(float, ADAGRAD);

template void percentileClipping(float * g, float *gnorm_vec, int step, const int n);
template void percentileClipping(half * g, float *gnorm_vec, int step, const int n);
