/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/limits.hpp"

namespace cv
{
namespace cuda
{
namespace device
{
namespace median_filter
{
	
template <class Ptr2D>
__device__ __forceinline__ void swap(Ptr2D &ptr, int x, int y, int a, int b, int rows)
{
    typename Ptr2D::elem_type val = ptr(a * rows + y, x);
    ptr(a * rows + y, x) = ptr(b * rows + y, x);
    ptr(b * rows + y, x) = val;
}

///////////////////////////////////////////////////////////////
// MOG2

__global__ void medianFilter(const PtrStepSzf frame, PtrStepSzf history, int nFrame)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

	history(nFrame * frame.rows + y, x) = frame(y, x);
}

void median_filter_gpu(PtrStepSzf frame, PtrStepSzf history, int nFrame, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid(divUp(frame.cols, block.x), divUp(frame.rows, block.y));
	
	cudaSafeCall(cudaFuncSetCacheConfig(medianFilter, cudaFuncCachePreferL1));
	medianFilter<<<grid, block, 0, stream>>>(frame, history, nFrame);
	
    cudaSafeCall(cudaGetLastError());
	
    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());
}

__global__ void getMedianImage(PtrStepSzf history, PtrStepSzf dst, int nFrame)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int left = 0;
	int right = nFrame - 1;
	int k = nFrame / 2;
   
	while (left <= right) {
		if (left == right) {
            dst(y, x) = history(left * dst.rows + y, x);
            break;
		}

		int pivot = left + (right - left) / 2;
		float pivotValue = history(pivot * dst.rows + y, x);
		swap(history, x, y, pivot, right, dst.rows);
		int storeIndex = left;

		for (int i = left; i < right; i++) {
			if (history(i * dst.rows + y, x) < pivotValue) {
				swap(history, x, y, i, storeIndex, dst.rows);
				storeIndex++;
			}
		}
		swap(history, x, y, storeIndex, right, dst.rows);
		pivot = storeIndex;

		if (pivot == k) {
			dst(y, x) = history(k * dst.rows + y, x);
			break;
		} else if (k < pivot) {
			right = pivot - 1;
		} else {
			left = pivot + 1;
		}	
	}
}

void getMedianImage_gpu(PtrStepSzf history, PtrStepSzf dst, int nFrame, cudaStream_t stream)
{
	dim3 block(32, 8);
	dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    cudaSafeCall(cudaFuncSetCacheConfig(getMedianImage, cudaFuncCachePreferL1));

	getMedianImage<<<grid, block, 0, stream>>>(history, dst, nFrame);

	cudaSafeCall(cudaGetLastError());

    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());
}
} // namespace median_filter
} // namespace device
} // namespace cuda
} // namespace cv

#endif /* CUDA_DISABLER */
