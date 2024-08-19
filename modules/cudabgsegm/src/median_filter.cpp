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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

Ptr<cuda::MedianFilter> cv::cuda::createMedianFilter(int, double, bool)
{
    throw_no_cuda();
    return Ptr<cuda::MedianFilter>();
}

#else

namespace cv { namespace cuda { namespace device
{
    namespace median_filter
    {
        void median_filter_gpu(PtrStepSzf frame, PtrStepSzf history, int nFrame, cudaStream_t stream);
        void getMedianImage_gpu(PtrStepSzf history, PtrStepSzf dst, int nFrame, cudaStream_t stream);
    }
}}}


namespace
{
const int historyLength = 50;
class MedianFilterImpl CV_FINAL : public cuda::MedianFilter
{
public:
	MedianFilterImpl();
	~MedianFilterImpl();
	
	void apply(InputArray image);
	void apply(InputArray image, Stream &stream);
	
	void getMedianImage(OutputArray medianImage) const;
	void getMedianImage(OutputArray medianImage, Stream &stream) const;

	void clear();
	
	
private:
    void initialize(Size frameSize, int frameType, Stream &stream);
	
	
	Size frameSize_;
	int frameType_;
	int nframes_;
	
	GpuMat history_;
};
	
MedianFilterImpl::MedianFilterImpl() : frameSize_(0, 0), frameType_(0), nframes_(0)
{
}

MedianFilterImpl::~MedianFilterImpl()
{
}

void MedianFilterImpl::apply(InputArray image)
{
    apply(image, Stream::Null());
}

void MedianFilterImpl::apply(InputArray _frame, Stream &stream)
{
    using namespace cv::cuda::device::median_filter;

    GpuMat frame = _frame.getGpuMat();

    if (nframes_ == 0)
        initialize(frame.size(), frame.type(), stream);

    ++nframes_;

    median_filter_gpu(frame, history_, nframes_, StreamAccessor::getStream(stream));
}

void MedianFilterImpl::getMedianImage(OutputArray medianImage) const
{
    getMedianImage(medianImage, Stream::Null());
}

void MedianFilterImpl::getMedianImage(OutputArray _medianImage, Stream &stream) const
{
    using namespace cv::cuda::device::median_filter;

    _medianImage.create(frameSize_, frameType_);
    GpuMat medianImage = _medianImage.getGpuMat();

    getMedianImage_gpu(history_, medianImage, nframes_, StreamAccessor::getStream(stream));
}

void MedianFilterImpl::clear() {
	nframes_ = 0;
}

void MedianFilterImpl::initialize(cv::Size frameSize, int frameType,
                                  Stream &stream) {
  using namespace cv::cuda::device::median_filter;

  CV_Assert(frameType == CV_32FC1);

  frameSize_ = frameSize;
  frameType_ = frameType;
  nframes_ = 0;

  history_.create(frameSize.height * historyLength, frameSize_.width, CV_32FC1);
}
} // namespace 

Ptr<cuda::MedianFilter> cv::cuda::createMedianFilter()
{
    return makePtr<MedianFilterImpl>();
}

#endif
