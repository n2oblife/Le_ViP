#include <boost/align/aligned_alloc.hpp>

#include <opencv2/cvconfig.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/core/ocl.hpp>
#include <opencv2/core/opencl/runtime/opencl_core.hpp>

/* Pb : Mat::getUMat() moves data, therefore invalidating pointers within OpenCL SVM data
SVM is needed when code+data needs a single address space, e.g. a linked-list. 
Most linked-lists have nodes and each node has a pointer to the next node. 
Those pointers are absolute memory address pointers within the same address space. 
A cpu could initially create the linked list and then give that linked list to the OpenCL compute device for use. 
Since they share a single address space, both cpu and OpenCL device can operate on the linked-list. 
The pointers are valid for both. Please note, a single address space does not imply simultaneous access by cpu and device.
*/
ocl::Kernel ManageMemory()
{
    cv::UMat svm_linked_list(1, 10000000, CV_8U, USAGE_ALLOCATE_SHARED_MEMORY);
    initialize_linked_list(svm_linked_list.getMat(ACCESS_RW)); // uses `Mat::data` and `size` to initialize a linked-list within the memory block
    ocl::Kernel my_list_operation = get_my_kernel();
    my_list_operation.args(KernelArg::ReadWrite(svm_linked_list)).run();
}


/*
Possible to look at How to Increase Performance by Minimizing Buffer Copies on Intel® Processor Graphics :
https://www.intel.com/content/www/us/en/developer/articles/training/getting-the-most-from-opencl-12-how-to-increase-performance-by-minimizing-buffer-copies-on-intel-processor-graphics.html
*/



struct AlignedDeleter
{
  void operator()(void* ptr) const
  {
    boost::alignment::aligned_free(ptr);
  };
};
auto mapAlignedMemToUMat(cl_context ctx, const cv::Mat& m, cv::UMat& um, int readWriteFlag = CL_MEM_READ_WRITE)
{
  cl_int status;
  cl_mem oclMem = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR | readWriteFlag, m.step[0] * m.rows, m.data, &status);
  if (status) throw(std::exception("Error in clCreateBuffer"));
  cv::ocl::convertFromBuffer(oclMem, m.step[0], m.rows, m.cols, m.type(), um);  // calls clRetain... and thus increments ref-counter for oclMem
  status = clReleaseMemObject(oclMem);
  if (status) throw(std::exception("Error in clReleaseMemObject"));
}
main(..)
{
    // to adapt depending of config
    int cols = 512;
    int rows = 400; // cols * rows is a multiple of 64
    
    std::unique_ptr<unsigned char, AlignedDeleter> dataRead;
    dataRead.reset(static_cast<unsigned char*>(boost::alignment::aligned_alloc(4096, cols * rows)));
    std::unique_ptr<unsigned char, AlignedDeleter> dataWrite;
    dataWrite.reset(static_cast<unsigned char*>(boost::alignment::aligned_alloc(4096, cols * rows)));

    //...read the image data ...
    
    {  // within this scope the shared mem is not accessed by the host
        cv::Mat mRead(rows, cols, CV_8U, dataRead.get(), rows);   // just used to hold the image format
        cv::Mat mWrite(rows, cols, CV_8U, dataWrite.get(), rows);

        cv::ocl::Context CvCtx = cv::ocl::Context::getDefault();
        auto ctx = reinterpret_cast<cl_context>(CvCtx.ptr());
    
        cv::UMat uRead, uWrite;
        mapAlignedMemToUMat(ctx, mRead, uRead, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS);
        mapAlignedMemToUMat(ctx, mWrite, uWrite, CL_MEM_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS);
    
        ... do some fancy image processing, whith uRead as cv::InputArray and uWrite as cv::OutputArray. eg  uRead.copyTo(uWrite)  ...
    }
    //... check the results ...
}