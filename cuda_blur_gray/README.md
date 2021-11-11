### CUDA Blur & Gray Out
Using the kernels provided in Chapter 3 of the book Programming Massively Parallel Processors: A Hands-On Approach 3rd Edition,
the functions GrayOutImage and BlurImage implement the logic to copy from the host to the GPU device using CUDA, applying the
kernel and then copying the result back to the host. In the beginning, memory for the image itself and the result of the
computation is allocated. After copying the result back to the host, both of these memory allocations are freed.

In this project, OpenCV is used to read and write the image data.
