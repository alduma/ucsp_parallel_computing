#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;

const int kBlurSize = 10;

void GenerateImage(unsigned char* output, int height, int width,
  string filename) {
  Mat output_data(height, width, CV_8UC1, (void*)output);
  imshow(filename, output_data);
  imwrite(filename, output_data);
}

__global__
void BlurKernel(unsigned char* image, unsigned char* output,
  int height, int width) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col < width && row < height) {
    int pixel_value = 0;
    int pixels = 0;

    for (int blur_row = -kBlurSize; blur_row < kBlurSize + 1; ++blur_row) {
      for (int blur_col = -kBlurSize; blur_col < kBlurSize + 1; ++blur_col) {
        int cur_row = row + blur_row;
        int cur_col = col + blur_col;
        if (cur_row > -1 && cur_row < height && cur_col > -1 && cur_col < width) {
          pixel_value += image[cur_row * width + cur_col];
          pixels++;
        }
      }
    }
    output[row * width + col] = (unsigned char)(pixel_value / pixels);
  }
}

void BlurImage(unsigned char* Image, unsigned char* output, int height, int width,
  int channels) {
  unsigned char* dev_image;
  unsigned char* dev_output;

  cudaMalloc((void**)&dev_image, height * width * channels);
  cudaMalloc((void**)&dev_output, height * width * channels);

  cudaMemcpy(dev_image, Image, height * width, cudaMemcpyHostToDevice);

  dim3 Grid_image((int)ceil(width / 16.0), (int)ceil(height / 16.0));
  dim3 dimBlock(16, 16);
  BlurKernel << <Grid_image, dimBlock >> > (dev_image, dev_output, height, width);

  cudaMemcpy(output, dev_output, height * width * channels,
    cudaMemcpyDeviceToHost);

  cudaFree(dev_output);
  cudaFree(dev_image);
}

__global__
void ColorToGreyscaleConversion(unsigned char* img,
  unsigned char* output, int height,
  int width, int CHANNELS) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  // Compute each for each pixel its gray version
  if (col < width && row < height) {
    int grey_offset = row * width + col;
    int rgb_offset = grey_offset * CHANNELS;

    unsigned char r = img[rgb_offset + 0];
    unsigned char g = img[rgb_offset + 1];
    unsigned char b = img[rgb_offset + 2];

    output[grey_offset] = r * 0.299f + g * 0.587f + b * 0.114f;
  }
}

void GrayOutImage(unsigned char* Image, unsigned char* output, int height, int width,
  int channels) {
  unsigned char* dev_image;
  unsigned char* dev_output;

  cudaMalloc((void**)&dev_image, height * width * channels);
  cudaMalloc((void**)&dev_output, height * width);

  cudaMemcpy(dev_image, Image, height * width * channels, cudaMemcpyHostToDevice);

  dim3 Grid_image((int)ceil(width / 16.0), (int)ceil(height / 16.0));
  dim3 dimBlock(16, 16);
  ColorToGreyscaleConversion << <Grid_image, dimBlock >> > (dev_image, dev_output,
    height, width, channels);

  cudaMemcpy(output, dev_output, height * width, cudaMemcpyDeviceToHost);

  cudaFree(dev_output);
  cudaFree(dev_image);
}

int main() {
  Mat image_to_gray = imread("test.jpg");
  Mat image_to_blur = imread("test.jpg", IMREAD_GRAYSCALE);

  unsigned char* output_grayed =
    (unsigned char*)malloc(sizeof(unsigned char*) * image_to_gray.rows *
      image_to_gray.cols * image_to_gray.channels());

  GrayOutImage(image_to_gray.data, output_grayed, image_to_gray.rows,
    image_to_gray.cols, image_to_gray.channels());

  unsigned char* output_blurred =
    (unsigned char*)malloc(sizeof(unsigned char*) * image_to_blur.rows *
      image_to_blur.cols * image_to_blur.channels());

  BlurImage(image_to_blur.data, output_blurred, image_to_blur.rows,
    image_to_blur.cols, image_to_blur.channels());

  GenerateImage(output_grayed, image_to_gray.rows, image_to_gray.cols,
    "test_image_gray.jpg");
  GenerateImage(output_blurred, image_to_blur.rows, image_to_blur.cols,
    "test_image_blur.jpg");

  waitKey();
  return 0;
}