#include <cuda.h>
#include <iostream>
#include <cstdio>

void check_error(void)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(err);
  }
}
extern "C" {
int get_device_driver_version(const int device) {
    cudaSetDevice(device);
    check_error();
    int driver;
    cudaDriverGetVersion(&driver);
    check_error();
    return driver;
}

std::string get_device_name(const int device) {
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  check_error();
  std::string device_name(props.name);
  return device_name;
}
}