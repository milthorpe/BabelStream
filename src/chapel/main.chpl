
// Copyright (c) 2023 Josh Milthorpe,
// Oak Ridge National Laboratory
//
// For full license terms please see the LICENSE file distributed with this
// source code

/*
void check_error(void)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(err);
  }
}
*/

use ArgumentParser; 
use IO;
use Time;
use Math;
use CTypes;
use Stream;

config type eltType = real;
param VERSION_STRING = "4.0";

var numTimes: uint = 100;
var useFloat = false;
var outputAsCsv = false;
var mibibytes = false;
config param csv_separator = ",";

// Options for running the benchmark:
// - All 5 kernels (Copy, Add, Mul, Triad, Dot).
// - Triad only.
// - Nstream only.
enum Benchmark {All, Triad, Nstream}

// Selected run options.
var selection: Benchmark = Benchmark.All;

extern proc get_device_driver_version(const deviceIndex: int(32)): int(32);
extern proc get_device_name(const deviceIndex: int(32)): c_ptrConst(c_char);

proc getDeviceName(deviceIndex: int(32)) {
  const deviceName = get_device_name(deviceIndex);
  return string.createBorrowingBuffer(deviceName);
}

proc listDevices() {
    if here.gpus.size == 0 {
        stderr.writeln("No devices found.");
    } else {
        writeln("Devices:");
        writeln();
        forall deviceId in 0..#here.gpus.size do
          writeln(deviceId, ": ", getDeviceName(deviceId: int(32)));
        writeln();
    }
}


proc main(args: [] string) {
    var parser = new argumentParser();
    var listArg = parser.addFlag(name="list", defaultValue=false, help="List available devices");
    var deviceArg = parser.addOption(name="device", valueName="INDEX", defaultValue="0", help="Select device at INDEX");
    var arraySizeArg = parser.addOption(name="arraysize", opts=["-z", "--arraysize"], valueName="SIZE", defaultValue="33554432", help="Use SIZE elements in the array");
    var numTimesArg = parser.addOption(name="numtimes", opts=["-n", "--numtimes"], valueName="NUM", defaultValue="100", help="Run the test NUM times (NUM >= 2)");
    var floatArg = parser.addFlag(name="float", defaultValue=false, help="Use floats (rather than doubles)");
    var triadOnlyArg = parser.addFlag(name="triad-only", defaultValue=false, help="Only run triad");
    var nstreamOnlyArg = parser.addFlag(name="nstream-only", defaultValue=false, help="Only run nstream");
    var csvArg = parser.addFlag(name="csv", defaultValue=false, help="Output as csv table");
    var mibibytesArg = parser.addFlag(name="mibibytes", defaultValue=false, help="Use MiB=2^20 for bandwidth calculation (default MB=10^6)");
    
    parser.parseArgs(args);
    
    arraySize = arraySizeArg.value() : uint;

    numTimes = numTimesArg.value() : uint;
    if numTimes < 2 then
        halt("Number of times must be 2 or more");

    useFloat = floatArg.valueAsBool();
    if triadOnlyArg.valueAsBool() then
        selection = Benchmark.Triad;
    else if nstreamOnlyArg.valueAsBool() then
        selection = Benchmark.Nstream;
    outputAsCsv = csvArg.valueAsBool();
    mibibytes = mibibytesArg.valueAsBool();

    if !outputAsCsv {
      writeln("BabelStream");
      writeln("Version: ", VERSION_STRING);
      writeln("Implementation: Chapel");
    }

        if (listArg.valueAsBool()) {
        listDevices();
        exit(0);
    }

    deviceIndex = deviceArg.value(): int(32);

    if useFloat then
      run(real(32));
    else
      run(real(64));
    
}

proc run(type eltType) {
  if !outputAsCsv {
    if selection == Benchmark.All {
      writeln("Running kernels ", numTimes, " times");
    } else if selection == Benchmark.Triad {
      writeln("Running triad ", numTimes, " times");
      writeln("Number of elements: ", arraySize);
    }

    if eltType == real(32) then
      writeln("Precision: float");
    else
      writeln("Precision: double");


    if mibibytes {
      // MiB = 2^20
      writef("Array size: %.1dr MiB (=%.1dr GiB)\n", 
        arraySize*c_sizeof(eltType)*exp2(-20.0),
        arraySize*c_sizeof(eltType)*exp2(-30.0));
      writef("Total size: %.1dr MiB (=%.1dr GiB)\n", 
        3.0*arraySize*c_sizeof(eltType)*exp2(-20.0),
        3.0*arraySize*c_sizeof(eltType)*exp2(-30.0));
    } else {
      // MB = 10^6
      writef("Array size: %.1dr MB (=%.1dr GB)\n", 
        arraySize*c_sizeof(eltType)*1.0E-6,
        arraySize*c_sizeof(eltType)*1.0E-9);
      writef("Total size: %.1dr MB (=%.1dr GB)\n", 
        3.0*arraySize*c_sizeof(eltType)*1.0E-6,
        3.0*arraySize*c_sizeof(eltType)*1.0E-9);
    }
  }

  var stream = new chapelStream(eltType);
  stream.initArrays(startA:eltType, startB:eltType, startC:eltType);

  // Run the 5 main kernels
  var sum: eltType;
  
  const benchmarkRange = if selection == Benchmark.All then 0..#5 else 0..#1;
  var timings: [benchmarkRange, 0..#numTimes] real;

  var timer: stopwatch;

  // Main loop
  for k in 0..#numTimes {
      // Execute Copy
      timer.start();
      stream.copy();
      timer.stop();
      timings[0,k] = timer.elapsed();
      timer.clear();

      // Execute Mul
      timer.start();
      stream.mul();
      timer.stop();
      timings[1,k] = timer.elapsed();
      timer.clear();

      // Execute Add
      timer.start();
      stream.add();
      timer.stop();
      timings[2,k] = timer.elapsed();
      timer.clear();

      // Execute Triad
      timer.start();
      stream.triad();
      timer.stop();
      timings[3,k] = timer.elapsed();
      timer.clear();

      // Execute Dot
      timer.start();
      sum = stream.dot(eltType);
      timer.stop();
      timings[4,k] = timer.elapsed();
      timer.clear();
  }

  // Check solutions
  // Create host vectors

  // Display timing results
  if outputAsCsv {
    writeln("function", csv_separator, 
      "num_times", csv_separator, 
      "n_elements", csv_separator, 
      "sizeof", csv_separator, 
      (if (mibibytes) then "max_mibytes_per_sec" else "max_mbytes_per_sec"), csv_separator, 
      "min_runtime", csv_separator,
      "max_runtime", csv_separator,
      "avg_runtime");
  } else {
    writef("%<12s", "Function");
    writef("%<12s", (if (mibibytes) then "MiBytes/sec" else "MBytes/sec"));
    writef("%<12s", "Min (sec)");
    writef("%<12s", "Max");
    writef("%<12s", "Average");
    writeln();
  }

  if selection == Benchmark.All || selection == Benchmark.Nstream {
    var labels: [benchmarkRange] string;
    var sizes: [benchmarkRange] uint;

    if selection == Benchmark.All {
      labels = ["Copy", "Mul", "Add", "Triad", "Dot"];
      sizes = [
        2 * c_sizeof(eltType) * arraySize,
        2 * c_sizeof(eltType) * arraySize,
        3 * c_sizeof(eltType) * arraySize,
        3 * c_sizeof(eltType) * arraySize,
        2 * c_sizeof(eltType) * arraySize];
    } else if selection == Benchmark.Nstream {
      labels = ["Nstream"];
      sizes = [4 * c_sizeof(eltType) * arraySize];
    }

    for i in timings.dim(0) {
      // Get min/max; ignore the first result
      const minTime = min reduce timings[i, 1..];
      const maxTime = max reduce timings[i, 1..];

      // Calculate average; ignore the first result
      const average = + reduce timings[i, 1..] / (numTimes-1);

      // Display results
      if outputAsCsv {
        writeln(
          labels[i], csv_separator,
          numTimes, csv_separator,
          arraySize, csv_separator,
          c_sizeof(eltType), csv_separator,
          (if (mibibytes) then exp2(-20.0) else 1.0E-6) * sizes[i] / minTime, csv_separator,
          minTime, csv_separator,
          maxTime, csv_separator,
          average);
      } else {
        writef("%<12s%<12.3dr%<12.5dr%<12.5dr%<12.5dr\n",
          labels[i], (if (mibibytes) then exp2(-20.0) else 1.0E-6) * sizes[i] / minTime, minTime, maxTime, average);
          /*
          << std::left << std::setw(12) << labels[i]
          << std::left << std::setw(12) << std::setprecision(3) << 
            ((mibibytes) ? pow(2.0, -20.0) : 1.0E-6) * sizes[i] / (*minmax.first)
          << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
          << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
          << std::left << std::setw(12) << std::setprecision(5) << average
          << std::endl;
          */
      }
    }

  } 
      /*else if (selection == Benchmark::Triad)
  {
    // Display timing results
    double total_bytes = 3 * c_sizeof(eltType) * ARRAY_SIZE * num_times;
    double bandwidth = ((mibibytes) ? pow(2.0, -30.0) : 1.0E-9) * (total_bytes / timings[0][0]);

    if (output_as_csv)
    {
      std::cout
        << "function" << csv_separator
        << "num_times" << csv_separator
        << "n_elements" << csv_separator
        << "sizeof" << csv_separator
        << ((mibibytes) ? "gibytes_per_sec" : "gbytes_per_sec") << csv_separator
        << "runtime"
        << std::endl;
      std::cout
        << "Triad" << csv_separator
        << num_times << csv_separator
        << ARRAY_SIZE << csv_separator
        << c_sizeof(eltType) << csv_separator
        << bandwidth << csv_separator
        << timings[0][0]
        << std::endl;
    }
    else
    {
      std::cout
        << "--------------------------------"
        << std::endl << std::fixed
        << "Runtime (seconds): " << std::left << std::setprecision(5)
        << timings[0][0] << std::endl
        << "Bandwidth (" << ((mibibytes) ? "GiB/s" : "GB/s") << "):  "
        << std::left << std::setprecision(3)
        << bandwidth << std::endl;
    }
  }
  */
}

proc stream() {

}

/*
template <class T>
CUDAStream<T>::CUDAStream(const int ARRAY_SIZE, const int device_index)
{



  array_size = ARRAY_SIZE;

  // Allocate the host array for partial sums for dot kernels
  sums = (T*)malloc(c_sizeof(eltType) * DOT_NUM_BLOCKS);

  // Check buffers fit on the device
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  if (props.totalGlobalMem < 3*ARRAY_SIZE*c_sizeof(eltType))
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
#if defined(MANAGED)
  cudaMallocManaged(&d_a, ARRAY_SIZE*c_sizeof(eltType));
  check_error();
  cudaMallocManaged(&d_b, ARRAY_SIZE*c_sizeof(eltType));
  check_error();
  cudaMallocManaged(&d_c, ARRAY_SIZE*c_sizeof(eltType));
  check_error();
  cudaMallocManaged(&d_sum, DOT_NUM_BLOCKS*c_sizeof(eltType));
  check_error();
#elif defined(PAGEFAULT)
  d_a = (T*)malloc(c_sizeof(eltType)*ARRAY_SIZE);
  d_b = (T*)malloc(c_sizeof(eltType)*ARRAY_SIZE);
  d_c = (T*)malloc(c_sizeof(eltType)*ARRAY_SIZE);
  d_sum = (T*)malloc(c_sizeof(eltType)*DOT_NUM_BLOCKS);
#else
  cudaMalloc(&d_a, ARRAY_SIZE*c_sizeof(eltType));
  check_error();
  cudaMalloc(&d_b, ARRAY_SIZE*c_sizeof(eltType));
  check_error();
  cudaMalloc(&d_c, ARRAY_SIZE*c_sizeof(eltType));
  check_error();
  cudaMalloc(&d_sum, DOT_NUM_BLOCKS*c_sizeof(eltType));
  check_error();
#endif
}


template <class T>
CUDAStream<T>::~CUDAStream()
{
  free(sums);

#if defined(PAGEFAULT)
  free(d_a);
  free(d_b);
  free(d_c);
  free(d_sum);
#else
  cudaFree(d_a);
  check_error();
  cudaFree(d_b);
  check_error();
  cudaFree(d_c);
  check_error();
  cudaFree(d_sum);
  check_error();
#endif
}


template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;
}

template <class T>
void CUDAStream<T>::init_arrays(T initA, T initB, T initC)
{
  init_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, initA, initB, initC);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <class T>
void CUDAStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Copy device memory to host
#if defined(PAGEFAULT) || defined(MANAGED)
  cudaDeviceSynchronize();
  for (int i = 0; i < array_size; i++)
  {
    a[i] = d_a[i];
    b[i] = d_b[i];
    c[i] = d_c[i];
  }
#else
  cudaMemcpy(a.data(), d_a, a.size()*c_sizeof(eltType), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(b.data(), d_b, b.size()*c_sizeof(eltType), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(c.data(), d_c, c.size()*c_sizeof(eltType), cudaMemcpyDeviceToHost);
  check_error();
#endif
}


template <typename T>
__global__ void copy_kernel(const T * a, T * c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i];
}

template <class T>
void CUDAStream<T>::copy()
{
  copy_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void mul_kernel(T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  b[i] = scalar * c[i];
}

template <class T>
void CUDAStream<T>::mul()
{
  mul_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

template <class T>
void CUDAStream<T>::add()
{
  add_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = b[i] + scalar * c[i];
}

template <class T>
void CUDAStream<T>::triad()
{
  triad_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void nstream_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] += b[i] + scalar * c[i];
}

template <class T>
void CUDAStream<T>::nstream()
{
  nstream_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <class T>
__global__ void dot_kernel(const T * a, const T * b, T * sum, int array_size)
{
  __shared__ T tb_sum[TBSIZE];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t local_i = threadIdx.x;

  tb_sum[local_i] = 0.0;
  for (; i < array_size; i += blockDim.x*gridDim.x)
    tb_sum[local_i] += a[i] * b[i];

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
  {
    __syncthreads();
    if (local_i < offset)
    {
      tb_sum[local_i] += tb_sum[local_i+offset];
    }
  }

  if (local_i == 0)
    sum[blockIdx.x] = tb_sum[local_i];
}

template <class T>
T CUDAStream<T>::dot()
{
  dot_kernel<<<DOT_NUM_BLOCKS, TBSIZE>>>(d_a, d_b, d_sum, array_size);
  check_error();

#if defined(MANAGED) || defined(PAGEFAULT)
  cudaDeviceSynchronize();
  check_error();
#else
  cudaMemcpy(sums, d_sum, DOT_NUM_BLOCKS*c_sizeof(eltType), cudaMemcpyDeviceToHost);
  check_error();
#endif

  T sum = 0.0;
  for (int i = 0; i < DOT_NUM_BLOCKS; i++)
  {
#if defined(MANAGED) || defined(PAGEFAULT)
    sum += d_sum[i];
#else
    sum += sums[i];
#endif
  }

  return sum;
}

void listDevices(void)
{
  // Get number of devices
  int count;
  cudaGetDeviceCount(&count);
  check_error();

  // Print device names
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < count; i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}


std::string getDeviceName(const int device)
{
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  check_error();
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  cudaSetDevice(device);
  check_error();
  int driver;
  cudaDriverGetVersion(&driver);
  check_error();
  return std::to_string(driver);
}

template class CUDAStream<float>;
template class CUDAStream<double>;
*/
