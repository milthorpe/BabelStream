module Stream {
    use GPU;
    use IO;
    use ChplConfig;
    use OS.POSIX;

    config param useGPU = false;
    config param TBSIZE = 1024;

    param startScalar = 0.4;
    param startA = 0.1;
    param startB = 0.2;
    param startC = 0.0;
    var arraySize: uint;
    var deviceIndex: int(32);

    extern proc get_device_driver_version(const deviceIndex: int(32)): int(32);
    extern proc get_device_name(const deviceIndex: int(32)): c_ptrConst(c_char);

    proc getDeviceName(deviceIndex: int(32)) {
        const deviceName = get_device_name(deviceIndex);
        return try! string.createBorrowingBuffer(deviceName);
    }

    proc listDevices() {
        if here.gpus.size == 0 {
            writeln(stderr, "No devices found.");
        } else {
            writeln("Devices:");
            writeln();
            forall deviceId in 0..#here.gpus.size do
            writeln(deviceId, ": ", getDeviceName(deviceId: int(32)));
            writeln();
        }
    }

    proc makeChapelStream(initA: ?eltType, initB: eltType, initC: eltType) throws {
        // TODO make CHPL_GPU_BLOCK_SIZE a ChplConfig variable
        if arraySize % TBSIZE != 0 {
            halt("Array size must be a multiple of ", TBSIZE);   
        }

        const streamLocale: locale;
        if useGPU {
            streamLocale = here.gpus[deviceIndex];
            if deviceIndex >= here.gpus.size {
                halt("Invalid device index.");
            } else {
                if CHPL_GPU == "nvidia" {
                    writeln("Using CUDA device ", getDeviceName(deviceIndex));
                } else if CHPL_GPU == "amd" {
                    //writeln("Using ROCM device ", getDeviceName(deviceIndex));
                }
                writeln("Driver: ", get_device_driver_version(deviceIndex));
            }
        } else {
            streamLocale = here;
        }
        
        var stream: owned chapelStream(eltType)?;
        on streamLocale do stream = new chapelStream(initA, initB, initC);
        return stream: owned chapelStream(eltType);
    }

    class chapelStream {
        type eltType;
        const scalar = startScalar: eltType;
        const vectorDom = 0..#arraySize;
        var A: [vectorDom] eltType = noinit;
        var B: [vectorDom] eltType = noinit;
        var C: [vectorDom] eltType = noinit;

        proc init(initA: ?eltType, initB: eltType, initC: eltType) {
            this.eltType = eltType;
            init this;
            forall i in vectorDom {
                A[i] = initA;
                B[i] = initB;
                C[i] = initC;
            }
        }

        proc copy() {
            forall i in vectorDom do
                C[i] = A[i];
            //forall (a, c) in zip(A, C) do
            //    c = a;
        }

        proc add() {
            forall i in vectorDom do
                C[i] = A[i] + B[i];
            //forall (a, b, c) in zip(A, B, C) do
            //    c = a + b;
        }

        proc mul() {
            forall i in vectorDom do
                B[i] = scalar * C[i];
            //forall (b, c) in zip(B, C) do
            //    b = scalar * c;
        }

        proc triad() {
            forall i in vectorDom do
                A[i] = B[i] + scalar * C[i];
            //forall (a, b, c) in zip(A, B, C) do
            //    a = b + scalar * c;
        }

        proc nstream() {
            forall i in vectorDom do
                A[i] += B[i] + scalar * C[i];
            //forall (a, b, c) in zip(A, B, C) do
            //    a += b + scalar * c;
        }

        proc dot():eltType {
            var sum = 0: eltType;
            if useGPU {
                /* 
                Ordinary Chapel reductions don't yet work on GPU
                https://chapel-lang.org/docs/1.33/technotes/gpu.html#reductions-and-scans
                */
                const DOT_NUM_BLOCKS = min(arraySize/TBSIZE, 256);
                var blockSum: [0..#DOT_NUM_BLOCKS] eltType;
                // TODO strided loops are not supported. https://github.com/chapel-lang/chapel/issues/23497
                // Not sure about shared array either
                const numThreads = TBSIZE * DOT_NUM_BLOCKS; // __primitive("gpu blockDim x") * __primitive("gpu gridDim x");
                @assertOnGpu foreach i in 0..#numThreads {
                    var tbSum = createSharedArray(eltType, TBSIZE);
                    const localI = i % TBSIZE; //__primitive("gpu threadIdx x");
                    const blockDimX = TBSIZE; //__primitive("gpu blockDim x");
                    tbSum[localI] = 0: eltType;
                    var j = i;
                    while j < arraySize {
                        tbSum[localI] += A[j] * B[j];
                        j += numThreads;
                    }
                    /*
                    // https://github.com/chapel-lang/chapel/issues/23497
                    for j in 0..#arraySize by numThreads {
                        tbSum[localI] += A[j] * B[j];
                    }
                    */
                    var offset = blockDimX / 2;
                    while offset > 0 {
                        syncThreads();
                        if localI < offset {
                            tbSum[localI] += tbSum[localI+offset];
                        }
                        offset /= 2;
                    }
                    if localI == 0 {
                        const blockIdxX = i / TBSIZE; //__primitive("gpu blockIdx x");
                        blockSum[blockIdxX] = tbSum[localI];
                    }
                }
                sum = + reduce blockSum;
            } else {
                forall (a, b) in zip(A, B) with (+ reduce sum) {
                    sum += a * b;
                }
            }
            return sum;
        }
    }
}
