module Stream {
    use GPU;
    use ChplConfig;
    use OS.POSIX;

    param startScalar = 0.4;
    param startA = 0.1;
    param startB = 0.2;
    param startC = 0.0;
    config param TBSIZE = 1024;
    var arraySize: uint;
    var deviceIndex: int(32);

    proc makeChapelStream(initA: ?eltType, initB: eltType, initC: eltType) throws {
        // TODO make CHPL_GPU_BLOCK_SIZE a ChplConfig variable
        if (arraySize % TBSIZE != 0) {
            halt("Array size must be a multiple of ", TBSIZE);   
        }

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
        
        var stream: owned chapelStream(eltType)?;
        on here.gpus[deviceIndex] do stream = new chapelStream(initA, initB, initC);
        return stream: owned chapelStream(eltType);
    }

    class chapelStream {
        type eltType;
        var vectorDom = 0..#arraySize;
        var A: [vectorDom] eltType;
        var B: [vectorDom] eltType;
        var C: [vectorDom] eltType;

        proc init(initA: ?eltType, initB: eltType, initC: eltType) {
            this.eltType = eltType;
            A = initA;
            B = initB;
            C = initC;
        }

        proc copy() {
            foreach (a, c) in zip(A, C) do
                c = a;
        }

        proc add() {
            foreach (a, b, c) in zip(A, B, C) do
                c = a + b;
        }

        proc mul() {
            const scalar = startScalar: eltType;
            foreach (b, c) in zip(B, C) do
                b = scalar * c;
        }

        proc triad() {
            const scalar = startScalar: eltType;
            foreach (a, b, c) in zip(A, B, C) do
                a = b + scalar * c;
        }

        proc nstream() {
            const scalar = startScalar: eltType;
            foreach (a, b, c) in zip(A, B, C) do
                a += b + scalar * c;
        }

        proc dot():eltType {
            var sum = 0: eltType;
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
                while (j < arraySize) {
                    tbSum[localI] += A[j] * B[j];
                    j += numThreads;
                }
                /*
                for j in 0..#arraySize //by numThreads
                {
                    if j % numThreads == i {
                        tbSum[localI] += A[j] * B[j];
                    }
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
                if (localI == 0) {
                    const blockIdxX = i / TBSIZE; //__primitive("gpu blockIdx x");
                    blockSum[blockIdxX] = tbSum[localI];
                }
            }
            sum = + reduce blockSum;
            /*
            forall (a, b) in zip(A, B) with (+ reduce sum) {
                sum += a * b;
            }
            */
            return sum;
        }
    }
}
