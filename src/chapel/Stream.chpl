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
            for deviceId in 0..#here.gpus.size do
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
                    writeln("Using ROCM device ", getDeviceName(deviceIndex));
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
        param scalar = startScalar: eltType;
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
        }

        proc add() {
            forall i in vectorDom do
                C[i] = A[i] + B[i];
        }

        proc mul() {
            forall i in vectorDom do
                B[i] = scalar * C[i];
        }

        proc triad() {
            forall i in vectorDom do
                A[i] = B[i] + scalar * C[i];
        }

        proc nstream() {
            forall i in vectorDom do
                A[i] += B[i] + scalar * C[i];
        }

        proc dot():eltType {
            var sum = 0: eltType;
            forall (a, b) in zip(A, B) with (+ reduce sum) {
                sum += a * b;
            }
            return sum;
        }

        proc readArrays(ref a: [vectorDom] eltType, ref b: [vectorDom] eltType, ref c: [vectorDom] eltType) {
            a = A;
            b = B;
            c = C;
        }
    }
}
