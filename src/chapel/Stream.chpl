module Stream {
    use GPU;
    use ChplConfig;

    param startScalar = 0.4;
    param startA = 0.1;
    param startB = 0.2;
    param startC = 0.0;
    config param TBSIZE = 1024;
    var arraySize: uint;
    var deviceIndex: int(32);

    record data {
        type eltType;
        var a: [0..#arraySize] eltType;
        var b: [0..#arraySize] eltType;
        var c: [0..#arraySize] eltType;
    }

    record chapelStream {
        type eltType;
        var myData: data(eltType);

        proc init(type eltType) {
            this.eltType = eltType;
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
                    writeln("Using ROCM device ", getDeviceName(deviceIndex));
                }
                writeln("Driver: ", get_device_driver_version(deviceIndex));
            }
            
            on here.gpus[deviceIndex] do myData = new data(eltType);
        }

        proc initArrays(initA, initB, initC) {
            on here.gpus[deviceIndex] {
                myData.a = initA;
                myData.b = initB;
                myData.c = initC;
            }
        }

        proc copy() {
            on here.gpus[deviceIndex] {
                foreach (a, c) in zip(myData.a, myData.c) do
                    c = a;
            }
        }

        proc add() {
            on here.gpus[deviceIndex] {
                foreach (a, b, c) in zip(myData.a, myData.b, myData.c) do
                    c = a + b;
            }
        }

        proc mul() {
            on here.gpus[deviceIndex] {
                const scalar = startScalar: eltType;
                foreach (b, c) in zip(myData.b, myData.c) do
                    b = scalar * c;
            }
        }

        proc triad() {
            on here.gpus[deviceIndex] {
                const scalar = startScalar: eltType;
                foreach (a, b, c) in zip(myData.a, myData.b, myData.c) do
                    a = b + scalar * c;
            }
        }

        proc nstream() {
            on here.gpus[deviceIndex] {
                const scalar = startScalar: eltType;
                /*
                var a = myData.a;
                const b = myData.b;
                const c = myData.c;
                forall i in a.dom {
                    __primitive("gpu set blockSize", TBSIZE);
                    a[i] += b[i] + scalar * c[i];
                }
                */
                foreach (a, b, c) in zip(myData.a, myData.b, myData.c) do
                    a += b + scalar * c;
            }
        }

        proc dot(type eltType):eltType {
            var sum: eltType = 0;
            const DOT_NUM_BLOCKS = 256;
            var blockSum: [DOT_NUM_BLOCKS] eltType;

            on here.gpus[deviceIndex] {
                var tbSum = createSharedArray(eltType, TBSIZE);
                const a = myData.a;
                const b = myData.b;
                const numThreads = TBSIZE * DOT_NUM_BLOCKS; // __primitive("gpu blockDim x") * __primitive("gpu gridDim x");
                foreach i in 0..#numThreads {
                    const localI = __primitive("gpu threadIdx x");
                    tb_sum[localI] = 0: eltType
                    for j in i..#arraySize by numThreads {
                        tb_sum[localI] += a[i] * b[i];
                    }
                    var offset = blockDim.x / 2;
                    while offset > 0 {
                        syncThreads();
                        if localI < offset then
                            tbSum[localI] += tbSum[localI+offset];

                        offset /= 2;
                    }
                }
            }
            
            /*
            on here.gpus[deviceIndex] {
                const a = myData.a;
                const b = myData.b;
                forall (a, b) in zip(myData.a, myData.b) with (+ reduce sum) {
                    sum += a * b;
                }
            }
            */
            return sum;
        }
    }
}
