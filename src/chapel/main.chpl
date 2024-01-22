
// Copyright (c) 2023 Josh Milthorpe,
// Oak Ridge National Laboratory
//
// For full license terms please see the LICENSE file distributed with this
// source code

use ArgumentParser; 
use IO;
use Time;
use Math;
use CTypes;
use Stream;
use Version;

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

    if (listArg.valueAsBool()) {
        if useGPU {
          Stream.listDevices();
        } else {
          try! stderr.writeln("No devices found. (Did you mean to build BabelStream with -suseGPU=true?)");
        }
        exit(0);
    }

    if !outputAsCsv {
      writeln("BabelStream");
      writeln("Version: ", VERSION_STRING);
      writeln("Implementation: Chapel ", Version.chplVersion);
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

  // Create host vectors
  var stream: owned chapelStream(eltType) = makeChapelStream(startA:eltType, startB:eltType, startC:eltType);
  
  const benchmarkRange = if selection == Benchmark.All then 0..#5 else 0..#1;
  var timings: [benchmarkRange, 0..#numTimes] real;

  if selection == Benchmark.Triad then
    runTriad(stream, timings);
  else if selection == Benchmark.Nstream then
    runNstream(stream, timings);
  else {
    var sum = 0: eltType;
    sum = runAll(stream, timings);

  // Check solutions
  // TODO proper check
    var testA = stream.A;
    var testB = stream.B;
    var testProd = testA * testB;
    var testSum = + reduce testProd;
    writeln(sum, " ", testSum);
  }

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
      }
    }
  } else if selection == Benchmark.Triad {
    // Display timing results
    const totalBytes = 3 * c_sizeof(eltType) * arraySize * numTimes;
    const bandwidth = (if mibibytes then exp2(-30.0) else 1.0E-9) * (totalBytes / timings[0, 0]);

    if outputAsCsv {
      writeln(
        "function", csv_separator,
        "num_times", csv_separator,
        "n_elements", csv_separator,
        "sizeof", csv_separator,
        (if mibibytes then "gibytes_per_sec" else "gbytes_per_sec"), csv_separator,
        "runtime");
      writeln(
        "Triad", csv_separator,
        numTimes, csv_separator,
        arraySize, csv_separator,
        c_sizeof(eltType), csv_separator,
        bandwidth, csv_separator,
        timings[0, 0]);
    } else {
      writeln("--------------------------------");
      writef("Runtime (seconds): %<.5dr\n", timings[0, 0]);
      writef("Bandwidth (%s):  %<.3dr\n", (if mibibytes then "GiB/s" else "GB/s"), bandwidth);
    }

  } 
}

// Run the 5 main kernels
proc runAll(ref stream, ref timings): stream.eltType {
  var sum: stream.eltType;
  var timer: stopwatch;

  // Main loop
  for k in 0..#numTimes {
      // Execute Copy
      timer.start();
      on stream.locale do stream.copy();
      timer.stop();
      timings[0,k] = timer.elapsed();
      timer.clear();

      // Execute Mul
      timer.start();
      on stream.locale do stream.mul();
      timer.stop();
      timings[1,k] = timer.elapsed();
      timer.clear();

      // Execute Add
      timer.start();
      on stream.locale do stream.add();
      timer.stop();
      timings[2,k] = timer.elapsed();
      timer.clear();

      // Execute Triad
      timer.start();
      on stream.locale do stream.triad();
      timer.stop();
      timings[3,k] = timer.elapsed();
      timer.clear();

      // Execute Dot
      timer.start();
      on stream.locale do sum = stream.dot();
      timer.stop();
      timings[4,k] = timer.elapsed();
      timer.clear();
  }

  return sum;
}

// Run the Triad kernel
proc runTriad(ref stream, ref timings) {
  var timer: stopwatch;

  // Run triad in loop
  timer.start();
  for k in 0..#numTimes {
      // Execute Triad
      on stream.locale do stream.triad();
  }
  timer.stop();
  timings[0,0] = timer.elapsed();
  timer.clear();
}

// Run the Nstream kernel
proc runNstream(ref stream, ref timings) {
  var timer: stopwatch;

  // Run nstream in loop
  for k in 0..#numTimes {
      timer.start();
      on stream.locale do stream.nstream();
      timer.stop();
      timings[0,k] = timer.elapsed();
      timer.clear();
  }
}


