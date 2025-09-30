# BGJ-vs-BDGL
A CUDA-based tool for comparing BGJ and BDGL sieving algorithms

### Compile:
Use NVIDIA nvcc to compile the CUDA program (example):
```
nvcc -ccbin g++ -Xcompiler "-O3 -g -march=native -pthread" -gencode arch=compute_89,code=sm_89 -O3 test_la.cu -lcudart -o test_la
```

### Get help:
Run the binary with -h or --help. The program prints the exact usage below
```
Usage: exp/test_la [-h] --CSD CSD [--thread THREAD] --filter FILTER [--red [RED ...]] [--db DB]

Options:
  -h, --help   : Show this help message and exit
  -t, --thread : Number of threads to use (default: 1)
  --CSD        : Vector dimension (required)
  --filter     : Filter type and params, e.g. "bgj1:0.31,8192" (required)
  --red        : Reducer type, e.g. "int8:1.0 int4:1.05 int1-512:200" (optional)
  --db         : Path to db file (optional)
```
