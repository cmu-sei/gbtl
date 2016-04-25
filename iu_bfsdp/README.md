This is our implementation of BFS using dynamic parallelism.

You'll need CUDA 7 and a compiler that supports C++11 to run this. Once you have
that, just run `make` to compile it.

Once you've compiled it, you can run the benchmark as follows:

```
./main <scale> <edge factor> <source vertex> <device>
```
