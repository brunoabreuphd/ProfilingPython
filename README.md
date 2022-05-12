# Profiling Python Applications

This repository contains examples on how to profile Python applications using [`cProfile`](https://docs.python.org/3/library/profile.html#module-cProfile) and [`snakeviz`](http://jiffyclub.github.io/snakeviz/) modules.

## Creating the environment
To reproduce the same environment, I suggest using Anaconda. If you have it installed, you can use [environment.yml](./environment.yml) to create it using
```bash
conda create --name <EnvName> --file environment.yml
```
and activate it with
```bash
conda activate <EnvName>
```

## Profiling
In each example, you will find a `profile.sh` script that will run the Python code with the [`cProfile`](https://docs.python.org/3/library/profile.html#module-cProfile) module on. Running this script will generate a file `example.prof`, that contains the profiling data.


## Visualization
Even though it is possible to get statistics directly from [`cProfile`](https://docs.python.org/3/library/profile.html#module-cProfile), a great way to visualize the profiling results is with [`snakeviz`](http://jiffyclub.github.io/snakeviz/). It's very easy to use. For each example, you will find a `visualize.sh` script that, when run, will launch [`snakeviz`](http://jiffyclub.github.io/snakeviz/) in a browser tab. Have fun!
