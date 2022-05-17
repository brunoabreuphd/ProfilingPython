# Profiling Python Applications

This repository contains examples on how to profile Python applications using [`cProfile`](https://docs.python.org/3/library/profile.html#module-cProfile), [`snakeviz`](http://jiffyclub.github.io/snakeviz/), [`line_profiler`](https://github.com/pyutils/line_profiler), and [`memory_profiler`](https://pypi.org/project/memory-profiler/) modules.

## Environment setup
To reproduce the same environment, I suggest using [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) as your package manager. If you have it installed, you can use [environment.yml](./environment.yml) to create it using
```bash
conda env create -f environment.yml
```
and activate it with
```bash
conda activate profiling
```

## Time-based Profiling
Time-based profiling allows you to see how much time your application spends in each one of its components. 

### Application overview
We use the [`cProfile`](https://docs.python.org/3/library/profile.html#module-cProfile) module to profile an entire Python script. In each example folder, you will find a `time/app-overview` folder that contains the relevant code, along with a `profile.sh` script that will run the Python code with [`cProfile`](https://docs.python.org/3/library/profile.html#module-cProfile) on. This script will generate a file `example.prof`, that contains the profiling data.

#### Visualization
Even though it is possible to get statistics directly from [`cProfile`](https://docs.python.org/3/library/profile.html#module-cProfile), a great way to visualize the profiling results is with [`snakeviz`](http://jiffyclub.github.io/snakeviz/). It's very easy to use. For each example, you will find a `visualize.sh` script that, when run, will launch [`snakeviz`](http://jiffyclub.github.io/snakeviz/) in a browser tab. Below is how a typical result looks:

![image](https://user-images.githubusercontent.com/84105092/168884846-dd00fb5b-66f8-43f9-80e3-2db988e120f9.png)


### Line-by-line profile
Once you spotted what functions, methods or routines are consuming most of the time in your application, you may want to dig deeper into it to see exactly what instructions under each of them are the hot ones. For each example, in `time/line-by-line`, we use [`line_profiler`](https://github.com/pyutils/line_profiler) for that, which requires decorating the target function with `@profile`. The `profile.sh` script calls the relevant binary (`kernprof`) to generate the profiling data, which can then be visualized with the `visualize.sh` script. A typical output is:

![image](https://user-images.githubusercontent.com/84105092/168885257-e44b60fc-03f3-413c-83fc-a1920678e999.png)


## Memory-based Profiling
Understanding your Python application in terms of time is definitely an important step, but to characterize your application workload better, we also need to understand how it uses memory. 

### Application overview
We use the [`memory_profiler`](https://pypi.org/project/memory-profiler/) module to get an overview of how much memory a Python script is using as a function of time. For each example, the `memory/app-overview` folder contains the code to be profiled and a `profile.sh` script that uses the relevant binary (`mprof`) to generate the profiling data, which can be visualized using the `visualize.sh` script. A typical output is:

![image](https://user-images.githubusercontent.com/84105092/168886421-18e657d5-f81b-4c82-a54c-2265e06ba901.png)

### Line-by-line profile
We can also target individual functions with the `@profile` decorator. [`memory_profiler`](https://pypi.org/project/memory-profiler/) will then show the amount of memory that the process associated to the Python interpreter is using as your code evolves, line by line. For each example, under `memory/line-by-line`, the `profile.sh` script runs the profiler and shows the results. A typical output is:

![image](https://user-images.githubusercontent.com/84105092/168887141-c6379677-7cb1-4387-9349-772dd9ba145e.png)


## Jupyter notebooks
Profiling Jupyter notebooks directly involves jumping through some hoops. The simplest alternative is to copy the content of your cells into a Python script. It possible to get the same effect with the [`nbconvert`](https://nbconvert.readthedocs.io/en/latest/index.html) module:

```bash
jupyter nbconvert <YourNB>.ipynb --to script
```

which will generate a `<YourNB>.py` script. Sometimes it looks quite ugly, though.
