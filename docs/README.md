The Python programming language is increasingly popular.  It is a
versatile language for general purpose programming and accessible
for novice programmers.  However, it is also increasingly used for
scientific computing and can be used to develop code that runs on
GPGPUs.  Additionally, a number of libraries that are commonly used
in scientific computing, data science and machine learning can use
GPGPUs to improve performance.


## Learning outcomes

When you complete this training you will

  * have an understanding of the architecture and features of GPGPUs,
  * be able to transfer data between the host and the GPGPU device,
  * be able to do linear algebra computations on GPGPUs using
    scikit-cuda,
  * be able to generate random numbers on a GPGPU using curand,
  * be able to define your own kernels to run on GPGPUs,
  * use numba to generate kernels to run on GPGPUs,
  * run machine learning algorithms on GPGPUs,
  * speed up data science tasks using Rapids.


## Schedule

Total duration: 4 hours.

  | Subject                                     | Duration |
  |---------------------------------------------|----------|
  | introduction and motivation                 |  5 min.  |
  | GPGPU architecture and features             | 30 min.  |
  | moving data between host and device         | 15 min.  |
  | linear algebra on GPGPUs                    | 20 min.  |
  | Numba and concepts of GPU programming       | 90 min.  |
  | coffee break                                | 10 min.  |
  | writing your own kernels                    | 60 min.  |
  | data science with Rapids                    | 30 min.  |
  | wrap up                                     | 10 min.  |


## Training materials

Slides are available in the
 [GitHub repository](https://github.com/gjbex/Python-on-GPUs),
as well as example code and hands-on material.


## Target audience

This training is for you if you speed up your Python by using GPUs.


## Prerequisites

You will need experience programming in Python.  This is not a training that starts
from scratch.  Some familiarity with numpy is required as well.

If you plan to do Python GPU programming in a Linux or HPC environment (and you should), then familiarity with these environments is required as well.

More concretely, participants should already be comfortable with the following:

* running Python code in Jupyter or from the command line;
* variables, numbers, strings, booleans, and basic containers such as lists
  and dictionaries;
* `if`/`else` statements, `for` loops, and writing simple functions;
* importing modules and reading short Python scripts without needing every line
  explained;
* basic NumPy array operations such as creating arrays, reshaping them,
  slicing them, and applying vectorized computations;
* basic linear algebra with NumPy, for example matrix multiplication and
  reductions such as sums or means;
* reading and making small changes to short numerical Python scripts or
  notebooks;
* basic familiarity with Linux or HPC workflows if you want to run the code on
  remote GPU systems.

You do not need prior experience with PyCUDA, CuPy, cuPyNumeric, Numba GPU
programming, cuRAND, RAPIDS, or custom CUDA kernels. Those are part of the
training itself.

### Quick self-assessment

If you can do most of the tasks below without looking up basic Python syntax,
you are likely ready for this training.

* create a NumPy array, compute a slice from it, and calculate its sum or mean;
* multiply two matrices with NumPy and interpret the shape of the result;
* read a short script that imports `numpy`, builds arrays, and times a
  computation;
* write a function that applies a simple numerical formula to all elements of
  an array;
* explain the difference between a Python loop over elements and a vectorized
  array operation at a high level;
* make a small change to an example notebook or script and run it again;
* read a short traceback or runtime error and identify roughly where the
  problem occurred;
* run a Python script from the command line with one or two arguments.

If several of these items still feel difficult, the training will probably move
too fast. In that case, it is better to first refresh basic Python and NumPy
programming before taking this training.

### Software and access requirements

For following along hands-on, you need
* laptop or desktop with internet access.
* a system set up so you can connect to an HPC system, an account on an HPC
  system (e.g., VSC, CECI, ...), compute credits if that is required to run
  jobs on the HPC system if you want to use an HPC system;
* a Python environment that can run Jupyter Lab if you want to use your own system
  (this will only work if your system has an NVIDIA GPU and the required software installed);
* access to Google Colaboratory if you prefer not to install software.


## Level of the Material

For participants who already have basic Python and NumPy experience, the material in this training is approximately

* Introductory: 10 %
* Intermediate: 35 %
* Advanced: 55 %

These percentages describe the level of the GPU programming and accelerated
computing topics covered in the training, not the required entry level in
Python itself.


## Trainer(s)

  * Geert Jan Bex ([geertjan.bex@uhasselt.be](mailto:geertjan.bex@uhasselt.be))
