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

For following along hands-on, you need
* laptop or desktop with internet access.
* a system set up so you can connect to an HPC system, an account on an HPC
  system (e.g., VSC, CECI, ...), compute credits if that is required to run
  jobs on the HPC system if you want to use an HPC system;
* a Python environment that can run Jupyter Lab if you want to use your own system
  (this will only work if your system has an NVIDIA GPU and the required software installed);
* access to Google Colaboratory if you prefer not to install software.


## Levels

* Introductory: 20 %
* Intermediate: 40 %
* Advanced: 40 %


## Trainer(s)

  * Geert Jan Bex ([geertjan.bex@uhasselt.be](mailto:geertjan.bex@uhasselt.be))
