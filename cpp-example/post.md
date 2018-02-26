# Post

Even though PyTorch already provides a plethora of operations related to neural
networks, arbitrary tensor algebra, data wrangling and other purposes, you may
still find yourself in need of a more customized operation. For example, you
might want to use a novel activation function you found in a paper, or implement
an operation you developed as part of your research.

The easiest way of integrating such a custom operation in PyTorch is to write it
in Python by extending `Function` and `Module` as outlined [here](). This gives
you the full power of automatic differentiation (spares you from writing the
backward pass) as well as the usual expressiveness of Python. However, there may
be times when your operation is better implemented in C++. For example, your
code may need to be *really* fast because it is called very frequently in your
model or is very expensive even for few calls. Another plausible reason is that
it depends on or interacts with other C or C++ libraries. To address such cases,
PyTorch provides a straightforward mechanism of writing custom *C++ extensions*.

C++ extensions are a mechanism we have developed to allow users (you) to create
C++ operations defined *out-of-source*, i.e. separate from the PyTorch backend.
Note that this mechanism is *separate* from the way native PyTorch operations
are developed. However, once you have defined your operation as a C++ extension,
turning it into a native PyTorch function is largely a matter of boilerplate,
which you can tackle after the fact if you decide to contribute your extension
upstream.

## Motivation and Example

Let's walk through an example. If you are being chased or someone will fire you
if you don't get that op done by the end of the day, you can skip this section
and head straight to the implementation in the next section.

Let's say you've come up with a new kind of recurrent unit that you found to
have superior properties compared to the state of the art. This recurrent unit
is similar to an LSTM, but differs in that it lacks a *forget gate* and uses an
*Exponential Linear Unit* (ELU) as its internal activation function. Because
this unit never forgets, we'll call it *LLTM*, or *Long-Long-Term-Memory* unit.

The two ways in which LLTMs differ from vanilla LSTMs are significant enough
that we can't configure PyTorch's `LSTMCell` for our purposes, so we'll have to
create a custom cell. The first and easiest approach for this -- and likely in
all cases a good first step -- is to implement our desired functionality in
plain PyTorch with Python. For this, we need to subclass `torch.nn.Module` and
implement the forward pass of the LLTM. This would look something like this:

```py
import math
import torch
import torch.nn.functional as F

class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.Tensor(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_cell = state
        X = torch.cat([old_h, input], dim=1)

        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = F.linear(X, self.weights, self.bias)
        # Split the combined gate weight matrix into its components.
        gates = gate_weights.chunk(3, dim=1)

        input_gate = F.sigmoid(gates[0])
        output_gate = F.sigmoid(gates[1])
        # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])

        # Compute the new cell state.
        new_cell = old_cell + candidate_cell * input_gate
        # Compute the new hidden state and output.
        new_h = F.tanh(new_cell) * output_gate

        return new_h, new_cell
```

which we could then use as expected:

```py
import torch
from torch.autograd import Variable

X = Variable(torch.randn(batch_size, input_features))
h = Variable(torch.randn(batch_size, state_size))
C = Variable(torch.randn(batch_size, state_size))

rnn = LLTM(input_features, state_size)

new_h, new_C = rnn(X, (h, C))
```

Naturally, if at all possible and plausible, you should use this approach to
extend PyTorch. Since PyTorch has highly optimized implementations of its
operations for CPU *and* GPU, powered by libraries such as NVIDIA cuDNN, Intel
MKL or NNPACK, PyTorch code like above will often be fast enough. However, we
can also see why, under certain circumstances, there is room for further
performance improvements. The most obvious reason is that PyTorch has no
knowledge of the *algorithm* you are implementing. It knows only of the
individual operations you use to compose your algorithm. As such, PyTorch must
execute your operations individually, one after the other. Since each individual
call to the implementation (or *kernel*) of an operation, which may involve
launch of a CUDA kernel, has a certain amount of overhead, this overhead may
become significant across many function calls. Furthermore, the Python
interpreter that is running our code can itself slow down our program
significantly.

A definite method of speeding things up is therefore to rewrite parts in C++ (or
CUDA) and *fuse* particular groups of operations. Fusing means combining the
implementations of many kernels into a single kernel, which profits from fewer
kernel launches as well as other optimizations we can perform with increased
visibility of the global flow of data.

Let's see how we can use C++ extensions to implement a *fused* version of the
LLTM. We'll begin by writing it in plain C++, using the *ATen* library that
powers much of PyTorch's backend, and see how easily it lets us translate our
Python code. We'll then speed things up even more by moving parts of model to
CUDA kernels and benefit from the massive parallelism GPUs provide.

## Writing a C++ Extension

C++ extensions come in two flavors: They can be built "ahead of time" with
`setuptools`, or "just in time" via `torch.utils.cpp_extension.load`. We'll look
at the first approach first and discuss the latter later.

### Building With `setuptools`

For the "ahead of time" flavor, we build our C++ extension by writing a
`setup.py` script that uses setuptools to compile our C++ code. For the LLTM, it
looks as simple as this:

```py
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='lltm',
      ext_modules=[CppExtension('lltm', ['lltm.cpp'])]
      cmdclass={'build_ext': BuildExtension})
```

In this code, `CppExtension` is a convenience wrapper around
`setuptools.Extension` that passes the correct include paths and sets the
language of the extension to C++. The equivalent vanilla `setuptools` code would
simply be:

```
setuptools.Extension(
   name='lltm',
   sources=['lltm.cpp'],
   include_dirs=torch.utils.cpp_extension.include_paths(),
   language='c++')
```

`BuildExtension` performs a number of required configuration steps and checks
and also manages mixed compilation in the case of mixed C++/CUDA extensions. And
that's all we really need to know about building C++ extensions for now! More
advanced documentation is available [here]().

### Writing An Op

Finally, the code for our C++ extension goes into `lltm.cpp`. Let's start
implementing the LLTM in C++! One function we'll require for the backward pass
is the derivative of the sigmoid. This is a small enough piece of code to
discuss the overall environment that is available to us when writing C++
extensions:

```c++
#include <torch/torch.h>
#include <iostream>

at::Tensor d_sigmoid(at::Tensor z) {
  auto s = at::sigmoid(z);
  return (1 - s) * s;
}
```

`torch/torch.h` is the one-stop header to include all the necessary PyTorch bits
to write C++ extensions. It includes:

- The ATen library, which is our primary API for tensor computation,
- Pybind11, which is how we create Python bindings for our C++ code,
- Headers that manage the details of interaction between ATen and pybind11.

The implementation of `d_sigmoid` shows how to use the ATen API. PyTorch's
tensor and variable interface is generated automatically from the ATen library,
so we can more or less translate our Python implementation 1:1 into C++.
`at::Tensor` will be our primary datatype for all computations. Its full API can
be inspected [here](doc/Tensor.h). Notice also that we can include `<iostream>`
or *any other C or C++ header* -- we have the full power of C++11 (including
`auto`) at our disposal.

#### Forward Pass

Next we can port our entire forward pass to C++:

```c++
#include <vector>

std::vector<at::Tensor> lltm_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell) {
  auto X = at::cat({old_h, input}, /*dim=*/1);

  auto gate_weights = at::addmm(bias, X, weights.transpose(0, 1));
  auto gates = gate_weights.chunk(3, /*dim=*/1);

  auto input_gate = at::sigmoid(gates[0]);
  auto output_gate = at::sigmoid(gates[1]);
  auto candidate_cell = at::elu(gates[2], /*alpha=*/1.0);

  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = at::tanh(new_cell) * output_gate;

  return {new_h,
          new_cell,
          input_gate,
          output_gate,
          candidate_cell,
          X,
          gate_weights};
}
```

#### Backward Pass

At this time, PyTorch's C++ interface does not support automatic
differentiation. This is something the PyTorch team is working on, but it is not
available yet. As such, we have to also implement the backward pass of our LLTM,
which computes the derivative of the loss with respect to each input of the
forward pass. Ultimately, we will plop both the forward and backward function
into a `torch.nn.Function` to create a nice Python binding. The backward
function is slightly more involved, so we'll not dig deeper into the code (if
you are interested, Alex Graves' thesis is a good read for more information on
this: http://www.cs.toronto.edu/~graves/phd.pdf):

```c++
// tanh'(z) = 1 - tanh^2(z)
at::Tensor d_tanh(at::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
at::Tensor d_elu(at::Tensor z, at::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<at::Tensor> lltm_backward(
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gate_weights,
    at::Tensor weights,
    at::Tensor old_cell) {
  auto d_output_gate = at::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, /*dim=*/1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
      at::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights).split(d_input_gate.size(1), /*dim=*/1);

  return {d_X[0], d_X[1], d_weights, d_bias, d_old_cell};
}
```

### Binding to Python

Once you have your operation written in C++ and ATen, you can use pybind11 to
bind your C++ functions or classes into Python in a very simple manner.
Questions or issues you have about this part of PyTorch C++ extensions will
largely be addressed by [pybind11
documentation](http://pybind11.readthedocs.io/en/master/).

For our extensions, the necessary binding code spans only four lines:

```c++
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}
```

One bit to note here is the macro `TORCH_EXTENSION_NAME`. The torch extension
build will define it as the name you give your extension in the `setup.py`
script. In this case, the value of `TORCH_EXTENSION_NAME` would be "lltm". This
is to avoid having to maintain the name of the extension in two places (the
build script and your C++ code), as a mismatch between the two can lead to nasty
and hard to track issues.

### Using Your Extension

We are now set to import our extension in PyTorch. At this point, your directory
structure could look something like this:

```
pytorch/
  lltm-extension/
    lltm.cpp
    setup.py
```

Now, run `python setup.py install` to build and install your extension. This
should look something like this:

```
running install
running bdist_egg
running egg_info
writing lltm.egg-info/PKG-INFO
writing dependency_links to lltm.egg-info/dependency_links.txt
writing top-level names to lltm.egg-info/top_level.txt
reading manifest file 'lltm.egg-info/SOURCES.txt'
writing manifest file 'lltm.egg-info/SOURCES.txt'
installing library code to build/bdist.linux-x86_64/egg
running install_lib
running build_ext
building 'lltm' extension
gcc -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I~/local/miniconda/lib/python3.6/site-packages/torch/lib/include -I~/local/miniconda/lib/python3.6/site-packages/torch/lib/include/TH -I~/local/miniconda/lib/python3.6/site-packages/torch/lib/include/THC -I~/local/miniconda/include/python3.6m -c lltm.cpp -o build/temp.linux-x86_64-3.6/lltm.o -DTORCH_EXTENSION_NAME=lltm -std=c++11
cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
g++ -pthread -shared -B ~/local/miniconda/compiler_compat -L~/local/miniconda/lib -Wl,-rpath=~/local/miniconda/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.6/lltm.o -o build/lib.linux-x86_64-3.6/lltm.cpython-36m-x86_64-linux-gnu.so
creating build/bdist.linux-x86_64/egg
copying build/lib.linux-x86_64-3.6/lltm_cuda.cpython-36m-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/egg
copying build/lib.linux-x86_64-3.6/lltm.cpython-36m-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/egg
creating stub loader for lltm.cpython-36m-x86_64-linux-gnu.so
byte-compiling build/bdist.linux-x86_64/egg/lltm.py to lltm.cpython-36.pyc
creating build/bdist.linux-x86_64/egg/EGG-INFO
copying lltm.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
copying lltm.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying lltm.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying lltm.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
writing build/bdist.linux-x86_64/egg/EGG-INFO/native_libs.txt
zip_safe flag not set; analyzing archive contents...
__pycache__.lltm.cpython-36: module references __file__
creating 'dist/lltm-0.0.0-py3.6-linux-x86_64.egg' and adding 'build/bdist.linux-x86_64/egg' to it
removing 'build/bdist.linux-x86_64/egg' (and everything under it)
Processing lltm-0.0.0-py3.6-linux-x86_64.egg
removing '/data/users/psag/miniconda/lib/python3.6/site-packages/lltm-0.0.0-py3.6-linux-x86_64.egg' (and everything under it)
creating /data/users/psag/miniconda/lib/python3.6/site-packages/lltm-0.0.0-py3.6-linux-x86_64.egg
Extracting lltm-0.0.0-py3.6-linux-x86_64.egg to /data/users/psag/miniconda/lib/python3.6/site-packages
lltm 0.0.0 is already the active version in easy-install.pth

Installed ~/local/miniconda/lib/python3.6/site-packages/lltm-0.0.0-py3.6-linux-x86_64.egg
Processing dependencies for lltm==0.0.0
Finished processing dependencies for lltm==0.0.0
```

A small note on compilers: Due to ABI versioning issues, the compiler you use to
build your C++ extension must be *ABI-compatible* with the compiler PyTorch was
built. In practice, this means that you must use GCC version 4.9 and above. For
Ubuntu 16.04 and other more-recent Linux distributions, this should be the
default compiler already. On MacOS, you will have to download GCC (e.g. `brew
install gcc` will give you GCC 7 at the time of this writing). In the worst
case, you can build PyTorch from source with your compiler and then build the
extension with that same compiler.

Once your extension is built, you can simply import it in Python, using the name
you specified in your `setup.py` script. Just be sure to `import torch` first,
as this will resolve some symbols that the dynamic linker must see:

```py
In [1]: import torch
In [2]: import lltm
In [3]: lltm.forward
Out[3]: <function lltm.PyCapsule.forward>
```

If we call `help()` on the function or module, we can see that its signature
matches our C++ code:

```py
In[4] help(lltm.forward)
forward(...) method of builtins.PyCapsule instance
    forward(arg0: at::Tensor, arg1: at::Tensor, arg2: at::Tensor, arg3: at::Tensor, arg4: at::Tensor) -> List[at::Tensor]

    LLTM forward
```

Since we are now able to call our C++ functions from Python, we can wrap them
with `torch.nn.Function` and `torch.nn.Module` to make them first class citizens
of PyTorch:

```py
import math
import torch

# Our module!
import lltm

class LLTMFunction(torch.nn.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights, old_cell]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        d_old_h, d_input, d_weights, d_bias, d_old_cell = lltm.backward(
            grad_h, grad_cell, *ctx.saved_variables)
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.Tensor(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)
```

#### Performance Comparison

Now that we are able to use and call our C++ code from PyTorch, we can run a
small benchmark to see how much performance we gained from rewriting our op in
C++. We'll run the LLTM forwards and backwards a few times and measure the
duration:

```py
import torch

batch_size = 16
input_features = 32
state_size = 128

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)

rnn = LLTM(input_features, state_size)

forward = 0
backward = 0
for _ in range(100000):
    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    forward += time.time() - start

    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    backward += time.time() - start

print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5,
                                                       backward * 1e6/1e5))
```

If we run this code with the original LLTM we wrote in pure Python at the start
of this post, we get the following numbers (on my machine):

```sh
Forward: 349.335 us | Backward 443.523 us
```

and with our new C++ version:

```sh
Forward: 506.480 us | Backward 444.694 us
```

We can already see a significant speedup for the forward function (more than
30%). For the backward function a speedup is visible, albeit not major one. The
backward pass I wrote above was not particularly optimized and could definitely
be improved. Also, PyTorch's automatic differentiation engine can automatically
parallelize computation graphs, may use a more efficient flow of operations
overall, and is also implemented in C++, so it's expected to be fast.
Nevertheless, this is a good start.

#### Performance on GPU Devices

run on GPU

### JIT Compiling Extensions

Previously, I mentioned there were two ways of building C++ extensions: using
`setuptools` or just in time (JIT). Having covered the former, let's elaborate
on the latter. The JIT compilation mechanism provides you with a way of
compiling and loading your extensions on the fly by calling a simple function in
PyTorch's API called `torch.utils.cpp_extension.load`. For the LLTM, this would
look as simple as this:

```py
lltm = torch.utils.cpp_extension.load(name="lltm", sources=["lltm.cpp"])
```

Here, we provide the function with the same information as for `setuptools`. In
the background, this will do the following:

1. Create a temporary directory `/tmp/torch_extensions/lltm`,
2. Emit a [Ninja](https://ninja-build.org/) build file into that temporary directory,
3. Compile your source files into a shared library,
4. Import this shared library as a Python module.

In fact, if you pass `verbose=True` to `cpp_extension.load()`, you will be
informed about the process:

```
Using /tmp/torch_extensions as PyTorch extensions root...
Creating extension directory /tmp/torch_extensions/lltm...
Emitting ninja build file /tmp/torch_extensions/lltm/build.ninja...
Building extension module lltm...
Loading extension module lltm...
```

The resulting Python module will be exactly the same as produced by setuptools,
but removes the requirement of having to maintain a separate `setup.py` build
file. If your setup is more complicated and you do need the full power of
`setuptools`, you *can* write your own `setup.py` -- but in many cases this JIT
mechanism will do just fine. The first time you run through this line, it will
take some time, as the extension is compiling in the background. Since we use
the Ninja build system to build your sources, re-compilation is incremental and
thus re-loading the extension when you run your Python module a second time is
fast and has low overhead.

## Writing a Mixed C++/CUDA extension

To really take our implementation to the next level, we can parallelize parts of
our forward and backward pass with CUDA and run our operations on a GPU. For the
LLTM, this has the prospect of being particularly effective, as there are a
large number of pointwise operations in sequence, that can all be fused and
parallelized in a single CUDA kernel. Let's see how we could write such a CUDA
kernel and integrate it with PyTorch using this extension mechanism.
