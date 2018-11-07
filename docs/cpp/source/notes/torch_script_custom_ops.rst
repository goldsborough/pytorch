Extending TorchScript with Custom C++ Operators
===============================================

The PyTorch 1.0 release introduced a new programming model to PyTorch called
`*TorchScript* <https://pytorch.org/docs/master/jit.html>`_. TorchScript is a
subset of the Python programming language which can be parsed, compiled and
optimized by the TorchScript compiler. Further, compiled TorchScript models have
the option of being serialized into an on-disk file format, which you can
subsequently load and run from pure C++ (or Python) to perform inference.

TorchScript supports a large subset of operations provided by the ``torch``
package, allowing you to express many kinds of complex models purely as a series
of tensor operations from PyTorch's "standard library". Nevertheless, there may
be times where you find yourself in need of extending TorchScript with a custom
C++ or CUDA function. While we recommend that you only resort to this option if
your idea cannot be expressed (efficiently enough) in Python (as a simple Python
function), we do provide a very friendly and simple interface for defining
custom C++ kernels using `ATen <https://pytorch.org/cppdocs/#aten>`_, PyTorch's
high performance C++ tensor library, and binding them into TorchScript. Once
bound, you will be able to embed these custom kernels (or "ops") into your
TorchScript model, and execute them both in Python and in their serialized form
directly in C++.

The following paragraphs give an example of writing a TorchScript custom op to
call into `OpenCV <https://www.opencv.org>`, a computer vision library written
in C++. We will discuss the API we provide to operate on tensors in C++, how to
efficiently convert them to third party tensor formats (in this case, OpenCV
``Mat``s), how to register your operator with the TorchScript runtime and
finally how to compile the operator and use it in Python for research or in C++
for inference purposes. We will also touch upon writing a CUDA kernel and
highlight differences to custom ops that purely use C++.

# Note on C++ extensions somewhere

Implementing the Custom Operator in C++
---------------------------------------

A common use case for extending PyTorch (or TorchScript) is calling into
third-party C++ libraries. You may find yourself lacking some operation for your
computer vision or NLP model (or another domain), and upon scavenging Google you
find an implementation in C++. The challenge of making this C++ implementation
usable in your PyTorch models in Python, and further embedding this operator in
your scripted or traced TorchScript models (for later inference), may seem
daunting -- but you'll see that it's not. For this tutorial, we'll be exposing
the `warpPerspective
<https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspective>`_
function, which applies a perspective transformation to an image, from OpenCV's
C++ API to TorchScript as a custom operator.

The first step is to write the implementation of our custom operator in C++.
Let's call the file for this implementation ``op.cpp``:

.. code-block:: cpp

  #include <opencv2/opencv.hpp>
  #include <torch/script.h>

  torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
    cv::Mat image_mat(/*rows=*/image.size(0),
                      /*cols=*/image.size(1),
                      /*type=*/CV_32FC1,
                      /*data=*/image.data<float>());
    cv::Mat warp_mat(/*rows=*/3,
                     /*cols=*/3,
                     /*type=*/CV_32FC1,
                     /*data=*/warp.data<float>());

    cv::Mat output_mat;
    cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{64, 64});

    return torch::from_blob(output.ptr<float>(), /*sizes=*/{64, 64}).clone();
  }

Fortunately, the code for this is quite short. At the top of the file, we
include the OpenCV2 C++ header file, ``opencv2/opencv.hpp``, alongside the
``torch/script.h`` header which exposes all the necessary goodies from PyTorch's
C++ API that we need to write custom TorchScript operators. Our function
``warp_perspective`` takes two arguments: an input ``image`` and the ``warp``
transformation matrix we wish to apply to the image. The type of these inputs is
``torch::Tensor``, PyTorch's tensor type in C++ (which is also the underlying
type of all tensors in Python). See `this note
<https://pytorch.org/cppdocs/notes/tensor_basics.html>`_ for more information
about *ATen*, the library that provides the ``Tensor`` class to PyTorch, and
`this tutorial <https://pytorch.org/cppdocs/notes/tensor_creation.html>`_ if you
are interested how to create new tensor objects in C++ yourself. The return type
of our ``warp_perspective`` function will also be a ``torch::Tensor``.

Inside of our function, the first thing we need to do is convert our PyTorch
(actually ATen) tensors to OpenCV matrices, as OpenCV's ``warpPerspective``
expects ``cv::Mat`` objects as inputs. Fortunately, there is a way to do this
**without copying any** data. In the first few lines,

.. code-block:: cpp

  cv::Mat image_mat(/*rows=*/image.size(0),
                    /*cols=*/image.size(1),
                    /*type=*/CV_32FC1,
                    /*data=*/image.data<float>());

we are calling `this constructor
<https://docs.opencv.org/trunk/d3/d63/classcv_1_1Mat.html#a922de793eabcec705b3579c5f95a643e>`_
of the OpenCV ``Mat`` class to convert our tensor to a ``Mat`` object. We pass
it the number of rows and columns of the original ``image`` tensor, the datatype
(which we'll fix as ``float32`` for this example), and finally a raw pointer to
the underlying data -- a ``float*``. What is special about this constructor of
the ``Mat`` class is that it does not copy the input data. Instead, it will
simply reference this memory for all operations performed on the ``Mat``. If an
in-place operation is performed on the ``image_mat``, this will be reflected in
the original ``image`` tensor (and vice-versa). This allows us to call
subsequent OpenCV routines with the library's native matrix type, even though
we're actually storing the data in a PyTorch Tensor.

Binding the Custom Operator into Python
---------------------------------------

Using the TorchScript Custom Operator in Python
-----------------------------------------------

Using the TorchScript Custom Operator in C++
--------------------------------------------

Writing a Custom Operator that uses CUDA
----------------------------------------

Conclusion
----------
