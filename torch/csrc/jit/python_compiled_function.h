#pragma once

#include <Python.h>

namespace torch { namespace jit { namespace python {

void initCompilerMixin(PyObject *module);

}}}
