// include/tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

// Tensor structure
// ---------------------------------------------
typedef struct {
  float *data;   // flattened data
  float *grad;   // flattened gradients (data shape)
  int *shape;    // array of dimensions
  size_t ndim;   // number of dimensions
  size_t size;   // total number of elements (product of shape)
} Tensor;

// Constructors & Destructors
// ---------------------------------------------
Tensor *tensor_create(int *shape, size_t ndim); // allocates data + grad
Tensor *tensor_zeros(int *shape, size_t ndim);  // zero initialised
Tensor *tensor_rand(int *shape, size_t ndim);   // random initialised
void tensor_free(Tensor *t);                    // deallocates

// Utility functions
// ---------------------------------------------
size_t tensor_index(Tensor *t, int *idxs);      // converts multi-index to flat index
void tensor_print(Tensor *t);

// Tensor operations (Forward)
// ---------------------------------------------
Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_sub(Tensor *a, Tensor *b);
Tensor *tensor_mul(Tensor *a, Tensor *b); // element-wise
Tensor *tensor_div(Tensor *a, Tensor *b); // element-wise
Tensor *tensor_dot(Tensor *a, Tensor *b); // matrix multiplication (2D tensors)
Tensor *tensor_sum(Tensor *a, int axis);  // sum along axis (-1 for all)
Tensor *tensor_mean(Tensor *a, int axis); // mean along axis (-1 for all)

#endif
