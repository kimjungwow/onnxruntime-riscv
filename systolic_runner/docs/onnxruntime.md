# ONNX Runtime

## Overview

For a general overview of ONNX runtime please refer to [HighLevelDesign.md](/docs/HighLevelDesign.md)

This fork:

* Adds a naive CPU-only fallback to existing SIMD code in mlas (Microsoft Linear Algebra Subroutines) which allows for compilation and use on risc-v targets (and potentially other platforms).

* Adds a Systolic backend to onnxruntime, described subsequently in more detail

## Systolic Backend

The added systolic backend is modeled off of the existing CPU backend, in that it shares the same routines for memory allocation and placement.

The systolic backend supports ONNX's QLinearConv and QLinearMatmul operators. In addition to these, we also define our own custom schema and fusion set which is intended to be used with models produced by the quantization script

## Custom Operators

We register and handle `QLinearConv_nhwc`, a variant of QLinearConv that expects its input tensors to be in NHWC format and produces an output with that layout. 

Note that the weights for this operator must be in a funky format: in NHWC format, with contents already pre-transposed and ready to be dotted with the output of im2col. Particularly, the weights we multiply by are given by the `M / groups` x `kernel_h * kernel_w * C / groups` matrix starting at `&weights[0] + group_id * (M / groups) kernel_h * kernel_w * C / groups`. For more information, refer to the `_get_weights_in_transposed_nhwc_format` method in [quantizer.py](/systolic_runner/quantization/quantize.py)

## Fusion

Because systolic can handle a fused matmul + relu, we perform operator fusion on QLinearConv followed by Relu.