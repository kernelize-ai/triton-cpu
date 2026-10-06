// RUN: triton-opt %s -split-input-file --tritoncpu-rewrite-tensor-desc-args | FileCheck %s

// The descriptor is passed as its base pointer and rebuilt from the shape and
// stride arguments the frontend adds after it.
// CHECK-LABEL: tt.func public @kernel
// CHECK-SAME: (%[[BASE:.*]]: !tt.ptr<f32>, %[[S0:.*]]: i32, %[[S1:.*]]: i32, %[[T0:.*]]: i64, %[[T1:.*]]: i64, %[[N:.*]]: i32)
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]], [%[[S0]], %[[S1]]], [%[[T0]], %[[T1]]] : <f32>, <32x16xf32>
// CHECK: tt.descriptor_load %[[DESC]]
tt.func public @kernel(%desc: !tt.tensordesc<32x16xf32>, %desc.shape.0: i32, %desc.shape.1: i32, %desc.stride.0: i64, %desc.stride.1: i64, %n: i32) -> tensor<32x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.descriptor_load %desc[%n, %c0_i32] : !tt.tensordesc<32x16xf32> -> tensor<32x16xf32>
  tt.return %0 : tensor<32x16xf32>
}

// -----

// Only kernel arguments are rewritten: a non-kernel function receives its
// descriptors from its callers.
// CHECK-LABEL: tt.func private @helper
// CHECK-SAME: (%{{.*}}: !tt.tensordesc<32x16xf32>,
// CHECK-NOT: tt.make_tensor_descriptor
tt.func private @helper(%desc: !tt.tensordesc<32x16xf32>, %desc.shape.0: i32, %desc.shape.1: i32, %desc.stride.0: i64, %desc.stride.1: i64, %n: i32) -> tensor<32x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.descriptor_load %desc[%n, %c0_i32] : !tt.tensordesc<32x16xf32> -> tensor<32x16xf32>
  tt.return %0 : tensor<32x16xf32>
}
