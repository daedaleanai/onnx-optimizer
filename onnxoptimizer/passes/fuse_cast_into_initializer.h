/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// If a cast is applied to an nitializer, this pass creates a new initializer with the cast operation already applied
// and replaces the Cast node with the updated initializer.
//
// Before:
//   A is in the initializer list
//   X = Cast(A)
//   Z = X + Y
// After:
//   B is in the initializer list
//   Z = B + Y
//

namespace ONNX_NAMESPACE {
namespace optimization {

namespace {

template <typename T, typename A, typename B>
void copyData(std::vector<A>& dst, const B* src, const size_t len) {
    dst.reserve(len);
    for (size_t i=0; i<len; ++i) {
        dst.push_back(static_cast<T>(src[i]));
    }
}

}  // namespace

struct FuseCastIntoInitializer final : public PredicateBasedPass {
  explicit FuseCastIntoInitializer()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Partial,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_cast_into_initializer";
  }

  bool patternMatchPredicate(Node *node) override {
    // Matches on Cast nodes that have an initializer (kParam) as their input and the initializer does not already
    // have the type specified in the "to" attribute (kto) of the Cast node. If the latter is the case this is an nop
    // cast that will be handled by the EliminateNopCast pass.
    return node->kind() == kCast && node->input()->node()->kind() == kParam && node->input()->elemType() != node->i(kto);
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    const Tensor* oldTensor = nullptr;
    for (auto &initializer : graph.initializers()) {
        if (initializer.hasName() && initializer.name() == node->input()->uniqueName()) {
            oldTensor = &initializer;
            break;
        }
    }

    if (oldTensor == nullptr) {
        return false;
    }

    Tensor newTensor;
    newTensor.elem_type() = node->i(kto);
    newTensor.sizes() = oldTensor->sizes();
    size_t num_elements = oldTensor->sizes().empty() ? 1 :num_elements = oldTensor->size_from_dim(0); 

    switch (newTensor.elem_type()) {
    case TensorProto_DataType_FLOAT:
      switch (oldTensor->elem_type()) {
        case TensorProto_DataType_UINT8:
        case TensorProto_DataType_INT8:
        case TensorProto_DataType_UINT16:
        case TensorProto_DataType_INT16:
        case TensorProto_DataType_INT32:
          copyData<float>(newTensor.floats(), oldTensor->data<int32_t>(), num_elements);
          break;
        case TensorProto_DataType_INT64:
          copyData<float>(newTensor.floats(), oldTensor->data<int64_t>(), num_elements);
          break;
        case TensorProto_DataType_DOUBLE:
          copyData<float>(newTensor.floats(), oldTensor->data<double>(), num_elements);
          break;
        case TensorProto_DataType_UINT32:
        case TensorProto_DataType_UINT64:
          copyData<float>(newTensor.floats(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;
    
    case TensorProto_DataType_UINT8:
      switch (oldTensor->elem_type()) {
        case TensorProto_DataType_FLOAT:
          copyData<uint8_t>(newTensor.int32s(), oldTensor->data<float>(), num_elements);
          break;
        case TensorProto_DataType_INT64:
          copyData<uint8_t>(newTensor.int32s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case TensorProto_DataType_DOUBLE:
          copyData<uint8_t>(newTensor.int32s(), oldTensor->data<double>(), num_elements);
          break;
        case TensorProto_DataType_UINT32:
        case TensorProto_DataType_UINT64:
          copyData<uint8_t>(newTensor.int32s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case TensorProto_DataType_INT8:
      switch (oldTensor->elem_type()) {
        case TensorProto_DataType_FLOAT:
          copyData<int8_t>(newTensor.int32s(), oldTensor->data<float>(), num_elements);
          break;
        case TensorProto_DataType_INT64:
          copyData<int8_t>(newTensor.int32s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case TensorProto_DataType_DOUBLE:
          copyData<int8_t>(newTensor.int32s(), oldTensor->data<double>(), num_elements);
          break;
        case TensorProto_DataType_UINT32:
        case TensorProto_DataType_UINT64:
          copyData<int8_t>(newTensor.int32s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case TensorProto_DataType_UINT16:
      switch (oldTensor->elem_type()) {
        case TensorProto_DataType_FLOAT:
          copyData<uint16_t>(newTensor.int32s(), oldTensor->data<float>(), num_elements);
          break;
        case TensorProto_DataType_INT64:
          copyData<uint16_t>(newTensor.int32s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case TensorProto_DataType_DOUBLE:
          copyData<uint16_t>(newTensor.int32s(), oldTensor->data<double>(), num_elements);
          break;
        case TensorProto_DataType_UINT32:
        case TensorProto_DataType_UINT64:
          copyData<uint16_t>(newTensor.int32s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case TensorProto_DataType_INT16:
      switch (oldTensor->elem_type()) {
        case TensorProto_DataType_FLOAT:
          copyData<int16_t>(newTensor.int32s(), oldTensor->data<float>(), num_elements);
          break;
        case TensorProto_DataType_INT64:
          copyData<int16_t>(newTensor.int32s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case TensorProto_DataType_DOUBLE:
          copyData<int16_t>(newTensor.int32s(), oldTensor->data<double>(), num_elements);
          break;
        case TensorProto_DataType_UINT32:
        case TensorProto_DataType_UINT64:
          copyData<int16_t>(newTensor.int32s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case TensorProto_DataType_INT32:
      switch (oldTensor->elem_type()) {
        case TensorProto_DataType_FLOAT:
          copyData<int32_t>(newTensor.int32s(), oldTensor->data<float>(), num_elements);
          break;
        case TensorProto_DataType_INT64:
          copyData<int32_t>(newTensor.int32s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case TensorProto_DataType_DOUBLE:
          copyData<int32_t>(newTensor.int32s(), oldTensor->data<double>(), num_elements);
          break;
        case TensorProto_DataType_UINT32:
        case TensorProto_DataType_UINT64:
          copyData<int32_t>(newTensor.int32s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case TensorProto_DataType_INT64:
      switch (oldTensor->elem_type()) {
        case TensorProto_DataType_FLOAT:
          copyData<int64_t>(newTensor.int64s(), oldTensor->data<float>(), num_elements);
          break;
        case TensorProto_DataType_UINT8:
        case TensorProto_DataType_INT8:
        case TensorProto_DataType_UINT16:
        case TensorProto_DataType_INT16:
        case TensorProto_DataType_INT32:
          copyData<int64_t>(newTensor.int64s(), oldTensor->data<int32_t>(), num_elements);
          break;
        case TensorProto_DataType_DOUBLE:
          copyData<int64_t>(newTensor.int64s(), oldTensor->data<double>(), num_elements);
          break;
        case TensorProto_DataType_UINT32:
        case TensorProto_DataType_UINT64:
          copyData<int64_t>(newTensor.int64s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case TensorProto_DataType_DOUBLE:
      switch (oldTensor->elem_type()) {
        case TensorProto_DataType_FLOAT:
          copyData<double>(newTensor.doubles(), oldTensor->data<float>(), num_elements);
          break;
        case TensorProto_DataType_UINT8:
        case TensorProto_DataType_INT8:
        case TensorProto_DataType_UINT16:
        case TensorProto_DataType_INT16:
        case TensorProto_DataType_INT32:
          copyData<double>(newTensor.doubles(), oldTensor->data<int32_t>(), num_elements);
          break;
        case TensorProto_DataType_INT64:
          copyData<double>(newTensor.doubles(), oldTensor->data<int64_t>(), num_elements);
          break;
        case TensorProto_DataType_UINT32:
        case TensorProto_DataType_UINT64:
          copyData<double>(newTensor.doubles(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case TensorProto_DataType_UINT32:
      switch (oldTensor->elem_type()) {
        case TensorProto_DataType_FLOAT:
          copyData<uint32_t>(newTensor.uint64s(), oldTensor->data<float>(), num_elements);
          break;
        case TensorProto_DataType_UINT8:
        case TensorProto_DataType_INT8:
        case TensorProto_DataType_UINT16:
        case TensorProto_DataType_INT16:
        case TensorProto_DataType_INT32:
          copyData<uint32_t>(newTensor.uint64s(), oldTensor->data<int32_t>(), num_elements);
          break;
        case TensorProto_DataType_INT64:
          copyData<uint32_t>(newTensor.uint64s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case TensorProto_DataType_DOUBLE:
          copyData<uint32_t>(newTensor.uint64s(), oldTensor->data<double>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case TensorProto_DataType_UINT64:
      switch (oldTensor->elem_type()) {
        case TensorProto_DataType_FLOAT:
          copyData<uint64_t>(newTensor.uint64s(), oldTensor->data<float>(), num_elements);
          break;
        case TensorProto_DataType_UINT8:
        case TensorProto_DataType_INT8:
        case TensorProto_DataType_UINT16:
        case TensorProto_DataType_INT16:
        case TensorProto_DataType_INT32:
          copyData<uint64_t>(newTensor.uint64s(), oldTensor->data<int32_t>(), num_elements);
          break;
        case TensorProto_DataType_INT64:
          copyData<uint64_t>(newTensor.uint64s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case TensorProto_DataType_DOUBLE:
          copyData<uint64_t>(newTensor.uint64s(), oldTensor->data<double>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    default:
        return false;
    }
    
    Value* newValue = graph.addInitializerAndInput(newTensor);
    if (!tryReplacingAllUsesWith(node->output(), newValue)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
