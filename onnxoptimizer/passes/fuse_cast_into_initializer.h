/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

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

enum DataType {
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  STRING = 8,       // not supported
  BOOL = 9,         // not supported
  FLOAT16 = 10,     // not supported
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,   // not supported
  COMPLEX128 = 15,  // not supported
  BFLOAT16 = 16     // not supported
};

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
    case DataType::FLOAT:
      switch (oldTensor->elem_type()) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::INT32:
          copyData<float>(newTensor.floats(), oldTensor->data<int32_t>(), num_elements);
          break;
        case DataType::INT64:
          copyData<float>(newTensor.floats(), oldTensor->data<int64_t>(), num_elements);
          break;
        case DataType::DOUBLE:
          copyData<float>(newTensor.floats(), oldTensor->data<double>(), num_elements);
          break;
        case DataType::UINT32:
        case DataType::UINT64:
          copyData<float>(newTensor.floats(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;
    
    case DataType::UINT8:
      switch (oldTensor->elem_type()) {
        case DataType::FLOAT:
          copyData<uint8_t>(newTensor.int32s(), oldTensor->data<float>(), num_elements);
          break;
        case DataType::INT64:
          copyData<uint8_t>(newTensor.int32s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case DataType::DOUBLE:
          copyData<uint8_t>(newTensor.int32s(), oldTensor->data<double>(), num_elements);
          break;
        case DataType::UINT32:
        case DataType::UINT64:
          copyData<uint8_t>(newTensor.int32s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case DataType::INT8:
      switch (oldTensor->elem_type()) {
        case DataType::FLOAT:
          copyData<int8_t>(newTensor.int32s(), oldTensor->data<float>(), num_elements);
          break;
        case DataType::INT64:
          copyData<int8_t>(newTensor.int32s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case DataType::DOUBLE:
          copyData<int8_t>(newTensor.int32s(), oldTensor->data<double>(), num_elements);
          break;
        case DataType::UINT32:
        case DataType::UINT64:
          copyData<int8_t>(newTensor.int32s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case DataType::UINT16:
      switch (oldTensor->elem_type()) {
        case DataType::FLOAT:
          copyData<uint16_t>(newTensor.int32s(), oldTensor->data<float>(), num_elements);
          break;
        case DataType::INT64:
          copyData<uint16_t>(newTensor.int32s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case DataType::DOUBLE:
          copyData<uint16_t>(newTensor.int32s(), oldTensor->data<double>(), num_elements);
          break;
        case DataType::UINT32:
        case DataType::UINT64:
          copyData<uint16_t>(newTensor.int32s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case DataType::INT16:
      switch (oldTensor->elem_type()) {
        case DataType::FLOAT:
          copyData<int16_t>(newTensor.int32s(), oldTensor->data<float>(), num_elements);
          break;
        case DataType::INT64:
          copyData<int16_t>(newTensor.int32s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case DataType::DOUBLE:
          copyData<int16_t>(newTensor.int32s(), oldTensor->data<double>(), num_elements);
          break;
        case DataType::UINT32:
        case DataType::UINT64:
          copyData<int16_t>(newTensor.int32s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case DataType::INT32:
      switch (oldTensor->elem_type()) {
        case DataType::FLOAT:
          copyData<int32_t>(newTensor.int32s(), oldTensor->data<float>(), num_elements);
          break;
        case DataType::INT64:
          copyData<int32_t>(newTensor.int32s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case DataType::DOUBLE:
          copyData<int32_t>(newTensor.int32s(), oldTensor->data<double>(), num_elements);
          break;
        case DataType::UINT32:
        case DataType::UINT64:
          copyData<int32_t>(newTensor.int32s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case DataType::INT64:
      switch (oldTensor->elem_type()) {
        case DataType::FLOAT:
          copyData<int64_t>(newTensor.int64s(), oldTensor->data<float>(), num_elements);
          break;
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::INT32:
          copyData<int64_t>(newTensor.int64s(), oldTensor->data<int32_t>(), num_elements);
          break;
        case DataType::DOUBLE:
          copyData<int64_t>(newTensor.int64s(), oldTensor->data<double>(), num_elements);
          break;
        case DataType::UINT32:
        case DataType::UINT64:
          copyData<int64_t>(newTensor.int64s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case DataType::DOUBLE:
      switch (oldTensor->elem_type()) {
        case DataType::FLOAT:
          copyData<double>(newTensor.doubles(), oldTensor->data<float>(), num_elements);
          break;
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::INT32:
          copyData<double>(newTensor.doubles(), oldTensor->data<int32_t>(), num_elements);
          break;
        case DataType::INT64:
          copyData<double>(newTensor.doubles(), oldTensor->data<int64_t>(), num_elements);
          break;
        case DataType::UINT32:
        case DataType::UINT64:
          copyData<double>(newTensor.doubles(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case DataType::UINT32:
      switch (oldTensor->elem_type()) {
        case DataType::FLOAT:
          copyData<uint32_t>(newTensor.uint64s(), oldTensor->data<float>(), num_elements);
          break;
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::INT32:
          copyData<uint32_t>(newTensor.uint64s(), oldTensor->data<int32_t>(), num_elements);
          break;
        case DataType::INT64:
          copyData<uint32_t>(newTensor.uint64s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case DataType::DOUBLE:
          copyData<uint32_t>(newTensor.uint64s(), oldTensor->data<double>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case DataType::UINT64:
      switch (oldTensor->elem_type()) {
        case DataType::FLOAT:
          copyData<uint64_t>(newTensor.uint64s(), oldTensor->data<float>(), num_elements);
          break;
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::INT32:
          copyData<uint64_t>(newTensor.uint64s(), oldTensor->data<int32_t>(), num_elements);
          break;
        case DataType::INT64:
          copyData<uint64_t>(newTensor.uint64s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case DataType::DOUBLE:
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
