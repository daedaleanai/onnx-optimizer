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

template <typename A, typename B>
void copyData(std::vector<A>& dst, const B* src, const size_t len) {
    dst.reserve(len);
    for (size_t i=0; i<len; ++i) {
        dst.push_back(src[i]);
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
    case 1:
      switch (oldTensor->elem_type()) {
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
          copyData(newTensor.floats(), oldTensor->data<int32_t>(), num_elements);
          break;
        case 7:
          copyData(newTensor.floats(), oldTensor->data<int64_t>(), num_elements);
          break;
        case 11:
          copyData(newTensor.floats(), oldTensor->data<double>(), num_elements);
          break;
        case 12:
        case 13:
          copyData(newTensor.floats(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;
    
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
      switch (oldTensor->elem_type()) {
        case 1:
          copyData(newTensor.int32s(), oldTensor->data<float>(), num_elements);
          break;
        case 7:
          copyData(newTensor.int32s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case 11:
          copyData(newTensor.int32s(), oldTensor->data<double>(), num_elements);
          break;
        case 12:
        case 13:
          copyData(newTensor.int32s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case 7:
      switch (oldTensor->elem_type()) {
        case 1:
          copyData(newTensor.int64s(), oldTensor->data<float>(), num_elements);
          break;
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
          copyData(newTensor.int64s(), oldTensor->data<int32_t>(), num_elements);
          break;
        case 11:
          copyData(newTensor.int64s(), oldTensor->data<double>(), num_elements);
          break;
        case 12:
        case 13:
          copyData(newTensor.int64s(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case 11:
      switch (oldTensor->elem_type()) {
        case 1:
          copyData(newTensor.doubles(), oldTensor->data<float>(), num_elements);
          break;
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
          copyData(newTensor.doubles(), oldTensor->data<int32_t>(), num_elements);
          break;
        case 7:
          copyData(newTensor.doubles(), oldTensor->data<int64_t>(), num_elements);
          break;
        case 12:
        case 13:
          copyData(newTensor.doubles(), oldTensor->data<uint64_t>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    case 12:
    case 13:
      switch (oldTensor->elem_type()) {
        case 1:
          copyData(newTensor.uint64s(), oldTensor->data<float>(), num_elements);
          break;
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
          copyData(newTensor.uint64s(), oldTensor->data<int32_t>(), num_elements);
          break;
        case 7:
          copyData(newTensor.uint64s(), oldTensor->data<int64_t>(), num_elements);
          break;
        case 11:
          copyData(newTensor.uint64s(), oldTensor->data<double>(), num_elements);
          break;
        default:
          return false;
        }
        break;

    default:
        return false;
    }
    
    Value* newValue = graph.addInitializerAndInput(newTensor);
    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), newValue);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
