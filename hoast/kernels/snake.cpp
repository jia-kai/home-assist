#include <cmath>
#include <immintrin.h>
#include <memory>
#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/core/parallel.hpp>
#include <openvino/op/op.hpp>

extern "C" __m256 _ZGVdN8v_sinf(__m256);

class HoastSnake : public ov::op::Op {
public:
  OPENVINO_OP("HoastSnake", "hoast");

  /** Construct an unconnected operation for deserialization. */
  HoastSnake() = default;

  /** Construct a fused residual sine activation.
   * Args:
   *   inputs:
   *     Float32 data, alpha, and scale tensors.
   *
   */
  explicit HoastSnake(const ov::OutputVector &inputs) : ov::op::Op(inputs) {
    constructor_validate_and_infer_types();
  }

  /** Preserve the data shape and validate the floating-point interface. */
  void validate_and_infer_types() override {
    NODE_VALIDATION_CHECK(this, get_input_size() == 3, "Expected three inputs");
    for (size_t i = 0; i < 3; ++i)
      NODE_VALIDATION_CHECK(this, get_input_element_type(i) == ov::element::f32,
                            "HoastSnake requires float32 inputs");
    set_output_type(0, ov::element::f32, get_input_partial_shape(0));
  }

  /** Clone this operation for graph transformations.
   * Args:
   *   inputs:
   *     Replacement data, alpha, and scale tensors.
   *
   */
  std::shared_ptr<ov::Node>
  clone_with_new_inputs(const ov::OutputVector &inputs) const override {
    return std::make_shared<HoastSnake>(inputs);
  }

  /** Advertise the CPU evaluation implementation. */
  bool has_evaluate() const override { return true; }

  /** Visit the attribute-free operation during serialization.
   * Args:
   *   visitor:
   *     OpenVINO attribute visitor; this operation has no attributes.
   *
   */
  bool visit_attributes(ov::AttributeVisitor &visitor) override {
    (void)visitor;
    return true;
  }

  /** Evaluate x + scale * sin(alpha*x)^2 in one AVX2 pass.
   * Args:
   *   outputs:
   *     One writable float32 output shaped (batch, channels, time).
   *
   *   inputs:
   *     Contiguous float32 data plus rank-at-most-three scalar coefficients or
   *     per-channel coefficients shaped (channels, 1) or (1, channels, 1).
   *
   */
  bool evaluate(ov::TensorVector &outputs,
                const ov::TensorVector &inputs) const override {
    const auto shape = inputs.at(0).get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Expected batch, channels, time");
    OPENVINO_ASSERT(inputs[0].is_continuous() && inputs[1].is_continuous() &&
                    inputs[2].is_continuous());
    const size_t channels = shape[1];
    const size_t width = shape[2];
    const size_t alpha_size = inputs[1].get_size();
    const size_t scale_size = inputs[2].get_size();
    OPENVINO_ASSERT((alpha_size == 1 || alpha_size == channels) &&
                    (scale_size == 1 || scale_size == channels));
    for (size_t index = 1; index < 3; ++index) {
      const auto coefficient_shape = inputs[index].get_shape();
      OPENVINO_ASSERT(coefficient_shape.size() <= 3,
                      "Coefficient rank exceeds data rank");
      if (inputs[index].get_size() != 1) {
        OPENVINO_ASSERT(
            coefficient_shape.size() >= 2 && coefficient_shape.back() == 1 &&
                coefficient_shape[coefficient_shape.size() - 2] == channels,
            "Expected per-channel broadcasting");
      }
    }
    outputs.at(0).set_shape(shape);
    const float *data = inputs[0].data<const float>();
    const float *alpha = inputs[1].data<const float>();
    const float *scale = inputs[2].data<const float>();
    float *output = outputs[0].data<float>();
    ov::parallel_for(shape[0] * channels, [&](size_t row) {
      const float a = alpha[alpha_size == 1 ? 0 : row % channels];
      const float s = scale[scale_size == 1 ? 0 : row % channels];
      const __m256 av = _mm256_set1_ps(a);
      const __m256 sv = _mm256_set1_ps(s);
      size_t column = 0;
      for (; column + 8 <= width; column += 8) {
        const size_t offset = row * width + column;
        const __m256 x = _mm256_loadu_ps(data + offset);
        const __m256 sine = _ZGVdN8v_sinf(_mm256_mul_ps(av, x));
        const __m256 squared = _mm256_mul_ps(sine, sine);
        _mm256_storeu_ps(output + offset,
                         _mm256_add_ps(x, _mm256_mul_ps(sv, squared)));
      }
      for (; column < width; ++column) {
        const size_t offset = row * width + column;
        const float sine = std::sin(a * data[offset]);
        output[offset] = data[offset] + s * (sine * sine);
      }
    });
    return true;
  }
};

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>(
    {std::make_shared<ov::OpExtension<HoastSnake>>()}))
