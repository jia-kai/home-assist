#pragma once

#include "esphome/components/microphone/microphone.h"
#include "esphome/core/component.h"
#include "esphome/core/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace esphome::xvf_microphone {

/// Downsample one 48 kHz signed stereo 32-bit XVF output into 16 kHz mono PCM.
class XvfMicrophone final : public microphone::Microphone, public Component {
public:
  /// Advertise the exact output format before voice components validate runtime
  /// use.
  XvfMicrophone() {
    this->audio_stream_info_ = audio::AudioStreamInfo(16, 1, 16000);
  }

  /// Retain the physical 48 kHz/32-bit stereo microphone shared by both
  /// channels.
  void set_source(microphone::Microphone *source) { source_ = source; }

  /// Select ASR (zero) or wake (one) from the interleaved I²S input.
  void set_channel(uint8_t channel) { channel_ = channel; }

  /// Initialize after the physical I²S microphone advertises its stream format.
  float get_setup_priority() const override {
    return esphome::setup_priority::LATE;
  }

  /// Attach an in-task callback that avoids extra I²S channel allocation.
  void setup() override {
    if (source_ == nullptr || channel_ > 1) {
      ESP_LOGE("xvf_mic", "status=failed reason=invalid_source");
      mark_failed();
      return;
    }
    const auto format = source_->get_audio_stream_info();
    if (format.get_sample_rate() != 48000 ||
        format.get_bits_per_sample() != 32 || format.get_channels() != 2) {
      ESP_LOGE("xvf_mic", "status=failed reason=incompatible_i2s_format");
      mark_failed();
      return;
    }
    source_->add_data_callback(
        [this](const std::vector<uint8_t> &input) { this->receive_(input); });
  }

  /// Begin a microphone listener without taking another I²S peripheral.
  void start() override {
    if (is_failed() || active_.exchange(true))
      return;
    state_ = microphone::STATE_RUNNING;
    source_->start();
  }

  /// Remove this listener, leaving the I²S device running for other listeners.
  void stop() override {
    if (!active_.exchange(false))
      return;
    state_ = microphone::STATE_STOPPED;
    source_->stop();
  }

protected:
  microphone::Microphone *source_{nullptr};
  /// Physical stereo I²S source, sampled at 48 kHz in 32-bit slots.
  uint8_t channel_{0};
  /// Chosen interleaved slot (ASR=0 or wake=1).
  std::atomic<bool> active_{false};
  /// Whether downstream microphone clients currently request this channel.
  std::array<int16_t, 33> history_{};
  /// Delay line of signed 16-bit audio for the anti-aliasing FIR filter.
  uint32_t history_index_{0};
  /// Next circular history position.
  uint8_t phase_{0};
  /// Input sample counter modulo 3 for the 48→16 kHz decimation.

  /// Filter the selected I²S slot at a 7 kHz cutoff and emit 16 kHz PCM.
  void receive_(const std::vector<uint8_t> &input) {
    if (!active_.load(std::memory_order_relaxed))
      return;
    if (input.size() % 8 != 0) {
      ESP_LOGW("xvf_mic", "status=dropped reason=partial_stereo_frame");
      return;
    }
    // Symmetric 33-tap Hamming-windowed FIR, 7 kHz cutoff at 48 kHz.
    // Coefficients sum to 32768 in Q15; generated with scipy.signal.firwin.
    static constexpr int16_t TAPS[33] = {
        45,    57,    22,   -77,  -186, -164, 98,   481,  608,  122,   -878,
        -1641, -1125, 1224, 4849, 8182, 9534, 8182, 4849, 1224, -1125, -1641,
        -878,  122,   608,  481,  98,   -164, -186, -77,  22,   57,    45,
    };
    std::vector<uint8_t> result;
    result.reserve(input.size() / 12 + 2);
    for (size_t frame = 0; frame < input.size(); frame += 8) {
      const uint8_t *sample = input.data() + frame + channel_ * 4;
      // XVF I²S uses signed little-endian 32-bit slots, with 16-bit high word.
      const int16_t high_word =
          static_cast<int16_t>(static_cast<uint16_t>(sample[2]) |
                               (static_cast<uint16_t>(sample[3]) << 8));
      history_[history_index_] = high_word;
      history_index_ = (history_index_ + 1) % history_.size();
      if (++phase_ != 3)
        continue;
      phase_ = 0;
      int64_t sum = 0;
      for (size_t tap = 0; tap < history_.size(); tap++) {
        const size_t index = (history_index_ + tap) % history_.size();
        sum += static_cast<int64_t>(history_[index]) * TAPS[tap];
      }
      const int32_t value =
          std::max<int64_t>(-32768, std::min<int64_t>(32767, sum >> 15));
      result.push_back(static_cast<uint8_t>(value));
      result.push_back(static_cast<uint8_t>(value >> 8));
    }
    if (!result.empty())
      data_callbacks_.call(result);
  }
};

} // namespace esphome::xvf_microphone
