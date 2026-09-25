#pragma once

// Receive timestamped RTP/L16 separately from ESPHome's wake and command API.
#include "esphome.h"
#include "esphome/components/audio/audio.h"
#include "esphome/components/speaker/speaker.h"

#include <arpa/inet.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <esp_heap_caps.h>

namespace hoast {

/// Feed the XVF3800's I²S far-end input on the host's NTP presentation
/// timeline.
class AecReceiver final : public esphome::Component {
public:
  /// Configure the dedicated PCM speaker sink feeding the XMOS I²S reference.
  explicit AecReceiver(esphome::speaker::Speaker *speaker)
      : speaker_(speaker) {}

  /// Start one low-priority network/presentation task after ESPHome setup.
  void setup() override {
    speaker_->set_audio_stream_info(
        esphome::audio::AudioStreamInfo(16, 2, 48000));
    // Keep the XMOS far-end clock running through gaps and before first RTP
    // data.
    speaker_->start();
    frames_ = static_cast<Frame *>(heap_caps_calloc(
        420, sizeof(Frame), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (frames_ == nullptr) {
      ESP_LOGE("aec_ref", "status=failed reason=psram_allocation");
      mark_failed();
      return;
    }
    if (xTaskCreate(task_entry_, "aec_reference", 6144, this, 4, nullptr) !=
        pdPASS) {
      ESP_LOGE("aec_ref", "status=failed reason=task_allocation");
      mark_failed();
    }
  }

  /// Return the ESPHome setup priority of the reference receiver.
  float get_setup_priority() const override {
    return esphome::setup_priority::LATE;
  }

protected:
  /// One 10 ms mono reference chunk, before duplication to stereo I²S.
  struct Frame {
    uint32_t timestamp;
    /// First-sample RTP timestamp at 48 kHz, modulo 2³².
    uint16_t count;
    /// Number of mono PCM samples in this packet (at most 480).
    std::array<int16_t, 480> pcm;
    /// Native little-endian mono signed PCM samples.
  };

  esphome::speaker::Speaker *speaker_;
  /// Reference-only PCM sink backed by the XMOS I²S output.
  Frame *frames_{nullptr};
  /// 4.2 seconds of 48 kHz/16-bit mono reference frames allocated in PSRAM.
  size_t first_{0};
  /// Oldest occupied slot in the PSRAM ring buffer.
  size_t size_{0};
  /// Number of valid packets in frames_; maximum 420 ten-ms frames.
  uint32_t stream_ssrc_{0};
  /// Active RTP SSRC, independent of its sequence generation.
  uint32_t anchor_rtp_{0};
  /// RTP timestamp from the latest RTCP sender report.
  int64_t anchor_unix_us_{0};
  /// Host presentation time corresponding to anchor_rtp_.
  bool anchored_{false};
  /// True after an RTCP report has mapped the current stream to host time.
  int64_t last_clock_log_us_{0};
  /// Last logged missing-NTP warning, rate limited.
  uint32_t last_report_ms_{0};
  /// Last status-report time in ESP monotonic milliseconds.
  uint32_t late_packets_{0};
  /// Stale reference packets discarded after their presentation deadline.
  uint32_t full_packets_{0};
  /// Frames rejected by the downstream I²S queue.
  uint32_t overflow_packets_{0};
  /// Future frames rejected by the 4.2-second PSRAM buffer.
  uint32_t missing_packets_{0};
  /// RTP sequence numbers missing before a received packet.
  uint16_t last_sequence_{0};
  /// Last accepted RTP sequence number in this SSRC generation.
  bool sequence_seen_{false};
  /// Whether last_sequence_ belongs to the current SSRC.

  /// Enter the receiver task without blocking ESPHome's audio inference loop.
  static void task_entry_(void *argument) {
    static_cast<AecReceiver *>(argument)->run_();
    vTaskDelete(nullptr);
  }

  /// Read network time as Unix microseconds, or zero until SNTP is ready.
  static int64_t unix_us_() {
    timeval tv{};
    gettimeofday(&tv, nullptr);
    if (tv.tv_sec < 1'700'000'000)
      return 0;
    return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
  }

  /// Read an unsigned network-order 32-bit integer without alignment
  /// assumptions.
  static uint32_t be32_(const uint8_t *data) {
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) | data[3];
  }

  /// Reset the queued reference when RTP switches to a new sender generation.
  void reset_(uint32_t ssrc) {
    first_ = 0;
    size_ = 0;
    stream_ssrc_ = ssrc;
    anchored_ = false;
    sequence_seen_ = false;
    ESP_LOGI("aec_ref", "status=stream_start ssrc=%" PRIu32, ssrc);
  }

  /// Accept a 48 kHz mono L16 RTP packet, dropping malformed/oversized packets.
  void receive_rtp_(const uint8_t *packet, size_t length) {
    if (length < 12 || packet[0] != 0x80 || (packet[1] & 0x7f) != 0 ||
        (length - 12) % 2 != 0 || length - 12 > 960 || length == 12)
      return;
    const uint32_t ssrc = be32_(packet + 8);
    if (stream_ssrc_ != ssrc)
      reset_(ssrc);
    const uint16_t sequence =
        (static_cast<uint16_t>(packet[2]) << 8) | packet[3];
    if (sequence_seen_ &&
        sequence != static_cast<uint16_t>(last_sequence_ + 1)) {
      const uint16_t gap = sequence - static_cast<uint16_t>(last_sequence_ + 1);
      if (gap >= 0x8000)
        return; // Stale or reordered packet cannot join the ordered I²S cache.
      missing_packets_ += gap;
    }
    last_sequence_ = sequence;
    sequence_seen_ = true;
    if (size_ == 420) {
      overflow_packets_++;
      return;
    }
    Frame frame{};
    frame.timestamp = be32_(packet + 4);
    frame.count = (length - 12) / 2;
    for (size_t i = 0; i < frame.count; i++) {
      const uint16_t value =
          (static_cast<uint16_t>(packet[12 + i * 2]) << 8) | packet[13 + i * 2];
      frame.pcm[i] = static_cast<int16_t>(value);
    }
    frames_[(first_ + size_) % 420] = frame;
    size_++;
  }

  /// Map this RTP generation's sample clock to the sender's Unix presentation
  /// time.
  void receive_rtcp_(const uint8_t *packet, size_t length) {
    if (length != 28 || packet[0] != 0x80 || packet[1] != 200 ||
        packet[2] != 0 || packet[3] != 6)
      return;
    if (be32_(packet + 4) != stream_ssrc_)
      return;
    const uint32_t ntp_sec = be32_(packet + 8);
    if (ntp_sec < 2208988800U)
      return;
    anchor_unix_us_ =
        static_cast<int64_t>(ntp_sec - 2208988800U) * 1000000 +
        (static_cast<uint64_t>(be32_(packet + 12)) * 1000000 >> 32);
    anchor_rtp_ = be32_(packet + 16);
    anchored_ = true;
  }

  /// Present due frames to ESPHome's bounded I²S source; drop stale audio.
  void present_() {
    int64_t now = unix_us_();
    if (!now) {
      if (last_clock_log_us_ == 0) {
        ESP_LOGW("aec_ref", "status=waiting reason=ntp_unsynchronized");
        last_clock_log_us_ = 1;
      }
      return;
    }
    if (!anchored_ || size_ == 0)
      return;
    while (size_ != 0) {
      auto &frame = frames_[first_];
      const int32_t delta = static_cast<int32_t>(frame.timestamp - anchor_rtp_);
      const int64_t presentation =
          anchor_unix_us_ + static_cast<int64_t>(delta) * 1000000 / 48000;
      // The speaker source and I²S DMA buffer samples before its physical clock
      // edge.
      if (presentation > now + 50000) {
        break;
      }
      if (presentation >= now - 20000) {
        std::array<int16_t, 960> stereo{};
        for (size_t sample = 0; sample < frame.count; sample++) {
          stereo[sample * 2] = frame.pcm[sample];
          stereo[sample * 2 + 1] = frame.pcm[sample];
        }
        const size_t written =
            speaker_->play(reinterpret_cast<const uint8_t *>(stereo.data()),
                           frame.count * 4, 0);
        if (written != frame.count * 4)
          full_packets_++;
      } else {
        late_packets_++;
      }
      first_ = (first_ + 1) % 420;
      size_--;
    }
  }

  /// Aggregate loss diagnostics so serial logging cannot starve I²S playback.
  void report_drops_() {
    const uint32_t now = esphome::millis();
    if (now - last_report_ms_ < 1000)
      return;
    last_report_ms_ = now;
    if (late_packets_ || full_packets_ || overflow_packets_ ||
        missing_packets_) {
      ESP_LOGW("aec_ref",
               "status=dropped late=%" PRIu32 " i2s_full=%" PRIu32
               " future_full=%" PRIu32 " rtp_missing=%" PRIu32,
               late_packets_, full_packets_, overflow_packets_,
               missing_packets_);
      late_packets_ = full_packets_ = overflow_packets_ = missing_packets_ = 0;
    }
  }

  /// Poll RTP and RTCP sockets without touching the microphone or wake tasks.
  void run_() {
    int rtp = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    int rtcp = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (rtp < 0 || rtcp < 0) {
      ESP_LOGE("aec_ref", "status=failed reason=socket_creation");
      return;
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(5070);
    if (bind(rtp, reinterpret_cast<sockaddr *>(&address), sizeof(address)) !=
        0) {
      ESP_LOGE("aec_ref", "status=failed reason=rtp_bind");
      close(rtp);
      close(rtcp);
      return;
    }
    address.sin_port = htons(5071);
    if (bind(rtcp, reinterpret_cast<sockaddr *>(&address), sizeof(address)) !=
        0) {
      ESP_LOGE("aec_ref", "status=failed reason=rtcp_bind");
      close(rtp);
      close(rtcp);
      return;
    }
    while (true) {
      fd_set ready;
      FD_ZERO(&ready);
      FD_SET(rtp, &ready);
      FD_SET(rtcp, &ready);
      timeval timeout{0, 5000};
      int result =
          ::select(std::max(rtp, rtcp) + 1, &ready, nullptr, nullptr, &timeout);
      if (result > 0) {
        uint8_t packet[1024];
        if (FD_ISSET(rtp, &ready)) {
          ssize_t length = recv(rtp, packet, sizeof(packet), 0);
          if (length > 0)
            receive_rtp_(packet, length);
        }
        if (FD_ISSET(rtcp, &ready)) {
          ssize_t length = recv(rtcp, packet, sizeof(packet), 0);
          if (length > 0)
            receive_rtcp_(packet, length);
        }
      }
      present_();
      report_drops_();
    }
  }
};

} // namespace hoast
