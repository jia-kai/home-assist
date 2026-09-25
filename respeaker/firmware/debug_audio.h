#pragma once

// Optional on-demand diagnostic PCM stream; never blocks the microphone
// callback.
#include "esphome.h"
#include "esphome/components/i2c/i2c.h"
#include "esphome/components/micro_wake_word/micro_wake_word.h"
#include "esphome/components/microphone/microphone.h"

#include <arpa/inet.h>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstring>
#include <freertos/stream_buffer.h>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

namespace hoast {

/// Expose live stereo I²S microphone PCM on a TCP socket for development.
class DebugAudio final : public esphome::Component,
                         protected esphome::i2c::I2CDevice {
public:
  /// Keep shared microphone capture and XMOS output routing under one owner.
  DebugAudio(esphome::microphone::Microphone *microphone,
             esphome::micro_wake_word::MicroWakeWord *wake,
             esphome::i2c::I2CBus *bus)
      : microphone_(microphone), wake_(wake) {
    set_i2c_bus(bus);
    set_i2c_address(0x2c);
  }

  /// Subscribe to the capture callback and launch an independent TCP writer.
  void setup() override {
    data_ = xStreamBufferCreate(65536, 1);
    if (!data_) {
      ESP_LOGE("debug_audio", "status=failed reason=buffer_allocation");
      mark_failed();
      return;
    }
    microphone_->add_data_callback([this](const std::vector<uint8_t> &data) {
      if (active_.load(std::memory_order_relaxed)) {
        const size_t copied =
            xStreamBufferSend(data_, data.data(), data.size(), 0);
        if (copied != data.size())
          dropped_.fetch_add(1, std::memory_order_relaxed);
      }
    });
    if (xTaskCreate(task_entry_, "debug_pcm", 12288, this, 3, nullptr) !=
        pdPASS) {
      ESP_LOGE("debug_audio", "status=failed reason=task_allocation");
      mark_failed();
    }
  }

  /// Route raw mic/far-end into diagnostic I²S slots from the main task.
  void loop() override {
    const int request = request_.exchange(0);
    if (request == 1 || request == 3 || request == 4) {
      pending_mode_ = request;
      route_attempts_ = 0;
      wake_->stop();
    }
    if (pending_mode_ != 0) {
      if (wake_->is_running())
        return;
      uint8_t command[] = {35, static_cast<uint8_t>(19 | 0x80), 3};
      uint8_t result[3]{};
      const auto error =
          write_read(command, sizeof(command), result, sizeof(result));
      if (error != esphome::i2c::ERROR_OK || result[0]) {
        if (++route_attempts_ < 200)
          return;
        ESP_LOGW("debug_audio",
                 "status=rejected reason=route_read error=%d status=%d",
                 static_cast<int>(error), result[0]);
        pending_mode_ = 0;
        wake_->start();
        route_ready_.store(-1);
        return;
      }
      old_category_ = result[1];
      old_source_ = result[2];
      if (pending_mode_ == 4) {
        uint8_t left_command[] = {35, static_cast<uint8_t>(15 | 0x80), 3};
        uint8_t left_result[3]{};
        const auto left_error = write_read(left_command, sizeof(left_command),
                                           left_result, sizeof(left_result));
        if (left_error != esphome::i2c::ERROR_OK || left_result[0]) {
          if (++route_attempts_ < 200)
            return;
          ESP_LOGW("debug_audio",
                   "status=rejected reason=left_route_read error=%d status=%u",
                   static_cast<int>(left_error), left_result[0]);
          pending_mode_ = 0;
          wake_->start();
          route_ready_.store(-1);
          return;
        }
        old_left_category_ = left_result[1];
        old_left_source_ = left_result[2];
        const uint8_t left_route[] = {35, 15, 2, 12, 0};
        if (write(left_route, sizeof(left_route)) != esphome::i2c::ERROR_OK) {
          ESP_LOGE("debug_audio", "status=rejected reason=left_route_write");
          pending_mode_ = 0;
          wake_->start();
          route_ready_.store(-1);
          return;
        }
        left_routed_ = true;
      }
      const uint8_t route[] = {
          35, 19, 2,
          static_cast<uint8_t>(pending_mode_ == 1 || pending_mode_ == 4 ? 1
                                                                        : 12),
          0};
      if (write(route, sizeof(route)) != esphome::i2c::ERROR_OK) {
        ESP_LOGW("debug_audio", "status=rejected reason=route_write");
        if (left_routed_) {
          const uint8_t original[] = {35, 15, 2, old_left_category_,
                                      old_left_source_};
          if (write(original, sizeof(original)) != esphome::i2c::ERROR_OK)
            ESP_LOGE("debug_audio", "status=failed reason=left_route_rollback");
          left_routed_ = false;
        }
        pending_mode_ = 0;
        wake_->start();
        route_ready_.store(-1);
        return;
      }
      pending_mode_ = 0;
      raw_routed_ = true;
      microphone_->start();
      ESP_LOGI("debug_audio", "status=diagnostic_route_active category=%d",
               route[3]);
      route_ready_.store(1);
    }
    if (request == 2) {
      active_.store(false);
      if (raw_routed_) {
        const uint8_t route[] = {35, 19, 2, old_category_, old_source_};
        if (write(route, sizeof(route)) != esphome::i2c::ERROR_OK) {
          ESP_LOGE("debug_audio", "status=failed reason=route_restore");
          route_ready_.store(-1);
          request_.store(2);
          return;
        }
        if (left_routed_) {
          const uint8_t left_route[] = {35, 15, 2, old_left_category_,
                                        old_left_source_};
          if (write(left_route, sizeof(left_route)) != esphome::i2c::ERROR_OK) {
            ESP_LOGE("debug_audio", "status=failed reason=left_route_restore");
            route_ready_.store(-1);
            request_.store(2);
            return;
          }
          left_routed_ = false;
        }
        raw_routed_ = false;
        microphone_->stop();
        wake_->start();
        ESP_LOGI("debug_audio", "status=wake_route_restored");
      }
      route_ready_.store(0);
    }
  }

  /// Delay setup until both microphone and XMOS bus are initialized.
  float get_setup_priority() const override {
    return esphome::setup_priority::LATE;
  }

  /// Arm one authenticated debug connection for at most ten seconds.
  void arm(const std::string &token) {
    if (token.size() != token_.size()) {
      ESP_LOGW("debug_audio", "status=rejected reason=invalid_token_length");
      return;
    }
    portENTER_CRITICAL(&token_lock_);
    if (session_active_) {
      portEXIT_CRITICAL(&token_lock_);
      ESP_LOGW("debug_audio", "status=rejected reason=session_active");
      return;
    }
    std::memcpy(token_.data(), token.data(), token_.size());
    deadline_ms_ = esphome::millis() + 10000;
    armed_ = true;
    portEXIT_CRITICAL(&token_lock_);
    ESP_LOGI("debug_audio", "status=armed timeout_ms=10000");
  }

protected:
  esphome::microphone::Microphone *microphone_;
  /// ESPHome microphone shared by wake-word and voice assistant.
  esphome::micro_wake_word::MicroWakeWord *wake_;
  /// Wake-word engine paused while channel 1 is routed to raw mic 0.
  StreamBufferHandle_t data_{nullptr};
  /// Best-effort diagnostic queue; overflow never blocks wake detection.
  std::atomic<bool> active_{false};
  /// Whether a TCP subscriber may enqueue microphone callback data.
  std::atomic<int> request_{0};
  /// Route request: 1=raw, 2=restore, 3=far-end, 4=dual delay, 0=none.
  std::atomic<int> route_ready_{0};
  /// I²C route result: 1=raw active, -1=error, 0=normal.
  int pending_mode_{0};
  /// Requested channel-1 diagnostic category pending on the main loop.
  unsigned route_attempts_{0};
  /// Bounded nonblocking I²C read attempts after wake inference stops.
  std::atomic<uint32_t> dropped_{0};
  /// Count of truncated microphone callback copies.
  bool raw_routed_{false};
  /// Whether raw microphone 0 temporarily replaces channel 1.
  uint8_t old_category_{0};
  /// Saved XMOS category for channel 1.
  uint8_t old_source_{0};
  /// Saved XMOS source index for channel 1.
  bool left_routed_{false};
  /// Whether channel 0 carries the SHF reference for a dual-delay capture.
  uint8_t old_left_category_{0};
  /// Saved XMOS output category for channel 0.
  uint8_t old_left_source_{0};
  /// Saved XMOS source index for channel 0.
  portMUX_TYPE token_lock_ = portMUX_INITIALIZER_UNLOCKED;
  /// Protect the one-time credential and arm/session state across tasks.
  std::array<char, 64> token_{};
  /// Hex-encoded 32-byte session credential; never logged.
  uint32_t deadline_ms_{0};
  /// Monotonic arm expiry, wrapping with the ESP uptime counter.
  bool armed_{false};
  /// A debug connection may be accepted only after an API action.
  bool session_active_{false};
  /// Prevent API actions from replacing an in-progress session.

  /// Check arm expiry and clear stale credentials before opening a socket.
  bool waiting_for_client_() {
    portENTER_CRITICAL(&token_lock_);
    if (armed_ && static_cast<int32_t>(esphome::millis() - deadline_ms_) >= 0) {
      armed_ = false;
      token_.fill(0);
    }
    const bool waiting = armed_;
    portEXIT_CRITICAL(&token_lock_);
    return waiting;
  }

  /// Consume the credential once, comparing every byte without early exit.
  bool authenticate_(const std::array<char, 64> &candidate) {
    portENTER_CRITICAL(&token_lock_);
    unsigned difference = 0;
    for (size_t i = 0; i < token_.size(); i++)
      difference |= static_cast<unsigned>(candidate[i] ^ token_[i]);
    const bool accepted =
        armed_ && !session_active_ && difference == 0 &&
        static_cast<int32_t>(esphome::millis() - deadline_ms_) < 0;
    if (accepted) {
      armed_ = false;
      session_active_ = true;
      token_.fill(0);
    }
    portEXIT_CRITICAL(&token_lock_);
    return accepted;
  }

  /// Launch the TCP server without blocking ESPHome's main loop.
  static void task_entry_(void *argument) {
    static_cast<DebugAudio *>(argument)->serve_();
    vTaskDelete(nullptr);
  }

  /// Open the debug TCP port only during a short API-authorized arm window.
  void serve_() {
    while (true) {
      if (!waiting_for_client_()) {
        vTaskDelay(pdMS_TO_TICKS(50));
        continue;
      }
      int server = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      sockaddr_in address{};
      address.sin_family = AF_INET;
      address.sin_addr.s_addr = INADDR_ANY;
      address.sin_port = htons(5072);
      if (server < 0 ||
          bind(server, reinterpret_cast<sockaddr *>(&address),
               sizeof(address)) != 0 ||
          listen(server, 1) != 0) {
        ESP_LOGE("debug_audio", "status=failed reason=socket_bind");
        if (server >= 0)
          close(server);
        vTaskDelay(pdMS_TO_TICKS(500));
        continue;
      }
      while (waiting_for_client_()) {
        fd_set ready;
        FD_ZERO(&ready);
        FD_SET(server, &ready);
        timeval poll{0, 200000};
        if (::select(server + 1, &ready, nullptr, nullptr, &poll) <= 0)
          continue;
        int client = accept(server, nullptr, nullptr);
        if (client < 0)
          continue;
        handle_client_(client);
      }
      close(server);
    }
  }

  /// Authorize one connection before reading any diagnostic mode or PCM.
  void handle_client_(int client) {
    timeval timeout{2, 0};
    setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    std::array<char, 64> candidate{};
    if (recv(client, candidate.data(), candidate.size(), MSG_WAITALL) !=
            candidate.size() ||
        !authenticate_(candidate)) {
      ESP_LOGW("debug_audio", "status=rejected reason=unauthorized_client");
      close(client);
      return;
    }
    const uint32_t session_deadline = esphome::millis() + 65000;
    char mode = 0;
    if (recv(client, &mode, 1, MSG_WAITALL) != 1 ||
        (mode != 'P' && mode != 'R' && mode != 'F' && mode != 'D')) {
      close(client);
      end_session_();
      return;
    }
    if (mode != 'P') {
      route_ready_.store(0);
      request_.store(mode == 'R' ? 1 : (mode == 'D' ? 4 : 3));
      for (int i = 0; i < 200 && route_ready_.load() == 0; i++)
        vTaskDelay(pdMS_TO_TICKS(10));
      if (route_ready_.load() != 1) {
        close(client);
        request_.store(2);
        end_session_();
        return;
      }
    }
    xStreamBufferReset(data_);
    active_.store(true);
    // Raw 48 kHz, stereo interleaved signed 32-bit little-endian PCM.
    // P: ASR/wake; R: ASR/raw mic; F: ASR/far-end; D: far-end/raw mic.
    static const uint8_t hello[] = {'H', 'D', 'B', 'G', 1, 2, 32, 0};
    if (send(client, hello, sizeof(hello), 0) == sizeof(hello)) {
      uint8_t buffer[2048];
      while (static_cast<int32_t>(esphome::millis() - session_deadline) < 0) {
        size_t size = xStreamBufferReceive(data_, buffer, sizeof(buffer),
                                           pdMS_TO_TICKS(250));
        if (size == 0) {
          char probe;
          if (recv(client, &probe, 1, MSG_DONTWAIT) == 0)
            break;
          continue;
        }
        size_t sent = 0;
        while (sent < size) {
          ssize_t result = send(client, buffer + sent, size - sent, 0);
          if (result <= 0)
            break;
          sent += result;
        }
        if (sent != size)
          break;
      }
    }
    active_.store(false);
    close(client);
    if (mode != 'P') {
      request_.store(2);
      for (int i = 0; i < 200 && route_ready_.load() == 1; i++)
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    ESP_LOGI("debug_audio", "status=session_end dropped=%" PRIu32,
             dropped_.exchange(0));
    end_session_();
  }

  /// Release the one-client guard after the debug socket and routing close.
  void end_session_() {
    portENTER_CRITICAL(&token_lock_);
    session_active_ = false;
    portEXIT_CRITICAL(&token_lock_);
  }
};

/// Singleton advanced by the ESPHome main-loop interval after startup.
inline DebugAudio *debug_audio{nullptr};

} // namespace hoast
