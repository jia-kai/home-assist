#pragma once

#include "esphome/components/i2c/i2c.h"
#include "esphome/core/log.h"

#include <cstdint>
#include <cstring>

namespace hoast {

/** Read the DSP's linear reference gain; leave gain unchanged on failure.
 *
 * Args:
 *     bus:
 *         I²C bus connected to the XVF3800 at address 0x2c.
 *
 *     gain:
 *         Receives the little-endian IEEE-754 amplitude multiplier on success.
 *
 * Returns:
 *     True on a successful read; false on bus errors, pending, or invalid
 * status.
 */
inline bool read_reference_gain(esphome::i2c::I2CBus *bus, float &gain) {
  const uint8_t request[] = {35, 0x81, 5};
  uint8_t response[5]{};
  if (bus->write_readv(0x2c, request, sizeof(request), response,
                       sizeof(response)) != esphome::i2c::ERROR_OK) {
    ESP_LOGW("xmos_ctrl", "status=retry reason=reference_gain_read");
    return false;
  }
  if (response[0] == 1 || response[0] == 64)
    return false;
  if (response[0] != 0) {
    ESP_LOGE("xmos_ctrl", "status=failed reason=reference_gain_status code=%u",
             response[0]);
    return false;
  }
  const uint32_t bits = static_cast<uint32_t>(response[1]) |
                        (static_cast<uint32_t>(response[2]) << 8) |
                        (static_cast<uint32_t>(response[3]) << 16) |
                        (static_cast<uint32_t>(response[4]) << 24);
  static_assert(sizeof(gain) == sizeof(bits));
  std::memcpy(&gain, &bits, sizeof(gain));
  return true;
}

/** Apply AUDIO_MGR_REF_GAIN=1.0, succeeding only after matching DSP readback.
 *
 * Args:
 *     bus:
 *         I²C bus connected to the XVF3800 at address 0x2c.
 *
 * Returns:
 *     True if unity is confirmed; false if the caller must retry. Does not
 * write flash or change microphone gain, routing, or the system delay.
 */
inline bool set_reference_unity_gain(esphome::i2c::I2CBus *bus) {
  float gain = 0.0f;
  if (!read_reference_gain(bus, gain))
    return false;
  if (gain == 1.0f) {
    ESP_LOGI("xmos_ctrl", "status=reference_gain_ready gain=1.0");
    return true;
  }
  ESP_LOGI("xmos_ctrl",
           "status=reference_gain_configure previous=%f target=1.0", gain);
  const uint8_t frame[] = {35, 1, 4, 0, 0, 0x80, 0x3f};
  if (bus->write_readv(0x2c, frame, sizeof(frame), nullptr, 0) !=
      esphome::i2c::ERROR_OK) {
    ESP_LOGW("xmos_ctrl", "status=retry reason=reference_gain_write");
    return false;
  }
  if (!read_reference_gain(bus, gain))
    return false;
  if (gain != 1.0f) {
    ESP_LOGE("xmos_ctrl", "status=failed reason=reference_gain_verify gain=%f",
             gain);
    return false;
  }
  ESP_LOGI("xmos_ctrl", "status=reference_gain_ready gain=1.0");
  return true;
}

/// Light the XVF3800 ring with the I²C command used by the selected DSP image.
inline void set_ring(esphome::i2c::I2CBus *bus, uint8_t command, uint32_t rgb) {
  uint8_t frame[51] = {20, command, 48};
  for (size_t index = 0; index < 12; ++index) {
    frame[3 + index * 4] = static_cast<uint8_t>(rgb);
    frame[4 + index * 4] = static_cast<uint8_t>(rgb >> 8);
    frame[5 + index * 4] = static_cast<uint8_t>(rgb >> 16);
    frame[6 + index * 4] = 0;
  }
  if (bus->write_readv(0x2c, frame, sizeof(frame), nullptr, 0) !=
      esphome::i2c::ERROR_OK)
    ESP_LOGE("xmos_ctrl", "status=failed reason=led_ring_write command=%u",
             command);
}

/// Route one I²S output slot and return whether the XMOS readback agrees.
inline bool set_output_route(esphome::i2c::I2CBus *bus, uint8_t command,
                             uint8_t category, uint8_t source) {
  if (category == 255)
    return true;
  const uint8_t frame[] = {35, command, 2, category, source};
  if (bus->write_readv(0x2c, frame, sizeof(frame), nullptr, 0) !=
      esphome::i2c::ERROR_OK) {
    ESP_LOGW("xmos_ctrl", "status=retry reason=output_route_write command=%u",
             command);
    return false;
  }
  const uint8_t request[] = {35, static_cast<uint8_t>(command | 0x80), 3};
  uint8_t response[3]{};
  if (bus->write_readv(0x2c, request, sizeof(request), response,
                       sizeof(response)) != esphome::i2c::ERROR_OK)
    return false;
  if (response[0] == 1 || response[0] == 64)
    return false;
  if (response[0] != 0 || response[1] != category || response[2] != source) {
    ESP_LOGE("xmos_ctrl",
             "status=failed reason=output_route_verify command=%u status=%u "
             "category=%u source=%u",
             command, response[0], response[1], response[2]);
    return false;
  }
  ESP_LOGI("xmos_ctrl",
           "status=output_route_ready command=%u category=%u source=%u",
           command, category, source);
  return true;
}

} // namespace hoast
