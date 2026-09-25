#pragma once

#include "esphome/components/i2c/i2c.h"
#include "esphome/core/log.h"

#include <cstdint>

namespace hoast {

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
