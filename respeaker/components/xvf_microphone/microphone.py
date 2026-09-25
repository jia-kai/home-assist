"""Expose one 16 kHz mono channel from 48 kHz stereo XVF3800 I²S PCM."""

import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import audio, microphone
from esphome.const import CONF_CHANNEL, CONF_ID, CONF_MICROPHONE

DEPENDENCIES = ["microphone"]
AUTO_LOAD = ["audio"]

xvf_microphone_ns = cg.esphome_ns.namespace("xvf_microphone")
XvfMicrophone = xvf_microphone_ns.class_(
    "XvfMicrophone", microphone.Microphone, cg.Component
)


def _output_stream_limits(config: dict) -> dict:
    """Describe the actual 16 kHz mono output of the FIR decimator.

    Args:
        config:
            ESPHome microphone platform configuration being validated.

    Returns:
        The validated configuration with fixed output stream bounds.

    """
    audio.set_stream_limits(
        min_bits_per_sample=16,
        max_bits_per_sample=16,
        min_channels=1,
        max_channels=1,
        min_sample_rate=16000,
        max_sample_rate=16000,
    )(config)
    return config


CONFIG_SCHEMA = cv.All(
    microphone.MICROPHONE_SCHEMA.extend(
        {
            cv.GenerateID(): cv.declare_id(XvfMicrophone),
            cv.Required(CONF_MICROPHONE): cv.use_id(microphone.Microphone),
            cv.Required(CONF_CHANNEL): cv.int_range(0, 1),
        }
    ).extend(cv.COMPONENT_SCHEMA),
    _output_stream_limits,
)


async def to_code(config: dict) -> None:
    """Instantiate a FIR decimator sharing the underlying I²S microphone.

    Args:
        config:
            Validated source microphone ID and interleaved channel index.

    """
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    await microphone.register_microphone(var, config)
    source = await cg.get_variable(config[CONF_MICROPHONE])
    cg.add(var.set_source(source))
    cg.add(var.set_channel(config[CONF_CHANNEL]))
