"""Host contract tests for XMOS reference-gain I²C control."""

import shutil
import subprocess
from pathlib import Path

import pytest

_I2C_STUB = r"""
#pragma once
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace esphome::i2c {
enum ErrorCode { ERROR_OK = 0, ERROR_UNKNOWN = 1 };

struct Transaction {
  std::vector<uint8_t> request;
  std::vector<uint8_t> response;
  ErrorCode error = ERROR_OK;
};

class I2CBus {
public:
  std::vector<Transaction> transactions;
  size_t next = 0;

  // Validate every frame, address, read size, and transaction ordering.
  ErrorCode write_readv(uint8_t address, const uint8_t *write_buffer,
                        size_t write_count, uint8_t *read_buffer,
                        size_t read_count) {
    assert(address == 0x2c);
    assert(next < transactions.size());
    const auto &transaction = transactions[next++];
    assert(std::vector<uint8_t>(write_buffer, write_buffer + write_count) ==
           transaction.request);
    assert(read_count == transaction.response.size());
    assert((read_buffer != nullptr) == (read_count != 0));
    if (read_count != 0)
      std::copy(transaction.response.begin(), transaction.response.end(),
                read_buffer);
    return transaction.error;
  }
};
} // namespace esphome::i2c
"""

_LOG_STUB = r"""
#pragma once
#define ESP_LOGE(...) ((void)0)
#define ESP_LOGW(...) ((void)0)
#define ESP_LOGI(...) ((void)0)
#define ESP_LOGD(...) ((void)0)
"""

_CONTROL_FIXTURE = r"""
#include "xmos_control.h"
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  assert(argc == 2);
  const std::string scenario = argv[1];
  std::fprintf(stderr, "scenario=%s\n", argv[1]);
  using esphome::i2c::ERROR_OK;
  using esphome::i2c::ERROR_UNKNOWN;
  using esphome::i2c::I2CBus;
  const std::vector<uint8_t> read_command = {35, 0x81, 5};
  const std::vector<uint8_t> set_command = {35, 1, 4, 0, 0, 128, 63};
  const std::vector<uint8_t> unity = {0, 0, 0, 128, 63};
  const std::vector<uint8_t> eight = {0, 0, 0, 0, 65};

  if (scenario.rfind("read-", 0) == 0) {
    auto response = unity;
    auto error = ERROR_OK;
    bool expected_success = false;
    float expected_gain = -123.5f;
    if (scenario == "read-unity") {
      expected_success = true;
      expected_gain = 1.0f;
    } else if (scenario == "read-eight") {
      response = eight;
      expected_success = true;
      expected_gain = 8.0f;
    } else if (scenario == "read-fractional") {
      response = {0, 0, 0, 192, 63};
      expected_success = true;
      expected_gain = 1.5f;
    } else if (scenario == "read-all-bytes") {
      response = {0, 0xdb, 0x0f, 0x49, 0x40};
      expected_success = true;
      expected_gain = 3.1415927f;
    } else if (scenario == "read-bus-error") {
      error = ERROR_UNKNOWN;
    } else {
      assert(scenario.rfind("read-status-", 0) == 0);
      response[0] = static_cast<uint8_t>(std::stoi(scenario.substr(12)));
    }
    I2CBus bus{{{read_command, response, error}}};
    float gain = -123.5f;
    assert(hoast::read_reference_gain(&bus, gain) == expected_success);
    assert(gain == expected_gain);
    assert(bus.next == bus.transactions.size());
    return 0;
  }

  assert(scenario.rfind("set-", 0) == 0);
  I2CBus bus{{{read_command, eight}}};
  bool expected_success = false;
  if (scenario == "set-noop") {
    bus.transactions[0].response = unity;
    expected_success = true;
  } else if (scenario == "set-initial-bus-error") {
    bus.transactions[0].error = ERROR_UNKNOWN;
  } else if (scenario.rfind("set-initial-status-", 0) == 0) {
    bus.transactions[0].response[0] =
        static_cast<uint8_t>(std::stoi(scenario.substr(19)));
  } else {
    bus.transactions.push_back({set_command, {}});
    if (scenario == "set-write-error") {
      bus.transactions.back().error = ERROR_UNKNOWN;
    } else {
      bus.transactions.push_back({read_command, unity});
      if (scenario == "set-success") {
        expected_success = true;
      } else if (scenario == "set-readback-bus-error") {
        bus.transactions.back().error = ERROR_UNKNOWN;
      } else if (scenario == "set-wrong-readback") {
        bus.transactions.back().response = eight;
      } else if (scenario == "set-near-unity-readback") {
        bus.transactions.back().response = {0, 1, 0, 128, 63};
      } else {
        assert(scenario.rfind("set-readback-status-", 0) == 0);
        bus.transactions.back().response[0] =
            static_cast<uint8_t>(std::stoi(scenario.substr(20)));
      }
    }
  }
  assert(hoast::set_reference_unity_gain(&bus) == expected_success);
  assert(bus.next == bus.transactions.size());
}
"""


@pytest.fixture(scope="module")
def control_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the project header against embedded, scripted ESPHome stubs.

    Args:
        tmp_path_factory:
            Pytest factory providing an isolated directory for generated files.

    Returns:
        Executable path; skip only when no supported C++ compiler is installed.
    """
    compiler = next(
        (path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip("No C++ compiler available for the XMOS host fixture")
    tmp_path = tmp_path_factory.mktemp("xmos_control")
    i2c_header = tmp_path / "esphome" / "components" / "i2c" / "i2c.h"
    log_header = tmp_path / "esphome" / "core" / "log.h"
    i2c_header.parent.mkdir(parents=True)
    log_header.parent.mkdir(parents=True)
    i2c_header.write_text(_I2C_STUB, encoding="utf-8")
    log_header.write_text(_LOG_STUB, encoding="utf-8")
    source = tmp_path / "control_fixture.cpp"
    source.write_text(_CONTROL_FIXTURE, encoding="utf-8")
    binary = tmp_path / "control_fixture"
    header = Path(__file__).resolve().parents[1] / "firmware" / "xmos_control.h"
    assert header.is_file(), f"Project header not found: {header}"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-pedantic",
            "-I",
            str(tmp_path),
            "-I",
            str(header.parent),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"C++ compile failed: {result.args}\n{result.stdout}\n{result.stderr}"
    )
    return binary


def _run_scenario(control_binary: Path, scenario: str) -> None:
    """Execute a scripted transaction sequence and expose complete diagnostics.

    Args:
        control_binary:
            Compiled host executable using the project XMOS helper header.

        scenario:
            Name selecting the fixture's expected calls, responses, and outcome.

    """
    result = subprocess.run(
        [str(control_binary), scenario],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, (
        f"Host scenario failed: {scenario}, exit={result.returncode}\n"
        f"{result.stdout}\n{result.stderr}"
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "read-unity",
        "read-eight",
        "read-fractional",
        "read-all-bytes",
        "read-bus-error",
        "read-status-1",
        "read-status-64",
        "read-status-2",
        "read-status-255",
    ],
)
def test_read_reference_gain(control_binary: Path, scenario: str) -> None:
    """Decode little-endian floats and preserve output on every failed read.

    Args:
        control_binary:
            Compiled host fixture enforcing the exact I²C read command.

        scenario:
            Successful gain value, bus error, pending status, or invalid status.

    """
    _run_scenario(control_binary, scenario)


@pytest.mark.parametrize(
    "scenario",
    [
        "set-noop",
        "set-success",
        "set-initial-bus-error",
        "set-initial-status-1",
        "set-initial-status-64",
        "set-initial-status-2",
        "set-write-error",
        "set-readback-bus-error",
        "set-readback-status-1",
        "set-readback-status-64",
        "set-readback-status-2",
        "set-wrong-readback",
        "set-near-unity-readback",
    ],
)
def test_set_reference_unity_gain(control_binary: Path, scenario: str) -> None:
    """Avoid redundant writes and require an exact unity readback after setting.

    Args:
        control_binary:
            Compiled host fixture enforcing ordered read, write, and readback.

        scenario:
            No-op, gain correction, or failure at a particular transaction stage.

    """
    _run_scenario(control_binary, scenario)
