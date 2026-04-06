"""Test the OUPES Mega 1 BMS implementation."""

import asyncio
from collections.abc import Buffer, Callable, Iterable
from typing import Any
from uuid import UUID

from bleak import BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic
from bleak.backends.device import BLEDevice
import pytest
from aiobmsble.bms.oupes_mega_bms import BMS, _crc8_smbus, _parse_packet
from tests.bluetooth import generate_ble_device
from tests.conftest import MockBleakClient
from tests.test_basebms import BMSBasicTests

# ── Helpers to build valid BLE notification packets ───────────────────────────


def _make_pkt(*tlv_entries: tuple[int, int]) -> bytearray:
    """Build a 20-byte standard TLV notification packet.

    Each entry is ``(attr, value)``.  Byte 0 = 0x00, byte 1 = 0x01 (type),
    last byte = CRC-8/SMBUS.  TLV format: [0x0A][length][attr][val…] where
    ``length = 1 + len(val_bytes)`` (length field counts the attr byte too).
    """
    pkt = bytearray(20)
    pkt[0] = 0x00
    pkt[1] = 0x01
    pos = 2
    for attr, value in tlv_entries:
        val_bytes = bytes([value]) if value <= 0xFF else value.to_bytes(2, "little")
        length = 1 + len(val_bytes)  # attr byte + value bytes
        pkt[pos] = 0x0A
        pkt[pos + 1] = length
        pkt[pos + 2] = attr
        pkt[pos + 3 : pos + 3 + len(val_bytes)] = val_bytes
        pos += 2 + length
    pkt[19] = _crc8_smbus(bytes(pkt[:19]))
    return pkt


def _make_compact_pkt(attr: int, value: int) -> bytearray:
    """Build a 20-byte compact (type 0x81) notification packet for a single attr."""
    pkt = bytearray(20)
    pkt[0] = 0x00
    pkt[1] = 0x81
    pkt[2] = 2  # length
    pkt[3] = attr
    pkt[4] = value & 0xFF
    pkt[5] = (value >> 8) & 0xFF
    pkt[19] = _crc8_smbus(bytes(pkt[:19]))
    return pkt


# ── Realistic test notifications ──────────────────────────────────────────────
# attr 3=80 (SoC), 21=200 (total input W), 4=150 (AC output W), 32=878 (87.8°F)
_PKT_MAIN = _make_pkt((3, 80), (21, 200))
# attr 4=150 (AC out), 32=878 (87.8 °F = 31.0 °C)
_PKT_TEMPS = _make_pkt((4, 150), (32, 878))
# attr 30=120 (runtime 120 min → 7200 s; only meaningful when discharging)
_PKT_RUNTIME = _make_pkt((30, 120))
# attr 105=1 (inverter protection active)
_PKT_FAULT = _make_pkt((105, 1))
# compact-mode attr 79=75 (module SoC)
_PKT_MODULE_SOC = _make_compact_pkt(79, 75)
# attr 101=1 (slot index), attr 80=856 (85.6 °F = 29.8 °C) — ext battery slot
_PKT_SLOT = _make_pkt((101, 1), (80, 856))
# attr 53=104 (B2 input 104W), attr 54=50 (B2 output 50W) — per-slot power
_PKT_SLOT_POWER = _make_pkt((53, 104), (54, 50))

# Discharging scenario: no input power, AC output present
_PKT_DISCHARGING = _make_pkt((3, 60), (21, 0))
_PKT_DISCHARGING_POWER = _make_pkt((4, 300))
_PKT_DISCHARGING_RUNTIME = _make_pkt((30, 90))  # 90 min → 5400 s

# Idle/charging-with-max-runtime sentinel (5940 → should not set runtime)
_PKT_IDLE_RUNTIME = _make_pkt((30, 5940))

# Handshake-only packets that should produce no attributes
_PKT_HANDSHAKE_80 = bytearray(b"\x01\x80" + b"\x00" * 17 + b"\x00")
_PKT_HANDSHAKE_82 = bytearray(b"\x01\x82" + b"\x00" * 17 + b"\x00")

# Too-short packet
_PKT_TOO_SHORT = bytearray(b"\x01")


# ── Mock BleakClient ──────────────────────────────────────────────────────────


class MockOUPESBleakClient(MockBleakClient):
    """Simulate an OUPES Mega 1 BLE device.

    Starts a background streaming task on ``start_notify`` that continuously
    delivers the test notification burst.  This mirrors the device's push-based
    streaming protocol and lets both single-update and keep-alive tests work.
    """

    # Notifications to fire as a repeating stream; subclasses may override.
    NOTIFICATIONS: list[bytearray] = [_PKT_MAIN, _PKT_TEMPS]

    async def start_notify(
        self,
        char_specifier: BleakGATTCharacteristic | int | str | UUID,
        callback: Callable,
        **kwargs: Any,
    ) -> None:
        """Start the notification subscription and launch the streaming task."""
        await super().start_notify(char_specifier, callback, **kwargs)
        asyncio.create_task(self._notification_stream())

    async def _notification_stream(self) -> None:
        """Deliver notification packets repeatedly until disconnected."""
        await asyncio.sleep(0)  # yield once so init sequence and _async_update.clear() run first
        while self._connected and self._notify_callback is not None:
            for pkt in type(self).NOTIFICATIONS:
                self._notify_callback("rx_char", pkt)
            await asyncio.sleep(0)  # yield between bursts so the event loop can process waits


class MockOUPESDischarging(MockOUPESBleakClient):
    """Variant that fires a discharging scenario."""

    NOTIFICATIONS: list[bytearray] = [_PKT_DISCHARGING, _PKT_DISCHARGING_POWER, _PKT_DISCHARGING_RUNTIME]


class MockOUPESFault(MockOUPESBleakClient):
    """Variant that includes an inverter fault."""

    NOTIFICATIONS: list[bytearray] = [_PKT_MAIN, _PKT_TEMPS, _PKT_FAULT]


class MockOUPESExtBattery(MockOUPESBleakClient):
    """Variant that includes ext-battery module data."""

    NOTIFICATIONS: list[bytearray] = [
        _PKT_MAIN,
        _PKT_SLOT,       # sets slot 1, attr 80
        _PKT_MODULE_SOC, # attr 79 for slot 1
        _PKT_SLOT_POWER, # attr 53=104 (B2 input), attr 54=50 (B2 output) for slot 1
    ]


class MockOUPESIdleRuntime(MockOUPESBleakClient):
    """Variant with charging-mode 5940-minute runtime sentinel (should not set runtime)."""

    NOTIFICATIONS: list[bytearray] = [_PKT_MAIN, _PKT_IDLE_RUNTIME]


# ── BMSBasicTests ─────────────────────────────────────────────────────────────


class TestBasicBMS(BMSBasicTests):
    """Run the shared basic BMS sanity checks."""

    bms_class = BMS


# ── Unit tests for protocol helpers ──────────────────────────────────────────


def test_crc8_smbus_known_value() -> None:
    """Verify CRC-8/SMBUS output against a known-good init packet."""
    # Keepalive packet: last byte is CRC of first 19 bytes
    pkt = bytes.fromhex("0180030254010000000000000000000000000076")
    assert _crc8_smbus(pkt[:19]) == pkt[19]


def test_crc8_smbus_empty() -> None:
    """Empty input should return 0x00."""
    assert _crc8_smbus(b"") == 0x00


def test_parse_packet_standard() -> None:
    """Standard TLV packet: attr 3 = 80."""
    pkt = _make_pkt((3, 80))
    result = _parse_packet(pkt)
    assert result[3] == 80


def test_parse_packet_multiple_attrs() -> None:
    """Multiple TLV entries in one packet."""
    pkt = _make_pkt((3, 80), (21, 200))
    result = _parse_packet(pkt)
    assert result[3] == 80
    assert result[21] == 200


def test_parse_packet_compact() -> None:
    """Compact form (type 0x81) is parsed correctly."""
    pkt = _make_compact_pkt(79, 75)
    result = _parse_packet(pkt)
    assert result[79] == 75


def test_parse_packet_handshake_types() -> None:
    """Type 0x80 and 0x82 packets return empty dict."""
    assert _parse_packet(_PKT_HANDSHAKE_80) == {}
    assert _parse_packet(_PKT_HANDSHAKE_82) == {}


def test_parse_packet_too_short() -> None:
    """Packets shorter than 3 bytes return empty dict."""
    assert _parse_packet(_PKT_TOO_SHORT) == {}


def test_parse_packet_unknown_type_no_0A() -> None:
    """Non-0x81 packet without 0x0A tags returns empty dict."""
    pkt = bytearray(20)
    pkt[1] = 0x02  # unknown type, no 0x0A tags
    assert _parse_packet(pkt) == {}


# ── Token packet builder ──────────────────────────────────────────────────────


def test_build_token_packet_with_secret() -> None:
    """Token packet embeds the secret at bytes [4:14] with correct CRC."""
    bms = BMS(generate_ble_device(), secret="bd236b1695")
    pkt = bms._build_token_packet()
    assert pkt[0] == 0x01
    assert pkt[1] == 0x06
    assert pkt[4:14] == b"bd236b1695"
    assert pkt[19] == _crc8_smbus(pkt[:19])


def test_build_token_packet_no_secret() -> None:
    """Without a secret the token bytes are all zero."""
    bms = BMS(generate_ble_device())
    pkt = bms._build_token_packet()
    assert pkt[4:14] == b"\x00" * 10
    assert pkt[19] == _crc8_smbus(pkt[:19])


def test_build_token_packet_long_secret_truncated() -> None:
    """Secrets longer than 10 ASCII characters are truncated to 10."""
    bms = BMS(generate_ble_device(), secret="abcdefghijklmno")
    pkt = bms._build_token_packet()
    assert pkt[4:14] == b"abcdefghij"


# ── Notification handler unit test ────────────────────────────────────────────


def test_notification_handler_accumulates() -> None:
    """Notification handler stores attrs and signals event on SoC receipt."""
    bms = BMS(generate_ble_device())
    assert not bms._msg_event.is_set()

    bms._notification_handler("char", _PKT_MAIN)  # attr 3 sets the event
    assert bms._data[3] == 80
    assert bms._data[21] == 200
    assert bms._msg_event.is_set()


def test_notification_handler_ext_battery_slot() -> None:
    """Slot index from attr 101 routes ext-battery attrs correctly."""
    bms = BMS(generate_ble_device())
    bms._notification_handler("char", _PKT_SLOT)  # attr 101=1, attr 80=856

    assert bms._current_slot == 1
    assert 1 in bms._ext_batteries
    assert bms._ext_batteries[1][80] == 856


def test_notification_handler_ignores_empty() -> None:
    """Packets that parse to empty dicts do not change state."""
    bms = BMS(generate_ble_device())
    bms._notification_handler("char", _PKT_HANDSHAKE_80)
    assert bms._data == {}
    assert not bms._msg_event.is_set()


def test_notification_handler_ext_battery_53_54() -> None:
    """Attrs 53 and 54 are routed to ext_batteries (not main data) after slot context."""
    bms = BMS(generate_ble_device())
    bms._notification_handler("char", _PKT_SLOT)       # attr 101=1 establishes slot 1
    bms._notification_handler("char", _PKT_SLOT_POWER) # attr 53=104, attr 54=50

    # Must be in ext_batteries[slot], NOT in main data
    assert 53 not in bms._data
    assert 54 not in bms._data
    assert bms._ext_batteries[1][53] == 104
    assert bms._ext_batteries[1][54] == 50


# ── _build_sample unit tests ──────────────────────────────────────────────────


def _bms_with_data(attrs: dict[int, int], ext: dict[int, dict[int, int]] | None = None) -> BMS:
    """Return a BMS instance pre-loaded with raw attribute data."""
    bms = BMS(generate_ble_device())
    bms._data = attrs
    bms._ext_batteries = ext or {}
    return bms


def test_build_sample_charging() -> None:
    """Charging scenario: SoC, temperature, net power, no runtime."""
    bms = _bms_with_data({3: 80, 21: 200, 4: 150, 32: 878})
    sample = bms._build_sample()

    assert sample["battery_level"] == 80
    assert sample["battery_charging"] is True
    assert sample["power"] == 50.0  # 200W in − 150W out
    assert sample["temp_values"] == [31.0]  # 87.8°F → 31.0°C
    assert "runtime" not in sample


def test_build_sample_discharging() -> None:
    """Discharging: negative net power, runtime present."""
    bms = _bms_with_data({3: 60, 21: 0, 4: 300, 30: 90})
    sample = bms._build_sample()

    assert sample["battery_charging"] is False
    assert sample["power"] == -300.0
    assert sample["runtime"] == 5400  # 90 min → 5400 s


def test_build_sample_runtime_sentinel_skipped() -> None:
    """5940-minute runtime sentinel is not propagated when discharging."""
    bms = _bms_with_data({3: 80, 21: 0, 30: 5940})
    sample = bms._build_sample()
    assert "runtime" not in sample


def test_build_sample_problem_code() -> None:
    """Attr 105 is mapped to problem_code."""
    bms = _bms_with_data({3: 50, 105: 1})
    sample = bms._build_sample()
    assert sample["problem_code"] == 1


def test_build_sample_no_power_data() -> None:
    """When no power attrs are present, power key is omitted."""
    bms = _bms_with_data({3: 50})
    sample = bms._build_sample()
    assert "power" not in sample


def test_build_sample_ext_battery() -> None:
    """Per-module SoC and temperature are included when available."""
    bms = _bms_with_data(
        {3: 80},
        ext={1: {79: 82, 80: 856}, 2: {79: 78}},
    )
    sample = bms._build_sample()
    assert sample["pack_battery_levels"] == [82, 78]
    assert sample["pack_count"] == 2
    # module temp from slot 1: 85.6°F = 29.8°C
    assert 29.8 in sample.get("temp_values", [])


def test_build_sample_empty() -> None:
    """Empty data returns only battery_charging (from absent input power)."""
    bms = _bms_with_data({})
    sample = bms._build_sample()
    assert sample == {"battery_charging": False}


# ── Full integration tests (with MockBleakClient) ─────────────────────────────


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speed up tests: zero delays and patch BleakClient with a safe mock."""
    monkeypatch.setattr("aiobmsble.bms.oupes_mega_bms.BMS._NOTIFY_DELAY", 0.0)
    monkeypatch.setattr("aiobmsble.bms.oupes_mega_bms.BMS._PKT_DELAY", 0.0)
    # Patch BleakClient so BMS() can be instantiated without a real BT stack.
    # Integration tests that need a specific mock call patch_bleak_client() to
    # override this default; the last monkeypatch.setattr wins.
    monkeypatch.setattr("aiobmsble.basebms.BleakClient", MockBleakClient)


async def test_update_charging(
    patch_bleak_client,
    keep_alive_fixture: bool,
) -> None:
    """Happy-path charging scenario returns correct BMSSample."""
    patch_bleak_client(MockOUPESBleakClient)
    bms = BMS(generate_ble_device(), keep_alive=keep_alive_fixture)

    result: BMSSample = await bms.async_update()

    assert result["battery_level"] == 80
    assert result["battery_charging"] is True
    assert result["power"] == 50.0  # 200 in − 150 out
    assert result["temp_values"] == [31.0]
    assert result.get("problem", False) is False

    # second call should work regardless of keep_alive mode
    await bms.async_update()
    assert bms.is_connected is keep_alive_fixture

    await bms.disconnect()


async def test_update_discharging(patch_bleak_client) -> None:
    """Discharging scenario includes runtime and negative power."""
    patch_bleak_client(MockOUPESDischarging)
    bms = BMS(generate_ble_device())

    result: BMSSample = await bms.async_update()

    assert result["battery_charging"] is False
    assert result["power"] == -300.0
    assert result["runtime"] == 5400
    await bms.disconnect()


async def test_update_fault(patch_bleak_client) -> None:
    """Inverter fault sets problem_code and causes problem=True."""
    patch_bleak_client(MockOUPESFault)
    bms = BMS(generate_ble_device())

    result: BMSSample = await bms.async_update()

    assert result["problem_code"] == 1
    assert result["problem"] is True
    await bms.disconnect()


async def test_update_ext_battery(patch_bleak_client) -> None:
    """External battery module data populates pack_battery_levels."""
    patch_bleak_client(MockOUPESExtBattery)
    bms = BMS(generate_ble_device())

    result: BMSSample = await bms.async_update()

    assert result.get("pack_battery_levels") == [75]
    assert result.get("pack_count") == 1
    await bms.disconnect()


async def test_update_idle_runtime_sentinel(patch_bleak_client) -> None:
    """5940-minute sentinel during charging is not included as runtime."""
    patch_bleak_client(MockOUPESIdleRuntime)
    bms = BMS(generate_ble_device())

    result: BMSSample = await bms.async_update()

    assert "runtime" not in result
    await bms.disconnect()


async def test_device_info(patch_bleak_client) -> None:
    """Device info is returned from standard BT service 0x180A."""
    patch_bleak_client(MockOUPESBleakClient)
    bms = BMS(generate_ble_device())
    info = await bms.device_info()
    # device_info() returns dynamic GATT 0x180A fields (manufacturer, model, …)
    # not the static INFO class-level keys (default_manufacturer, default_model)
    assert {"manufacturer", "model"}.issubset(info)


async def test_update_keepalive(
    patch_bleak_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keepalive task is started when keep_alive=True and cancelled on reconnect."""
    patch_bleak_client(MockOUPESBleakClient)
    bms = BMS(generate_ble_device(), keep_alive=True)

    # reduce keepalive interval so it can fire quickly in tests
    monkeypatch.setattr("aiobmsble.bms.oupes_mega_bms.BMS._KEEPALIVE_INTERVAL", 0.01)

    await bms.async_update()
    assert bms._keepalive_task is not None
    assert not bms._keepalive_task.done()

    await asyncio.sleep(0.05)  # let keepalive fire at least once

    await bms.disconnect()
    # After disconnect, the keepalive task should self-terminate
    await asyncio.sleep(0.02)
    assert bms._keepalive_task.done()
