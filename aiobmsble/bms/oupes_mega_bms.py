"""Module to support OUPES Mega 1 power station.

This driver was reverse-engineered from an Android HCI snoop capture of the
official Cleanergy app paired to an OUPES Mega 1.

The device is a power station rather than a traditional BMS, so the data set
differs from the standard BMS profile:

  Available:
    battery_level (SoC %)           — attr  3
    temperature (main unit, °C)     — attr 32  (firmware always reports °F×10)
    temp_values (all sensors, °C)   — attrs 32 + per-module attr 80
    power (net W, +charge/-dchrg)   — attrs 21, 4, 6, 7, 8
    battery_charging                — attr 21 (total input) > 0
    runtime (s, discharge only)     — attr 30  (minutes; sentinel 5940 = idle)
    problem_code                    — attr 105 (1 = inverter protection active)
    pack_battery_levels             — per-module attr 79
    pack_count                      — number of modules reporting

  Per-module attrs (stored in _ext_batteries[slot]; NOT in BMSSample):
    attr 53  — B2 Input Power (W): power entering the B2 via its secondary
               MPPT/DC port (solar panel or external DC source).
               B2 solar input is ALSO counted in attr 21's system total —
               it does NOT flow through attr 22 (grid) or attr 23 (main solar).
    attr 54  — B2 Output Power (W): total power leaving the B2 (chain-cable
               discharge toward the main unit plus B2 USB ports combined).
    attr 78  — multiplexed: raw value ≤ 6000 = runtime remaining (minutes);
               raw value ≥ 44000 = battery pack voltage (mV).  On current
               Mega 1 firmware only slot 2 broadcasts voltage readings;
               slot 1 never does.
    attr 79  — per-module SoC (direct %, 0–100)  → pack_battery_levels
    attr 80  — per-module temperature (°F × 10)  → temp_values

  NOT available from this device:
    voltage, current — the protocol only exposes watt readings, not V/A.

Note: upstream contribution to aiobmsble requires voltage + current per
CONTRIBUTING.md ("How to qualify as a BMS"). This driver is intended for
personal/fork use or as a starting point should OUPES expose those values
in a future firmware.

Protocol details:
    ../../../oupes-mega-hass/custom_components/oupes_mega/protocol.py

Project: aiobmsble fork, https://pypi.org/p/aiobmsble/
License: Apache-2.0, http://www.apache.org/licenses/
"""

import asyncio
from typing import Final

from bleak.backends.characteristic import BleakGATTCharacteristic
from bleak.backends.device import BLEDevice

from aiobmsble import BMSInfo, BMSSample, MatcherPattern
from aiobmsble.basebms import BaseBMS

# ── CRC-8 SMBUS (polynomial 0x07, init 0x00) ──────────────────────────────────
# Distinct from CRC-8/MAXIM-DOW exported by basebms.crc8.


def _crc8_smbus(data: bytes) -> int:
    """CRC-8/SMBUS over *data* (polynomial 0x07, initial value 0x00)."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


# ── Attribute sets ─────────────────────────────────────────────────────────────

# Attrs that belong to a per-slot battery module (slot index carried in attr 101).
# 53 = B2 Input Power, 54 = B2 Output Power, 78 = runtime/voltage (muxed),
# 79 = SoC (%), 80 = temperature (°F×10).
_EXT_BATTERY_ATTRS: Final[frozenset[int]] = frozenset({53, 54, 78, 79, 80})


# ── Packet parser ──────────────────────────────────────────────────────────────


def _parse_packet(data: bytearray) -> dict[int, int]:
    """Parse a 20-byte BLE notification into ``{attr: raw_value}``.

    Formats observed in HCI capture:

    * **Type 0x00 / 0x01** — standard TLV stream: ``[0x0a][length][attr][val…]``
    * **Type 0x81** — compact form (firmware omits the 0x0a tag):
      ``[length][attr][val…]``, length 1–4 only
    * **Type 0x80 / 0x82** — handshake / end-of-group markers; no payload
    """
    results: dict[int, int] = {}
    if len(data) < 3:
        return results

    pkt_type = data[1]

    if pkt_type in (0x80, 0x82):
        return results  # no TLV payload

    i = 2
    while i < len(data) - 1:  # last byte is CRC / checksum
        if data[i] == 0x0A and i + 2 < len(data):
            # Standard form: [0x0a][length][attr][val…]
            length = data[i + 1]
            if length >= 1 and i + 2 + length <= len(data) - 1:
                attr = data[i + 2]
                val_bytes = data[i + 3 : i + 2 + length]
                results[attr] = (
                    int.from_bytes(val_bytes, "little") if val_bytes else 0
                )
            i += 2 + length
        elif (
            pkt_type == 0x81
            and 1 <= data[i] <= 4
            and i + 1 + data[i] <= len(data) - 1
        ):
            # Compact form (type 0x81 only): [length][attr][val…]
            length = data[i]
            attr = data[i + 1]
            val_bytes = data[i + 2 : i + 1 + length]
            results[attr] = (
                int.from_bytes(val_bytes, "little") if val_bytes else 0
            )
            i += 1 + length
        else:
            i += 1

    return results


# ── BMS class ──────────────────────────────────────────────────────────────────


class BMS(BaseBMS):
    """OUPES Mega 1 power station BLE driver.

    Pass the per-device pairing token (e.g. ``"bd236b1695"``) as the
    ``secret`` constructor argument.  Without it, init packet 6 will contain
    all-zero bytes, which may cause the device to reject the session.
    """

    INFO: BMSInfo = {
        "default_manufacturer": "OUPES",
        "default_model": "Mega 1",
    }

    accept_secret: bool = True  # per-device token passed via ``secret``

    # ── GATT identifiers ───────────────────────────────────────────────────────
    _SERVICE_UUID: Final[str] = "00001910-0000-1000-8000-00805f9b34fb"
    _NOTIFY_UUID: Final[str] = "00002b10-0000-1000-8000-00805f9b34fb"
    _WRITE_UUID: Final[str] = "00002b11-0000-1000-8000-00805f9b34fb"

    # ── Keepalive ──────────────────────────────────────────────────────────────
    # Without a write every ~10 s the device terminates the session.
    _KEEPALIVE_PKT: Final[bytes] = bytes.fromhex(
        "0180030254010000000000000000000000000076"
    )
    _KEEPALIVE_INTERVAL: Final[float] = 9.0  # slightly under the 10 s device timeout

    # ── Timing constants ───────────────────────────────────────────────────────
    _NOTIFY_DELAY: Final[float] = 0.2  # wait after start_notify before sending init
    _PKT_DELAY: Final[float] = 0.01   # inter-packet delay in init sequence
    _COLLECT_TIMEOUT: Final[float] = 8.0  # seconds to wait for first notification burst

    # ── Init sequence ──────────────────────────────────────────────────────────
    # 10 fixed packets + 1 token packet (None = built at runtime from secret).
    # Sent immediately after subscribing to notifications.
    _INIT_PKTS: Final[tuple[bytes | None, ...]] = (
        bytes.fromhex("0100019901010101010101010101010101010192"),
        bytes.fromhex("010101010101010101010101010101010101018f"),
        bytes.fromhex("0102000000000000000000000000000000000082"),
        bytes.fromhex("01030000000000000000000000000000000000a8"),
        bytes.fromhex("010400000000000000000000000000000000007e"),
        bytes.fromhex("0105000000000000000000000000000000000054"),
        None,  # ← token packet, built in _build_token_packet()
        bytes.fromhex("0107000000000000000000000000000000000000"),
        bytes.fromhex("0108000000000000000000000000000000000081"),
        bytes.fromhex("01890000000000000000000000000000000000c0"),
        bytes.fromhex("0180020101000000000000000000000000000016"),
    )

    def __init__(
        self,
        ble_device: BLEDevice,
        keep_alive: bool = True,
        secret: str = "",
        logger_name: str = "",
    ) -> None:
        """Initialize private BMS members."""
        super().__init__(ble_device, keep_alive, secret, logger_name)
        self._data: dict[int, int] = {}
        self._ext_batteries: dict[int, dict[int, int]] = {}
        self._current_slot: int = 1
        self._keepalive_task: asyncio.Task[None] | None = None

    # ── BMS ABC ────────────────────────────────────────────────────────────────

    @staticmethod
    def matcher_dict_list() -> list[MatcherPattern]:
        """Provide BluetoothMatcher definition."""
        return [
            {
                "service_uuid": "00001910-0000-1000-8000-00805f9b34fb",
                "connectable": True,
            }
        ]

    @staticmethod
    def uuid_services() -> tuple[str, ...]:
        """Return list of 128-bit UUIDs of services required by BMS."""
        return ("00001910-0000-1000-8000-00805f9b34fb",)

    @staticmethod
    def uuid_rx() -> str:
        """Return UUID of the notification characteristic (RX)."""
        return "00002b10-0000-1000-8000-00805f9b34fb"

    @staticmethod
    def uuid_tx() -> str:
        """Return UUID of the write characteristic (TX)."""
        return "00002b11-0000-1000-8000-00805f9b34fb"

    # ── Connection lifecycle ───────────────────────────────────────────────────

    def _build_token_packet(self) -> bytes:
        """Build init packet 6 with the per-device token at bytes [4:14].

        The token is the ASCII pairing string from ``self._secret``, truncated
        to 10 bytes and zero-padded if shorter.  CRC-8/SMBUS covers bytes [0:19].
        """
        token = (
            self._secret.encode("ascii")[:10].ljust(10, b"\x00")
            if self._secret
            else b"\x00" * 10
        )
        pkt = bytearray(20)
        pkt[0] = 0x01
        pkt[1] = 0x06
        pkt[4:14] = token
        pkt[19] = _crc8_smbus(bytes(pkt[:19]))
        return bytes(pkt)

    async def _init_connection(
        self, char_notify: BleakGATTCharacteristic | int | str | None = None
    ) -> None:
        """Subscribe to notifications, send init sequence, start keepalive if needed."""
        # Cancel any stale keepalive from a previous connection attempt
        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            self._keepalive_task = None

        # Base class clears frame/event and starts_notify on uuid_rx()
        await super()._init_connection()
        await asyncio.sleep(self._NOTIFY_DELAY)

        token_pkt = self._build_token_packet()
        for pkt in self._INIT_PKTS:
            await self._client.write_gatt_char(
                self._WRITE_UUID,
                pkt if pkt is not None else token_pkt,
                response=False,
            )
            await asyncio.sleep(self._PKT_DELAY)

        if self._keep_alive:
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _keepalive_loop(self) -> None:
        """Periodically write the keepalive packet to prevent session expiry (10 s)."""
        try:
            while True:
                await asyncio.sleep(self._KEEPALIVE_INTERVAL)
                await self._client.write_gatt_char(
                    self._WRITE_UUID, self._KEEPALIVE_PKT, response=False
                )
        except Exception:  # noqa: BLE001 – self-terminates on disconnect or cancel
            pass

    # ── Notification handler ───────────────────────────────────────────────────

    def _notification_handler(
        self, _sender: BleakGATTCharacteristic, data: bytearray
    ) -> None:
        """Accumulate notification payloads into ``self._data`` / ``self._ext_batteries``."""
        self._log.debug("RX BLE data: %s", data.hex())

        parsed = _parse_packet(data)
        if not parsed:
            return

        if 101 in parsed:
            slot = parsed[101]
            self._current_slot = slot
            if slot not in self._ext_batteries:
                self._ext_batteries[slot] = {}

        for attr, val in parsed.items():
            if attr in _EXT_BATTERY_ATTRS:
                self._ext_batteries[self._current_slot][attr] = val
            elif attr != 101:
                self._data[attr] = val

        if 3 in self._data:  # battery_level received — signal that data is ready
            self._msg_event.set()

    # ── Data update ────────────────────────────────────────────────────────────

    async def _async_update(self) -> BMSSample:
        """Wait for one notification burst and return the accumulated BMSSample."""
        self._msg_event.clear()
        self._data.clear()
        self._ext_batteries.clear()

        try:
            await asyncio.wait_for(
                self._msg_event.wait(), timeout=self._COLLECT_TIMEOUT
            )
        except TimeoutError:
            if not self._data:
                raise
            self._log.debug(
                "collect timeout with partial data (%d attrs)", len(self._data)
            )

        return self._build_sample()

    # ── Sample construction ────────────────────────────────────────────────────

    def _build_sample(self) -> BMSSample:
        """Convert the accumulated attribute dict into a :class:`BMSSample`."""
        data = self._data
        sample: BMSSample = {}

        # SoC — attr 3 is a direct percentage value
        if 3 in data:
            sample["battery_level"] = data[3]

        # Temperature — attr 32 is (°F × 10); convert to °C
        # Example: raw 878 = 87.8 °F = 31.0 °C
        temp_values: list[float] = []
        if 32 in data:
            temp_values.append(round((data[32] / 10.0 - 32.0) * 5.0 / 9.0, 1))

        # Per-module temperatures — attr 80 per battery module (°F × 10)
        for slot_data in self._ext_batteries.values():
            if 80 in slot_data:
                temp_values.append(
                    round((slot_data[80] / 10.0 - 32.0) * 5.0 / 9.0, 1)
                )

        if temp_values:
            sample["temp_values"] = temp_values
            # BaseBMS._add_missing_values() will derive `temperature` as fmean(temp_values)

        # Power — net watts: positive = charging, negative = discharging.
        # attr 21 = total system input (grid + main-unit solar + B2 secondary-port
        # solar — note B2 solar bypasses attr 23 and flows directly into attr 21).
        # attrs 4/6/7/8 = AC / DC-12V / USB-C / USB-A output.
        # Per-slot B2 input/output power (attrs 53/54) are stored in
        # _ext_batteries[slot] for callers that need them but are not in BMSSample.
        input_w: int = data.get(21, 0)
        output_w: int = sum(data.get(a, 0) for a in (4, 6, 7, 8))
        if 21 in data or any(a in data for a in (4, 6, 7, 8)):
            sample["power"] = float(input_w - output_w)

        # Charging indicator — any active input source
        sample["battery_charging"] = input_w > 0

        # Runtime — attr 30 in minutes; sentinel 5940 (= 99 h) means idle/charging
        if 30 in data and not sample["battery_charging"]:
            runtime_min = data[30]
            if runtime_min < 5940:
                sample["runtime"] = runtime_min * 60

        # Inverter protection / thermal warning
        if 105 in data:
            sample["problem_code"] = data[105]

        # Per-module SoC (attr 79 = direct %, per slot)
        module_socs = [
            slot_data[79]
            for slot_data in self._ext_batteries.values()
            if 79 in slot_data
        ]
        if module_socs:
            sample["pack_battery_levels"] = module_socs
            sample["pack_count"] = len(module_socs)

        return sample
