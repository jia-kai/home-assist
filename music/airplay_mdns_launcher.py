"""Start Music Assistant with live AirPlay discovery and a cached mDNS fallback.

MUSIC_ASSISTANT_IP is required. The launcher refreshes a live receiver snapshot
before MA starts and every minute thereafter. If discovery fails, it advertises
the last valid snapshot until the receiver is visible again. Snapshots are stored
with the Music Assistant data at /data/homepod-mdns.json.
"""

import ipaddress
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, ServiceListener, Zeroconf

logger = logging.getLogger(__name__)
_SERVICE_TYPES = (
    "_airplay._tcp.local.",
    "_raop._tcp.local.",
    "_companion-link._tcp.local.",
    "_mediaremotetv._tcp.local.",
)
_REQUIRED_SERVICE_TYPES = frozenset(("_airplay._tcp.local.", "_raop._tcp.local."))
_STATIC_MARKER = b"ha_static"
_DEFAULT_COMMAND = (
    "/usr/local/bin/entrypoint.sh",
    "--data-dir",
    "/data",
    "--cache-dir",
    "/data/.cache",
)


@dataclass(slots=True)
class AirPlayService:
    """Serializable mDNS service record used for discovery cache and fallback."""

    service_type: str
    """mDNS service type, including the local domain."""

    name: str
    """Fully qualified mDNS service-instance name."""

    server: str
    """Fully qualified SRV target hostname."""

    port: int
    """TCP port from the SRV record."""

    addresses: list[str]
    """IPv4 and IPv6 addresses from resolved A and AAAA records."""

    properties: dict[str, str]
    """UTF-8 mDNS TXT record key-value pairs."""


class AirPlayListener(ServiceListener):
    """Collect live receiver records while excluding this launcher's fallback."""

    target_address: str
    """Receiver IP address that selects matching records."""

    _services: dict[tuple[str, str], AirPlayService]
    """Live matching records keyed by service type and instance name."""

    _lock: threading.Lock
    """Synchronizes Zeroconf callback updates with snapshot collection."""

    def __init__(self, target_address: str) -> None:
        """Initialize collection for one required receiver address.

        Args:
            target_address:
                Normalized IPv4 or IPv6 receiver address.

        """
        self.target_address = target_address
        self._services = {}
        self._lock = threading.Lock()

    def add_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        """Resolve a newly advertised service.

        Args:
            zeroconf:
                Active mDNS resolver used to fetch the service record.

            service_type:
                Browsed mDNS service type.

            name:
                Fully qualified mDNS service-instance name.

        """
        self._store_service(zeroconf, service_type, name)

    def update_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        """Refresh a service after its mDNS data changes.

        Args:
            zeroconf:
                Active mDNS resolver used to fetch the service record.

            service_type:
                Browsed mDNS service type.

            name:
                Fully qualified mDNS service-instance name.

        """
        self._store_service(zeroconf, service_type, name)

    def remove_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        """Discard a service after its mDNS goodbye record.

        Args:
            zeroconf:
                Active mDNS resolver that delivered the removal event.

            service_type:
                Browsed mDNS service type.

            name:
                Fully qualified mDNS service-instance name.

        """
        del zeroconf
        with self._lock:
            self._services.pop((service_type, name), None)

    def services(self) -> list[AirPlayService]:
        """Return a stable copy of currently resolved live service records.

        Returns:
            Matching service records sorted by service-instance name.

        """
        with self._lock:
            return sorted(self._services.values(), key=lambda service: service.name)

    def _store_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        """Resolve and retain one live service record.

        Args:
            zeroconf:
                Active mDNS resolver used to fetch the service record.

            service_type:
                Browsed mDNS service type.

            name:
                Fully qualified mDNS service-instance name.

        """
        info = zeroconf.get_service_info(service_type, name, timeout=3_000)
        if info is None or _STATIC_MARKER in info.properties:
            return
        addresses = info.parsed_addresses(IPVersion.All)
        if self.target_address not in addresses:
            return
        if info.server is None or info.port is None:
            raise ValueError(f"Resolved service lacks an SRV endpoint: {name}")
        properties = {
            key.decode("utf-8", errors="backslashreplace"): value.decode(
                "utf-8", errors="backslashreplace"
            )
            for key, value in info.properties.items()
            if value is not None
        }
        service = AirPlayService(
            service_type=service_type,
            name=info.name,
            server=info.server,
            port=info.port,
            addresses=addresses,
            properties=properties,
        )
        with self._lock:
            self._services[(service_type, name)] = service


def get_configured_ip() -> str:
    """Return the required configured receiver address.

    Returns:
        Normalized IPv4 or IPv6 receiver address.

    Raises:
        RuntimeError: If MUSIC_ASSISTANT_IP is missing or invalid.

    """
    configured_ip = os.environ.get("MUSIC_ASSISTANT_IP")
    if not configured_ip:
        raise RuntimeError("MUSIC_ASSISTANT_IP is required")
    try:
        return str(ipaddress.ip_address(configured_ip))
    except ValueError as error:
        raise RuntimeError("MUSIC_ASSISTANT_IP must be an IP address") from error


def get_source_address(target_address: str) -> str:
    """Return the local address selected by the route to the receiver.

    Args:
        target_address:
            Receiver IPv4 or IPv6 address.

    Returns:
        Local source address on the receiver's routed interface.

    """
    target = ipaddress.ip_address(target_address)
    family = socket.AF_INET6 if target.version == 6 else socket.AF_INET
    endpoint: tuple[str, int] | tuple[str, int, int, int]
    if family == socket.AF_INET6:
        endpoint = (target_address, 5353, 0, 0)
    else:
        endpoint = (target_address, 5353)
    with socket.socket(family, socket.SOCK_DGRAM) as probe:
        probe.connect(endpoint)
        return str(probe.getsockname()[0])


def require_complete_snapshot(services: list[AirPlayService], target_address: str) -> None:
    """Validate that a snapshot can recreate the configured AirPlay receiver.

    Args:
        services:
            Service records to validate.

        target_address:
            Required receiver address present in every record.

    Raises:
        RuntimeError: If the snapshot lacks required service types or endpoint data.

    """
    found_types = {service.service_type for service in services}
    missing_types = _REQUIRED_SERVICE_TYPES - found_types
    if missing_types:
        raise RuntimeError(f"mDNS snapshot is missing service types: {sorted(missing_types)}")
    for service in services:
        if target_address not in service.addresses:
            raise RuntimeError(f"mDNS snapshot service does not contain {target_address}: {service.name}")
        if not service.server or not 1 <= service.port <= 65_535:
            raise RuntimeError(f"mDNS snapshot has an invalid SRV endpoint: {service.name}")


def read_snapshot(path: Path, target_address: str) -> list[AirPlayService]:
    """Load and validate cached receiver records.

    Args:
        path:
            JSON snapshot path.

        target_address:
            Required receiver address present in every record.

    Returns:
        Complete cached receiver service records.

    Raises:
        RuntimeError: If the snapshot is unavailable or invalid JSON.

        TypeError: If the decoded JSON does not match the snapshot schema.

    """
    try:
        raw_services = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"No live mDNS records and no snapshot at {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid mDNS snapshot JSON: {path}") from error
    if not isinstance(raw_services, list):
        raise TypeError(f"mDNS snapshot must be a JSON list: {path}")
    services: list[AirPlayService] = []
    for raw_service in raw_services:
        if not isinstance(raw_service, dict):
            raise TypeError(f"mDNS snapshot has an invalid service record: {path}")
        try:
            service_type = raw_service["service_type"]
            name = raw_service["name"]
            server = raw_service["server"]
            port = raw_service["port"]
            addresses = raw_service["addresses"]
            properties = raw_service["properties"]
        except KeyError as error:
            raise RuntimeError(f"mDNS snapshot has a missing field: {path}") from error
        if (
            not isinstance(service_type, str)
            or not isinstance(name, str)
            or not isinstance(server, str)
            or not isinstance(port, int)
            or not isinstance(addresses, list)
            or not isinstance(properties, dict)
            or not all(isinstance(address, str) for address in addresses)
            or not all(isinstance(key, str) and isinstance(value, str) for key, value in properties.items())
        ):
            raise RuntimeError(f"mDNS snapshot has invalid field types: {path}")
        for address in addresses:
            try:
                ipaddress.ip_address(address)
            except ValueError as error:
                raise RuntimeError(f"mDNS snapshot has an invalid address: {address}") from error
        services.append(
            AirPlayService(
                service_type=service_type,
                name=name,
                server=server,
                port=port,
                addresses=addresses,
                properties=properties,
            )
        )
    require_complete_snapshot(services, target_address)
    return services


def write_snapshot(path: Path, services: list[AirPlayService]) -> None:
    """Atomically save a complete live receiver mDNS snapshot.

    Args:
        path:
            Destination JSON snapshot path.

        services:
            Complete live receiver records to persist.

    """
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    output = json.dumps([asdict(service) for service in services], indent=2, sort_keys=True) + "\n"
    temporary_path.write_text(output, encoding="utf-8")
    temporary_path.replace(path)


def discover_services(
    zeroconf: Zeroconf, target_address: str, timeout_seconds: float
) -> list[AirPlayService]:
    """Collect complete live AirPlay records for one receiver.

    Args:
        zeroconf:
            Active mDNS resolver bound to the receiver's network interface.

        target_address:
            Receiver address that selects matching records.

        timeout_seconds:
            Maximum collection duration in seconds.

    Returns:
        Complete matching service records, or an empty list when unavailable.

    """
    listener = AirPlayListener(target_address)
    browser = ServiceBrowser(zeroconf, list(_SERVICE_TYPES), listener)
    try:
        time.sleep(timeout_seconds)
    finally:
        browser.cancel()
    services = listener.services()
    try:
        require_complete_snapshot(services, target_address)
    except RuntimeError:
        return []
    return services


class FallbackAdvertiser:
    """Register cached AirPlay records while the live receiver is undiscoverable."""

    _zeroconf: Zeroconf
    """mDNS responder that owns registered fallback records."""

    _registered: list[ServiceInfo]
    """Fallback records currently registered with the responder."""

    def __init__(self, zeroconf: Zeroconf) -> None:
        """Initialize an inactive fallback advertiser.

        Args:
            zeroconf:
                Active mDNS responder used to register fallback records.

        """
        self._zeroconf = zeroconf
        self._registered = []

    def start(self, services: list[AirPlayService]) -> None:
        """Advertise a cached receiver snapshot until explicitly stopped.

        Args:
            services:
                Validated cached receiver records to advertise.

        """
        if self._registered:
            return
        for service in services:
            properties = {key.encode(): value.encode() for key, value in service.properties.items()}
            properties[_STATIC_MARKER] = b"1"
            info = ServiceInfo(
                type_=service.service_type,
                name=service.name,
                addresses=[ipaddress.ip_address(address).packed for address in service.addresses],
                port=service.port,
                properties=properties,
                server=service.server,
            )
            self._zeroconf.register_service(info)
            self._registered.append(info)
        logger.warning("Advertising cached AirPlay mDNS records")

    def stop(self) -> None:
        """Withdraw all currently advertised fallback records."""
        for info in reversed(self._registered):
            self._zeroconf.unregister_service(info)
        self._registered.clear()

    @property
    def active(self) -> bool:
        """Return whether fallback records are currently advertised."""
        return bool(self._registered)


def main() -> int:
    """Launch MA and reconcile live AirPlay discovery once each minute.

    Returns:
        Music Assistant's process exit status.

    """
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    target_address = get_configured_ip()
    cache_path = Path("/data/homepod-mdns.json")
    source_address = get_source_address(target_address)
    command = tuple(sys.argv[1:]) or _DEFAULT_COMMAND
    stop_event = threading.Event()

    def request_shutdown(signum: int, frame: object) -> None:
        """Request graceful shutdown after a container termination signal.

        Args:
            signum:
                Received POSIX signal number.

            frame:
                Interrupted interpreter frame, unused by this handler.

        """
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    with Zeroconf(interfaces=[source_address], ip_version=IPVersion.All) as zeroconf:
        advertiser = FallbackAdvertiser(zeroconf)
        live_services = discover_services(zeroconf, target_address, timeout_seconds=5)
        if live_services:
            write_snapshot(cache_path, live_services)
            logger.info("Refreshed live AirPlay mDNS snapshot")
        else:
            advertiser.start(read_snapshot(cache_path, target_address))
        music_assistant = subprocess.Popen(command)
        try:
            while music_assistant.poll() is None and not stop_event.wait(60):
                live_services = discover_services(zeroconf, target_address, timeout_seconds=5)
                if live_services:
                    write_snapshot(cache_path, live_services)
                    advertiser.stop()
                    logger.info("Refreshed live AirPlay mDNS snapshot")
                    continue
                if not advertiser.active:
                    advertiser.start(read_snapshot(cache_path, target_address))
        finally:
            if music_assistant.poll() is None:
                music_assistant.terminate()
                try:
                    music_assistant.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    music_assistant.kill()
                    music_assistant.wait()
            advertiser.stop()
    return music_assistant.returncode


if __name__ == "__main__":
    raise SystemExit(main())
