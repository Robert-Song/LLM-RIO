"""Service-owned private port reservations, retained until verified process teardown."""

from __future__ import annotations

import socket


class PortAllocator:
    def __init__(self, start: int, end: int) -> None:
        if not 1 <= start <= end <= 65535:
            raise ValueError("Private ports must be between 1 and 65535")
        self.start = start
        self.end = end
        self._owners: dict[int, str] = {}

    @staticmethod
    def available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind(("127.0.0.1", port))
            except OSError:
                return False
        return True

    def reserve(self, owner: str) -> int:
        # Calls are synchronous on the service event loop, so reservation is atomic.
        for port in range(self.start, self.end + 1):
            if port not in self._owners and self.available(port):
                self._owners[port] = owner
                return port
        raise RuntimeError(f"Private port range {self.start}-{self.end} is exhausted")

    def release(self, port: int, owner: str) -> None:
        if self._owners.get(port) != owner:
            raise RuntimeError("Only the port reservation owner may release it")
        del self._owners[port]

    def snapshot(self) -> dict[int, str]:
        return dict(self._owners)
