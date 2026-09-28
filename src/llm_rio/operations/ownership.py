"""Process-lifetime ownership, acquired before recovery and released after teardown.

Locks are never unlinked: replacing an inode would admit two concurrent owners.
All cooperating service instances on a host must run as the same service account.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path


@contextmanager
def owner_lock(resource: str, *, root: Path | None = None) -> Iterator[None]:
    root = root or Path("/tmp") / f"llm-rio-owners-{os.getuid()}"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise RuntimeError(
            f"Ownership lock directory must be private to this service account: {root}"
        )
    path = root / (hashlib.sha256(resource.encode()).hexdigest() + ".lock")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"Another service owns {resource}; stop it before continuing"
            ) from None
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n{resource}\n".encode())
        yield
    finally:
        os.close(descriptor)


def database_resource(path: Path) -> str:
    return f"database:{path.resolve()}"


@contextmanager
def gpu_ownership(uuids: list[str]) -> Iterator[None]:
    with ExitStack() as locks:
        for uuid in sorted(set(uuids)):
            locks.enter_context(owner_lock(f"gpu:{uuid}"))
        yield
