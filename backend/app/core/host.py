"""What "localhost" means once the app is in a container.

The image runs the app and nothing else, so a customer whose RAG system or
embedding server sits on the same machine will type the address they always
type — localhost:3000 — and inside the container that is the container itself.
Teaching them that the host is called host.docker.internal, and that it differs
by platform, is not something this product should be asking.

So the image declares the name instead of the app guessing at it:
RBL_HOST_GATEWAY is set in the Dockerfile, with --add-host in the run command
to make it resolve on Linux (Docker Desktop resolves it already). Every request
that leaves the app translates a loopback host into it. Nothing is translated
when the variable is empty, which is the case when start.sh runs the app
directly — there localhost already means what the user means, and a translation
would be the bug.

Deliberately not sniffing /.dockerenv: that file is a Docker implementation
detail, absent under containerd and CRI-O, and even when present it says only
"in a container" — never what the host is called.
"""

import os
from urllib.parse import urlsplit, urlunsplit

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def host_gateway(url: str) -> str:
    """Swap a loopback host for the declared gateway; anything else is left be.

    Only the host changes — scheme, port, path and query stay the user's, so
    what the UI shows them keeps matching what they typed.
    """
    gateway = os.environ.get("RBL_HOST_GATEWAY", "").strip()
    if not gateway or not url:
        return url
    # A pasted "localhost:3000/v1" has no scheme, and urlsplit reads the host as
    # the scheme when it is missing. Prefixing "//" puts it back in the netloc;
    # the prefix comes off again below so the shape handed back is the one
    # handed in.
    prefixed = url if "//" in url else "//" + url
    parts = urlsplit(prefixed)
    if not parts.hostname or parts.hostname.lower() not in _LOCAL_HOSTS:
        return url
    host = gateway if parts.port is None else f"{gateway}:{parts.port}"
    resolved = urlunsplit(parts._replace(netloc=host))
    return resolved if "//" in url else resolved[2:]
