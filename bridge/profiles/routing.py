"""Tailscale IP-to-profile routing (Phase H.d).

Maps stable Tailscale IPs to profile names so the system auto-detects
which device is talking and switches to the right profile. This avoids
cookies (fragile — cleared on whim) and works reliably because every
device on a tailnet keeps the same IP across reconnects.

Config lives at ``data/profile_routing.yaml`` (shared across profiles,
since it maps devices TO profiles, not within one):

    tailscale_ip:
      100.64.0.10:                    # your actual Tailscale IPs go here
        profile: work
        nickname: Work Laptop
      100.64.0.20:
        profile: family
        nickname: My Phone

(Tailscale uses the CGNAT range 100.64.0.0/10 — replace the example
addresses above with the real stable IPs you see in `tailscale status`
on your own tailnet.)

Backward compatible: old ``{ip: "profile_name"}`` string values still
parse (treated as profile with empty nickname).

Resolution is read-time (no caching beyond the yaml parse) so edits
take effect on the next request without a restart.

Public API:
    load_routing()              -> RoutingConfig
    resolve_profile_for_ip(ip)  -> str | None
    save_routing(config)        -> Path
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from bridge.utils import REPO_ROOT, atomic_write_text

log = logging.getLogger("bridge.profiles.routing")


def _routing_path() -> Path:
    return REPO_ROOT / "data" / "profile_routing.yaml"


@dataclass(frozen=True)
class DeviceRoute:
    """One IP→profile mapping with an optional human-friendly nickname."""
    profile: str
    nickname: str = ""


@dataclass
class RoutingConfig:
    """Parsed routing table. ``tailscale_ip`` maps IP strings to
    DeviceRoute entries (profile name + optional nickname)."""
    tailscale_ip: dict[str, DeviceRoute] = field(default_factory=dict)


def load_routing() -> RoutingConfig:
    """Read the routing yaml, returning an empty config on any failure.
    Never raises — a missing or broken file just means "no IP routing,
    fall back to manual profile selection."

    Backward compatible: old ``{ip: "profile_name"}`` string values
    parse as DeviceRoute(profile=..., nickname="").
    """
    path = _routing_path()
    if not path.exists():
        return RoutingConfig()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("profile_routing.yaml unreadable (%s); ignoring", e)
        return RoutingConfig()
    if not isinstance(raw, dict):
        return RoutingConfig()
    ip_map = raw.get("tailscale_ip") or {}
    if not isinstance(ip_map, dict):
        return RoutingConfig()
    cleaned: dict[str, DeviceRoute] = {}
    for k, v in ip_map.items():
        ip = str(k).strip()
        if not ip:
            continue
        if isinstance(v, str):
            profile = v.strip().lower()
            if profile:
                cleaned[ip] = DeviceRoute(profile=profile)
        elif isinstance(v, dict):
            profile = str(v.get("profile", "")).strip().lower()
            nickname = str(v.get("nickname", "")).strip()
            if profile:
                cleaned[ip] = DeviceRoute(profile=profile, nickname=nickname)
    return RoutingConfig(tailscale_ip=cleaned)


def resolve_profile_for_ip(ip: str | None) -> str | None:
    """Return the profile name mapped to ``ip``, or None if unmapped."""
    if not ip:
        return None
    cfg = load_routing()
    route = cfg.tailscale_ip.get(ip.strip())
    return route.profile if route else None


def save_routing(config: RoutingConfig) -> Path:
    """Write the routing config to disk. Pretty-printed YAML."""
    path = _routing_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    if config.tailscale_ip:
        ip_block: dict = {}
        for ip, route in config.tailscale_ip.items():
            entry: dict = {"profile": route.profile}
            if route.nickname:
                entry["nickname"] = route.nickname
            ip_block[ip] = entry
        out["tailscale_ip"] = ip_block
    atomic_write_text(
        path,
        yaml.dump(out, default_flow_style=False, allow_unicode=True),
    )
    return path


def get_routing_dict() -> list[dict]:
    """Convenience for the API: return the routing entries as a list
    of {ip, profile, nickname} dicts for JSON serialization."""
    cfg = load_routing()
    return [
        {"ip": ip, "profile": r.profile, "nickname": r.nickname}
        for ip, r in cfg.tailscale_ip.items()
    ]


def discover_tailscale_peers() -> list[dict]:
    """Query the local Tailscale daemon for peers on the tailnet.

    Returns a list of dicts with keys: hostname, ip, os, online.
    Empty list on any failure (socket missing, tailscale not running,
    etc.) — the Settings UI falls back to manual IP entry.

    Talks to tailscaled via its Unix socket at
    ``/var/run/tailscale/tailscaled.sock``. The socket is mounted
    read-only from the host in docker-compose.yml.
    """
    import httpx
    SOCKET = "/var/run/tailscale/tailscaled.sock"
    try:
        transport = httpx.HTTPTransport(uds=SOCKET)
        with httpx.Client(transport=transport, timeout=5.0) as client:
            resp = client.get("http://local-tailscaled.sock/localapi/v0/status")
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.debug("tailscale peer discovery failed: %s", e)
        return []

    peers: list[dict] = []
    for _key, peer in (data.get("Peer") or {}).items():
        ips = peer.get("TailscaleIPs") or []
        ipv4 = next((ip for ip in ips if "." in ip), ips[0] if ips else "")
        if not ipv4:
            continue
        peers.append({
            "hostname": peer.get("HostName", ""),
            "ip": ipv4,
            "os": peer.get("OS", ""),
            "online": bool(peer.get("Online", False)),
        })
    peers.sort(key=lambda p: (not p["online"], p["hostname"].lower()))
    return peers


def set_routing_list(entries: list[dict]) -> Path:
    """Convenience for the API: overwrite the routing table from a list
    of ``{ip, profile, nickname?}`` dicts. Validates profile names."""
    from bridge.profiles import validate_name
    cleaned: dict[str, DeviceRoute] = {}
    for entry in entries:
        ip_s = str(entry.get("ip", "")).strip()
        profile_s = validate_name(str(entry.get("profile", "")))
        nickname = str(entry.get("nickname", "")).strip()
        if ip_s:
            cleaned[ip_s] = DeviceRoute(profile=profile_s, nickname=nickname)
    return save_routing(RoutingConfig(tailscale_ip=cleaned))
