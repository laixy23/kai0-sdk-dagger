#!/usr/bin/env python3
"""Commission the four confirmed USB-CAN adapters; never send application frames.

Default/--plan is read-only. --apply needs root and only uses `ip link` to
rename DOWN interfaces, set 1 Mbps, and put them UP. --allow-up-rename also
permits temporarily taking confirmed, idle UP / 1 Mbps adapters DOWN to rename
them. It does not import a robot
SDK, open CAN sockets, enable motors, or send joint/gripper commands.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ADAPTERS = {
    "004A003C4648571620303731": ("follower_left", "can_left_slave"),
    "003000414648570A20303731": ("follower_right", "can_right_slave"),
    "002C002A4648570220393831": ("master_left", "can_left_mas"),
    "001F00434648571720303731": ("master_right", "can_right_mas"),
}
BITRATE = 1_000_000


class Refused(RuntimeError):
    pass


def call(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def receive_entries(stats_path: Path = Path("/proc/net/can/stats")) -> int:
    try:
        stats = stats_path.read_text()
    except OSError as exc:
        raise Refused(f"Cannot check active CAN receivers: {stats_path}: {exc}") from exc
    match = re.search(r"^\s*(\d+)\s+current receive list entries \(CRCV\)", stats, re.M)
    if match is None:
        raise Refused("CAN receiver statistics are unrecognized; refusing network changes")
    count = int(match.group(1))
    if count:
        raise Refused(f"{count} active CAN receive-list entries; stop existing CAN clients first")
    return count


def usb_identity(interface: str, sys_net: Path = Path("/sys/class/net")) -> dict:
    device = (sys_net / interface / "device").resolve()
    for ancestor in (device, *device.parents):
        if (ancestor / "idVendor").exists():
            try:
                return {
                    "serial": (ancestor / "serial").read_text().strip(),
                    "vendor": (ancestor / "idVendor").read_text().strip(),
                    "product": (ancestor / "idProduct").read_text().strip(),
                    "usb_interface": device.name,
                }
            except OSError as exc:
                raise Refused(f"Cannot identify USB-CAN {interface}: {exc}") from exc
    return {"serial": None, "usb_interface": device.name}


def link_data(ip: str) -> list[dict]:
    return json.loads(call([ip, "-j", "-details", "link", "show"]).stdout)


def build_plan(links: list[dict], identities: dict[str, dict], *, allow_up_rename: bool = False) -> list[dict]:
    names = {link["ifname"]: link for link in links}
    selected: dict[str, list[tuple[dict, dict]]] = {serial: [] for serial in ADAPTERS}
    for link in links:
        if link.get("linkinfo", {}).get("info_kind") != "can":
            continue
        identity = identities.get(link["ifname"], {})
        serial = identity.get("serial")
        if serial in selected:
            if identity.get("vendor") != "1d50" or identity.get("product") != "606f":
                raise Refused(f"Unexpected USB vendor/product for adapter {serial}")
            selected[serial].append((link, identity))
    plan = []
    for serial, (role, target) in ADAPTERS.items():
        candidates = selected[serial]
        if len(candidates) != 1:
            raise Refused(f"Expected exactly one {role} adapter {serial}; found {len(candidates)}")
        link, identity = candidates[0]
        name = link["ifname"]
        if target in names and target != name:
            raise Refused(f"Target name {target} is occupied by another interface")
        data = link.get("linkinfo", {}).get("info_data", {})
        up = "UP" in link.get("flags", [])
        bitrate = data.get("bittiming", {}).get("bitrate")
        modes = data.get("ctrlmode", [])
        # An unknown mode representation must not be assumed safe for reuse.
        known_modes = isinstance(modes, list) and all(isinstance(m, str) for m in modes)
        listen_only = known_modes and any(m.upper().replace("_", "-") == "LISTEN-ONLY" for m in modes)
        loopback = known_modes and any(m.upper() == "LOOPBACK" for m in modes)
        normal = (bitrate == BITRATE and known_modes and not listen_only and not loopback
                  and data.get("state") == "ERROR-ACTIVE")
        reusable = name == target and normal
        rename_confirmed_up = allow_up_rename and name != target and normal
        if up and not reusable and not rename_confirmed_up:
            raise Refused(f"{name} is UP but not already named/configured for {role}; refusing to interrupt it")
        plan.append({
            "role": role, "serial": serial, "name_before": name,
            "target_name": target, "usb_interface": identity["usb_interface"],
            "was_up": up, "skip": up and reusable,
            "original_link": link,
        })
    return plan


def discover(ip: str, *, allow_up_rename: bool = False) -> list[dict]:
    receive_entries()
    links = link_data(ip)
    identities = {
        link["ifname"]: usb_identity(link["ifname"])
        for link in links if link.get("linkinfo", {}).get("info_kind") == "can"
    }
    return build_plan(links, identities, allow_up_rename=allow_up_rename)


def apply(ip: str, plan: list[dict], *, allow_up_rename: bool = False) -> list[dict]:
    """Best-effort rollback returns modified interfaces DOWN under their old names."""
    if os.geteuid() != 0:
        raise Refused("--apply requires root; run this exact reviewed script with sudo")
    # Catch replug, rename, or a new client between planning and committing.
    refreshed = discover(ip, allow_up_rename=allow_up_rename)
    signature = lambda rows: [(r["serial"], r["name_before"], r["was_up"], r["skip"]) for r in rows]
    if signature(refreshed) != signature(plan):
        raise Refused("CAN identity/state changed since planning; run again after checking cables")
    touched = []
    try:
        for item in plan:
            if item["skip"]:
                continue
            # UP interfaces require the explicit commissioning flag, normal
            # 1 Mbps state, a confirmed USB serial and no active CAN receivers.
            receive_entries()
            touched.append(dict(item, current_name=item["name_before"]))
            current = touched[-1]
            if item["was_up"]:
                call([ip, "link", "set", "dev", current["current_name"], "down"])
            if current["current_name"] != item["target_name"]:
                call([ip, "link", "set", "dev", current["current_name"], "name", item["target_name"]])
                current["current_name"] = item["target_name"]
            call([ip, "link", "set", "dev", current["current_name"], "type", "can",
                  "bitrate", str(BITRATE), "listen-only", "off"])
            call([ip, "link", "set", "dev", current["current_name"], "up"])
        # Verify every requested link, not just successful subprocess statuses.
        verified = discover(ip)
        if not all(row["skip"] for row in verified):
            raise RuntimeError("Post-activation name/state/bitrate validation failed")
        return verified
    except BaseException as exc:
        errors = []
        for item in reversed(touched):
            try:
                call([ip, "link", "set", "dev", item["current_name"], "down"])
                if item["current_name"] != item["name_before"]:
                    call([ip, "link", "set", "dev", item["current_name"], "name", item["name_before"]])
            except BaseException as rollback_exc:
                errors.append(str(rollback_exc))
        detail = "; rollback errors: " + "; ".join(errors) if errors else "; touched interfaces restored DOWN under original names"
        raise RuntimeError(f"CAN activation failed: {exc}{detail}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Read-only plan (default)")
    mode.add_argument("--apply", action="store_true", help="Explicitly apply network configuration; requires root")
    parser.add_argument("--allow-up-rename", action="store_true",
                        help="Explicitly rename confirmed UP / 1 Mbps adapters after all CAN clients stop")
    parser.add_argument("--receipt", type=Path, help="Write a JSON record after successful completion")
    args = parser.parse_args()
    ip = shutil.which("ip")
    if not ip:
        raise Refused("iproute2 `ip` executable is required")
    plan = discover(ip, allow_up_rename=args.allow_up_rename)
    report = {"checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "mode": "apply" if args.apply else "plan", "bitrate": BITRATE,
              "allow_up_rename": args.allow_up_rename,
              "application_can_frames_sent": 0, "motor_enable_commands_sent": 0,
              "plan": plan}
    if args.apply:
        report["verified"] = apply(ip, plan, allow_up_rename=args.allow_up_rename)
    if args.receipt:
        args.receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[REFUSED] {exc}", file=sys.stderr)
        raise SystemExit(1)
