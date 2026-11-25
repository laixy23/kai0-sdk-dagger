#!/usr/bin/env python3
"""Continuous receive-only native master data, isolated by handoff phase.

Actual priming uses raw 2A*/26* feedback. FA forwarding uses hardware-originated
155/156/157/159 command packets as teaching inputs, never as measured pose.
Kernel source timestamps are retained; no SDK cache or zero-value fallback exists.
"""
from __future__ import annotations

import math
from collections import Counter
import struct
import threading
import time

from prepare_follower_enabled import checked_enabled_feedback
from prepare_follower_mode import REQUIRED_IDS, decode_feedback


MASTER_CHANNELS = ("can_left_mas", "can_right_mas")
ACTUAL_IDS = tuple(REQUIRED_IDS)
ACTUAL_POSE_IDS = (0x2A5, 0x2A6, 0x2A7, 0x2A8)
FA_IDS = (0x155, 0x156, 0x157, 0x159)
MAX_AGE = {"actual": 0.2, "fa": 0.15}
MAX_POSE_SPAN = {"actual": 0.05, "fa": 0.02}
MAX_FUTURE_S = 0.05


def _gripper_command(raw, *, key, status, data):
    """Preserve signed native WIDTH while matching the legacy follower clamp."""
    context = f"id=0x{key:03X} raw_gripper_um={raw} status={status} hex={bytes(data).hex()}"
    if abs(raw) > 150000:
        raise RuntimeError("Native WIDTH exceeds SDK signed +/-150000 um filter; " + context)
    return min(80000, max(0, raw))


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _phase_ids(kind):
    if kind == "actual":
        return ACTUAL_IDS
    if kind == "fa":
        return FA_IDS
    raise ValueError("Native phase must be 'actual' or 'fa'")


def decode_native_snapshot(kind, frames, *, since_timestamp, now, epoch=0, after=None):
    """Decode a complete, fresh and coherent phase; incomplete/stale is pending."""
    required = _phase_ids(kind)
    if not _finite(now) or not _finite(since_timestamp) or since_timestamp <= 0:
        raise ValueError("Invalid native phase/source clock")
    if not all(key in frames for key in required):
        return None
    stamps = {}
    for key in required:
        data, stamp = frames[key]
        if len(data) != 8 or not _finite(stamp):
            raise RuntimeError(f"Malformed native frame 0x{key:03X}")
        age = now - stamp
        if age < -MAX_FUTURE_S:
            raise RuntimeError(f"Future native frame 0x{key:03X}")
        if stamp < since_timestamp or age > MAX_AGE[kind]:
            return None
        stamps[f"0x{key:03X}"] = stamp
    if after is not None:
        if not isinstance(after, dict) or not all(key in after and _finite(after[key]) for key in stamps):
            raise ValueError("Previous component stamps must cover this native phase")
        if not all(stamp > after[key] for key, stamp in stamps.items()):
            return None
    pose_ids = ACTUAL_POSE_IDS if kind == "actual" else FA_IDS
    pose_stamps = [frames[key][1] for key in pose_ids]
    if max(pose_stamps) - min(pose_stamps) > MAX_POSE_SPAN[kind]:
        return None
    if kind == "actual":
        row = decode_feedback(frames, now, since_timestamp=since_timestamp)
        joints, _ = checked_enabled_feedback(row, now=now, since_timestamp=since_timestamp)
        if row["ctrl_mode"] != 1:
            raise RuntimeError("Native actual priming requires CAN1 control")
        gripper = row["gripper_um"]
    else:
        joints = [value for key in FA_IDS[:3] for value in struct.unpack(">ii", frames[key][0])]
        data = frames[0x159][0]
        # V3 gripper control uses bit 2 (0x04) of byte 6 for ANGLE mode.
        if data[6] not in (0x00, 0x01, 0x02, 0x03):
            raise RuntimeError(f"Native teaching gripper status is not WIDTH: id=0x159 raw_gripper_um={struct.unpack('>i', data[:4])[0]} status={data[6]} hex={data.hex()}")
        gripper = struct.unpack(">i", data[:4])[0]
    grip_id = 0x2A8 if kind == "actual" else 0x159
    grip_data = frames[grip_id][0]
    command_gripper = _gripper_command(gripper, key=grip_id,
                                      status=grip_data[7] if kind == "actual" else grip_data[6],
                                      data=grip_data)
    return {"kind": kind, "epoch": epoch, "joint_mdeg": joints,
            "gripper_um": gripper, "raw_gripper_um": gripper,
            "command_gripper_um": command_gripper, "source_time": min(pose_stamps),
            "gripper_effort_mnm": struct.unpack(">h", grip_data[4:6])[0],
            "gripper_status_byte": grip_data[6], "gripper_byte7": grip_data[7],
            "gripper_hex": grip_data.hex(), "gripper_source_time": frames[grip_id][1],
            "component_stamps": stamps}


class NativeMasterCache:
    """Thread-safe raw cache; phases clear all old native values atomically."""
    def __init__(self, *, wall=time.time):
        self._wall = wall
        self._lock = threading.Lock()
        self._kind = None
        self._epoch = 0
        self._since = None
        self._frames = {}
        self._failure = None
        self._counts = Counter()
        self._last_rx = {}
        self._rejections = Counter()
        self._last_rejection = None

    def begin_phase(self, kind, since_timestamp):
        _phase_ids(kind)
        if (not _finite(since_timestamp) or since_timestamp <= 0
                or since_timestamp - self._wall() > MAX_FUTURE_S):
            raise ValueError("Invalid/future native phase start")
        with self._lock:
            self._epoch += 1
            self._kind, self._since = kind, since_timestamp
            self._frames = {}
            self._failure = None
            self._counts = Counter()
            self._last_rx = {}
            self._rejections = Counter()
            self._last_rejection = None
            return self._epoch

    def ingest(self, message):
        """Store only physical RX, never another host socket's control loopback."""
        with self._lock:
            key = getattr(message, 'arbitration_id', None)
            stamp = getattr(message, 'timestamp', None)
            try:
                observed_data = bytes(getattr(message, 'data', b''))
            except (TypeError, ValueError):
                observed_data = b''
            def reject(reason):
                self._rejections[reason] += 1
                self._last_rejection = {'id': f'0x{key:03X}' if type(key) is int else str(key),
                                        'reason': reason, 'timestamp_sec': stamp,
                                        'hex': observed_data.hex()}
                return False
            try:
                if getattr(message, "is_rx", None) is not True:
                    return reject('host_tx_or_missing_rx_marker')
                if getattr(message, "is_error_frame", False):
                    reject('can_error_frame')
                    raise RuntimeError("SocketCAN error frame in native master listener")
                if self._kind is None:
                    return False
                if (getattr(message, "is_remote_frame", False)
                        or getattr(message, "is_extended_id", False)
                        or getattr(message, "is_fd", False)):
                    return reject('rtr_extended_or_fd')
                key = message.arbitration_id
                stamp = message.timestamp
                if not _finite(stamp):
                    reject('invalid_timestamp')
                    raise RuntimeError("Invalid native CAN source timestamp")
                if stamp < self._since:
                    return reject('old_phase')
                name = f'0x{key:03X}'
                self._counts[name] += 1
                self._last_rx[name] = {'timestamp_sec': stamp, 'hex': observed_data.hex()}
                if key not in _phase_ids(self._kind):
                    return reject('not_phase_required_id')
                age = self._wall() - stamp
                if age < -MAX_FUTURE_S:
                    reject('future')
                    raise RuntimeError(f"Future native CAN frame id={name} hex={observed_data.hex()} source={stamp}")
                if age > MAX_AGE[self._kind]:
                    return reject('stale')
                data = bytes(message.data)
                if len(data) != 8 or getattr(message, "dlc", len(data)) != 8:
                    reject('short_or_invalid_dlc')
                    raise RuntimeError(f"Native master frame must contain eight bytes id={name} hex={data.hex()}")
                previous = self._frames.get(key)
                if previous is not None and stamp <= previous[1]:
                    return reject('not_advanced')
                if self._kind == "fa" and key == 0x159:
                    if data[6] not in (0x00, 0x01, 0x02, 0x03):
                        reject('non_width_status')
                        raise RuntimeError(f"Native teaching gripper is not WIDTH id=0x159 raw_gripper_um={struct.unpack('>i', data[:4])[0]} status={data[6]} hex={data.hex()}")
                    try:
                        _gripper_command(struct.unpack(">i", data[:4])[0], key=key, status=data[6], data=data)
                    except RuntimeError:
                        reject('width_outlier')
                        raise
                self._frames[key] = (data, stamp)
                return True
            except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                self._failure = exc
                raise

    def snapshot(self, after=None):
        with self._lock:
            if self._failure is not None:
                raise RuntimeError("Native master cache rejected a frame: " + str(self._failure))
            if self._kind is None:
                return None
            kind, epoch, since, frames = self._kind, self._epoch, self._since, dict(self._frames)
        result = decode_native_snapshot(kind, frames, since_timestamp=since,
                                        now=self._wall(), epoch=epoch, after=after)
        with self._lock:
            # A transition racing this decode must not return the previous phase.
            if epoch != self._epoch:
                return None
            if self._failure is not None:
                raise RuntimeError("Native master cache failed during decoding: " + str(self._failure))
        return result

    def diagnostics(self, after=None):
        """Report this phase's physical RX and rejection evidence without CAN I/O."""
        now = self._wall()
        with self._lock:
            kind, epoch, since = self._kind, self._epoch, self._since
            required = _phase_ids(kind) if kind is not None else ()
            frames = dict(self._frames)
            counts = dict(self._counts)
            last_rx = {key: dict(value, age_s=now - value['timestamp_sec'])
                       for key, value in self._last_rx.items()}
            rejections, rejected = dict(self._rejections), self._last_rejection
            failure = str(self._failure) if self._failure is not None else None
        missing = [f'0x{key:03X}' for key in required if key not in frames]
        stale = [f'0x{key:03X}' for key in required if key in frames
                 and not -MAX_FUTURE_S <= now - frames[key][1] <= MAX_AGE[kind]]
        unadvanced = [f'0x{key:03X}' for key in required if key in frames and after is not None
                      and f'0x{key:03X}' in after and frames[key][1] <= after[f'0x{key:03X}']]
        pose_ids = ACTUAL_POSE_IDS if kind == 'actual' else FA_IDS
        pose_stamps = [frames[key][1] for key in pose_ids if key in frames]
        return {'kind': kind, 'epoch': epoch, 'since_timestamp': since,
                'rx_counts': counts, 'last_rx': last_rx, 'missing_ids': missing,
                'stale_ids': stale, 'unadvanced_ids': unadvanced,
                'pose_span_s': max(pose_stamps) - min(pose_stamps) if len(pose_stamps) == len(pose_ids) else None,
                'rejections': rejections, 'last_rejection': rejected, 'failure': failure,
                'native_actual_rx_after_fa': {key: count for key, count in counts.items()
                                               if kind == 'fa' and key.startswith('0x2A')}}


def _default_bus_factory(channel):
    import can  # Opening a receive-only socket creates no SDK and sends no CAN.
    # Observe other hardware IDs for failure diagnostics (e.g. 2A* still arriving
    # after FA). Only the explicitly required phase IDs can enter forwarding.
    return can.Bus(channel=channel, interface="socketcan", receive_own_messages=False)


class NativeMasterReader:
    """Keep one passive listener open before FA and across every phase change."""
    def __init__(self, channel, shutdown=lambda: False, *, bus_factory=_default_bus_factory,
                 wall=time.time, mono=time.monotonic):
        if channel not in MASTER_CHANNELS:
            raise ValueError("Native master reader requires an explicitly mapped master CAN channel")
        self.channel = channel
        self._shutdown = shutdown
        self._factory = bus_factory
        self._wall, self._mono = wall, mono
        self._cache = NativeMasterCache(wall=wall)
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread = None
        self._running = False
        self._failure = None

    def start(self):
        with self._state_lock:
            if self._thread is not None:
                if self._thread.is_alive() and not self._stop.is_set():
                    return self
                raise RuntimeError("Native master listener cannot be restarted after stopping/failure")
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        bus = None
        try:
            bus = self._factory(self.channel)
            with self._state_lock:
                self._running = True
            self._started.set()
            while not self._stop.is_set() and not self._shutdown():
                message = bus.recv(timeout=0.02)
                if message is not None:
                    self._cache.ingest(message)
        except Exception as exc:
            if not self._stop.is_set():
                with self._state_lock:
                    self._failure = exc
        finally:
            if bus is not None:
                try:
                    bus.shutdown()  # Close only; this listener never starts TX tasks.
                except Exception as exc:
                    with self._state_lock:
                        if self._failure is None:
                            self._failure = exc
            with self._state_lock:
                self._running = False
            self._started.set()  # Failed opening also unblocks wait_started.

    def _require_running(self):
        with self._state_lock:
            if self._failure is not None:
                raise RuntimeError("Native master listener failed: " + str(self._failure))
            if not self._running or self._stop.is_set():
                raise RuntimeError("Native master listener is not running")

    def wait_started(self, timeout=1.0):
        if not _finite(timeout) or timeout < 0:
            raise ValueError("Invalid native listener startup timeout")
        if self._thread is None:
            raise RuntimeError("Native listener start() has not been called")
        if not self._started.wait(timeout):
            raise TimeoutError("Native master CAN listener did not open before timeout")
        self._require_running()
        return True

    def begin_phase(self, kind, since_timestamp):
        self._require_running()
        return self._cache.begin_phase(kind, since_timestamp)

    def snapshot(self, after=None):
        self._require_running()
        result = self._cache.snapshot(after=after)
        self._require_running()
        return result

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
            if thread.is_alive():
                raise RuntimeError("Native master listener did not stop within one second")

    def diagnostics(self, after=None):
        result = self._cache.diagnostics(after=after)
        with self._state_lock:
            result.update(listener_running=self._running,
                          listener_failure=str(self._failure) if self._failure is not None else None)
        return result
