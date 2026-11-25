#!/usr/bin/env python3
"""Add calibrated USB pedal input around the unchanged keyboard Bash entry.

The existing Bash/supervisor still owns all robot startup and shutdown. This
outer process owns only its child process group, PTY and private input pipes.
It never imports ROS/SDK, sends robot commands, or stops unrelated services.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import os
import pty
import select
import selectors
import signal
import subprocess
import sys
import termios
import time
import tty

from foot_pedal_device import FootPedalDevice, PedalGestures, load_config
def drain_pty(master_fd, output_fd):
    if master_fd is None:
        return
    while True:
        try:
            ready, _, _ = select.select([master_fd], [], [], 0)
        except (OSError, ValueError):
            return
        if not ready:
            return
        try:
            data = os.read(master_fd, 65536)
        except OSError as exc:
            if exc.errno == errno.EIO:
                return
            raise
        if not data:
            return
        try:
            os.write(output_fd, data)
        except OSError:
            return


def run_pedal_session(command, config, *, device=None, input_fd=0, output_fd=1,
                      standby_on_pedal_stop=False):
    config = load_config(config)
    standby_enabled = bool(standby_on_pedal_stop) and len(config['bindings']) == 2
    pedals = FootPedalDevice(config) if device is None else device
    process = None
    master_fd = slave_fd = command_read = command_write = ready_read = ready_write = None
    saved_terminal = None
    saved_handlers = {}
    terminal_signal = None
    requested_signal = None
    pedal_stop_requested = False

    def emit(message):
        try:
            os.write(output_fd, message.encode())
        except OSError:
            pass

    def forward_child_signal(signum):
        nonlocal requested_signal
        if requested_signal is None:
            requested_signal = signum
        if process is not None and process.poll() is None:
            try:
                # Exactly the foreground group created for this Bash invocation.
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    def handle_os_signal(signum, _frame=None):
        nonlocal terminal_signal
        if terminal_signal is None:
            terminal_signal = signum
        forward_child_signal(signum)

    def drain_queued_pedal_events():
        pedal_fds = list(pedals.filenos)
        while pedal_fds:
            readable = select.select(pedal_fds, [], [], 0)[0]
            if not readable:
                break
            for descriptor in readable:
                try:
                    events = pedals.read_events(descriptor)
                except (OSError, RuntimeError):
                    pedals.close()
                    return
                if not events:
                    return
            pedal_fds = list(pedals.filenos)

    def close_round_descriptors():
        nonlocal master_fd, slave_fd, command_read, command_write, ready_read, ready_write
        for descriptor in (master_fd, slave_fd, command_read, command_write, ready_read, ready_write):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        master_fd = slave_fd = command_read = command_write = ready_read = ready_write = None

    def wait_for_standby_restart():
        nonlocal terminal_signal
        drain_queued_pedal_events()
        if not pedals.filenos:
            emit('\n[踏板] 踏板已断开，退出待机模式。\n')
            return False
        gestures = PedalGestures(config['bindings'])
        standby_entered_at = time.monotonic()
        emit('\n' + '=' * 70 + '\n')
        emit('[待机] 本轮采集与保存已完成，四臂转发已停止。\n')
        emit('[待机] 重新铺好衣物并确认工作区安全后：\n')
        emit('       - 踩「左双踩」→ 开启新一轮采集（自动归位并开始模型运行）\n')
        emit('       - 键盘按 Ctrl+C → 彻底退出程序\n')
        emit('       （待机期间单踩与右双踩均忽略，防止铺衣服时误触发）\n')
        emit('=' * 70 + '\n')

        with selectors.DefaultSelector() as selector:
            selector.register(input_fd, selectors.EVENT_READ, ('keyboard', None))
            for descriptor in pedals.filenos:
                selector.register(descriptor, selectors.EVENT_READ, ('pedal', None))

            def read_standby_pedal_events(fd):
                try:
                    return pedals.read_events(fd)
                except (OSError, RuntimeError) as exc:
                    for descriptor in pedals.filenos:
                        try:
                            selector.unregister(descriptor)
                        except KeyError:
                            pass
                    pedals.close()
                    emit('\n[踏板] 输入已断开：' + str(exc) + '；退出待机模式。\n')
                    return None

            while terminal_signal is None:
                pedal_events = []
                try:
                    ready_keys = selector.select(timeout=0.02)
                except InterruptedError:
                    continue
                for key, _ in ready_keys:
                    kind = key.data[0]
                    if kind == 'pedal':
                        events = read_standby_pedal_events(key.fd)
                        if events is None:
                            return False
                        pedal_events.extend(events)
                        continue
                    try:
                        data = os.read(key.fd, 65536)
                    except OSError:
                        data = b''
                    if not data:
                        if terminal_signal is None:
                            terminal_signal = signal.SIGHUP
                        return False
                    if b'\x03' in data:
                        if terminal_signal is None:
                            terminal_signal = signal.SIGINT
                        return False
                pedal_fds = list(pedals.filenos)
                if pedal_fds:
                    for descriptor in select.select(pedal_fds, [], [], 0)[0]:
                        events = read_standby_pedal_events(descriptor)
                        if events is None:
                            return False
                        pedal_events.extend(events)
                processing_now = time.monotonic()
                pedal_events.sort(key=lambda event: event[4] if len(event) == 5 else processing_now)
                for event in pedal_events:
                    eligible = len(event) != 5 or event[4] >= standby_entered_at
                    tokens = gestures.feed(event, processing_now, allow_short=eligible)
                    if 'resume' in tokens:
                        emit('\n[踏板] 左双踩：开启新一轮采集（请清空工作区，双从臂即将归位并开始模型运行）……\n')
                        return True
                gestures.tick(time.monotonic())
        return False

    try:
        if not os.isatty(input_fd) or not os.isatty(output_fd):
            raise RuntimeError('Pedal DAgger requires an interactive terminal')
        pedals.open()  # Identify/grab this pedal before the old Bash can start arms.
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            saved_handlers[signum] = signal.signal(signum, handle_os_signal)
        saved_terminal = termios.tcgetattr(input_fd)
        tty.setcbreak(input_fd)

        while True:
            gestures = PedalGestures(config['bindings'])
            process = None
            requested_signal = None
            pedal_stop_requested = False
            ready = False
            ready_seen_monotonic = None
            ready_buffer = b''

            def deliver(tokens):
                nonlocal pedal_stop_requested
                for token in tokens:
                    if requested_signal is not None:
                        break
                    if token == 'stop':
                        pedal_stop_requested = True
                        if standby_enabled:
                            emit('\n[踏板] 右踏板双踩：停止本轮并保存剩余模型推理段，完成后将进入待机等待下一轮。\n')
                        else:
                            emit('\n[踏板] 右踏板双踩：停止本轮并保存剩余 inference，不恢复推理。\n')
                        forward_child_signal(signal.SIGINT)
                    elif ready and token in ('left', 'save', 'resume', 'delete'):
                        try:
                            os.write(command_write, token.encode('ascii') + b'\n')
                        except (BrokenPipeError, BlockingIOError):
                            emit('\n[踏板] 操作未送达；键盘仍可用，右双踩仍可结束本轮。\n')

            try:
                master_fd, slave_fd = pty.openpty()
                command_read, command_write = os.pipe()
                ready_read, ready_write = os.pipe()
                # The unchanged Bash entry owns fd9 for its round lock. Keep inherited
                # pedal endpoints away from that fixed descriptor, regardless of how
                # many event interfaces this device exposes.
                inherited_command = fcntl.fcntl(command_read, fcntl.F_DUPFD_CLOEXEC, 20)
                os.close(command_read)
                command_read = inherited_command
                inherited_ready = fcntl.fcntl(ready_write, fcntl.F_DUPFD_CLOEXEC, 20)
                os.close(ready_write)
                ready_write = inherited_ready
                os.set_blocking(command_write, False)
                os.set_blocking(ready_read, False)
                environment = dict(os.environ, DAGGER_PEDAL_COMMAND_FD=str(command_read),
                                   DAGGER_PEDAL_READY_FD=str(ready_write))
                if terminal_signal is not None:
                    return 128 + terminal_signal
                process = subprocess.Popen(command, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                                           start_new_session=True, close_fds=True,
                                           pass_fds=(command_read, ready_write), env=environment)
                for descriptor in (slave_fd, command_read, ready_write):
                    os.close(descriptor)
                slave_fd = command_read = ready_write = None
                if terminal_signal is not None:
                    forward_child_signal(terminal_signal)
                if len(config['bindings']) == 2:
                    if standby_enabled:
                        emit('[踏板] 左单踩示教/新段，右单踩保存，左双踩恢复/待机开新一轮，右双踩结束本轮进入待机；删除确认用键盘 w。\n')
                        emit('[踏板] 左/右单踩等待0.5秒，双踩不会先发单踩；彻底退出程序请按终端 Ctrl+C。\n')
                    else:
                        emit('[踏板] 左单踩示教/新段，右单踩保存，左双踩恢复，右双踩退出；删除确认仍可用键盘 w。\n')
                        emit('[踏板] 左/右单踩等待0.5秒，双踩不会先发单踩；准备期间仅右双踩和键盘 Ctrl+C 可退出。\n')
                else:
                    emit('[踏板] 左单踩示教/新段，中单踩保存、中双踩删除确认，右单踩恢复、右双踩退出。\n')
                    emit('[踏板] 中/右单踩等待0.5秒，双踩不会先发单踩；准备期间仅右双踩和键盘 Ctrl+C 可退出。\n')
                with selectors.DefaultSelector() as selector:
                    selector.register(input_fd, selectors.EVENT_READ, ('keyboard', None))
                    selector.register(master_fd, selectors.EVENT_READ, ('child', None))
                    selector.register(ready_read, selectors.EVENT_READ, ('ready', None))
                    for descriptor in pedals.filenos:
                        selector.register(descriptor, selectors.EVENT_READ, ('pedal', None))

                    def read_pedal_events(fd):
                        nonlocal gestures
                        try:
                            return pedals.read_events(fd)
                        except (OSError, RuntimeError) as exc:
                            for descriptor in pedals.filenos:
                                try:
                                    selector.unregister(descriptor)
                                except KeyError:
                                    pass
                            pedals.close()
                            gestures = PedalGestures(config['bindings'])
                            emit('\n[踏板] 输入已断开：' + str(exc) + '；键盘继续可用。\n')
                            return None

                    while process.poll() is None:
                        pedal_events = []
                        try:
                            ready_keys = selector.select(timeout=0.02)
                        except InterruptedError:
                            continue
                        for key, _ in ready_keys:
                            kind = key.data[0]
                            if kind == 'pedal':
                                events = read_pedal_events(key.fd)
                                if events is None:
                                    pedal_events.clear()
                                    break
                                pedal_events.extend(events)
                                continue
                            try:
                                data = os.read(key.fd, 65536)
                            except OSError as exc:
                                if kind == 'child' and exc.errno == errno.EIO:
                                    data = b''
                                else:
                                    raise
                            if kind == 'ready':
                                if data:
                                    ready_buffer += data
                                    if b'ready\n' in ready_buffer:
                                        ready = True
                                        ready_seen_monotonic = time.monotonic()
                                        emit('\n[踏板] 控制已就绪；保存确认期间单踩只确认保留。\n')
                                        selector.unregister(key.fd)
                                else:
                                    selector.unregister(key.fd)
                                continue
                            if not data:
                                if kind == 'keyboard':
                                    selector.unregister(key.fd)
                                    if terminal_signal is None:
                                        terminal_signal = signal.SIGHUP
                                    if requested_signal is None:
                                        forward_child_signal(signal.SIGHUP)
                                else:
                                    selector.unregister(key.fd)
                                continue
                            if kind == 'keyboard':
                                if requested_signal is None:
                                    os.write(master_fd, data)
                            else:
                                os.write(output_fd, data)
                        # Keyboard/child output may have taken time since selector's
                        # snapshot. Read pedal input queued meanwhile before deciding
                        # that the single-click window expired.
                        pedal_fds = list(pedals.filenos)
                        if pedal_fds:
                            for descriptor in select.select(pedal_fds, [], [], 0)[0]:
                                events = read_pedal_events(descriptor)
                                if events is None:
                                    pedal_events.clear()
                                    break
                                pedal_events.extend(events)
                        # Process all device queues in kernel-time order before running
                        # click timers. A delayed read must not split a real double
                        # into two single actions, or turn a pre-READY press into a new command.
                        processing_now = time.monotonic()
                        pedal_events.sort(key=lambda event: event[4] if len(event) == 5 else processing_now)
                        for event in pedal_events:
                            eligible = ready and (len(event) != 5 or event[4] >= ready_seen_monotonic)
                            deliver(gestures.feed(event, processing_now, allow_short=eligible))
                        if requested_signal is None:
                            deliver(gestures.tick(time.monotonic()))
                code = process.wait()
            finally:
                if process is not None and process.poll() is None:
                    if requested_signal is None:
                        forward_child_signal(signal.SIGINT)
                    # Let the original owner perform its normal driver-first cleanup.
                    # Keep exporting even if an outer input error led here. In
                    # particular, do not fill its PTY or terminate a slow video export.
                    # Another real Ctrl+C is still forwarded to the original owner.
                    while process.poll() is None:
                        drain_pty(master_fd, output_fd)
                        time.sleep(0.02)
                if master_fd is not None:
                    drain_pty(master_fd, output_fd)
                close_round_descriptors()
                process = None

            clean_pedal_stop = (
                standby_enabled
                and terminal_signal is None
                and pedal_stop_requested
                and code in (0, 130, -signal.SIGINT)
                and bool(pedals.filenos)
            )
            if clean_pedal_stop:
                if wait_for_standby_restart():
                    continue
                if terminal_signal is not None:
                    return 128 + terminal_signal
                return 2
            if code == 0 and requested_signal is not None:
                return 128 + requested_signal
            return 128 - code if code < 0 else code
    except (OSError, RuntimeError, ValueError) as exc:
        emit('\n[踏板错误] ' + str(exc) + '\n')
        return 2
    finally:
        if process is not None and process.poll() is None:
            if requested_signal is None:
                forward_child_signal(signal.SIGINT)
            while process.poll() is None:
                drain_pty(master_fd, output_fd)
                time.sleep(0.02)
        if master_fd is not None:
            drain_pty(master_fd, output_fd)
        pedals.close()
        close_round_descriptors()
        if saved_terminal is not None:
            termios.tcsetattr(input_fd, termios.TCSADRAIN, saved_terminal)
        for signum, handler in saved_handlers.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--standby', action='store_true',
                        help='Stay in standby after a clean right-double-click round stop; left double-click starts a new round, Ctrl+C exits.')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A Bash command after -- is required')
    try:
        return run_pedal_session(command, args.config, standby_on_pedal_stop=args.standby)
    except (OSError, RuntimeError, ValueError) as exc:
        print('[踏板错误] ' + str(exc) + '\n请先运行 run_dagger_pedal.sh --calibrate。', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
