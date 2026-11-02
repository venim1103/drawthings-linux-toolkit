"""Local lifecycle for the guarded MOSSkr2 image preset."""

import argparse
import fcntl
import json
import os
import select
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
RUNNER = ROOT / "tools/run_q6p_canary_once.sh"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"start_ticks": fields[19], "uid": Path(f"/proc/{pid}").stat().st_uid,
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_process(state):
    return (state.get("identity") is not None
            and state["identity"] == process_identity(state["pid"])
            and state["identity"]["uid"] == os.getuid())


def snapshot(bundle, runtime_paths):
    result = {}
    paths = [bundle / "bundle.json", *sorted((bundle / "models").rglob("*")),
             *map(Path, runtime_paths)]
    for path in paths:
        if path.is_dir():
            continue
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"not a regular session input: {path}")
        if path.is_relative_to(bundle / "models"):
            if path.is_symlink() or info.st_nlink != 1 or not os.access(path, os.W_OK):
                raise ValueError(f"session models must be independent writable files: {path}")
        result[str(path)] = [str(path.resolve()), info.st_dev, info.st_ino,
                             info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink]
    return result


def validate_inputs(state):
    if snapshot(Path(state["bundle"]), state["runtime_paths"]) != state["inputs"]:
        raise ValueError("session inputs changed; stop and restart to fully revalidate the bundle")


def listener_owned(state):
    processes = [state["pid"]]
    descriptors = set()
    for pid in processes:
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            processes.extend(int(child) for child in children if int(child) not in processes)
            for descriptor in Path(f"/proc/{pid}/fd").iterdir():
                try:
                    descriptors.add(os.readlink(descriptor))
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, ProcessLookupError):
            continue
    address = f"0100007F:{state['port']:04X}"
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        if fields[1] == address and fields[3] == "0A" and f"socket:[{fields[9]}]" in descriptors:
            return True
    return False


def stop_owned(state):
    if not owned_process(state):
        return
    descriptor = os.pidfd_open(state["pid"])
    try:
        if not owned_process(state):
            return
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        if not select.select([descriptor], [], [], 10)[0]:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            if not select.select([descriptor], [], [], 10)[0]:
                raise RuntimeError("owned server did not stop")
    except ProcessLookupError:
        pass
    finally:
        os.close(descriptor)


def start_session(arguments, directory, state_path, runner_arguments):
    if state_path.exists() and owned_process(json.loads(state_path.read_text())):
        raise ValueError("session already running; use it or stop it first")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", arguments.port))
    bundle = arguments.bundle.resolve()
    run = directory / f"run-{time.time_ns()}"
    run.mkdir(mode=0o700)
    plan_directory = run / "plan"
    runner_arguments = [*runner_arguments, "--server-weights-cache", str(arguments.cache_gib)]
    subprocess.run(["bash", str(RUNNER), *runner_arguments, "--dry-run",
                    "--output-dir", str(plan_directory)], check=True)
    launch = json.loads((plan_directory / "server-launch.json").read_text())
    runtime_paths = [str(RUNNER), str(ROOT / "tools/dt_image.sh"), str(Path(__file__).resolve()),
                     str(PYTHON.resolve()), launch["command"][0],
                     launch["environment"]["DT_VALIDATION_NATIVE"]]
    guard_python = Path(launch["command"][0]).with_suffix(".py")
    if guard_python.exists():
        runtime_paths.append(str(guard_python))
    inputs = snapshot(bundle, runtime_paths)
    subprocess.run([str(PYTHON), str(ROOT / "tools/dt_moss_convert.py"),
                    "--bundle-model", str(bundle)], check=True)
    if snapshot(bundle, runtime_paths) != inputs:
        raise ValueError("session inputs changed during bundle verification")
    environment = os.environ.copy()
    for key in ("LD_PRELOAD", "DT_SAMPLE_CPU_STACKS", "DT_TARGETED_PROFILE_CLIENT_LOG"):
        environment.pop(key, None)
    environment.update(launch["environment"])
    progress = run / "current-client.log"
    progress.touch(exist_ok=False)
    environment["DT_VALIDATION_CLIENT_LOG"] = str(progress)
    with (run / "server.log").open("x") as output:
        server = subprocess.Popen(launch["command"], env=environment, stdin=subprocess.DEVNULL,
                                  stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    state = {"pid": server.pid, "identity": process_identity(server.pid), "phase": "starting",
             "port": arguments.port, "mode": arguments.mode, "cache_gib": arguments.cache_gib,
             "bundle": str(bundle), "runtime_paths": runtime_paths, "inputs": inputs,
             "run_directory": str(run), "model": json.loads((bundle / "bundle.json").read_text())["model"],
             "requests": 0, "created_unix_sec": time.time()}
    try:
        write_json(state_path, state)
        subprocess.run(["bash", str(RUNNER), *runner_arguments, "--connect-only", "--server-pid",
                        str(server.pid), "--preflight-only", "--output-dir", str(run / "preflight")], check=True)
        if server.poll() is not None or not listener_owned(state):
            raise ValueError("session does not own the expected local listener")
        validate_inputs(state)
        state["phase"] = "ready"
        write_json(state_path, state)
    except BaseException:
        stop_owned(state)
        server.wait(timeout=10)
        state["phase"] = "failed"
        write_json(state_path, state)
        raise
    print(f"SESSION=READY pid={server.pid} weights_cache_gib={arguments.cache_gib} logs={run}", flush=True)


def use_session(arguments, state, state_path, runner_arguments):
    if not owned_process(state) or state["phase"] != "ready":
        raise ValueError("no ready session; start one first")
    if state["bundle"] != str(arguments.bundle.resolve()) or state["mode"] != arguments.mode:
        raise ValueError("session bundle/weight policy mismatch; restart for a different configuration")
    if arguments.cache_gib != state["cache_gib"]:
        raise ValueError("session weights cache mismatch; restart to change its size")
    validate_inputs(state)
    if not listener_owned(state):
        raise ValueError("session no longer owns the expected local listener")
    output = Path(runner_arguments[runner_arguments.index("--output-dir") + 1])
    if output.exists() or output.is_symlink():
        raise ValueError(f"output directory already exists: {output}")
    run = Path(state["run_directory"])
    progress = run / "current-client.log"
    started = time.monotonic()
    state["phase"] = "generating"
    write_json(state_path, state)
    command = ["bash", str(RUNNER), *runner_arguments, "--server-weights-cache", str(state["cache_gib"]),
               "--connect-only", "--server-pid", str(state["pid"])]
    code = 1
    offset = (run / "server.log").stat().st_size
    try:
        code = subprocess.run(command, env=dict(os.environ, DT_SESSION_PROGRESS_LOG=str(progress),
                               DT_SESSION_SERVER_LOG=str(run / "server.log"))).returncode
        if code == 0:
            validate_inputs(state)
            if not owned_process(state):
                raise ValueError("session exited during generation")
        return code
    finally:
        elapsed = time.monotonic() - started
        if code != 0 or sys.exc_info()[0] is not None:
            stop_owned(state)
            state["phase"] = "failed"
        else:
            state["phase"] = "ready"
        state["requests"] += 1
        state["last_request_seconds"] = elapsed
        write_json(state_path, state)
        if output.is_dir():
            with (run / "server.log").open("rb") as source, (output / "session-server.log").open("xb") as target:
                source.seek(offset)
                target.write(source.read())
            write_json(output / "session-request.json", {
                "pid": state["pid"], "request": state["requests"], "elapsed_sec": elapsed,
                "weights_cache_gib": state["cache_gib"], "returncode": code, "phase": state["phase"],
                "session_server_log": str(run / "server.log"),
            })
        print(f"SESSION_REQUEST seconds={elapsed:.3f} pid={state['pid']} phase={state['phase']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "use", "status", "stop", "model"))
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--mode", choices=("offload", "gpu"), default="gpu")
    parser.add_argument("--cache-gib", type=int, default=None)
    parser.add_argument("--bundle", type=Path)
    arguments, runner_arguments = parser.parse_known_args()
    if runner_arguments[:1] == ["--"]:
        runner_arguments = runner_arguments[1:]
    if not 1024 <= arguments.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    if arguments.cache_gib is not None and not 0 <= arguments.cache_gib <= 8:
        parser.error("weights cache must be 0..8 integer GiB")
    directory = ROOT / "output" / f"image-session-{arguments.port}"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path = directory / "state.json"
    if arguments.action == "status":
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        status = dict(state)
        status.pop("inputs", None)
        status["alive"] = bool(state) and owned_process(state)
        print(json.dumps(status, indent=2))
        return 0
    with (directory / "request.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("session busy; another lifecycle operation or generation is running")
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if arguments.action == "stop":
            if state:
                stop_owned(state)
                state["phase"] = "stopped"
                write_json(state_path, state)
            print("SESSION=STOPPED")
        elif arguments.action == "model":
            if not state or not owned_process(state) or state["phase"] != "ready":
                raise ValueError("no ready session; start one first")
            validate_inputs(state)
            if not listener_owned(state):
                raise ValueError("session no longer owns the expected local listener")
            print(state["model"])
        elif arguments.action == "start":
            arguments.cache_gib = arguments.cache_gib if arguments.cache_gib is not None else 0
            start_session(arguments, directory, state_path, runner_arguments)
        else:
            if not state:
                raise ValueError("no ready session; start one first")
            arguments.cache_gib = arguments.cache_gib if arguments.cache_gib is not None else state["cache_gib"]
            return use_session(arguments, state, state_path, runner_arguments)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)