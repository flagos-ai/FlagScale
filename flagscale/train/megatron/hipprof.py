import os
import shlex
import subprocess


def _format_command(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def hipprof_session_control(action):
    if action not in ("start", "stop"):
        raise ValueError(f"Unsupported hipprof session action: {action}")

    session_id = os.environ.get("HIPPROF_SESSION_ID", "")
    if not session_id:
        raise RuntimeError(
            "hipprof profiling requires a launcher-provided session. "
            "Configure experiment.runner.hipprof_bin_path and "
            "experiment.runner.hipprof_output_dir."
        )

    hipprof_bin = os.environ.get("HIPPROF_BIN_PATH", "") or "hipprof"
    command = [hipprof_bin, "--session-client", session_id, f"--{action}"]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"hipprof executable not found: {hipprof_bin}. Configure "
            "experiment.runner.hipprof_bin_path."
        ) from error
    except subprocess.CalledProcessError as error:
        output = []
        if error.stdout:
            output.append(f"stdout:\n{error.stdout.strip()}")
        if error.stderr:
            output.append(f"stderr:\n{error.stderr.strip()}")
        details = "\n".join(output)
        if details:
            details = "\n" + details
        raise RuntimeError(
            f"hipprof session {action} failed with exit code {error.returncode}: "
            f"{_format_command(command)}{details}"
        ) from error
