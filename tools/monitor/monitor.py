import argparse
import datetime
import json
import re
import smtplib
import sys
import time
from datetime import datetime
from email.mime.text import MIMEText
from getpass import getpass

import paramiko
import requests
import yaml


def read_config(config_file):
    """Read and parse configuration from YAML file"""
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        print("Loaded configuration:")
        print(yaml.dump(config))
        return config
    except FileNotFoundError:
        print(f"{datetime.now()} - Configuration file not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"{datetime.now()} - Error parsing configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"{datetime.now()} - Error reading configuration file: {e}")
        sys.exit(1)


def read_logs(remote_host, remote_user, remote_port, remote_log_path):
    """Connect to remote host and read log file using SSH/SFTP"""
    print(f"{datetime.now()} - Connecting to remote host...")
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=remote_host, username=remote_user, port=remote_port)

        sftp = ssh.open_sftp()
        with sftp.open(remote_log_path, "r") as remote_file:
            print("Loading log file content...")
            contents = remote_file.readlines()
            print("Loading finish")

        sftp.close()
        ssh.close()

        return contents
    except Exception as e:
        print(f"{datetime.now()} - Error reading remote log file: {e}")
        sys.exit(1)


def parse_logs(log_lines):
    """Parse log file contents using regex pattern matching"""
    print(f"{datetime.now()} - Parsing log content...")

    pattern = (
        r"\[+(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]+\s*"
        r"iteration\s+(?P<iteration>\d+\s*/\s*\d+)\s+\|\s*"
        r"consumed samples:\s+(?P<consumed_samples>\d+)\s+\|\s*"
        r"elapsed time per iteration \(ms\):\s+(?P<elapsed_time_per_iteration_ms>\d+\.\d+)\s+\|\s*"
        r"throughput per GPU \(TFLOP/s/GPU\):\s+(?P<throughput_per_GPU_TFLOPs_per_GPU>\d+\.\d+)\s+\|\s*"
        r"learning rate:\s+(?P<learning_rate>\d+\.?\d*E?-?\d*)\s+\|\s*"
        r"global batch size:\s+(?P<global_batch_size>\d+)\s+\|\s*"
        r"lm loss:\s+(?P<lm_loss>\d+\.\d+E?[-\+]?\d*)\s+\|\s*"
        r"load_balancing_loss:\s+(?P<load_balancing_loss>\d+\.\d+E?[-\+]?\d*)\s+\|\s*"
        r"loss scale:\s+(?P<loss_scale>\d+\.\d+)\s+\|\s*"
        r"grad norm:\s+(?P<grad_norm>\d+\.\d+)\s+\|\s*"
        r"num zeros:\s+(?P<num_zeros>[\d.]+)\s+\|\s*"
        r"params norm:\s+(?P<params_norm>\d+\.\d+)\s+\|\s*"
        r"number of skipped iterations:\s+(?P<number_of_skipped_iterations>\d+)\s+\|\s*"
        r"number of nan iterations:\s+(?P<number_of_nan_iterations>\d+)\s*"
    )

    lines_to_keep = []
    for line in log_lines:
        match = re.search(pattern, line)
        if match:
            parsed_data = match.groupdict()
            if parsed_data["iteration"].startswith("1/"):
                lines_to_keep = []
            lines_to_keep.append(parsed_data)
    return lines_to_keep


def check_logs(logs):
    """Analyze log entries to detect training issues"""
    print(f"{datetime.now()} - Checking log content...")
    recent_logs = logs[-10:]
    last_log = logs[-1]
    last_timestamp = datetime.strptime(
        last_log["timestamp"], "%Y-%m-%d %H:%M:%S"
    )
    last_elapsed = float(last_log["elapsed_time_per_iteration_ms"])
    now = datetime.now()
    time_delta_ms = (now - last_timestamp).total_seconds() * 1000

    loss_abnormal = False
    min_loss = 10000000000
    max_loss = -10000000000
    for _log in logs:
        cur_loss = float(_log["lm_loss"])
        min_loss = min(min_loss, cur_loss)
        max_loss = min(max_loss, cur_loss)
        if cur_loss <= 0 or max_loss > min_loss + 5:
            loss_abnormal = True

    if time_delta_ms > 3 * last_elapsed:
        status_code = 1
        message = "Training appears to be stuck."
    elif historical_elapsed := [
        float(entry["elapsed_time_per_iteration_ms"]) for entry in logs[:-1]
    ]:
        average_elapsed = sum(historical_elapsed) / len(historical_elapsed)
        if last_elapsed > 2 * average_elapsed:
            status_code = 2
            message = "Training appears to be slowing down."
        else:
            status_code = 0
            message = "Training is normal."
    elif loss_abnormal:
        status_code = 3
        message = "Model loss appears as 0, negative, or abnormal peak, suspecting training issues"
    else:
        status_code = 0
        message = "Training is normal."

    return status_code, message, recent_logs


def send_email(
    smtp_server, source_email, source_email_password, target_email, subject, body
):
    """Send email notification using SMTP"""
    print(f"{datetime.now()} - Sending email: {subject}")
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = source_email
        msg["To"] = target_email

        server = smtplib.SMTP(smtp_server, 25)
        server.login(source_email, source_email_password)
        server.sendmail(source_email, [target_email], msg.as_string())
        server.quit()
        print(f"{datetime.now()} - Email sent successfully.")
    except Exception as e:
        print(f"{datetime.now()} - Error sending email: {e}")


def send_feishu(robot_url, message, body):
    """Send a message to the Feishu robot."""
    print(f"{datetime.now()} - Sending to Feishu robot: {message}")
    card_json = json.dumps(
        {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": "monitor: "+ message,
                        "content": [
                            [{
                                "tag": "text",
                                "text": body
                            }]
                        ]
                    }
                }
            }
        }
    )
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url=robot_url, data=card_json, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to send message: {e}")


def main():
    """Main function to handle command line arguments and monitoring loop"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Training Monitor")
    parser.add_argument(
        "--notice",
        type=str,
        choices=["email", "feishu"],
        required=True,
        help="Notification method (email/feishu)",
    )
    args = parser.parse_args()

    if args.notice == "email":
        # Load configuration from YAML file
        config = read_config("config-email.yaml")

        # Email notification configuration
        smtp_server = config["smtp_server"]
        source_email = config["source_email"]
        target_email = config["target_email"]
        check_interval = int(config["check_interval"])

        # Securely get email password
        source_email_password = getpass(f"Enter the password for {source_email}: ")

    elif args.notice == "feishu":
        # Load configuration from YAML file
        config = read_config("config-feishu.yaml")

        check_interval = int(config["check_interval"])

        robot_url = getpass(f"Enter the URL for Feishu robot: ")

    while True:
        logs = read_logs(
            config["remote_host"],
            config["remote_user"],
            int(config["remote_port"]),
            config["remote_log_path"],
        )
        useful_logs = parse_logs(logs)
        status_code, message, recent_logs = check_logs(useful_logs)
        recent_logs_str = json.dumps(recent_logs, indent=4)

        if status_code != 0:
            if args.notice == "email":
                send_email(
                    smtp_server,
                    source_email,
                    source_email_password,
                    target_email,
                    message,
                    recent_logs_str,
                )
            if args.notice == "feishu":
                send_feishu(
                    robot_url,
                    message,
                    recent_logs_str,
                )

        print(f"Waiting {check_interval} seconds for next check...")
        time.sleep(check_interval)


if __name__ == "__main__":
    main()