# Training Log Monitor

This monitoring tool is used to monitor training logs on remote servers, check for anomalies in the logs, send reminders via email or Feishu robot. This tool aims to ensure that problems during the training process can be identified in a timely manner, including timely monitoring of training jamming or slowing down.

# NOTE

For email reminders:
   This program requires a password to be entered during runtime, so please ensure that it is used in a secure environment to avoid the risk of password leakage.

For Feishu robot reminders:
   This program requires a URL of Feishu Robot to be entered during runtime, so please ensure that it is used in a secure environment to avoid the risk of URL leakage. Configuration method reference link https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot, set the keyword to "monitor".

Training anomaly monitoring relies on historical training data analysis using various statistical methods. Please manually observe the logs for a period of time to ensure that at least the first 10 iterations are normal.

## Features

- Monitors a remote log file for training status.
- Sends corresponding abnormal information prompt emails based on log analysis results, including sample content for clarity.
- Configurable check interval.

## Run

The monitoring program runs on the user's local machine.

## Prerequisites

Before running the script, ensure you have a password-free SSH login to the remote host.


## Installation

   ```bash
   git clone https://github.com/FlagOpen/FlagScale.git
   cd FlagScale/tools/monitor
   pip install -r requirements.txt
   ```

## Configuration

1. For Email:
   Modify the provided configuration file [config-email.yaml](config-email.yaml) example to set actual values:

2. For Feishu Root

   Modify the provided configuration file [config-feishu.yaml](config-feishu.yaml) example to set actual values:

## Usage

1. For Email:

   ```bash
   python monitor.py --notice email
   ```

   You will then be prompted to enter your source email's password.

2. For Feishu robot:

   ```bash
   python monitor.py --notice feishu
   ```

   You will then be prompted to enter Feishu robot URL.

## Next steps

We will add monitoring perspectives, including:
- Prompt when training ends.
- Perform communication group-based monitoring.
- Monitor hardware utilization anomalies.
- More user needs...
