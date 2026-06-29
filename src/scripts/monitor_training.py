#!/usr/bin/env python3
"""Monitor DPO training progress and send Telegram alerts at milestones.

Usage:
    uv run src/scripts/monitor_training.py --session overnight

Sends alerts at 25%, 50%, 75%, 100% progress milestones.
"""

import argparse
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

HOME_ENV_FILE = Path.home() / ".env"
if HOME_ENV_FILE.exists():
    load_dotenv(HOME_ENV_FILE)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")

logger = logging.getLogger(__name__)

MILESTONES = [25, 50, 75, 100]
POLL_INTERVAL = 60


def send_telegram_message(message: str) -> bool:
    """Send a message via Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_USER_ID:
        logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_USER_ID not set")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_USER_ID,
        "text": message,
        "parse_mode": "Markdown",
    }

    try:
        response = httpx.post(url, json=data, timeout=10)
        response.raise_for_status()
        return True
    except httpx.HTTPError as e:
        logger.error("Failed to send Telegram message: %s", e)
        return False


def get_training_progress(session_name: str) -> tuple[int | None, int | None]:
    """Extract progress from tmux session.

    Returns (current_iteration, total_iterations) or (None, None) if not found.
    """
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session_name, "-p"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout

        for line in reversed(output.split("\n")):
            match = re.search(r"^\s+(\d+)%\|.+#\|\s*(\d+)/(\d+)", line)
            if match:
                current = int(match.group(2))
                total = int(match.group(3))
                return current, total

    except subprocess.TimeoutExpired:
        logger.warning("tmux capture timed out")
    except Exception as e:
        logger.warning("Error getting progress: %s", e)

    return None, None


def calculate_percentage(current: int, total: int) -> float:
    """Calculate progress percentage."""
    if total == 0:
        return 0.0
    return (current / total) * 100


def format_eta(eta_seconds: float | None) -> str:
    """Format ETA in human-readable format."""
    if eta_seconds is None:
        return "Unknown"

    hours = int(eta_seconds // 3600)
    minutes = int((eta_seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def parse_eta_from_tmux(session_name: str) -> float | None:
    """Extract ETA in seconds from tmux progress line."""
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session_name, "-p"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout

        for line in output.split("\n")[-20:]:
            match = re.search(r"\[.*<(\d+):(\d+):(\d+)", line)
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                seconds = int(match.group(3))
                return hours * 3600 + minutes * 60 + seconds

    except Exception:
        pass

    return None


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor DPO training progress")
    parser.add_argument(
        "--session",
        default="overnight",
        help="tmux session name (default: overnight)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=POLL_INTERVAL,
        help=f"Poll interval in seconds (default: {POLL_INTERVAL})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting monitoring for session '%s'", args.session)
    logger.info("Poll interval: %ds, Milestones: %s%%", args.interval, MILESTONES)

    last_reported_milestone = 0
    last_progress_log = 0

    while True:
        current, total = get_training_progress(args.session)

        if current is None or total is None:
            logger.warning("Could not parse progress, retrying...")
            time.sleep(args.interval)
            continue

        percentage = calculate_percentage(current, total)
        eta_seconds = parse_eta_from_tmux(args.session)

        for milestone in MILESTONES:
            if last_reported_milestone < milestone <= percentage:
                message = (
                    f"🎯 *DPO Training Milestone*\n\n"
                    f"*Progress:* {milestone}% ({current}/{total})\n"
                    f"*ETA:* {format_eta(eta_seconds)}\n\n"
                    f"Session: `{args.session}`"
                )
                if send_telegram_message(message):
                    logger.info("Sent milestone alert: %s%%", milestone)
                last_reported_milestone = milestone

        progress_bucket = int(percentage // 10) * 10
        if progress_bucket > last_progress_log and progress_bucket > 0:
            logger.info("Progress: %.1f%% (%d/%d)", percentage, current, total)
            last_progress_log = progress_bucket

        if current >= total:
            message = (
                f"✅ *DPO Training Complete!*\n\n"
                f"*Final:* {total}/{total} iterations\n"
                f"Session: `{args.session}`"
            )
            send_telegram_message(message)
            logger.info("Training complete!")
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
