"""Telegram bot interface — CFO assistant POC.

A thin Telegram front-end that wraps the same assessment pipeline as the
REST API. Designed to live on a small VM (Fly.io / Cloudflare Tunnel /
Render free tier) and answer commands from a CFO directly in their phone.

Commands
--------
/start            — welcome + capability overview
/companies        — list known digital twins
/assess <name>    — run an assessment for the company
/predict          — list the 8 pre-registered predictions
/help             — show this help

Setup
-----
    pip install python-telegram-bot
    export TIMESTONE_TELEGRAM_TOKEN=...        # from @BotFather
    python -m timestone.interfaces.telegram.bot

The runtime hosting is out of scope for this repo — pair this module with
any small Python host that can keep a long-poll session open.
"""
from .bot import build_application, main  # noqa: F401
