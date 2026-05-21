"""Telegram bot for the TimeStone CFO assistant POC.

Run:
    pip install python-telegram-bot
    export TIMESTONE_TELEGRAM_TOKEN=<token from @BotFather>
    python -m timestone.interfaces.telegram.bot
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

try:
    from telegram import Update
    from telegram.constants import ParseMode
    from telegram.ext import (
        Application,
        ApplicationBuilder,
        CommandHandler,
        ContextTypes,
    )
    HAS_TELEGRAM = True
except ImportError:  # pragma: no cover
    HAS_TELEGRAM = False
    Update = object  # type: ignore[assignment,misc]
    ContextTypes = object  # type: ignore[assignment,misc]

from ...application.assess_company import AssessOptions, assess_company
from ...infrastructure.paths import REPO_ROOT
from ...repositories.company import CompanyRepository

logger = logging.getLogger(__name__)

WELCOME = (
    "*TimeStone AI · CFO Assistant*\n\n"
    "I predict the success probability, NPV and payback of business "
    "transformations using Monte Carlo simulation calibrated on real "
    "historical cases.\n\n"
    "*Commands*\n"
    "/companies — list digital twins I can assess\n"
    "/assess <name> — run a fresh assessment\n"
    "/predict — read the 8 pre-registered predictions\n"
    "/help — this message"
)


# ----- Handlers -----

async def start(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN)


async def help_cmd(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN)


async def companies(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    repo = CompanyRepository()
    twins = repo.list_all()
    if not twins:
        await update.message.reply_text("No digital twins available.")
        return
    lines = ["*Known digital twins*\n"]
    for t in twins:
        rev = getattr(getattr(t, "metrics", None), "annual_revenue_usd", None)
        rev_str = f" · ${rev/1e9:.1f}B rev" if rev else ""
        lines.append(f"• `{t.company_name}` — {t.industry}{rev_str}")
    lines.append("\nUse `/assess <name>` to score one.")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def assess(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Usage: `/assess <company name>` — e.g. `/assess Kazakhstan Temir Zholy (KTZ)`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    name = " ".join(args)
    repo = CompanyRepository()
    twin = repo.load_by_name(name)
    if twin is None:
        await update.message.reply_text(
            f"Company `{name}` not found. Try `/companies`.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    await update.message.reply_text(
        f"Running assessment for *{twin.company_name}* — please hold.",
        parse_mode=ParseMode.MARKDOWN,
    )
    try:
        # Lighter run on the bot host — same engine, smaller iter count.
        opts = AssessOptions(scenario_count=100, iterations=200)
        report = assess_company(twin, options=opts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("assess failed")
        await update.message.reply_text(f"Assessment failed: `{exc}`", parse_mode=ParseMode.MARKDOWN)
        return

    lines = [f"*{report.company_name} · top-{len(report.top_recommendations)}*\n"]
    for r in report.top_recommendations:
        npv_str = f"${r.mean_npv/1e6:+.1f}M"
        lines.append(
            f"*#{r.rank}. {r.scenario_name}*\n"
            f"  P(NPV>0): `{r.success_probability*100:.0f}%` · "
            f"NPV: `{npv_str}` · "
            f"Payback: `{r.payback_years:.1f}y`\n"
            f"  → _{r.headline}_"
        )
    lines.append(f"\n_Run id_: `{report.run_id}`")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def predict(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    p = REPO_ROOT / "predictions" / "2026-05-20.json"
    if not p.exists():
        await update.message.reply_text("No pre-registered predictions found.")
        return
    data = json.loads(p.read_text())
    preds = data.get("predictions", [])
    if not preds:
        await update.message.reply_text("No pre-registered predictions found.")
        return
    lines = ["*Pre-registered predictions* (committed 20 May 2026)\n"]
    for pr in preds:
        co = pr.get("company") or pr.get("co", "?")
        p_pos = pr.get("p", 0) * 100
        npv = pr.get("npv", 0)
        win = pr.get("window", "")
        lines.append(f"• *{co}* — P(NPV>0) `{p_pos:.0f}%`, NPV `${npv:+.0f}M`, {win}")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# ----- Wiring -----

def build_application(token: Optional[str] = None) -> "Application":
    if not HAS_TELEGRAM:
        raise RuntimeError("python-telegram-bot is not installed. Run: pip install python-telegram-bot")
    tok = token or os.environ.get("TIMESTONE_TELEGRAM_TOKEN")
    if not tok:
        raise RuntimeError("Set TIMESTONE_TELEGRAM_TOKEN env var (get a token from @BotFather).")
    app = ApplicationBuilder().token(tok).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("companies", companies))
    app.add_handler(CommandHandler("assess", assess))
    app.add_handler(CommandHandler("predict", predict))
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = build_application()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":  # pragma: no cover
    main()
