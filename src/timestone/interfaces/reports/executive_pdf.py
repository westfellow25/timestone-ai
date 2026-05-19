"""Executive PDF report generator.

Takes an AssessmentReport plus the underlying scenarios/cases and produces
a polished 5-6 page PDF suitable for handing to a CFO or CEO.

Layout:
  - Cover page: company name, run ID, date, summary metrics
  - Page 1: Executive summary - narrative + KPI card
  - Page 2-3: Top recommendations, one per expander
  - Page 4: Methodology - NPV model, shocks, case library
  - Page 5: Sources appendix - all cases that informed the priors
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, PageBreak, Paragraph,
    Spacer, Table, TableStyle, KeepTogether,
)

from ...domain.report import AssessmentReport, Recommendation


# Brand palette
COLOR_PRIMARY = colors.HexColor("#1B2A4E")        # deep navy
COLOR_ACCENT = colors.HexColor("#667EEA")         # soft indigo
COLOR_SUCCESS = colors.HexColor("#22C55E")
COLOR_WARN = colors.HexColor("#F59E0B")
COLOR_DANGER = colors.HexColor("#EF4444")
COLOR_MUTED = colors.HexColor("#6B7280")
COLOR_BG = colors.HexColor("#F9FAFB")


@dataclass
class ReportContext:
    """Data needed to render a report. assess_company packages this."""
    report: AssessmentReport
    scenarios_payload: Dict        # full scenarios.json content
    simulation_payload: Dict       # full simulation.json content
    cases_by_id: Dict[str, Dict]   # case ID -> raw case dict (for sources page)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def _build_styles():
    base = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle("h1", parent=base["Heading1"],
                             textColor=COLOR_PRIMARY, fontSize=24,
                             spaceAfter=14, leading=28),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                             textColor=COLOR_PRIMARY, fontSize=15,
                             spaceBefore=14, spaceAfter=8, leading=18),
        "h3": ParagraphStyle("h3", parent=base["Heading3"],
                             textColor=COLOR_PRIMARY, fontSize=12,
                             spaceBefore=10, spaceAfter=4, leading=14),
        "body": ParagraphStyle("body", parent=base["BodyText"],
                               fontSize=10, leading=14, alignment=TA_JUSTIFY,
                               spaceAfter=8, textColor=colors.black),
        "muted": ParagraphStyle("muted", parent=base["BodyText"],
                                fontSize=9, leading=12,
                                textColor=COLOR_MUTED),
        "cover_title": ParagraphStyle("cover_title", parent=base["Heading1"],
                                      fontSize=42, leading=48, alignment=TA_LEFT,
                                      textColor=COLOR_PRIMARY, spaceAfter=10),
        "cover_sub": ParagraphStyle("cover_sub", parent=base["BodyText"],
                                    fontSize=14, leading=18, alignment=TA_LEFT,
                                    textColor=COLOR_MUTED),
        "headline_strong": ParagraphStyle("hs", parent=base["BodyText"],
                                          fontSize=11, leading=14,
                                          textColor=COLOR_SUCCESS,
                                          fontName="Helvetica-Bold"),
        "headline_caution": ParagraphStyle("hc", parent=base["BodyText"],
                                           fontSize=11, leading=14,
                                           textColor=COLOR_WARN,
                                           fontName="Helvetica-Bold"),
        "headline_pilot": ParagraphStyle("hp", parent=base["BodyText"],
                                         fontSize=11, leading=14,
                                         textColor=COLOR_DANGER,
                                         fontName="Helvetica-Bold"),
    }
    return styles


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
def fmt_money(x: float) -> str:
    if x is None:
        return "-"
    a = abs(x)
    if a >= 1e9:
        return f"${x / 1e9:.2f}B"
    if a >= 1e6:
        return f"${x / 1e6:.1f}M"
    if a >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:.0f}"


def fmt_pct(x: Optional[float], decimals: int = 1) -> str:
    if x is None:
        return "-"
    return f"{x * 100:.{decimals}f}%"


def headline_style(success_p: float, styles) -> ParagraphStyle:
    if success_p >= 0.80:
        return styles["headline_strong"]
    if success_p >= 0.60:
        return styles["headline_caution"]
    return styles["headline_pilot"]


# ---------------------------------------------------------------------------
# Page templates - header + footer
# ---------------------------------------------------------------------------
def _draw_header_footer(canvas, doc, run_id: str, company_name: str):
    canvas.saveState()
    # Footer
    canvas.setStrokeColor(COLOR_MUTED)
    canvas.setLineWidth(0.3)
    canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(COLOR_MUTED)
    canvas.drawString(2 * cm, 1.1 * cm, f"TimeStone AI    run {run_id}    {company_name}")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _cover_page(ctx: ReportContext, styles) -> List:
    flow = []
    flow.append(Spacer(1, 4 * cm))
    flow.append(Paragraph("TimeStone AI", styles["cover_sub"]))
    flow.append(Spacer(1, 0.5 * cm))
    flow.append(Paragraph("Transformation Assessment", styles["cover_title"]))
    flow.append(Spacer(1, 0.3 * cm))
    flow.append(Paragraph(ctx.report.company_name, styles["cover_sub"]))
    flow.append(Spacer(1, 5 * cm))

    # Cover metrics card
    rows = [
        ["Run ID", ctx.report.run_id],
        ["Generated", ctx.report.generated_at[:19].replace("T", " ") + " UTC"],
        ["Scenarios simulated", f"{ctx.report.total_scenarios:,}"],
        ["Monte Carlo iterations", f"{ctx.report.config_summary['iterations']:,}"],
        ["Case library size", f"{ctx.report.case_library_size}"],
        ["High-risk scenarios", f"{ctx.report.failure_rate_among_scenarios:.0%}"],
    ]
    t = Table(rows, colWidths=[6 * cm, 9 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), COLOR_MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.black),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, COLOR_MUTED),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    flow.append(t)
    flow.append(PageBreak())
    return flow


def _executive_summary(ctx: ReportContext, styles) -> List:
    flow = []
    flow.append(Paragraph("Executive Summary", styles["h1"]))

    top = ctx.report.top_recommendations[0] if ctx.report.top_recommendations else None
    narrative = (
        f"TimeStone analysed <b>{ctx.report.total_scenarios:,} transformation scenarios</b> "
        f"for <b>{ctx.report.company_name}</b>, each evaluated through "
        f"<b>{ctx.report.config_summary['iterations']:,} Monte Carlo iterations</b> "
        f"calibrated against <b>{ctx.report.case_library_size} real corporate transformations</b> "
        f"with documented promised-versus-actual outcomes."
    )
    flow.append(Paragraph(narrative, styles["body"]))

    if top:
        recommendation = (
            f"Our top recommendation is <b>{top.scenario_name}</b> with a "
            f"<b>{fmt_pct(top.success_probability, 0)}</b> probability of producing a "
            f"positive NPV. Mean NPV is <b>{fmt_money(top.mean_npv)}</b>, median payback "
            f"<b>{top.payback_years:.1f} years</b>, and the prior is grounded in "
            f"<b>{len(top.based_on_cases)} similar real transformations</b>."
        )
        flow.append(Paragraph(recommendation, styles["body"]))

    flow.append(Paragraph(
        f"Across the full set, <b>{ctx.report.failure_rate_among_scenarios:.0%}</b> of "
        f"simulated scenarios are flagged as high-risk (P(NPV>0) below 50 percent), "
        f"consistent with published industry baselines that report transformation "
        f"failure rates in the 60-70 percent range.", styles["body"]))

    flow.append(Spacer(1, 0.4 * cm))
    flow.append(Paragraph("Key metrics", styles["h2"]))

    # KPI table
    avg_p = sum(r.success_probability for r in ctx.report.top_recommendations) / max(1, len(ctx.report.top_recommendations))
    avg_npv = sum(r.mean_npv for r in ctx.report.top_recommendations) / max(1, len(ctx.report.top_recommendations))
    rows = [
        ["Top-3 mean P(NPV>0)", fmt_pct(avg_p, 0)],
        ["Top-3 mean NPV", fmt_money(avg_npv)],
        ["Scenarios flagged high-risk", f"{ctx.report.failure_rate_among_scenarios:.0%}"],
        ["Case library coverage", f"{ctx.report.case_library_size} cases"],
    ]
    t = Table(rows, colWidths=[8 * cm, 7 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), COLOR_MUTED),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, COLOR_MUTED),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    flow.append(t)
    flow.append(PageBreak())
    return flow


def _recommendation_block(rec: Recommendation, ctx: ReportContext, styles) -> List:
    flow = []
    flow.append(Paragraph(f"#{rec.rank}. {rec.scenario_name}", styles["h2"]))
    flow.append(Paragraph(rec.headline, headline_style(rec.success_probability, styles)))

    metrics = [
        ["P(NPV > 0)", fmt_pct(rec.success_probability, 0)],
        ["Mean NPV", fmt_money(rec.mean_npv)],
        ["Mean ROI", f"{rec.mean_roi:.1f}x"],
        ["Median payback", f"{rec.payback_years:.1f} years"],
        ["Risk level", rec.risk_level],
    ]
    t = Table(metrics, colWidths=[6 * cm, 9 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), COLOR_MUTED),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (-1, -1), 0.4, COLOR_MUTED),
        ("LINEBELOW", (0, 0), (-1, -2), 0.2, colors.HexColor("#E5E7EB")),
    ]))
    flow.append(t)

    if rec.description:
        flow.append(Spacer(1, 0.2 * cm))
        flow.append(Paragraph(f"<i>{rec.description}</i>", styles["muted"]))

    if rec.based_on_cases:
        flow.append(Spacer(1, 0.3 * cm))
        flow.append(Paragraph("Empirical prior", styles["h3"]))
        prior_text = (
            f"This recommendation is calibrated against "
            f"<b>{len(rec.based_on_cases)} real transformations</b>: "
            f"{', '.join(rec.based_on_cases)}."
        )
        flow.append(Paragraph(prior_text, styles["body"]))

        prior_metrics = []
        if rec.empirical_failure_rate is not None:
            prior_metrics.append(["Failure rate in similar cases", fmt_pct(rec.empirical_failure_rate, 0)])
        if rec.p10_revenue_uplift is not None:
            prior_metrics.append(["Observed P10 revenue uplift", fmt_pct(rec.p10_revenue_uplift)])
        if rec.p90_revenue_uplift is not None:
            prior_metrics.append(["Observed P90 revenue uplift", fmt_pct(rec.p90_revenue_uplift)])
        if prior_metrics:
            t2 = Table(prior_metrics, colWidths=[8 * cm, 7 * cm])
            t2.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), COLOR_MUTED),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]))
            flow.append(t2)

    flow.append(Spacer(1, 0.6 * cm))
    return flow


def _recommendations_section(ctx: ReportContext, styles) -> List:
    flow = []
    flow.append(Paragraph("Top Recommendations", styles["h1"]))
    flow.append(Paragraph(
        "The three scenarios below have the highest probability of producing a "
        "positive 5-year NPV across all simulations. Each is grounded in real "
        "historical transformations with documented outcomes.", styles["body"]))
    flow.append(Spacer(1, 0.3 * cm))

    for rec in ctx.report.top_recommendations:
        block = KeepTogether(_recommendation_block(rec, ctx, styles))
        flow.append(block)

    flow.append(PageBreak())
    return flow


def _methodology_section(ctx: ReportContext, styles) -> List:
    flow = []
    flow.append(Paragraph("Methodology", styles["h1"]))
    flow.append(Paragraph(
        "TimeStone evaluates transformation scenarios using a Monte Carlo "
        "simulation that combines a discounted cash flow model with empirical "
        "priors built from a curated library of real corporate transformations.",
        styles["body"]))

    flow.append(Paragraph("Financial model", styles["h2"]))
    flow.append(Paragraph(
        f"Each scenario is evaluated over a {ctx.report.config_summary['horizon_years']}-year "
        f"horizon at a {ctx.report.config_summary['discount_rate']:.0%} discount rate. "
        f"Capex is paid in year zero; benefits ramp up at 40 / 70 / 95 / 100 percent "
        f"in years 1 through 4 after implementation. Revenue uplift is applied to "
        f"baseline revenue; cost reduction is applied to operating costs "
        f"(not revenue - a common mis-modelling error).", styles["body"]))

    flow.append(Paragraph("Risk model", styles["h2"]))
    flow.append(Paragraph(
        "Each Monte Carlo iteration draws revenue and cost impact from "
        "Gaussian distributions scaled by the scenario's risk level (low / medium / "
        "high). Per-iteration shocks are applied: market downturn (8 percent "
        "probability, -30 percent to revenue impact), competitive response "
        "(15 percent, -20 percent), execution failure (variable - see below). "
        "Implementation timelines suffer delays up to +80 percent for high-risk "
        "projects; investment is subject to overruns up to +50 percent.", styles["body"]))

    flow.append(Paragraph("Empirical priors", styles["h2"]))
    flow.append(Paragraph(
        f"For each scenario, TimeStone retrieves up to five most similar real "
        f"transformations from a library of {ctx.report.case_library_size} cases, "
        f"scored on industry, company size, transformation type, and geography "
        f"overlap. The empirical distribution of actual revenue uplift and cost "
        f"reduction among retrieved cases is blended with the rule-based prior "
        f"(more weight to empirical at higher sample sizes). The observed "
        f"failure rate of similar cases overrides the default execution_failure_prob, "
        f"clipped to the 2-60 percent band.", styles["body"]))

    flow.append(Paragraph("Configuration snapshot", styles["h2"]))
    cfg = ctx.report.config_summary
    rows = [[k, str(v)] for k, v in cfg.items()]
    t = Table(rows, colWidths=[7 * cm, 8 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), COLOR_MUTED),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    flow.append(t)
    flow.append(PageBreak())
    return flow


def _sources_appendix(ctx: ReportContext, styles) -> List:
    flow = []
    flow.append(Paragraph("Sources", styles["h1"]))
    flow.append(Paragraph(
        "The following real corporate transformations informed the priors used "
        "in the top recommendations.", styles["body"]))
    flow.append(Spacer(1, 0.3 * cm))

    # Deduplicate case IDs across all top recommendations
    used_ids: List[str] = []
    seen = set()
    for rec in ctx.report.top_recommendations:
        for cid in rec.based_on_cases:
            if cid not in seen:
                used_ids.append(cid)
                seen.add(cid)

    for cid in used_ids:
        case = ctx.cases_by_id.get(cid)
        if case is None:
            continue
        t = case.get("transformation", {})
        fin = case.get("financials", {})
        status = t.get("status", "?")
        status_color = {"success": COLOR_SUCCESS, "partial": COLOR_WARN, "failed": COLOR_DANGER}.get(status, COLOR_MUTED)
        flow.append(Paragraph(
            f"<b>{case.get('company', cid)}</b>  "
            f"<font color='{status_color.hexval()}'>[{status.upper()}]</font>  "
            f"<font color='{COLOR_MUTED.hexval()}'>{case.get('industry', '?')} - "
            f"{case.get('geography', '?')} - {t.get('start_year', '?')}</font>",
            styles["h3"]))
        desc_bits = []
        if t.get("description"):
            desc_bits.append(t["description"])
        if fin.get("actual_revenue_uplift_pct") is not None:
            desc_bits.append(f"actual revenue uplift {fmt_pct(fin['actual_revenue_uplift_pct'])}")
        if fin.get("actual_cost_reduction_pct") is not None:
            desc_bits.append(f"actual cost reduction {fmt_pct(fin['actual_cost_reduction_pct'])}")
        if fin.get("writeoff_usd"):
            desc_bits.append(f"writeoff {fmt_money(fin['writeoff_usd'])}")
        flow.append(Paragraph("  -  ".join(desc_bits), styles["body"]))

        if case.get("tacit_notes"):
            flow.append(Paragraph(f"<i>{case['tacit_notes']}</i>", styles["muted"]))

        sources = case.get("sources", [])
        if sources:
            src_text = "Sources: " + "; ".join(
                f"{s.get('type', '?')} - {s.get('title', '')} ({s.get('year', '?')})"
                for s in sources)
            flow.append(Paragraph(src_text, styles["muted"]))
        flow.append(Spacer(1, 0.2 * cm))

    return flow


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------
def generate_pdf(ctx: ReportContext, output_path: Path) -> Path:
    """Build the executive PDF and write it to output_path. Returns the path."""
    styles = _build_styles()

    def on_page(canvas, doc):
        _draw_header_footer(canvas, doc, ctx.report.run_id, ctx.report.company_name)

    doc = BaseDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"TimeStone Assessment - {ctx.report.company_name}",
        author="TimeStone AI", subject="Transformation assessment",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=on_page)])

    story: List = []
    story.extend(_cover_page(ctx, styles))
    story.extend(_executive_summary(ctx, styles))
    story.extend(_recommendations_section(ctx, styles))
    story.extend(_methodology_section(ctx, styles))
    story.extend(_sources_appendix(ctx, styles))

    doc.build(story)
    return output_path
