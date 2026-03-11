"""
STRATA Evolution Engine — Notifier Module
Sends email summaries of evolution progress.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

logger = logging.getLogger("strata.notifier")


def _get_config() -> dict | None:
    """Get email configuration from environment. Returns None if not configured."""
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    notify_email = os.environ.get("NOTIFY_EMAIL")

    if not all([smtp_user, smtp_pass, notify_email]):
        return None

    return {
        "host": os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": smtp_user,
        "pass": smtp_pass,
        "to": notify_email,
    }


def send_evolution_summary(
    generation: int,
    fitness: float,
    halluc_rate: float,
    metrics: dict[str, float],
    insight: dict | None,
    emergent_principles: list[str],
    stagnation_counter: int,
    elapsed_seconds: float,
    converged: bool = False,
    spec_text: str | None = None,
) -> bool:
    """Send an evolution progress email. Returns True if sent successfully."""
    config = _get_config()
    if not config:
        logger.debug("Email not configured, skipping notification")
        return False

    # Format elapsed time
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

    subject = (
        f"STRATA KONVERGIERT — Gen {generation} | Fitness {fitness:.4f}"
        if converged
        else f"STRATA Update — Gen {generation} | Fitness {fitness:.4f} | Halluc {halluc_rate:.2%}"
    )

    # Build HTML email
    status_color = "#06b6d4" if converged else "#22c55e"
    status_text = "KONVERGIERT" if converged else "AKTIV"

    metrics_rows = ""
    for k, v in sorted(metrics.items(), key=lambda x: x[0]):
        bar_pct = v * 100
        color = "#ef4444" if k in ("halluc_rate", "over_caution_score") else "#22c55e"
        metrics_rows += f"""
        <tr>
            <td style="padding:6px 12px;border-bottom:1px solid #222;font-size:13px;color:#a0a0b0">{k}</td>
            <td style="padding:6px 12px;border-bottom:1px solid #222;width:200px">
                <div style="background:#111;border-radius:4px;height:16px;overflow:hidden">
                    <div style="background:{color};height:100%;width:{bar_pct:.1f}%;border-radius:4px"></div>
                </div>
            </td>
            <td style="padding:6px 12px;border-bottom:1px solid #222;font-size:13px;color:#e0e0e8;text-align:right;font-weight:600">{v:.4f}</td>
        </tr>"""

    insight_html = ""
    if insight:
        insight_html = f"""
        <div style="background:#1a1a28;border:1px solid #6366f1;border-radius:8px;padding:16px;margin:16px 0">
            <h3 style="color:#818cf8;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px">Claude Insight</h3>
            <p style="color:#a0a0b0;font-size:13px;line-height:1.6;margin:0">{insight.get('analysis', '')}</p>
            <div style="margin-top:8px;font-size:11px;color:#707088">
                Bottleneck: <strong style="color:#f59e0b">{insight.get('bottleneck_metric', '—')}</strong>
                &nbsp;|&nbsp; Strategie: {insight.get('strategy', '—')}
            </div>
        </div>"""

    principles_html = ""
    if emergent_principles:
        items = "".join(
            f'<li style="padding:6px 12px;border-left:3px solid #6366f1;margin-bottom:6px;background:#1a1a28;border-radius:0 4px 4px 0;font-size:12px;color:#a0a0b0">{p}</li>'
            for p in emergent_principles[-10:]
        )
        principles_html = f"""
        <div style="margin:16px 0">
            <h3 style="color:#818cf8;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px">Emergente Prinzipien</h3>
            <ul style="list-style:none;padding:0;margin:0">{items}</ul>
        </div>"""

    spec_html = ""
    if converged and spec_text:
        spec_preview = spec_text[:2000] + ("..." if len(spec_text) > 2000 else "")
        spec_html = f"""
        <div style="background:#1a1a28;border:1px solid #06b6d4;border-radius:8px;padding:16px;margin:16px 0">
            <h3 style="color:#06b6d4;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px">Finale Spezifikation (Vorschau)</h3>
            <pre style="color:#a0a0b0;font-size:11px;line-height:1.5;white-space:pre-wrap;margin:0;font-family:monospace">{spec_preview}</pre>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<body style="background:#0a0a0f;margin:0;padding:20px;font-family:'Helvetica Neue',Arial,sans-serif">
<div style="max-width:640px;margin:0 auto;background:#12121a;border-radius:12px;overflow:hidden;border:1px solid #222">
    <div style="background:linear-gradient(135deg,#12121a,#1a1a28);padding:24px;text-align:center;border-bottom:1px solid #222">
        <h1 style="color:#818cf8;font-size:18px;letter-spacing:3px;margin:0">STRATA EVOLUTION ENGINE</h1>
        <div style="margin-top:8px">
            <span style="background:{status_color};color:#0a0a0f;padding:4px 12px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:1px">{status_text}</span>
        </div>
    </div>

    <div style="padding:24px">
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px">
            <div style="flex:1;min-width:120px;background:#0a0a0f;border-radius:8px;padding:12px;text-align:center">
                <div style="font-size:10px;color:#707088;text-transform:uppercase;letter-spacing:1px">Generation</div>
                <div style="font-size:24px;font-weight:700;color:#e0e0e8;margin-top:4px">{generation}</div>
            </div>
            <div style="flex:1;min-width:120px;background:#0a0a0f;border-radius:8px;padding:12px;text-align:center">
                <div style="font-size:10px;color:#707088;text-transform:uppercase;letter-spacing:1px">Fitness</div>
                <div style="font-size:24px;font-weight:700;color:#22c55e;margin-top:4px">{fitness:.4f}</div>
            </div>
            <div style="flex:1;min-width:120px;background:#0a0a0f;border-radius:8px;padding:12px;text-align:center">
                <div style="font-size:10px;color:#707088;text-transform:uppercase;letter-spacing:1px">Halluc-Rate</div>
                <div style="font-size:24px;font-weight:700;color:{'#22c55e' if halluc_rate < 0.05 else '#f59e0b' if halluc_rate < 0.15 else '#ef4444'};margin-top:4px">{halluc_rate:.2%}</div>
            </div>
            <div style="flex:1;min-width:120px;background:#0a0a0f;border-radius:8px;padding:12px;text-align:center">
                <div style="font-size:10px;color:#707088;text-transform:uppercase;letter-spacing:1px">Laufzeit</div>
                <div style="font-size:24px;font-weight:700;color:#e0e0e8;margin-top:4px">{time_str}</div>
            </div>
        </div>

        <h3 style="color:#818cf8;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px">Metriken</h3>
        <table style="width:100%;border-collapse:collapse;background:#0a0a0f;border-radius:8px;overflow:hidden">
            {metrics_rows}
        </table>

        {insight_html}
        {principles_html}
        {spec_html}
    </div>

    <div style="padding:16px 24px;border-top:1px solid #222;text-align:center">
        <p style="color:#707088;font-size:11px;margin:0">STRATA Evolution Engine | Dashboard: http://localhost:7860</p>
    </div>
</div>
</body>
</html>"""

    # Also build plain text fallback
    text_lines = [
        f"STRATA Evolution Engine — {'KONVERGIERT' if converged else 'Update'}",
        f"",
        f"Generation: {generation}",
        f"Fitness: {fitness:.4f}",
        f"Halluzinationsrate: {halluc_rate:.2%}",
        f"Stagnation: {stagnation_counter}",
        f"Laufzeit: {time_str}",
        f"",
        f"Metriken:",
    ]
    for k, v in sorted(metrics.items()):
        text_lines.append(f"  {k}: {v:.4f}")

    if insight:
        text_lines.extend([
            f"",
            f"Claude Insight: {insight.get('analysis', '')}",
            f"Bottleneck: {insight.get('bottleneck_metric', '—')}",
        ])

    if emergent_principles:
        text_lines.extend(["", "Emergente Prinzipien:"])
        for p in emergent_principles[-10:]:
            text_lines.append(f"  - {p}")

    plain_text = "\n".join(text_lines)

    # Send email
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = config["user"]
        msg["To"] = config["to"]

        msg.attach(MIMEText(plain_text, "plain", "utf-8"))
        msg.attach(MIMEText(html, "html", "utf-8"))

        with smtplib.SMTP(config["host"], config["port"]) as server:
            server.starttls()
            server.login(config["user"], config["pass"])
            server.send_message(msg)

        logger.info(f"Email summary sent to {config['to']} (Gen {generation})")
        return True

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False
