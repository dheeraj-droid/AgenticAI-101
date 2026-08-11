"""Mail delivery for the onboarding notifications.

Two messages go out once a customer is registered: one to the internal team with
the onboarding checklist, and one to the customer with the welcome email.

**Sending is off by default.** ``deliver`` writes a real ``.eml`` file to
``.runs/outbox/`` and records a ``mail_sent`` audit event. Actually transmitting
requires all of:

* ``SMTP_HOST`` configured, and
* ``allow_send=True`` (the CLI's ``--send`` flag), and
* for the customer-facing message only, a recipient on the allowlist.

That last condition is the important one: a form takes whatever address is typed
into it, and a demo run must not be able to email a real stranger.
"""

from __future__ import annotations

import os
import re
import smtplib
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Literal

from onboarding.core.audit import JsonlAuditSink
from onboarding.core.concepts import Concept, concept
from onboarding.core.config import paths
from onboarding.core.schemas import CustomerRecord, OnboardingTask, WelcomeEmail

Audience = Literal["team", "customer"]


def support_address() -> str:
    """Where every internal task list goes. One address, set once."""
    return os.environ.get("ONBOARDING_SUPPORT_EMAIL", team_address())


def allowlist() -> set[str]:
    """Customer addresses cleared to receive real mail.

    A demo form takes whatever email is typed into it. Without this, a typo
    sends real mail to a real stranger, and that cannot be undone. Set
    ONBOARDING_ALLOWED_RECIPIENTS to a comma-separated list; "*" disables the
    check entirely and is deliberately awkward to type by accident.
    """
    raw = os.environ.get("ONBOARDING_ALLOWED_RECIPIENTS", "")
    return {a.strip().lower() for a in raw.split(",") if a.strip()}


def is_allowed(address: str) -> bool:
    allowed = allowlist()
    if "*" in allowed:
        return True
    return address.strip().lower() in allowed


def outbox_dir() -> Path:
    override = os.environ.get("ONBOARDING_OUTBOX")
    return Path(override) if override else paths().runs / "outbox"


def team_address() -> str:
    return os.environ.get("ONBOARDING_TEAM_EMAIL", "onboarding-team@example.internal")


def smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST"))


def from_address() -> str:
    return os.environ.get("ONBOARDING_FROM_EMAIL", "onboarding@example.internal")


@dataclass(frozen=True, slots=True)
class OutboundMail:
    audience: Audience
    to: str
    subject: str
    body: str
    record_id: str
    run_id: str

    def as_message(self) -> EmailMessage:
        message = EmailMessage()
        message["From"] = from_address()
        message["To"] = self.to
        message["Subject"] = self.subject
        message["X-Onboarding-Run"] = self.run_id
        message["X-Onboarding-Record"] = self.record_id
        message["X-Onboarding-Audience"] = self.audience
        message["Date"] = datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S +0000")
        message.set_content(self.body)
        return message


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    mail: OutboundMail
    path: Path | None
    sent: bool
    reason: str = ""


@concept(Concept.ACTION)
def build_team_mail(
    record: CustomerRecord, tasks: list[OnboardingTask], run_id: str, *, registered: bool
) -> OutboundMail:
    """The internal notification: who signed up and what ops needs to do."""
    lines = [
        f"{record.company_name} has been onboarded." if registered
        else f"{record.company_name} reached onboarding but was not registered.",
        "",
        f"Record:  {record.record_id}",
        f"Contact: {record.primary_contact.full_name}",
        f"Plan:    {record.effective_plan}",
        f"Tier:    {record.tier}   Region: {record.region.upper()}",
        f"Value:   {record.commercial_terms.annual_contract_value_usd} "
        f"{record.commercial_terms.currency} over {record.commercial_terms.term_months} months",
        "",
        "Onboarding checklist:",
    ]
    for task in tasks:
        lines.append(
            f"  [{task.priority}] {task.title} — {task.owner_role}, due in {task.due_offset_days}d"
        )
    if not tasks:
        lines.append("  (none generated)")
    lines += ["", f"Run: {run_id}"]

    return OutboundMail(
        audience="team",
        to=support_address(),
        subject=f"[onboarding] {record.company_name} ({record.effective_plan})",
        body="\n".join(lines),
        record_id=record.record_id,
        run_id=run_id,
    )


@concept(Concept.ACTION)
def build_already_registered_mail(record: CustomerRecord, run_id: str) -> OutboundMail:
    """The duplicate path: tell them they already have an account.

    Deterministic text, no model involved — there is nothing to draft, and a
    known customer should get a consistent message every time.
    """
    body = "\n".join(
        [
            f"Hello {record.primary_contact.full_name},",
            "",
            f"Thanks for signing up to {record.company_name} again — it looks like you",
            "already have an account with us, so there is nothing more to do.",
            "",
            "If you were expecting a new workspace, or you think this is a mistake,",
            "just reply to this message and the onboarding team will take a look.",
            "",
            "Best regards,",
            "The Onboarding Team",
        ]
    )
    return OutboundMail(
        audience="customer",
        to=str(record.primary_contact.email),
        subject=f"You are already signed up, {record.company_name}",
        body=body,
        record_id=record.record_id,
        run_id=run_id,
    )


@concept(Concept.ACTION)
def build_customer_mail(record: CustomerRecord, email: WelcomeEmail, run_id: str) -> OutboundMail:
    """The customer-facing welcome, with masked placeholders resolved.

    The draft is written against ``<PERSON_1>``-style placeholders so the model
    never sees a real name. Substituting the real values back in happens here, in
    deterministic code, at the very last moment before delivery.
    """
    return OutboundMail(
        audience="customer",
        to=str(record.primary_contact.email),
        subject=personalise(email.subject, record),
        body=personalise(email.body, record),
        record_id=record.record_id,
        run_id=run_id,
    )


_PLACEHOLDER = re.compile(r"<(PERSON|EMAIL_ADDRESS|PHONE_NUMBER)_(\d+)>")


def personalise(text: str, record: CustomerRecord) -> str:
    """Replace masking placeholders with the real values, ordered as masked."""
    contacts = record.all_contacts
    names = [c.full_name for c in contacts]
    emails = [str(c.email) for c in contacts]
    phones = [c.phone or "" for c in contacts]
    table = {"PERSON": names, "EMAIL_ADDRESS": emails, "PHONE_NUMBER": phones}

    def substitute(match: re.Match[str]) -> str:
        values = table.get(match.group(1), [])
        index = int(match.group(2)) - 1
        if 0 <= index < len(values) and values[index]:
            return values[index]
        return match.group(0)

    return _PLACEHOLDER.sub(substitute, text)


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-")[:60]


@concept(Concept.ACTION, Concept.AUDIT_LOGGING)
def deliver(
    mail: OutboundMail,
    sink: JsonlAuditSink | None = None,
    *,
    allow_send: bool = False,
) -> DeliveryResult:
    """Write the message to the outbox, and transmit it only if truly permitted."""
    directory = outbox_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = directory / f"{stamp}-{mail.audience}-{_slug(mail.record_id)}.eml"
    path.write_text(mail.as_message().as_string(), encoding="utf-8")

    sent, reason = False, ""
    host = os.environ.get("SMTP_HOST")
    if not allow_send:
        reason = "written to the outbox; pass --send to transmit"
    elif not host:
        reason = "SMTP_HOST is not set, so nothing was transmitted"
    elif mail.audience == "customer" and not is_allowed(mail.to):
        reason = (
            f"{mail.to} is not on ONBOARDING_ALLOWED_RECIPIENTS, so nothing was sent. "
            "Add the address to send to it for real."
        )
    else:
        try:
            _transmit(mail, host)
            sent, reason = True, f"transmitted via {host}"
        except Exception as exc:
            reason = f"SMTP delivery failed: {type(exc).__name__}: {exc}"

    if sink is not None:
        sink.emit(
            "mail_sent",
            audience=mail.audience,
            subject=mail.subject,
            outbox_path=str(path),
            transmitted=sent,
            reason=reason,
        )
    return DeliveryResult(mail=mail, path=path, sent=sent, reason=reason)


def _transmit(mail: OutboundMail, host: str) -> None:
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    with smtplib.SMTP(host, port, timeout=30) as server:
        if os.environ.get("SMTP_STARTTLS", "1") not in ("0", "false"):
            server.starttls()
        if user and password:
            server.login(user, password)
        server.send_message(mail.as_message())


def list_outbox(directory: Path | None = None) -> list[Path]:
    target = directory or outbox_dir()
    return sorted(target.glob("*.eml")) if target.exists() else []
