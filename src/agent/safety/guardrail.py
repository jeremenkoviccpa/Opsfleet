"""Input guardrail: scope enforcement and prompt-injection defence.

Runs before the planner sees the message, in two stages:

  Stage 1 - deterministic patterns. Cheap, unbypassable by paraphrase-resistant
            wording, and costs nothing. Catches the classic instruction-override
            and PII-extraction phrasings outright.
  Stage 2 - an LLM classifier, used only when stage 1 is inconclusive. It
            decides scope (is this an analysis question at all?) rather than
            safety, because a classifier is exactly the wrong place to put a
            control that an attacker can talk to.

Design note: the guardrail is a *scope* control. The controls that actually
protect the data are non-LLM and sit downstream - the SQL validator and the
result masker. A jailbreak that gets past this classifier still cannot select a
denied column or see an unmasked value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..llm import LLMRouter, TurnBudget
from ..obs import metrics

ALLOW = "allow"
REFUSE_SCOPE = "refuse_out_of_scope"
REFUSE_PII = "refuse_pii_request"
BLOCK_INJECTION = "block_prompt_injection"
BLOCK_DESTRUCTIVE = "block_destructive_sql"

INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?(your\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?)", "instruction override"),
    (r"disregard\s+(all\s+)?(your\s+|the\s+)?(previous|prior|above|safety)\s+\w+", "instruction override"),
    (r"(reveal|show|print|repeat|output|dump|display|what is)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules?|persona|configuration)", "system prompt extraction"),
    (r"you\s+are\s+now\s+(a|an|in)\b", "persona hijack"),
    (r"\b(developer|debug|god|admin|root|dan)\s+mode\b", "mode escalation"),
    (r"pretend\s+(that\s+)?(you|there)\s+(are|is|have)\b", "persona hijack"),
    (r"(bypass|disable|turn\s+off|switch\s+off|remove)\s+(the\s+)?(safety|guardrails?|filters?|masking|pii|redaction|security)", "control bypass"),
    (r"without\s+(any\s+)?(masking|redaction|filtering|restrictions?)", "control bypass"),
    (r"\bnew\s+(system\s+)?(instructions?|rules?)\s*[:\-]", "instruction injection"),
    (r"</?(system|instructions?|admin)>", "tag injection"),
    (r"\bsudo\b|\boverride\b\s+(the\s+)?(policy|rules)", "control bypass"),
]

PII_REQUEST_PATTERNS = [
    (r"\b(email|e-mail)\s+(address(es)?|of|for)\b", "email addresses"),
    (r"\b(list|show|give|get|export|send)\s+(me\s+)?(the\s+)?(customers?|users?|clients?)('|’)?s?\s+(names?|emails?|addresses?|phone)", "customer identifiers"),
    (r"\b(full\s+)?names?\s+(and|,)\s*(email|address|phone)", "customer identifiers"),
    (r"\b(home\s+)?address(es)?\s+(of|for)\s+(the\s+)?(customers?|users?|top)", "postal addresses"),
    (r"\bphone\s+numbers?\b", "phone numbers"),
    (r"\b(personally identifiable|pii)\b.*\b(show|give|reveal|list)", "explicit PII request"),
    (r"\b(who|which person)\s+(exactly|specifically)\s+(is|are)\b.*\b(customer|user)\b", "re-identification"),
    (r"\bcontact\s+(details|info(rmation)?)\s+(of|for)\b", "contact details"),
]

DESTRUCTIVE_SQL_PATTERNS = [
    (r"\b(drop|truncate)\s+(table|database|schema)\b", "DDL"),
    (r"\bdelete\s+from\b", "DML"),
    (r"\bupdate\s+\w+\s+set\b", "DML"),
    (r"\binsert\s+into\b", "DML"),
    (r"\balter\s+table\b", "DDL"),
    (r"\bgrant\s+\w+\s+(on|to)\b", "privilege change"),
    (r"\bcreate\s+(or\s+replace\s+)?(table|view|function)\b", "DDL"),
    # Natural-language phrasings. Deliberately anchored on the object rather
    # than the verb, so analytical uses of "remove" ("remove cancelled orders
    # from the revenue calculation") do not trip the rule.
    (r"\b(delete|drop|truncate|wipe|erase|purge)\s+(all\s+|every\s+)?(the\s+)?"
     r"(rows?|records?|entries|tables?|databases?|data)\b", "natural-language DML"),
    (r"\b(delete|drop|truncate|wipe|erase)\s+(the\s+)?"
     r"(orders|users|customers|products|order_items|inventory_items)\b", "natural-language DML"),
    (r"\b(clear|empty)\s+(out\s+)?(the\s+)?(orders|users|products|order_items|table|database)\b",
     "natural-language DML"),
]

# Report-library deletion is a legitimate, supported action - it must not be
# confused with an attempt to write to the warehouse.
REPORT_OP_HINTS = re.compile(
    r"\b(report|reports|briefing|summary|summaries)\b.*\b(delete|remove|discard|purge|drop|erase)\b"
    r"|\b(delete|remove|discard|purge|erase)\b.*\b(report|reports|briefing|summary|summaries)\b",
    re.IGNORECASE,
)


@dataclass
class GuardrailVerdict:
    decision: str = ALLOW
    reasons: List[str] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)
    stage: str = "deterministic"
    user_message: str = ""
    confidence: float = 1.0

    @property
    def allowed(self) -> bool:
        return self.decision == ALLOW

    @property
    def blocked(self) -> bool:
        return self.decision in (BLOCK_INJECTION, BLOCK_DESTRUCTIVE)


REFUSAL_COPY = {
    REFUSE_SCOPE: (
        "I only answer questions about the retail sales, product, customer and inventory "
        "data — and I can write reports on top of it. That one is outside what I can help with.\n\n"
        "Try something like: *\"How did revenue trend over the last 6 months?\"* or "
        "*\"Compare Jeans against Shorts and tell me why they differ.\"*"
    ),
    REFUSE_PII: (
        "I can't return personal details — names, email addresses, postal addresses, phone "
        "numbers or exact locations. That restriction is enforced in the query layer, not "
        "just in my instructions.\n\n"
        "I *can* answer the underlying business question with pseudonymous customer IDs and "
        "cohort attributes: state, age band, acquisition channel, spend and order frequency. "
        "Want me to do that instead?\n\n"
        "If you genuinely need to contact these customers, that is a CRM export request and "
        "goes through your data-governance approval path."
    ),
    BLOCK_INJECTION: (
        "That message looks like an attempt to change my instructions rather than to ask about "
        "the data, so I've stopped there and logged it.\n\n"
        "I'm happy to keep going on the analysis — what would you like to know?"
    ),
    BLOCK_DESTRUCTIVE: (
        "I can't run statements that modify the warehouse. My database connection is read-only "
        "and only SELECT statements pass validation.\n\n"
        "If you meant to remove something from your **saved reports** library, say so explicitly "
        "— for example *\"delete the reports mentioning Northwind\"* — and I'll show you exactly "
        "what would be removed before anything happens."
    ),
}

CLASSIFIER_SYSTEM = """You classify messages sent to an internal retail data-analysis assistant.

The assistant answers questions about sales, orders, products, customers and inventory, writes \
analytical reports, and manages a library of saved reports.

Return ONLY this JSON object:
{"decision": "<allow|refuse_out_of_scope|refuse_pii_request>", "reason": "<max 15 words>", "confidence": <0.0-1.0>}

Rules:
- "allow": any question about the business data, a follow-up or clarification about a previous \
answer, a request for a report or chart, a request to manage/delete SAVED REPORTS, a greeting, \
or a question about what data is available.
- "refuse_out_of_scope": unrelated to the retail data and not conversational management \
(e.g. write me a poem, general coding help, world news, medical or legal advice).
- "refuse_pii_request": ONLY when the message explicitly asks for an identifying \
ATTRIBUTE - a name, email address, postal address, phone number or precise coordinates - \
or plainly wants to contact or personally identify someone.

Asking about customers is NOT a PII request. The assistant answers customer questions with \
pseudonymous ids and cohort attributes, so ranking, segmenting or profiling customers is \
always in scope. Refuse only the identifying attribute, never the business question.

"allow" (these are core capabilities, do not refuse them):
  "Who are our top 10 customers by total spend?"
  "Which customers are at risk of churning?"
  "What do our highest-value customers have in common?"
  "Show me the biggest spenders in Texas and their order frequency."
  "Break down revenue by customer age and acquisition channel."

"refuse_pii_request":
  "Give me the names and emails of our top customers."
  "What is the home address of customer 4821?"
  "Send me a contact list for the Texas cohort."

Be permissive. Vague or terse follow-ups like "why?", "and Texas?", "go deeper" are "allow".
When genuinely unsure, choose "allow" - the SQL validator and the result masker enforce the
data policy regardless of what you decide."""


class Guardrail:
    def __init__(self, router: Optional[LLMRouter] = None, tracer=None) -> None:
        self.router = router
        self.tracer = tracer
        self._injection = [(re.compile(p, re.IGNORECASE), label) for p, label in INJECTION_PATTERNS]
        self._pii = [(re.compile(p, re.IGNORECASE), label) for p, label in PII_REQUEST_PATTERNS]
        self._destructive = [(re.compile(p, re.IGNORECASE), label) for p, label in DESTRUCTIVE_SQL_PATTERNS]

    def check(self, message: str, budget: Optional[TurnBudget] = None) -> GuardrailVerdict:
        verdict = self._deterministic(message)
        if verdict.decision != ALLOW:
            self._record(verdict)
            return verdict
        verdict = self._classify(message, budget) if self.router else verdict
        self._record(verdict)
        return verdict

    # ---- stage 1 ---------------------------------------------------------
    def _deterministic(self, message: str) -> GuardrailVerdict:
        text = message or ""
        for pattern, label in self._injection:
            if pattern.search(text):
                return GuardrailVerdict(
                    decision=BLOCK_INJECTION,
                    reasons=[f"prompt injection: {label}"],
                    matched_rules=[pattern.pattern],
                    user_message=REFUSAL_COPY[BLOCK_INJECTION],
                )
        if not REPORT_OP_HINTS.search(text):
            for pattern, label in self._destructive:
                if pattern.search(text):
                    return GuardrailVerdict(
                        decision=BLOCK_DESTRUCTIVE,
                        reasons=[f"destructive SQL requested: {label}"],
                        matched_rules=[pattern.pattern],
                        user_message=REFUSAL_COPY[BLOCK_DESTRUCTIVE],
                    )
        for pattern, label in self._pii:
            if pattern.search(text):
                return GuardrailVerdict(
                    decision=REFUSE_PII,
                    reasons=[f"PII extraction attempt: {label}"],
                    matched_rules=[pattern.pattern],
                    user_message=REFUSAL_COPY[REFUSE_PII],
                )
        return GuardrailVerdict()

    # ---- stage 2 ---------------------------------------------------------
    def _classify(self, message: str, budget: Optional[TurnBudget]) -> GuardrailVerdict:
        try:
            parsed, _ = self.router.complete_json(
                purpose="guardrail",
                system=CLASSIFIER_SYSTEM,
                messages=[{"role": "user", "content": message}],
                default={"decision": ALLOW, "reason": "classifier unavailable", "confidence": 0.0},
                tier="fast",
                budget=budget,
            )
        except Exception as exc:
            # Fail open on scope, never on safety: stage 1 already ran, and the
            # SQL validator plus result masker still stand between the user and
            # the data.
            return GuardrailVerdict(
                decision=ALLOW, stage="llm_unavailable",
                reasons=[f"classifier unavailable, deterministic rules passed: {type(exc).__name__}"],
                confidence=0.0,
            )
        decision = str(parsed.get("decision", ALLOW))
        if decision not in (ALLOW, REFUSE_SCOPE, REFUSE_PII):
            decision = ALLOW
        return GuardrailVerdict(
            decision=decision,
            reasons=[str(parsed.get("reason", ""))],
            stage="llm",
            confidence=float(parsed.get("confidence", 0.5) or 0.5),
            user_message=REFUSAL_COPY.get(decision, ""),
        )

    @staticmethod
    def _record(verdict: GuardrailVerdict) -> None:
        if verdict.decision != ALLOW:
            metrics.incr("guardrail.blocked", decision=verdict.decision)
        if verdict.decision == BLOCK_INJECTION:
            metrics.incr("guardrail.injection_detected")
