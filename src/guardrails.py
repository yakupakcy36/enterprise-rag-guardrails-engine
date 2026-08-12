"""Multi-layered security guardrails: injection defense, Base64 de-obfuscation, and PII redaction."""

import base64
import re
from dataclasses import dataclass
from typing import List


@dataclass
class GuardrailResult:
    is_safe: bool
    sanitized_text: str
    risk_score: float  # [0.0 - 1.0]
    violation_reasons: List[str]
    pii_redacted_count: int


class SecurityGuardrails:
    """
    Enterprise security layer preventing:
    1. Direct & indirect prompt injections
    2. Base64-encoded / obfuscated instruction smuggling
    3. Adversarial chat template / delimiter tampering
    4. Sensitive PII & secret credential leakage
    """

    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompts|directives)",
        r"(?i)disregard\s+(the\s+)?(above|previous|system)\s+rules",
        r"(?i)you\s+are\s+now\s+(unrestricted|in\s+developer\s+mode|dan\b)",
        r"(?i)system\s*:\s*override",
        r"(?i)reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions|master\s+key)",
        r"(?i)bypass\s+(all\s+)?(safety|moderation|guardrails)",
        r"(?i)print\s+verbatim\s+the\s+prompt\s+above",
        r"(?i)start\s+response\s+with\s+[\"']yes[\"']\s+and\s+confirm\s+evil",
        r"(?i)base64\s+decode\s+and\s+execute",
    ]

    PII_PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "API_KEY": r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z-_]{35})\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "IPV4": r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}\b",
    }

    def __init__(self, risk_threshold: float = 0.70):
        self.risk_threshold = risk_threshold
        self._compiled_injections = [re.compile(p) for p in self.INJECTION_PATTERNS]
        self._compiled_pii = {k: re.compile(v) for k, v in self.PII_PATTERNS.items()}

    def _decode_potential_base64(self, text: str) -> str:
        """Scan for Base64-like tokens and decode them to inspect hidden payloads."""
        candidates = re.findall(r"[A-Za-z0-9+/]{16,}={0,2}", text)
        decoded_segments = []
        for token in candidates:
            if len(token) % 4 == 0:
                try:
                    raw = base64.b64decode(token, validate=True).decode("utf-8", errors="ignore")
                    if raw and any(c.isalpha() for c in raw):
                        decoded_segments.append(raw)
                except Exception:
                    continue
        return " ".join(decoded_segments)

    def inspect_query(self, query: str) -> GuardrailResult:
        """Inspects, de-obfuscates, and sanitizes incoming user query."""
        violations = []
        risk_score = 0.0

        if not query or not query.strip():
            return GuardrailResult(
                is_safe=False,
                sanitized_text="",
                risk_score=1.0,
                violation_reasons=["Query is empty or whitespace"],
                pii_redacted_count=0,
            )

        # 1. Direct pattern scan
        for pattern in self._compiled_injections:
            if pattern.search(query):
                violations.append("Prompt Injection / Instruction Override Detected")
                risk_score = 0.95
                break

        # 2. Obfuscated Base64 payload scan
        decoded_payload = self._decode_potential_base64(query)
        if decoded_payload:
            for pattern in self._compiled_injections:
                if pattern.search(decoded_payload):
                    violations.append("Obfuscated Base64 Prompt Injection Detected")
                    risk_score = max(risk_score, 0.98)
                    break

        # 3. Delimiter tampering checks
        if re.search(r"(```system|<\|im_start\|>|<\|im_end\|>|\[INST\]|\[\/INST\])", query):
            violations.append("Adversarial Delimiter Injection Detected")
            risk_score = max(risk_score, 0.90)

        # 4. PII & Credential Redaction
        sanitized = query
        pii_count = 0
        for pii_type, pii_regex in self._compiled_pii.items():
            matches = pii_regex.findall(sanitized)
            if matches:
                pii_count += len(matches)
                sanitized = pii_regex.sub(f"[REDACTED_{pii_type}]", sanitized)

        is_safe = risk_score < self.risk_threshold
        return GuardrailResult(
            is_safe=is_safe,
            sanitized_text=sanitized,
            risk_score=risk_score,
            violation_reasons=violations,
            pii_redacted_count=pii_count,
        )
