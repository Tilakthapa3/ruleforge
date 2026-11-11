"""RuleForge step 1: model clients, in proposer and reviewer roles.

The proposer drafts a detection rule; the reviewer critiques it.
They use different system prompts -- and ideally different models --
because the flaw in ScientistTwo's headline numbers was an in-house
grader grading its own drafts. Here the final score always comes
from held-out logs the proposer never saw, so even a shared model
cannot grade its own homework.

MockClient is offline and deterministic: it proposes one plausible
first rule, then applies one concrete refinement per round (a burst
threshold, then a publickey probe), and the reviewer returns one
concrete suggestion per round. That is enough to exercise the whole
loop and the scorer without a model running.

OllamaClient mirrors SOC Agent's: POST to a local Ollama model on
http://localhost:11434. Step 1 runs on mock only; Ollama is a
skeleton for step 2, when the loop runs live on the Mac.

Standard library only. The only network is the Ollama localhost
client, unused in step 1.
"""

import json
import os
import re
import urllib.request

from rules import CONDITION_HELP

PROPOSER_SYSTEM = (
    "You are a detection-rule researcher. Propose ONE Sigma-compatible "
    "detection rule for the given goal, as a JSON object with exactly "
    "these fields: title, logsource {category, product, service}, "
    "detection, level, description. The detection object has exactly "
    "two keys: 'selection' and 'condition'. The selection maps a field "
    "(message, src_ip, user, event) to a string or a list of substrings "
    "to match. Do NOT add any other keys under detection -- no src_ip, "
    "event, user, or message keys there. The condition is one of: the "
    "selection name; '<a> and <b>'; '<a> and not <b>'; '<a> or <b>'; "
    "or 'selection | count() by src_ip >= N' for a burst threshold. "
    + CONDITION_HELP +
    " "
    "The level is one of: informational, low, medium, high, critical. "
    "Here is an example of the exact shape to produce:\n"
    "```json\n"
    "{\n"
    '  "title": "SSH Brute Force - Failed Logins From One IP",\n'
    '  "logsource": {"category": "auth", "product": "linux", "service": "sshd"},\n'
    '  "detection": {\n'
    '    "selection": {"message": ["Failed password", "Failed publickey"]},\n'
    '    "condition": "selection | count() by src_ip >= 5"\n'
    "  },\n"
    '  "level": "high",\n'
    '  "description": "Flags repeated failed SSH logins from one source IP."\n'
    "}\n"
    "```\n"
    "Reply with ONLY a ```json fenced block containing the rule object."
)

REVIEWER_SYSTEM = (
    "You are an independent detection-rule reviewer. Critique the "
    "proposed rule using its train results: what it caught, what it "
    "missed, and which benign lines it wrongly matched. Read the "
    "numbers first and let them decide the verdict: if the catch rate "
    "is 0, the rule catches nothing, so the threshold is too high (or "
    "the selection matches nothing) -- say so and suggest LOWERING "
    "the threshold, never call it good and never suggest raising it. "
    "If the false-positive rate is high, suggest raising the "
    "threshold or narrowing the selection. Quote the actual TP, FP, "
    "FN, TN counts and the actual threshold from the rule as given -- "
    "never invent numbers. Only write 'VERDICT: good rule' when the "
    "catch rate is high AND the false-positive rate is low. Reply "
    "with a short verdict and exactly ONE concrete refinement "
    "suggestion, or the line 'VERDICT: good rule' when no "
    "improvement is needed. Never propose a rule yourself; only "
    "critique and suggest."
)

_FENCED_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
_ROLE_RE = re.compile(r"ROLE:\s*(proposer|reviewer)", re.IGNORECASE)
_ROUND_RE = re.compile(r"ROUND:\s*(\d+)")
_TRAIN_RE = re.compile(
    r"Train results:\s*TP=(\d+),\s*FP=(\d+),\s*FN=(\d+),\s*TN=(\d+)")


def _base_rule(message_values, condition):
    return {
        "title": "SSH Brute Force - Failed Logins From One IP",
        "logsource": {"category": "auth", "product": "linux",
                      "service": "sshd"},
        "detection": {
            "selection": {"message": message_values},
            "condition": condition,
        },
        "level": "high",
        "description": "Flags repeated failed SSH logins from one "
                       "source IP.",
    }


# The mock proposer's rule per round: round 1 is the plain rule, round
# 2 adds a burst threshold, round 3 also covers publickey probes.
MOCK_RULES = {
    1: _base_rule(["Failed password"], "selection"),
    2: _base_rule(["Failed password"],
                  "selection | count() by src_ip >= 5"),
    3: _base_rule(["Failed password", "Failed publickey"],
                  "selection | count() by src_ip >= 5"),
}


def _mock_review(round_no, prompt):
    """One concrete suggestion per round, grounded in the train counts."""
    train = _TRAIN_RE.search(prompt)
    counts = ""
    if train:
        counts = (" (train: {tp} attack lines caught, {fp} benign "
                  "lines wrongly matched)".format(
                      tp=train.group(1), fp=train.group(2)))
    if round_no <= 1:
        return (
            "VERDICT: needs work{counts}. The rule catches the "
            "brute-force bursts, but it also fires on ordinary typos: "
            "benign users who mistyped once and then logged in. "
            "Suggestion: require a burst -- only fire when 5 or more "
            "failures come from one source IP. Add a count threshold "
            "keyed on src_ip.".format(counts=counts)
        )
    if round_no == 2:
        return (
            "VERDICT: needs work{counts}. False positives are gone, "
            "but the rule still misses the attacker's publickey probe "
            "(an attack line it did not match). Suggestion: also match "
            "'Failed publickey' in the message selection.".format(
                counts=counts)
        )
    return (
        "VERDICT: good rule{counts}. It catches brute-force bursts, "
        "ignores single typos, and covers both password and publickey "
        "probes. Remaining gap: the successful login after a burst is "
        "not covered -- that needs a separate login-after-brute-force "
        "rule.".format(counts=counts)
    )


class MockClient:
    """Offline stand-in model: canned proposer/reviewer replies."""

    def complete(self, prompt):
        role_match = _ROLE_RE.search(prompt)
        role = role_match.group(1).lower() if role_match else "proposer"
        round_match = _ROUND_RE.search(prompt)
        round_no = int(round_match.group(1)) if round_match else 1
        if role == "reviewer":
            return _mock_review(round_no, prompt)
        rule = MOCK_RULES.get(round_no, MOCK_RULES[3])
        return ("```json\n" + json.dumps(rule, indent=2)
                + "\n```")


def parse_rule_reply(text):
    """Extract a rule dict from a model reply, or None.

    Accepts a ```json fenced block, or a bare JSON object anywhere in
    the text. Anything else returns None so the loop can reject the
    round and move on.
    """
    if not text:
        return None
    match = _FENCED_JSON_RE.search(text)
    candidate = match.group(1) if match else text
    candidate = candidate.strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except ValueError:
        pass
    start = candidate.find("{")
    if start < 0:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


class OllamaClient:
    """Local Ollama model via its chat endpoint (SOC Agent's shape)."""

    def __init__(self, model=None, base_url=None):
        self.model = model or os.environ.get("RULEFORGE_MODEL",
                                             "qwen2.5:7b")
        self.base_url = base_url or os.environ.get(
            "RULEFORGE_OLLAMA_URL", "http://localhost:11434")

    def complete(self, prompt):
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                "Ollama request failed (is 'ollama serve' running?): {exc}"
                .format(exc=exc)
            )
        return body["message"]["content"]
