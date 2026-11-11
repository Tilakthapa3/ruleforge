"""RuleForge step 1: rule representation, validation, and YAML output.

A rule is a plain JSON-serializable dict with Sigma-compatible fields:

    {
        "title": "SSH Brute Force - 5+ Failed Logins From One IP",
        "logsource": {"category": "auth", "product": "linux",
                      "service": "sshd"},
        "detection": {
            "selection": {
                "message": ["Failed password", "Failed publickey"],
            },
            "condition": "selection | count() by src_ip >= 5",
        },
        "level": "high",
        "description": "One sentence saying what the rule catches.",
    }

A selection is a set of field substring matches: every field must
match at least one of its listed substrings (OR within a field, AND
across fields). Supported fields are "message", "src_ip", "user",
and "event" (failed_password, failed_publickey, accepted, other).

Conditions use a small explicit grammar:

    selection                  match lines matching that selection
    a and b                    both selections match
    a and not b                first matches, second does not
    a or b                     either selection matches
    a | count() by src_ip >= N match lines whose source IP has at
                               least N selection-matching lines in
                               the same log file
    a | count() by src_ip > N  same, strictly more than N

Validation here is STRUCTURAL only: required fields present, known
fields, a condition that parses and names selections that exist, and
YAML that emits cleanly. It does not run the real `sigma check` --
that is a step-2 item for the Mac, where the Sigma toolchain lives.
The validator is one small function (validate_rule) so the real
checker can plug in behind it later.

Standard library only. No network, no shell execution.
"""

import json
import re

KNOWN_FIELDS = ("message", "src_ip", "user", "event")
LEVELS = ("informational", "low", "medium", "high", "critical")

_COUNT_RE = re.compile(
    r"^(\w+)\s*\|\s*count\(\)\s*by\s+src_ip\s*(>=|>)\s*(\d+)$")
_AND_NOT_RE = re.compile(r"^(\w+)\s+and\s+not\s+(\w+)$")
_AND_RE = re.compile(r"^(\w+)\s+and\s+(\w+)$")
_OR_RE = re.compile(r"^(\w+)\s+or\s+(\w+)$")
_SINGLE_RE = re.compile(r"^(\w+)$")


# One sentence stating the condition grammar, shared by the proposer
# prompt and the rejection note the loop feeds back after an invalid
# rule -- the model can only correct itself if it is told the rules.
CONDITION_HELP = (
    "Allowed conditions are exactly: a selection name; 'A and B'; "
    "'A and not B'; 'A or B', where A and B are selection names; or "
    "'selection | count() by src_ip >= N'. No time windows "
    "('over 1m' etc.) and no 'and'/'or' between two count() expressions."
)


def parse_condition(condition):
    """Parse a condition string into a small structure, or None.

    Returns one of:
      {"kind": "single", "name": ...}
      {"kind": "and", "a": ..., "b": ..., "negated": bool}
      {"kind": "or", "a": ..., "b": ...}
      {"kind": "count", "name": ..., "op": ">=" or ">", "n": int}
    """
    if not isinstance(condition, str):
        return None
    text = condition.strip()
    match = _COUNT_RE.match(text)
    if match:
        return {"kind": "count", "name": match.group(1),
                "op": match.group(2), "n": int(match.group(3))}
    match = _AND_NOT_RE.match(text)
    if match:
        return {"kind": "and", "a": match.group(1), "b": match.group(2),
                "negated": True}
    match = _AND_RE.match(text)
    if match:
        return {"kind": "and", "a": match.group(1), "b": match.group(2),
                "negated": False}
    match = _OR_RE.match(text)
    if match:
        return {"kind": "or", "a": match.group(1), "b": match.group(2)}
    match = _SINGLE_RE.match(text)
    if match:
        return {"kind": "single", "name": match.group(1)}
    return None


def condition_names(parsed):
    """Selection names a parsed condition refers to."""
    if parsed["kind"] == "single" or parsed["kind"] == "count":
        return [parsed["name"]]
    return [parsed["a"], parsed["b"]]


def validate_rule(rule):
    """Structural validation. Returns (ok, errors list)."""
    errors = []
    if not isinstance(rule, dict):
        return False, ["rule must be a JSON object"]
    title = rule.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append("title must be a non-empty string")
    logsource = rule.get("logsource")
    if not isinstance(logsource, dict):
        errors.append("logsource must be an object")
    else:
        for field in ("category", "product"):
            value = logsource.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append("logsource.{0} must be a non-empty string"
                              .format(field))
        service = logsource.get("service")
        if service is not None and (
                not isinstance(service, str) or not service.strip()):
            errors.append("logsource.service must be a non-empty string "
                          "when present")
    detection = rule.get("detection")
    selections = {}
    if not isinstance(detection, dict):
        errors.append("detection must be an object")
    else:
        condition = detection.get("condition")
        if not isinstance(condition, str) or not condition.strip():
            errors.append("detection.condition must be a non-empty string")
        for name, selection in detection.items():
            if name == "condition":
                continue
            selections[name] = selection
            if not isinstance(selection, dict) or not selection:
                errors.append("detection.{0} must be a non-empty object"
                              .format(name))
                continue
            for field, values in selection.items():
                if field not in KNOWN_FIELDS:
                    errors.append(
                        "detection.{0} uses unknown field {1!r} "
                        "(known fields: {2})".format(
                            name, field, ", ".join(KNOWN_FIELDS)))
                    continue
                if isinstance(values, str):
                    values = [values]
                if (not isinstance(values, list) or not values
                        or any(not isinstance(v, str) or not v
                               for v in values)):
                    errors.append(
                        "detection.{0}.{1} must be a non-empty string or "
                        "a non-empty list of non-empty strings".format(
                            name, field))
        if not selections:
            errors.append("detection must define at least one selection")
        if isinstance(condition, str) and condition.strip():
            parsed = parse_condition(condition)
            if parsed is None:
                errors.append(
                    "detection.condition does not parse: {0!r}".format(
                        condition))
            else:
                for name in condition_names(parsed):
                    if name not in selections:
                        errors.append(
                            "detection.condition refers to unknown "
                            "selection {0!r}".format(name))
    level = rule.get("level")
    if level not in LEVELS:
        errors.append("level must be one of: {0}".format(", ".join(LEVELS)))
    description = rule.get("description")
    if description is not None and not isinstance(description, str):
        errors.append("description must be a string when present")
    return (not errors), errors


def _yaml_str(value):
    """Render a string as a YAML scalar (JSON double-quoted is valid YAML)."""
    return json.dumps(value, ensure_ascii=False)


def emit_yaml(rule):
    """Render a rule dict as Sigma-style YAML text.

    Raises ValueError on an invalid rule, so callers validate first.
    """
    ok, errors = validate_rule(rule)
    if not ok:
        raise ValueError("cannot emit YAML for invalid rule: "
                         + "; ".join(errors))
    lines = ["title: " + _yaml_str(rule["title"])]
    logsource = rule["logsource"]
    lines.append("logsource:")
    lines.append("  category: " + _yaml_str(logsource["category"]))
    lines.append("  product: " + _yaml_str(logsource["product"]))
    if logsource.get("service"):
        lines.append("  service: " + _yaml_str(logsource["service"]))
    lines.append("detection:")
    for name, selection in rule["detection"].items():
        if name == "condition":
            continue
        lines.append("  {0}:".format(name))
        for field, values in selection.items():
            if isinstance(values, str):
                values = [values]
            lines.append("    {0}:".format(field))
            for value in values:
                lines.append("      - " + _yaml_str(value))
    lines.append("  condition: "
                 + _yaml_str(rule["detection"]["condition"]))
    lines.append("level: " + _yaml_str(rule["level"]))
    if rule.get("description"):
        lines.append("description: " + _yaml_str(rule["description"]))
    return "\n".join(lines) + "\n"


def rule_to_json(rule):
    """Render a rule dict as indented JSON text."""
    return json.dumps(rule, indent=2, ensure_ascii=False) + "\n"
