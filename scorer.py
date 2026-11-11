"""RuleForge step 1: log parsing, rule matching, and scoring.

A log line is parsed into a small record:

    {"raw": ..., "timestamp": ..., "message": ...,
     "src_ip": ..., "user": ..., "event": ...}

"event" is one of failed_password, failed_publickey, accepted, or
other. Parsing mirrors the sshd line shapes CyberBeep reads
("Failed password for invalid user admin from 203.0.113.45 ...",
"Accepted password for deploy from 203.0.113.45 ...").

Scoring is per log line: given a rule and two labeled sets (attack
lines, benign lines) it counts TP (attack lines matched), FP
(benign lines matched), FN, TN. The primary score is one number:

    score = attack catch rate - benign false-positive rate
          = TP / (TP + FN) - FP / (FP + TN)          (range -1..1)

Precision, recall, and F1 are reported alongside it.

Count conditions ("selection | count() by src_ip >= 5") are
evaluated per log file: a line matches only if its source IP has at
least that many selection-matching lines in the same file. That is
the honest step-1 reading -- the train and test files are separate
logs, so the count is scoped to each file. A merged chronological
log with per-line labels is a step-2 item.

Standard library only. No network, no shell execution.
"""

import re

from rules import parse_condition

_TS_RE = re.compile(r"^(\w{3})\s+(\d+) (\d{2}):(\d{2}):(\d{2})")
_PREFIX_RE = re.compile(r"^\S+ \S+(?:\[\d+\])?: ")
_SRC_IP_RE = re.compile(r"from (\d+\.\d+\.\d+\.\d+)")
_USER_RE = re.compile(r"for (?:invalid user )?(\S+) from")


def parse_line(line):
    """Parse one log line into a record, or None for a blank line."""
    if line is None:
        return None
    raw = line.rstrip("\n")
    if not raw.strip():
        return None
    timestamp = None
    match = _TS_RE.match(raw)
    if match:
        timestamp = match.group(0)
    rest = raw
    if match:
        rest = raw[match.end():].strip()
    message = _PREFIX_RE.sub("", rest, count=1)
    src_ip = None
    ip_match = _SRC_IP_RE.search(message)
    if ip_match:
        src_ip = ip_match.group(1)
    user = None
    user_match = _USER_RE.search(message)
    if user_match:
        user = user_match.group(1)
    if "Failed password" in message:
        event = "failed_password"
    elif "Failed publickey" in message:
        event = "failed_publickey"
    elif "Accepted" in message:
        event = "accepted"
    else:
        event = "other"
    return {"raw": raw, "timestamp": timestamp, "message": message,
            "src_ip": src_ip, "user": user, "event": event}


def parse_lines(lines):
    """Parse an iterable of lines into records, skipping blanks."""
    records = []
    for line in lines:
        record = parse_line(line)
        if record is not None:
            records.append(record)
    return records


def read_log_file(path):
    """Read a log file into parsed records."""
    with open(path, "r", encoding="utf-8") as handle:
        return parse_lines(handle)


def selection_matches(selection, record):
    """True when every field in the selection matches the record.

    A field matches when the record's value for it contains at least
    one of the listed substrings. A missing value never matches.
    """
    for field, values in selection.items():
        if isinstance(values, str):
            values = [values]
        value = record.get(field)
        if value is None:
            return False
        if not any(sub in value for sub in values):
            return False
    return True


def evaluate(rule, records):
    """Match a rule against a list of records; returns a list of bools.

    Count conditions are two-pass: first count selection matches per
    source IP across the whole list, then apply the threshold.
    """
    parsed = parse_condition(rule["detection"]["condition"])
    if parsed is None:
        raise ValueError("rule condition does not parse: {0!r}".format(
            rule["detection"]["condition"]))
    detection = rule["detection"]
    if parsed["kind"] == "count":
        base = [selection_matches(detection[parsed["name"]], record)
                for record in records]
        counts = {}
        for record, matched in zip(records, base):
            if matched and record.get("src_ip"):
                counts[record["src_ip"]] = counts.get(
                    record["src_ip"], 0) + 1
        threshold = parsed["n"]
        if parsed["op"] == ">":
            return [matched and counts.get(record.get("src_ip"), 0) > threshold
                    for record, matched in zip(records, base)]
        return [matched and counts.get(record.get("src_ip"), 0) >= threshold
                for record, matched in zip(records, base)]
    if parsed["kind"] == "single":
        selection = detection[parsed["name"]]
        return [selection_matches(selection, record) for record in records]
    left = [selection_matches(detection[parsed["a"]], record)
            for record in records]
    right = [selection_matches(detection[parsed["b"]], record)
             for record in records]
    if parsed["kind"] == "and":
        if parsed["negated"]:
            return [a and not b for a, b in zip(left, right)]
        return [a and b for a, b in zip(left, right)]
    return [a or b for a, b in zip(left, right)]


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def score_rule(rule, attack_records, benign_records):
    """Score a rule against labeled attack and benign records.

    Returns counts, rates, the primary score, precision/recall/F1,
    and the matched/missed record lists the report needs.
    """
    attack_hits = evaluate(rule, attack_records)
    benign_hits = evaluate(rule, benign_records)
    tp = sum(1 for hit in attack_hits if hit)
    fn = len(attack_hits) - tp
    fp = sum(1 for hit in benign_hits if hit)
    tn = len(benign_hits) - fp
    catch_rate = _safe_div(tp, tp + fn)
    fp_rate = _safe_div(fp, fp + tn)
    score = catch_rate - fp_rate
    precision = _safe_div(tp, tp + fp)
    recall = catch_rate
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "catch_rate": catch_rate, "fp_rate": fp_rate,
        "score": score, "precision": precision, "recall": recall,
        "f1": f1,
        "matched_attack": [record for record, hit in
                           zip(attack_records, attack_hits) if hit],
        "missed_attack": [record for record, hit in
                          zip(attack_records, attack_hits) if not hit],
        "matched_benign": [record for record, hit in
                           zip(benign_records, benign_hits) if hit],
    }


def zero_result():
    """The result of a round that produced no valid rule: score 0."""
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "catch_rate": 0.0, "fp_rate": 0.0,
            "score": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "matched_attack": [], "missed_attack": [],
            "matched_benign": []}
