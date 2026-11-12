"""RuleForge step 1: the research loop for detection rules.

You give it a detection goal in plain words; it invents candidate
rules, tests them against labeled logs, scores them, and hands you
the best one with a report. One loop, up to 3 rounds by default:

    goal in -> propose a rule -> validate (hard gate) -> score on the
    train split -> reviewer critiques -> next round applies it ->
    keep the best -> final score on the HELD-OUT test split -> report

The test split is never shown to the proposer, so the final score is
an honest check, not the model grading its own homework.

Safety rule, by design: this agent only PROPOSES. It reads log
files and writes rule files inside its own workspace directory. It
never deploys a rule, never touches a live system, and a human must
approve before any rule goes live.

Standard library only. No network except the optional Ollama
localhost client (step 2); step 1 runs on the mock client, offline.

Usage:
    python3 ruleforge.py --goal "catch SSH brute force with fewer false alarms"
    python3 ruleforge.py --goal "..." --model ollama --rounds 3
    python3 ruleforge.py --goal "..." --data-dir data --output-dir output

Exit code is 0 on a completed run, 2 on missing data or bad arguments.
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agents
import rules
import scorer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "data")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "output")


def load_split(data_dir, split):
    """Load one labeled split: (attack records, benign records)."""
    attack_path = os.path.join(data_dir, split, "attack.log")
    benign_path = os.path.join(data_dir, split, "benign.log")
    for path in (attack_path, benign_path):
        if not os.path.exists(path):
            raise FileNotFoundError("missing labeled log file: {0}".format(
                path))
    return (scorer.read_log_file(attack_path),
            scorer.read_log_file(benign_path))


def build_proposer_prompt(goal, round_no, current_rule, reviewer_note):
    """The proposer prompt: goal, round, previous rule, reviewer note."""
    parts = [
        agents.PROPOSER_SYSTEM,
        "",
        "ROLE: proposer",
        "ROUND: {0}".format(round_no),
        "Goal: {0}".format(goal),
    ]
    if current_rule is not None:
        parts.append("Current rule (improve on this):")
        parts.append(rules.rule_to_json(current_rule))
    if reviewer_note:
        parts.append("Reviewer note from the last round:")
        parts.append(reviewer_note)
    return "\n".join(parts)


def build_reviewer_prompt(goal, round_no, rule, train_result):
    """The reviewer prompt: goal, round, rule, and its train results."""
    return "\n".join([
        agents.REVIEWER_SYSTEM,
        "",
        "ROLE: reviewer",
        "ROUND: {0}".format(round_no),
        "Goal: {0}".format(goal),
        "Proposed rule:",
        rules.rule_to_json(rule),
        "Train results: TP={tp}, FP={fp}, FN={fn}, TN={tn}".format(
            tp=train_result["tp"], fp=train_result["fp"],
            fn=train_result["fn"], tn=train_result["tn"]),
        "Train score: {0:.4f} (catch rate {1:.4f}, false-positive "
        "rate {2:.4f})".format(train_result["score"],
                               train_result["catch_rate"],
                               train_result["fp_rate"]),
    ])


def run_loop(goal, client, rounds, train, test):
    """Run the research loop. Returns a report dict."""
    train_attack, train_benign = train
    test_attack, test_benign = test
    history = []
    best = None
    current_rule = None
    reviewer_note = ""
    last_verdict = ""
    for round_no in range(1, rounds + 1):
        prompt = build_proposer_prompt(goal, round_no, current_rule,
                                       reviewer_note)
        reply = client.complete(prompt)
        rule = agents.parse_rule_reply(reply)
        errors = []
        if rule is None:
            errors = ["could not parse a rule object from the model reply"]
        else:
            ok, errors = rules.validate_rule(rule)
            if not ok:
                rule = None
        if rule is None:
            result = scorer.zero_result()
            reviewer_note = ("The proposed rule was invalid: "
                             + "; ".join(errors) + ". "
                             + rules.CONDITION_HELP
                             + " Propose a valid rule.")
            history.append({"round": round_no, "rule": None,
                            "errors": errors, "result": result,
                            "verdict": "", "reply": reply})
            continue
        result = scorer.score_rule(rule, train_attack, train_benign)
        if best is None or result["score"] > best["result"]["score"]:
            best = {"round": round_no, "rule": rule, "result": result}
        current_rule = rule
        verdict = client.complete(build_reviewer_prompt(
            goal, round_no, rule, result))
        last_verdict = verdict
        reviewer_note = verdict
        history.append({"round": round_no, "rule": rule,
                        "errors": [], "result": result,
                        "verdict": verdict, "reply": reply})
    if best is not None:
        test_result = scorer.score_rule(best["rule"],
                                        test_attack, test_benign)
    else:
        test_result = scorer.zero_result()
    return {"goal": goal, "rounds": rounds, "history": history,
            "best": best, "test_result": test_result,
            "verdict": last_verdict}


def _summarize_lines(records):
    """One line per event kind: '12 failed-password attempts'."""
    counts = Counter(record["event"] for record in records)
    labels = {"failed_password": "failed-password attempts",
              "failed_publickey": "failed-publickey probes",
              "accepted": "successful logins",
              "other": "other lines"}
    return ", ".join("{0} {1}".format(counts[event], labels[event])
                     for event in sorted(counts))


def format_report(report):
    """Render the loop result as a plain-text report."""
    lines = [
        "RuleForge - research report",
        "Goal: " + report["goal"],
        "Rounds: {0}".format(report["rounds"]),
        "",
    ]
    for entry in report["history"]:
        result = entry["result"]
        if entry["rule"] is None:
            lines.append(
                "Round {0}: INVALID RULE - {1} (score 0.0000)".format(
                    entry["round"], "; ".join(entry["errors"])))
        else:
            lines.append(
                "Round {0}: {1} - train score {2:.4f} "
                "(catch {3:.4f}, false-positive {4:.4f})".format(
                    entry["round"], entry["rule"]["title"],
                    result["score"], result["catch_rate"],
                    result["fp_rate"]))
    lines.append("")
    best = report["best"]
    test_result = report["test_result"]
    if best is None:
        lines.append("No valid rule was proposed in any round.")
        lines.append("Nothing to score on the held-out test split.")
        return "\n".join(lines)
    rule = best["rule"]
    lines.append("Best rule: " + rule["title"])
    lines.append("  proposed in round {0}".format(best["round"]))
    lines.append("  level: " + rule["level"])
    lines.append("  condition: " + rule["detection"]["condition"])
    lines.append("  train score: {0:.4f}".format(best["result"]["score"]))
    lines.append("  test score: {0:.4f} (catch rate {1:.4f}, "
                 "false-positive rate {2:.4f})".format(
                     test_result["score"], test_result["catch_rate"],
                     test_result["fp_rate"]))
    lines.append("  test counts: TP {tp}, FP {fp}, FN {fn}, "
                 "TN {tn}".format(tp=test_result["tp"],
                                  fp=test_result["fp"],
                                  fn=test_result["fn"],
                                  tn=test_result["tn"]))
    lines.append("  precision: {0:.4f}  recall: {1:.4f}  "
                 "F1: {2:.4f}".format(test_result["precision"],
                                      test_result["recall"],
                                      test_result["f1"]))
    lines.append("")
    lines.append("What it catches on the held-out test split: "
                 "{0} of {1} attack lines ({2}).".format(
                     test_result["tp"],
                     test_result["tp"] + test_result["fn"],
                     _summarize_lines(test_result["matched_attack"])))
    if test_result["missed_attack"]:
        lines.append("What it misses:")
        for record in test_result["missed_attack"][:5]:
            lines.append("  - " + record["raw"])
        if len(test_result["missed_attack"]) > 5:
            lines.append("  - ... and {0} more".format(
                len(test_result["missed_attack"]) - 5))
    else:
        lines.append("What it misses: nothing.")
    if test_result["matched_benign"]:
        lines.append("False positives on the test split:")
        for record in test_result["matched_benign"][:5]:
            lines.append("  - " + record["raw"])
    else:
        lines.append("False positives on the test split: none.")
    lines.append("")
    lines.append("Reviewer's verdict: " + report["verdict"])
    lines.append("")
    lines.append("A human must approve before any rule goes live. "
                 "This agent only proposes.")
    return "\n".join(lines)


def save_report(report, output_dir):
    """Write the plain-text report inside the workspace.

    Step 3: the report used to be printed to stdout only, so a run's
    reasoning vanished once the terminal scrolled. It is now always
    saved as report.txt next to best_rule.json/yml.
    Returns the written path.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(format_report(report) + "\n")
    return report_path


def save_to_repo(report, repo_dir):
    """Write the best rule + report into a repo's sigma/ folder.

    Step 3, save-to-repo option. Only runs when the user passes
    --save-to-repo on the command line -- never by default, and a
    human must still approve before any rule goes live (the files
    are proposed rules in a repo folder, not a deployment).

    Writes <repo_dir>/sigma/ruleforge-ssh-brute-force.yml and
    <repo_dir>/sigma/ruleforge-report.txt. Refuses to overwrite an
    existing file, so a previous RuleForge save can never be
    silently replaced; delete or rename it first if you mean to
    replace it. Raises FileNotFoundError when repo_dir is not a
    directory, FileExistsError when a target file already exists,
    and returns [] when no valid rule was proposed.
    """
    if report["best"] is None:
        return []
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(
            "save-to-repo target is not a directory: {0}".format(repo_dir))
    sigma_dir = os.path.join(repo_dir, "sigma")
    os.makedirs(sigma_dir, exist_ok=True)
    rule = report["best"]["rule"]
    test_result = report["test_result"]
    header = (
        "# RuleForge proposal - NOT live until a human approves it.\n"
        "# Goal: {0}\n"
        "# Held-out test score: {1:.4f} (TP {2}, FP {3}, FN {4}, "
        "TN {5}); see ruleforge-report.txt in this folder.\n"
        "# Honest limit: the count is per log file, not per time\n"
        "# window, unlike CyberBeep's 5-in-300-seconds rule.\n".format(
            report["goal"], test_result["score"], test_result["tp"],
            test_result["fp"], test_result["fn"], test_result["tn"]))
    yaml_path = os.path.join(sigma_dir, "ruleforge-ssh-brute-force.yml")
    report_path = os.path.join(sigma_dir, "ruleforge-report.txt")
    for path in (yaml_path, report_path):
        if os.path.exists(path):
            raise FileExistsError(
                "refusing to overwrite existing file: {0}".format(path))
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(header + rules.emit_yaml(rule))
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(format_report(report) + "\n")
    return [yaml_path, report_path]


def save_best_rule(report, output_dir):
    """Write the best rule as JSON and YAML inside the workspace.

    Returns the written paths, or [] when no valid rule exists.
    """
    if report["best"] is None:
        return []
    os.makedirs(output_dir, exist_ok=True)
    rule = report["best"]["rule"]
    json_path = os.path.join(output_dir, "best_rule.json")
    yaml_path = os.path.join(output_dir, "best_rule.yml")
    with open(json_path, "w", encoding="utf-8") as handle:
        handle.write(rules.rule_to_json(rule))
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(rules.emit_yaml(rule))
    return [json_path, yaml_path]


def save_replies(report, output_dir):
    """Write each round's raw model reply inside the workspace.

    This is how a live-model failure stays diagnosable: the reply that
    produced an invalid rule is kept verbatim instead of vanishing.
    Returns the written paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for entry in report["history"]:
        reply_path = os.path.join(
            output_dir, "reply_round_{0}.txt".format(entry["round"]))
        with open(reply_path, "w", encoding="utf-8") as handle:
            handle.write(entry.get("reply") or "")
        paths.append(reply_path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="RuleForge step 1: research a detection rule.")
    parser.add_argument("--goal", required=True,
                        help="detection goal in plain words")
    parser.add_argument("--model", default="mock",
                        choices=["mock", "ollama"],
                        help="model backend (default: mock, offline)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="research rounds, 1-5 (default 3)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="labeled log directory (default: data/)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="where to write the best rule "
                             "(default: output/)")
    parser.add_argument("--save-to-repo", default=None, metavar="DIR",
                        help="also write the best rule + report into "
                             "DIR/sigma/ (step 3, only when given; "
                             "refuses to overwrite existing files)")
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.rounds > 5:
        parser.error("--rounds must be between 1 and 5")
    try:
        train = load_split(args.data_dir, "train")
        test = load_split(args.data_dir, "test")
    except FileNotFoundError as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 2
    client = agents.OllamaClient() if args.model == "ollama" \
        else agents.MockClient()
    try:
        report = run_loop(args.goal, client, args.rounds, train, test)
    except RuntimeError as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 2
    print(format_report(report))
    saved = save_best_rule(report, args.output_dir)
    saved = saved + [save_report(report, args.output_dir)]
    saved = saved + save_replies(report, args.output_dir)
    if args.save_to_repo is not None:
        try:
            saved = saved + save_to_repo(report, args.save_to_repo)
        except (FileNotFoundError, FileExistsError) as exc:
            print("error: {0}".format(exc), file=sys.stderr)
            return 2
    if saved:
        print("")
        print("Saved:")
        for path in saved:
            print("  " + path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
