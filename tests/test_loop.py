"""Tests for RuleForge step 1: the loop and the CLI.

Standard library unittest only. The end-to-end test calls main()
directly and captures stdout -- no subprocess, no network.
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents
import ruleforge
import rules

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")


class GarbageClient:
    """A model that never returns a usable rule."""

    def complete(self, prompt):
        return "hello, I am a language model and I have no rule for you"


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.train = ruleforge.load_split(DATA_DIR, "train")
        self.test = ruleforge.load_split(DATA_DIR, "test")

    def test_mock_loop_improves_across_rounds(self):
        report = ruleforge.run_loop(
            "catch SSH brute force", agents.MockClient(), 3,
            self.train, self.test)
        scores = [entry["result"]["score"] for entry in report["history"]]
        self.assertEqual(len(scores), 3)
        self.assertAlmostEqual(scores[0], 0.875 - 0.15)  # 0.725
        self.assertAlmostEqual(scores[1], 0.875)
        self.assertAlmostEqual(scores[2], 0.9375)
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])

    def test_mock_loop_keeps_the_best_rule(self):
        report = ruleforge.run_loop(
            "catch SSH brute force", agents.MockClient(), 3,
            self.train, self.test)
        best = report["best"]
        self.assertEqual(best["round"], 3)
        self.assertIn("Failed publickey",
                      best["rule"]["detection"]["selection"]["message"])
        self.assertIn("count()", best["rule"]["detection"]["condition"])

    def test_best_rule_scores_honestly_on_held_out_test(self):
        report = ruleforge.run_loop(
            "catch SSH brute force", agents.MockClient(), 3,
            self.train, self.test)
        test_result = report["test_result"]
        self.assertEqual((test_result["tp"], test_result["fp"],
                          test_result["fn"], test_result["tn"]),
                         (13, 0, 1, 18))
        self.assertAlmostEqual(test_result["score"], 13 / 14)
        self.assertAlmostEqual(test_result["precision"], 1.0)
        self.assertAlmostEqual(test_result["f1"], 26 / 27)

    def test_invalid_proposals_score_zero_and_loop_completes(self):
        report = ruleforge.run_loop(
            "catch SSH brute force", GarbageClient(), 3,
            self.train, self.test)
        self.assertIsNone(report["best"])
        for entry in report["history"]:
            self.assertIsNone(entry["rule"])
            self.assertEqual(entry["result"]["score"], 0.0)
            self.assertTrue(entry["errors"])
        self.assertIn("No valid rule", ruleforge.format_report(report))

    def test_invalid_round_feeds_the_grammar_back(self):
        # A model that first proposes a windowed condition (the live
        # run-3 failure) must see the allowed grammar in the next
        # round's prompt, and then be able to recover.
        class WindowClient:
            def __init__(self):
                self.prompts = []

            def complete(self, prompt):
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    return ("```json\n" + json.dumps({
                        "title": "x",
                        "logsource": {"category": "auth",
                                      "product": "linux",
                                      "service": "sshd"},
                        "detection": {
                            "selection": {"message": ["Failed password"]},
                            "condition": "selection | count() by "
                                         "src_ip over 1m >= 10"},
                        "level": "high",
                        "description": "x",
                    }) + "\n```")
                return agents.MockClient().complete(prompt)

        self.assertIn("No time windows", agents.PROPOSER_SYSTEM)
        client = WindowClient()
        report = ruleforge.run_loop(
            "catch SSH brute force", client, 2, self.train, self.test)
        self.assertIsNone(report["history"][0]["rule"])
        self.assertIn("No time windows", client.prompts[1])
        self.assertIn("between two count() expressions",
                      client.prompts[1])
        self.assertIsNotNone(report["best"])

    def test_one_round_loop(self):
        report = ruleforge.run_loop(
            "catch SSH brute force", agents.MockClient(), 1,
            self.train, self.test)
        self.assertEqual(len(report["history"]), 1)
        self.assertEqual(report["best"]["round"], 1)


class ParseReplyTests(unittest.TestCase):
    def test_fenced_json_parses(self):
        text = agents.MockClient().complete(
            "ROLE: proposer\nROUND: 1\nGoal: test")
        rule = agents.parse_rule_reply(text)
        self.assertIsInstance(rule, dict)
        self.assertEqual(rule["level"], "high")

    def test_garbage_returns_none(self):
        self.assertIsNone(agents.parse_rule_reply("no json here"))
        self.assertIsNone(agents.parse_rule_reply(""))
        self.assertIsNone(agents.parse_rule_reply(None))

    def test_bare_json_parses(self):
        rule = agents.parse_rule_reply('{"title": "x", "level": "low"}')
        self.assertEqual(rule["title"], "x")


class CliTests(unittest.TestCase):
    def test_mock_end_to_end_exits_zero_and_saves_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = ruleforge.main([
                    "--goal", "catch SSH brute force",
                    "--output-dir", tmp,
                ])
            self.assertEqual(code, 0)
            text = out.getvalue()
            self.assertIn("test score: 0.9286", text)
            self.assertIn("Reviewer's verdict", text)
            self.assertIn("A human must approve", text)
            json_path = os.path.join(tmp, "best_rule.json")
            yaml_path = os.path.join(tmp, "best_rule.yml")
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(yaml_path))
            with open(json_path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            ok, errors = rules.validate_rule(saved)
            self.assertTrue(ok, errors)
            with open(yaml_path, "r", encoding="utf-8") as handle:
                self.assertIn("count() by src_ip >= 5",
                              handle.read())

    def test_proposer_prompt_example_is_a_valid_rule(self):
        # The few-shot example in the proposer prompt must itself pass
        # validation: a bad example teaches the model a bad shape.
        example = agents.parse_rule_reply(agents.PROPOSER_SYSTEM)
        self.assertIsNotNone(example)
        ok, errors = rules.validate_rule(example)
        self.assertTrue(ok, errors)

    def test_reviewer_prompt_reads_the_score(self):
        # Run 4's live reviewer called a 0-catch rule "good" and
        # suggested raising the threshold -- backwards. The reviewer
        # prompt must now state the score-reading rules explicitly.
        prompt = agents.REVIEWER_SYSTEM
        self.assertIn("catch rate is 0", prompt)
        self.assertIn("LOWERING", prompt)
        self.assertIn("never call it good", prompt)
        self.assertIn("never invent numbers", prompt)
        self.assertIn("false-positive rate is high", prompt)

    def test_replies_are_saved_per_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = ruleforge.main([
                    "--goal", "catch SSH brute force",
                    "--output-dir", tmp,
                ])
            self.assertEqual(code, 0)
            for round_no in (1, 2, 3):
                path = os.path.join(
                    tmp, "reply_round_{0}.txt".format(round_no))
                self.assertTrue(os.path.exists(path))
                with open(path, "r", encoding="utf-8") as handle:
                    self.assertIn("```json", handle.read())

    def test_report_is_saved_to_output_dir(self):
        # Step 3: the report used to exist only on stdout.
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = ruleforge.main([
                    "--goal", "catch SSH brute force",
                    "--output-dir", tmp,
                ])
            self.assertEqual(code, 0)
            report_path = os.path.join(tmp, "report.txt")
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r", encoding="utf-8") as handle:
                saved = handle.read()
            self.assertIn("test score: 0.9286", saved)
            self.assertIn("A human must approve", saved)

    def test_save_to_repo_writes_sigma_files(self):
        # Step 3: --save-to-repo writes the rule + report under
        # the target repo's sigma/ folder, and nothing without it.
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repo")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(repo)
            out = io.StringIO()
            with redirect_stdout(out):
                code = ruleforge.main([
                    "--goal", "catch SSH brute force",
                    "--output-dir", out_dir,
                    "--save-to-repo", repo,
                ])
            self.assertEqual(code, 0)
            yaml_path = os.path.join(
                repo, "sigma", "ruleforge-ssh-brute-force.yml")
            report_path = os.path.join(
                repo, "sigma", "ruleforge-report.txt")
            self.assertTrue(os.path.exists(yaml_path))
            self.assertTrue(os.path.exists(report_path))
            with open(yaml_path, "r", encoding="utf-8") as handle:
                yaml_text = handle.read()
            self.assertIn("count() by src_ip >= 5", yaml_text)
            self.assertIn("NOT live until a human approves", yaml_text)
            self.assertIn("per log file, not per time", yaml_text)
            with open(report_path, "r", encoding="utf-8") as handle:
                self.assertIn("test score: 0.9286", handle.read())
            # Without the flag, no sigma/ folder is created.
            out2 = io.StringIO()
            with redirect_stdout(out2):
                code2 = ruleforge.main([
                    "--goal", "catch SSH brute force",
                    "--output-dir", os.path.join(tmp, "out2"),
                ])
            self.assertEqual(code2, 0)
            self.assertFalse(os.path.exists(
                os.path.join(tmp, "out2", "sigma")))

    def test_save_to_repo_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repo")
            os.makedirs(repo)
            out = io.StringIO()
            with redirect_stdout(out):
                code = ruleforge.main([
                    "--goal", "catch SSH brute force",
                    "--output-dir", os.path.join(tmp, "out"),
                    "--save-to-repo", repo,
                ])
            self.assertEqual(code, 0)
            yaml_path = os.path.join(
                repo, "sigma", "ruleforge-ssh-brute-force.yml")
            with open(yaml_path, "r", encoding="utf-8") as handle:
                before = handle.read()
            out2 = io.StringIO()
            with redirect_stdout(out2):
                code2 = ruleforge.main([
                    "--goal", "catch SSH brute force",
                    "--output-dir", os.path.join(tmp, "out2"),
                    "--save-to-repo", repo,
                ])
            self.assertEqual(code2, 2)
            with open(yaml_path, "r", encoding="utf-8") as handle:
                self.assertEqual(before, handle.read())

    def test_save_to_repo_missing_dir_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = ruleforge.main([
                "--goal", "catch SSH brute force",
                "--output-dir", os.path.join(tmp, "out"),
                "--save-to-repo", os.path.join(tmp, "no-such-repo"),
            ])
        self.assertEqual(code, 2)

    def test_missing_data_dir_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = ruleforge.main(["--goal", "x", "--data-dir", tmp])
        self.assertEqual(code, 2)

    def test_bad_rounds_exits_two(self):
        with self.assertRaises(SystemExit) as caught:
            ruleforge.main(["--goal", "x", "--rounds", "0"])
        self.assertEqual(caught.exception.code, 2)


class SafetyTests(unittest.TestCase):
    def test_no_subprocess_anywhere(self):
        # The agent must never gain a way to run shell commands.
        for name in ("ruleforge.py", "scorer.py", "rules.py", "agents.py"):
            path = os.path.join(BASE, name)
            with open(path, "r", encoding="utf-8") as handle:
                source = handle.read()
            self.assertNotIn("subprocess", source, name)
            self.assertNotIn("os.system", source, name)

    def test_network_is_localhost_ollama_only(self):
        path = os.path.join(BASE, "agents.py")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("localhost:11434", source)
        for banned in ("api.telegram.org", "https://api.", "http://api."):
            self.assertNotIn(banned, source)


if __name__ == "__main__":
    unittest.main()
