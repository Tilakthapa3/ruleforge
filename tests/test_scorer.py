"""Tests for RuleForge step 1: log parsing, matching, scoring."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rules
import scorer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOOD_RULE = {
    "title": "SSH Brute Force",
    "logsource": {"category": "auth", "product": "linux",
                  "service": "sshd"},
    "detection": {
        "selection": {"message": ["Failed password"]},
        "condition": "selection",
    },
    "level": "high",
}

# Hand-computed case: 3 attack lines (2 failed, 1 accepted) and
# 2 benign lines (1 failed typo, 1 accepted). The plain "Failed
# password" rule gives TP=2, FN=1, FP=1, TN=1.
ATTACK_LINES = [
    "Oct 30 02:00:01 server sshd[1]: Failed password for root "
    "from 10.1.1.1 port 1111 ssh2",
    "Oct 30 02:00:02 server sshd[2]: Failed password for admin "
    "from 10.1.1.1 port 1112 ssh2",
    "Oct 30 02:00:03 server sshd[3]: Accepted password for deploy "
    "from 10.1.1.1 port 1113 ssh2",
]
BENIGN_LINES = [
    "Oct 30 08:00:01 server sshd[4]: Failed password for alice "
    "from 10.2.2.2 port 2222 ssh2",
    "Oct 30 08:00:02 server sshd[5]: Accepted password for alice "
    "from 10.2.2.2 port 2223 ssh2",
]


def count_rule(op, threshold):
    rule = dict(GOOD_RULE)
    rule["detection"] = {
        "selection": {"message": ["Failed password"]},
        "condition": "selection | count() by src_ip {0} {1}".format(
            op, threshold),
    }
    return rule


class ParseTests(unittest.TestCase):
    def test_failed_password_line(self):
        record = scorer.parse_line(ATTACK_LINES[0])
        self.assertEqual(record["src_ip"], "10.1.1.1")
        self.assertEqual(record["user"], "root")
        self.assertEqual(record["event"], "failed_password")
        self.assertIn("Failed password", record["message"])

    def test_invalid_user_is_stripped_from_username(self):
        line = ("Oct 30 02:00:01 server sshd[1]: Failed password for "
                "invalid user admin from 10.1.1.1 port 1111 ssh2")
        self.assertEqual(scorer.parse_line(line)["user"], "admin")

    def test_accepted_line(self):
        record = scorer.parse_line(ATTACK_LINES[2])
        self.assertEqual(record["event"], "accepted")
        self.assertEqual(record["user"], "deploy")

    def test_publickey_line(self):
        line = ("Oct 30 02:00:01 server sshd[1]: Failed publickey for "
                "invalid user git from 10.1.1.1 port 1111 ssh2")
        self.assertEqual(scorer.parse_line(line)["event"],
                         "failed_publickey")

    def test_non_sshd_line_is_other(self):
        line = "Oct 30 12:35:51 server CRON[400]: (root) CMD (run-parts)"
        record = scorer.parse_line(line)
        self.assertEqual(record["event"], "other")
        self.assertIsNone(record["src_ip"])

    def test_blank_line_returns_none(self):
        self.assertIsNone(scorer.parse_line(""))
        self.assertIsNone(scorer.parse_line("   \n"))


class SelectionTests(unittest.TestCase):
    def test_substring_match(self):
        record = scorer.parse_line(ATTACK_LINES[0])
        self.assertTrue(scorer.selection_matches(
            {"message": ["Failed password"]}, record))
        self.assertFalse(scorer.selection_matches(
            {"message": ["Accepted"]}, record))

    def test_or_within_a_field_and_across_fields(self):
        record = scorer.parse_line(ATTACK_LINES[0])
        self.assertTrue(scorer.selection_matches(
            {"message": ["Accepted", "Failed password"]}, record))
        self.assertTrue(scorer.selection_matches(
            {"message": ["Failed password"], "src_ip": ["10.1.1.1"]},
            record))
        self.assertFalse(scorer.selection_matches(
            {"message": ["Failed password"], "src_ip": ["10.9.9.9"]},
            record))

    def test_missing_value_never_matches(self):
        record = scorer.parse_line(
            "Oct 30 12:35:51 server CRON[400]: (root) CMD (run-parts)")
        self.assertFalse(scorer.selection_matches(
            {"src_ip": ["10.1.1.1"]}, record))


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.attack = scorer.parse_lines(ATTACK_LINES)
        self.benign = scorer.parse_lines(BENIGN_LINES)

    def test_single_selection(self):
        self.assertEqual(scorer.evaluate(GOOD_RULE, self.attack),
                         [True, True, False])
        self.assertEqual(scorer.evaluate(GOOD_RULE, self.benign),
                         [True, False])

    def test_and_and_and_not(self):
        rule = dict(GOOD_RULE)
        rule["detection"] = {
            "selection": {"message": ["Failed password"]},
            "filter": {"src_ip": ["10.1.1.1"]},
            "condition": "selection and filter",
        }
        self.assertEqual(scorer.evaluate(rule, self.attack),
                         [True, True, False])
        self.assertEqual(scorer.evaluate(rule, self.benign),
                         [False, False])
        rule["detection"]["condition"] = "selection and not filter"
        self.assertEqual(scorer.evaluate(rule, self.attack),
                         [False, False, False])
        self.assertEqual(scorer.evaluate(rule, self.benign),
                         [True, False])

    def test_or(self):
        rule = dict(GOOD_RULE)
        rule["detection"] = {
            "selection": {"message": ["Failed password"]},
            "accepted": {"event": ["accepted"]},
            "condition": "selection or accepted",
        }
        self.assertEqual(scorer.evaluate(rule, self.attack),
                         [True, True, True])
        self.assertEqual(scorer.evaluate(rule, self.benign),
                         [True, True])

    def test_count_threshold_keeps_bursts_drops_singletons(self):
        # Attack IP has 2 failures (>= 2): both match. Benign IP has
        # 1: the typo line drops out.
        result = scorer.evaluate(count_rule(">=", 2), self.attack)
        self.assertEqual(result, [True, True, False])
        self.assertEqual(scorer.evaluate(count_rule(">=", 2), self.benign),
                         [False, False])

    def test_count_strict_greater(self):
        self.assertEqual(scorer.evaluate(count_rule(">", 2), self.attack),
                         [False, False, False])

    def test_unparseable_condition_raises(self):
        rule = dict(GOOD_RULE)
        rule["detection"] = {"selection": {"message": ["Failed password"]},
                             "condition": "selection |"}
        with self.assertRaises(ValueError):
            scorer.evaluate(rule, self.attack)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.attack = scorer.parse_lines(ATTACK_LINES)
        self.benign = scorer.parse_lines(BENIGN_LINES)

    def test_hand_computed_counts(self):
        result = scorer.score_rule(GOOD_RULE, self.attack, self.benign)
        self.assertEqual((result["tp"], result["fp"], result["fn"],
                          result["tn"]), (2, 1, 1, 1))
        self.assertAlmostEqual(result["catch_rate"], 2 / 3)
        self.assertAlmostEqual(result["fp_rate"], 1 / 2)
        self.assertAlmostEqual(result["score"], 2 / 3 - 1 / 2)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_count_rule_improves_score(self):
        plain = scorer.score_rule(GOOD_RULE, self.attack, self.benign)
        counted = scorer.score_rule(count_rule(">=", 2), self.attack,
                                    self.benign)
        self.assertEqual((counted["tp"], counted["fp"], counted["fn"],
                          counted["tn"]), (2, 0, 1, 2))
        self.assertAlmostEqual(counted["score"], 2 / 3)
        self.assertGreater(counted["score"], plain["score"])

    def test_zero_result(self):
        result = scorer.zero_result()
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["tp"], 0)


class FixtureTests(unittest.TestCase):
    def test_fixture_line_counts(self):
        counts = {}
        for split in ("train", "test"):
            for kind in ("attack", "benign"):
                path = os.path.join(BASE, "data", split, kind + ".log")
                counts[(split, kind)] = len(scorer.read_log_file(path))
        self.assertEqual(counts, {("train", "attack"): 16,
                                  ("train", "benign"): 20,
                                  ("test", "attack"): 14,
                                  ("test", "benign"): 18})

    def test_attack_ips_are_disjoint_across_splits(self):
        def attack_ips(split):
            path = os.path.join(BASE, "data", split, "attack.log")
            return {record["src_ip"] for record in
                    scorer.read_log_file(path)}
        self.assertFalse(attack_ips("train") & attack_ips("test"))

    def test_best_mock_rule_scores_fixtures(self):
        import agents
        rule = agents.MOCK_RULES[3]
        train_attack = scorer.read_log_file(
            os.path.join(BASE, "data", "train", "attack.log"))
        train_benign = scorer.read_log_file(
            os.path.join(BASE, "data", "train", "benign.log"))
        result = scorer.score_rule(rule, train_attack, train_benign)
        self.assertEqual((result["tp"], result["fp"], result["fn"],
                          result["tn"]), (15, 0, 1, 20))
        self.assertAlmostEqual(result["score"], 15 / 16)


if __name__ == "__main__":
    unittest.main()
