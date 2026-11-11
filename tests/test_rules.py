"""Tests for RuleForge step 1: rule validation and YAML output."""

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents
import rules

GOOD_RULE = {
    "title": "SSH Brute Force - 5+ Failed Logins From One IP",
    "logsource": {"category": "auth", "product": "linux",
                  "service": "sshd"},
    "detection": {
        "selection": {"message": ["Failed password",
                                  "Failed publickey"]},
        "condition": "selection | count() by src_ip >= 5",
    },
    "level": "high",
    "description": "Flags repeated failed SSH logins from one IP.",
}


class ConditionTests(unittest.TestCase):
    def test_single(self):
        self.assertEqual(rules.parse_condition("selection"),
                         {"kind": "single", "name": "selection"})

    def test_and_and_not(self):
        self.assertEqual(rules.parse_condition("a and b"),
                         {"kind": "and", "a": "a", "b": "b",
                          "negated": False})
        self.assertEqual(rules.parse_condition("a and not b"),
                         {"kind": "and", "a": "a", "b": "b",
                          "negated": True})

    def test_or(self):
        self.assertEqual(rules.parse_condition("a or b"),
                         {"kind": "or", "a": "a", "b": "b"})

    def test_count_forms(self):
        self.assertEqual(
            rules.parse_condition("selection | count() by src_ip >= 5"),
            {"kind": "count", "name": "selection", "op": ">=", "n": 5})
        self.assertEqual(
            rules.parse_condition("selection|count() by src_ip>3"),
            {"kind": "count", "name": "selection", "op": ">", "n": 3})

    def test_garbage_does_not_parse(self):
        for bad in ("", "selection |", "selection and", "a b",
                    "selection | count() by user >= 5",
                    # Real Sigma allows time windows in count
                    # correlations; this small language does not, and the
                    # live model reached for one in step 2 (Nov 2026).
                    "selection | count() by src_ip over 1m >= 10",
                    "selection | count() by src_ip over 1m >= 10 and not "
                    "selection | count() by src_ip over 5m >= 20",
                    None, 42):
            self.assertIsNone(rules.parse_condition(bad), bad)


class ValidateTests(unittest.TestCase):
    def test_good_rule_passes(self):
        ok, errors = rules.validate_rule(GOOD_RULE)
        self.assertTrue(ok, errors)
        self.assertEqual(errors, [])

    def test_mock_rules_all_pass(self):
        for rule in agents.MOCK_RULES.values():
            ok, errors = rules.validate_rule(rule)
            self.assertTrue(ok, errors)

    def test_not_a_dict(self):
        ok, errors = rules.validate_rule(["not", "a", "dict"])
        self.assertFalse(ok)
        self.assertTrue(errors)

    def test_missing_title(self):
        rule = copy.deepcopy(GOOD_RULE)
        del rule["title"]
        ok, errors = rules.validate_rule(rule)
        self.assertFalse(ok)
        self.assertTrue(any("title" in e for e in errors))

    def test_missing_detection(self):
        rule = copy.deepcopy(GOOD_RULE)
        del rule["detection"]
        ok, errors = rules.validate_rule(rule)
        self.assertFalse(ok)
        self.assertTrue(any("detection" in e for e in errors))

    def test_empty_selection(self):
        rule = copy.deepcopy(GOOD_RULE)
        rule["detection"]["selection"] = {}
        ok, errors = rules.validate_rule(rule)
        self.assertFalse(ok)

    def test_unknown_field(self):
        rule = copy.deepcopy(GOOD_RULE)
        rule["detection"]["selection"] = {"hostname": ["server"]}
        ok, errors = rules.validate_rule(rule)
        self.assertFalse(ok)
        self.assertTrue(any("hostname" in e for e in errors))

    def test_condition_names_unknown_selection(self):
        rule = copy.deepcopy(GOOD_RULE)
        rule["detection"]["condition"] = "selection and filter"
        ok, errors = rules.validate_rule(rule)
        self.assertFalse(ok)
        self.assertTrue(any("filter" in e for e in errors))

    def test_condition_does_not_parse(self):
        rule = copy.deepcopy(GOOD_RULE)
        rule["detection"]["condition"] = "selection |"
        ok, errors = rules.validate_rule(rule)
        self.assertFalse(ok)

    def test_bad_level(self):
        rule = copy.deepcopy(GOOD_RULE)
        rule["level"] = "urgent"
        ok, errors = rules.validate_rule(rule)
        self.assertFalse(ok)
        self.assertTrue(any("level" in e for e in errors))

    def test_bad_logsource(self):
        rule = copy.deepcopy(GOOD_RULE)
        rule["logsource"] = {"category": "auth"}
        ok, errors = rules.validate_rule(rule)
        self.assertFalse(ok)
        self.assertTrue(any("product" in e for e in errors))


class YamlTests(unittest.TestCase):
    def test_emits_expected_lines(self):
        text = rules.emit_yaml(GOOD_RULE)
        self.assertIn('title: "SSH Brute Force - 5+ Failed Logins From '
                      'One IP"', text)
        self.assertIn("category: \"auth\"", text)
        self.assertIn('  - "Failed password"', text)
        self.assertIn('  - "Failed publickey"', text)
        self.assertIn('condition: "selection | count() by src_ip >= 5"',
                      text)
        self.assertIn('level: "high"', text)

    def test_invalid_rule_raises(self):
        rule = copy.deepcopy(GOOD_RULE)
        rule["level"] = "urgent"
        with self.assertRaises(ValueError):
            rules.emit_yaml(rule)

    def test_json_round_trip(self):
        import json
        text = rules.rule_to_json(GOOD_RULE)
        self.assertEqual(json.loads(text), GOOD_RULE)


if __name__ == "__main__":
    unittest.main()
