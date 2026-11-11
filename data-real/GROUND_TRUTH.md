# data-real — ground truth (from CyberBeep's own artifacts)

Source: `~/workspace/work/cyberbeep/examples/sample_auth.log` (11 lines) +
`~/workspace/work/cyberbeep/incidents.jsonl` (2 incidents).

- Attack IP `203.0.113.45`: 6x "Failed password" (02:00:01–02:00:58) then
  "Accepted password for deploy" (02:01:10). CyberBeep flagged it HIGH
  (ssh_brute_force, 5 failures in 300s, at 02:00:44) and CRITICAL
  (login_after_brute_force, at 02:01:10). Those 7 lines are in
  `test/attack.log`. A burst rule (count >= 5) is expected to catch the
  6 failures and MISS the accepted login — that miss is a separate
  login-after-brute-force rule's job (same finding as step 1).
- Benign (4 lines): `198.51.100.7` has only 2 failures (below
  CyberBeep's threshold of 5), plus a CRON line and the preauth
  disconnect line from 203.0.113.45. The disconnect is session noise,
  not a login attempt, so it sits in benign: a burst rule must ignore
  it, same as the CRON noise in step 1's fixtures.
- This is a held-out check only (no train split): step 2 scores the
  live model's best rule against it after iterating on data/train.
- Sanity check (run on the VM 2026-11-11): the step-1 best rule scores
  TP 6 / FP 0 / FN 1 / TN 4 on this set — score 0.8571, F1 0.9231.
