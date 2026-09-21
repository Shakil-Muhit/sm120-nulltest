# Push time evidence (R1)

The repository `Shakil-Muhit/sm120-nulltest` is PRIVATE (GitHub API returns 404 unauthenticated; owner brief §10). GH Archive records public events only, so no server-side PushEvent exists for any push made while the repository is private. The evidence available now, labelled by kind:

| commit | subject | committer date (client-side, git) | local reflog push record | server-side evidence |
|---|---|---|---|---|
| e523ef9318e6 | registration + harness for Phase 1 sessions (pushed before session 1 started 11:01 UTC 2026-09-20) | 2026-09-20T16:55:21+06:00 | see docs/provenance/reflog_origin_main.txt | none while private — GitHub Events API `created_at` (needs owner token, 90-day retention) or GH Archive after the repository is public |
| f942b1439108 | gate-08 measurement-path fixes | 2026-09-20T17:18:19+06:00 | see docs/provenance/reflog_origin_main.txt | none while private — GitHub Events API `created_at` (needs owner token, 90-day retention) or GH Archive after the repository is public |
| 795fc1bf2ed1 | execution report | 2026-09-21T09:15:29+06:00 | see docs/provenance/reflog_origin_main.txt | none while private — GitHub Events API `created_at` (needs owner token, 90-day retention) or GH Archive after the repository is public |
| 954733316d0b | Phase 3 §2 part 1 | 2026-09-21T21:25:12+06:00 | see docs/provenance/reflog_origin_main.txt | none while private — GitHub Events API `created_at` (needs owner token, 90-day retention) or GH Archive after the repository is public |

Session-1 start on the pod: 2026-09-20T11:01:23+00:00 (evidence/session1_console.log). The registration commit e523ef9 was pushed at 2026-09-20 ~11:00 UTC (client push log below); the ordering registration-before-measurement therefore rests on client-side timestamps plus the pod's own console log until server-side evidence is obtained.

## How to obtain server-side evidence (owner action)
```
curl -s -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/Shakil-Muhit/sm120-nulltest/events?per_page=100 | jq '.[] | select(.type=="PushEvent") | {created_at, before: .payload.before, head: .payload.head, ref: .payload.ref}'
```
Record the `created_at` of the PushEvent whose `head` equals each registration commit into the table above. Once public, GH Archive (gharchive.org, hourly JSON) holds the same events permanently.

## OpenTimestamps
An OpenTimestamps proof of the file `docs/provenance/registration_hashes.txt` (commit hashes above) is attempted below; it proves existence from the stamping time only, never retroactively.
