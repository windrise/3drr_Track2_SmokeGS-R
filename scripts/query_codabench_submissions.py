#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from codabench_submit import BASE_URL, build_session, ensure_ok


def fetch_submissions(session, competition_id: str, phase_id: str | None) -> list[dict]:
    if phase_id is not None:
        url = f"{BASE_URL}/api/submissions/?phase={phase_id}"
    else:
        url = f"{BASE_URL}/api/submissions/?competition={competition_id}"
    resp = session.get(url)
    ensure_ok(resp, "fetch submissions")
    return resp.json()


def fetch_submission(session, submission_id: str) -> dict:
    resp = session.get(f"{BASE_URL}/api/submissions/{submission_id}/")
    ensure_ok(resp, f"fetch submission {submission_id}")
    return resp.json()


def score_map(submission: dict) -> dict[str, str]:
    return {score["column_key"]: score["score"] for score in submission.get("scores", [])}


def print_submission_row(submission: dict) -> None:
    scores = score_map(submission)
    created_when = submission.get("created_when", "")
    print(
        f"{submission['id']}\t{submission.get('status')}\t"
        f"{scores.get('PSNR', '-')}\t{scores.get('SSIM', '-')}\t"
        f"{created_when}\t{submission.get('filename', '')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Query Codabench submissions for this competition.")
    parser.add_argument("--competition-id", required=True)
    parser.add_argument("--cookie", required=True)
    parser.add_argument("--phase-id", default=None, help="Optional phase id, e.g. 23265")
    parser.add_argument("--submission-id", default=None, help="Fetch a single submission by id")
    parser.add_argument("--limit", type=int, default=20, help="Number of recent submissions to print")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to dump raw JSON")
    args = parser.parse_args()

    session = build_session(args.competition_id, args.cookie)

    if args.submission_id:
        payload = fetch_submission(session, args.submission_id)
        if args.save_json:
            args.save_json.parent.mkdir(parents=True, exist_ok=True)
            args.save_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    submissions = fetch_submissions(session, args.competition_id, args.phase_id)
    submissions = sorted(
        submissions,
        key=lambda item: item.get("created_when", ""),
        reverse=True,
    )
    if args.limit > 0:
        submissions = submissions[: args.limit]

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(submissions, ensure_ascii=False, indent=2), encoding="utf-8")

    print("id\tstatus\tPSNR\tSSIM\tcreated_when\tfilename")
    for submission in submissions:
        print_submission_row(submission)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
