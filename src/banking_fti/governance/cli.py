"""CLI entrypoint for Day-0 Banking GitFlow governance checks and PR promotion planning."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict

from banking_fti.governance.gitflow import (
    OpenPullRequest,
    PromotionAction,
    extract_release_version_from_commit,
    parse_conventional_commit,
    plan_develop_to_release_pr,
    plan_feature_to_develop_pr,
    plan_release_to_main_pr,
    validate_branch_name,
    validate_commit_messages,
)


def _read_git_commits(commit_range: str) -> list[str]:
    """Read raw commit messages for a git revision range using ASCII RS delimiters."""
    proc = subprocess.run(
        ["git", "log", commit_range, "--format=%B%x1e"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [chunk.strip() for chunk in proc.stdout.split("\x1e") if chunk.strip()]


def _collect_messages(commit_range: str | None, messages: Sequence[str]) -> list[str]:
    """Combine explicit CLI messages with commits read from an optional git range."""
    collected = list(messages)
    if commit_range:
        collected.extend(_read_git_commits(commit_range))
    return collected


def _extract_branch_ref(raw_endpoint: object) -> str:
    """Extract a branch name from either a flat string or GitHub REST API dict."""
    if isinstance(raw_endpoint, dict):
        return str(raw_endpoint["ref"])
    return str(raw_endpoint)


def _parse_open_prs(raw_json: str) -> list[OpenPullRequest]:
    """Deserialize JSON array of open PRs into OpenPullRequest objects."""
    loaded = json.loads(raw_json)
    if not isinstance(loaded, list):
        raise ValueError("--open-prs-json must decode to a JSON array.")
    return [
        OpenPullRequest(
            number=int(item["number"]),
            head=_extract_branch_ref(item["head"]),
            base=_extract_branch_ref(item["base"]),
            title=str(item["title"]),
            body=str(item.get("body") or ""),
        )
        for item in loaded
    ]


def _run_validate_branch(args: argparse.Namespace) -> int:
    result = validate_branch_name(
        args.branch,
        allow_release=args.allow_release,
        allow_protected=args.allow_protected,
    )
    if not result.valid:
        print(result.reason, file=sys.stderr)
        return 1
    print(f"Valid {result.branch_type} branch: {result.branch}")
    return 0


def _run_validate_commits(args: argparse.Namespace) -> int:
    try:
        if args.pr_title is not None:
            parse_conventional_commit(args.pr_title)
        messages = _collect_messages(args.commit_range, args.message)
        parsed = validate_commit_messages(messages, ignore_merges=True)
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Validated {len(parsed)} commit(s) against Conventional Commits 1.0.0.")
    return 0


def _run_plan_promotion(args: argparse.Namespace) -> int:
    try:
        open_prs = _parse_open_prs(args.open_prs_json)
        action: PromotionAction
        if args.stage == "feature-to-develop":
            if not args.feature_branch:
                raise ValueError("--feature-branch is required for feature-to-develop.")
            messages = _collect_messages(args.commit_range, args.message)
            action = plan_feature_to_develop_pr(
                feature_branch=args.feature_branch,
                commit_messages=messages,
                open_prs=open_prs,
            )
        elif args.stage == "develop-to-release":
            messages = _collect_messages(args.commit_range, args.message)
            existing_releases_raw = json.loads(args.existing_release_branches_json)
            if not isinstance(existing_releases_raw, list):
                raise ValueError("--existing-release-branches-json must decode to a JSON array.")
            existing_releases = [str(item) for item in existing_releases_raw]
            merged_releases_raw = json.loads(args.merged_release_branches_json)
            if not isinstance(merged_releases_raw, list):
                raise ValueError("--merged-release-branches-json must decode to a JSON array.")
            merged_releases = [str(item) for item in merged_releases_raw]
            latest_tag = args.latest_tag or None
            commits_since_release = (
                _read_git_commits(args.release_commit_range) if args.release_commit_range else None
            )
            action = plan_develop_to_release_pr(
                latest_main_tag=latest_tag,
                commits_since_main=messages,
                existing_release_branches=existing_releases,
                open_prs=open_prs,
                commits_since_release=commits_since_release,
                merged_release_branches=merged_releases,
            )
        else:
            if not args.release_branch:
                raise ValueError("--release-branch is required for release-to-main.")
            action = plan_release_to_main_pr(
                release_branch=args.release_branch,
                open_prs=open_prs,
            )
    except (ValueError, json.JSONDecodeError, KeyError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(asdict(action)))
    return 0


def _run_resolve_release_tag(args: argparse.Namespace) -> int:
    try:
        if args.message is not None:
            raw_message = str(args.message)
        else:
            proc = subprocess.run(
                ["git", "log", "-1", "--format=%B", str(args.commit_ref)],
                check=True,
                capture_output=True,
                text=True,
            )
            raw_message = proc.stdout
        tag = extract_release_version_from_commit(raw_message)
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(tag or "")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Banking GitFlow governance CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    branch_parser = subparsers.add_parser("validate-branch")
    branch_parser.add_argument("--branch", required=True)
    branch_parser.add_argument("--allow-release", action="store_true")
    branch_parser.add_argument("--allow-protected", action="store_true")

    commits_parser = subparsers.add_parser("validate-commits")
    commits_parser.add_argument("--pr-title", default=None)
    commits_parser.add_argument("--commit-range", default=None)
    commits_parser.add_argument("--message", action="append", default=[])

    promo_parser = subparsers.add_parser("plan-promotion")
    promo_parser.add_argument(
        "--stage",
        required=True,
        choices=["feature-to-develop", "develop-to-release", "release-to-main"],
    )
    promo_parser.add_argument("--feature-branch", default=None)
    promo_parser.add_argument("--release-branch", default=None)
    promo_parser.add_argument("--latest-tag", default="")
    promo_parser.add_argument("--commit-range", default=None)
    promo_parser.add_argument("--release-commit-range", default=None)
    promo_parser.add_argument("--message", action="append", default=[])
    promo_parser.add_argument("--existing-release-branches-json", default="[]")
    promo_parser.add_argument("--merged-release-branches-json", default="[]")
    promo_parser.add_argument("--open-prs-json", default="[]")

    tag_parser = subparsers.add_parser("resolve-release-tag")
    tag_parser.add_argument("--message", default=None)
    tag_parser.add_argument("--commit-ref", default="HEAD")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Banking GitFlow governance CLI and return an exit status code."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "validate-branch":
        return _run_validate_branch(args)
    if args.command == "validate-commits":
        return _run_validate_commits(args)
    if args.command == "resolve-release-tag":
        return _run_resolve_release_tag(args)
    return _run_plan_promotion(args)


if __name__ == "__main__":
    raise SystemExit(main())
