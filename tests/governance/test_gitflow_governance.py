"""Tests for Banking GitFlow governance, Conventional Branch/Commits, SemVer, and PR promotion."""

import json

import pytest

from banking_fti.governance.gitflow import (
    OpenPullRequest,
    SemVer,
    derive_next_semver,
    parse_conventional_commit,
    plan_develop_to_release_pr,
    plan_feature_to_develop_pr,
    plan_release_to_main_pr,
    validate_branch_name,
    validate_commit_messages,
)


@pytest.mark.parametrize(
    "branch",
    [
        "feature/gitflow-workflow-governance",
        "feature/01-causal-feature-pipeline",
        "feature/fix-drift-cooldown",
    ],
)
def test_validate_branch_name_accepts_valid_feature_branches(branch: str) -> None:
    result = validate_branch_name(branch)

    assert result.valid is True
    assert result.branch == branch
    assert result.branch_type == "feature"
    assert result.reason == "ok"


@pytest.mark.parametrize(
    "branch",
    [
        "develop",
        "main",
        "release/v0.1.0",
        "bugfix/login-error",
        "fix/header-bug",
        "chore/update-deps",
        "hotfix/urgent-fix",
        "docs/readme",
        "feature/UpperCase",
        "feature/snake_case",
        "feature/-leading-hyphen",
        "feature/trailing-hyphen-",
        "feature/double--hyphen",
        "feature/with.dot",
        "feature/",
        "",
    ],
)
def test_validate_branch_name_rejects_disallowed_local_branches(branch: str) -> None:
    result = validate_branch_name(branch)

    assert result.valid is False
    assert result.reason != ""


def test_validate_branch_name_respects_allow_release_and_allow_protected_flags() -> None:
    release_ok = validate_branch_name("release/v1.2.0", allow_release=True)
    assert release_ok.valid is True
    assert release_ok.branch_type == "release"

    release_bad = validate_branch_name("release/v01.2.0", allow_release=True)
    assert release_bad.valid is False

    develop_ok = validate_branch_name("develop", allow_protected=True)
    assert develop_ok.valid is True
    assert develop_ok.branch_type == "protected"

    main_ok = validate_branch_name("main", allow_protected=True)
    assert main_ok.valid is True
    assert main_ok.branch_type == "protected"


def test_parse_conventional_commit_parses_standard_body_and_footers() -> None:
    msg = (
        "feat(feature-store)!: enforce causal rolling window checks\n"
        "\n"
        "Ensure t+1 events never leak into t aggregations.\n"
        "\n"
        "BREAKING CHANGE: removed legacy window_mode argument\n"
        "Refs: #12"
    )
    parsed = parse_conventional_commit(msg)

    assert parsed.commit_type == "feat"
    assert parsed.scope == "feature-store"
    assert parsed.description == "enforce causal rolling window checks"
    assert parsed.breaking is True
    assert parsed.body == "Ensure t+1 events never leak into t aggregations."
    assert parsed.footers == {
        "BREAKING CHANGE": "removed legacy window_mode argument",
        "Refs": "#12",
    }


def test_parse_conventional_commit_detects_breaking_footer_without_bang() -> None:
    msg = "refactor(api): normalize prediction payload\n\nBREAKING CHANGE: renamed field"
    parsed = parse_conventional_commit(msg)

    assert parsed.commit_type == "refactor"
    assert parsed.breaking is True


@pytest.mark.parametrize(
    "bad_message",
    [
        "",
        "update readme",
        "unknown: add something",
        "feat:missing space after colon",
        "feat: Uppercase description",
        "feat: trailing period.",
        "feat(): empty scope",
        "feat(bad scope): spaces in scope",
        f"feat: {'a' * 70}",
        "feat: missing blank line\nbody starts immediately",
    ],
)
def test_parse_conventional_commit_rejects_non_conforming_messages(bad_message: str) -> None:
    with pytest.raises(ValueError):
        parse_conventional_commit(bad_message)


def test_validate_commit_messages_skips_git_merge_commits_when_enabled() -> None:
    commits = validate_commit_messages(
        [
            "Merge pull request #7 from org/feature/01-init",
            "Merge branch 'develop' into release/v0.1.0",
            "Merge c9920ee78b49bd69ae175edbb98cb3ba9e756461 "
            "into bf3af2c0ad1183842597310f67936f4c94420ed5",
            "fix(ci): handle empty commit diff",
        ],
        ignore_merges=True,
    )

    assert len(commits) == 1
    assert commits[0].commit_type == "fix"


def test_derive_next_semver_starts_at_initial_version_when_no_tag_exists() -> None:
    version = derive_next_semver(None, ["feat(gitflow): bootstrap governance"])

    assert version == SemVer(0, 1, 0)
    assert version.format() == "v0.1.0"


@pytest.mark.parametrize(
    ("latest_tag", "commits", "expected"),
    [
        ("v0.1.0", ["fix(ci): handle empty diff"], "v0.1.1"),
        ("v0.1.0", ["chore(deps): bump ruff"], "v0.1.1"),
        ("v0.1.2", ["feat(batch): add credit scorer"], "v0.2.0"),
        ("v0.2.0", ["feat(api)!: change response schema"], "v0.3.0"),
        ("v1.2.3", ["fix(api): guard NaN input"], "v1.2.4"),
        ("v1.2.3", ["feat(fraud): add amount slices"], "v1.3.0"),
        ("v1.2.3", ["feat(api)!: break payload"], "v2.0.0"),
        ("v1.2.3", ["chore(core): drop legacy field\n\nBREAKING CHANGE: removed x"], "v2.0.0"),
    ],
)
def test_derive_next_semver_applies_semver_rules_from_conventional_commits(
    latest_tag: str,
    commits: list[str],
    expected: str,
) -> None:
    version = derive_next_semver(latest_tag, commits)
    assert version.format() == expected


def test_derive_next_semver_rejects_empty_non_merge_commit_list() -> None:
    with pytest.raises(ValueError, match="no non-merge commits"):
        derive_next_semver("v0.1.0", ["Merge pull request #1 from org/feature/a"])


def test_plan_feature_to_develop_pr_is_idempotent_and_never_enables_automerge() -> None:
    created = plan_feature_to_develop_pr(
        feature_branch="feature/01-causal-pipeline",
        commit_messages=["feat(features): add causal rolling aggregations"],
        open_prs=[],
    )
    assert created.action == "create"
    assert created.head == "feature/01-causal-pipeline"
    assert created.base == "develop"
    assert created.title == "feat(features): add causal rolling aggregations"
    assert created.auto_merge is False
    assert parse_conventional_commit(created.title).commit_type == "feat"

    unchanged = plan_feature_to_develop_pr(
        feature_branch="feature/01-causal-pipeline",
        commit_messages=["feat(features): add causal rolling aggregations"],
        open_prs=[
            OpenPullRequest(
                number=11,
                head="feature/01-causal-pipeline",
                base="develop",
                title=created.title,
                body=created.body,
            )
        ],
    )
    assert unchanged.action == "noop"
    assert unchanged.existing_pr_number == 11
    assert unchanged.auto_merge is False

    updated = plan_feature_to_develop_pr(
        feature_branch="feature/01-causal-pipeline",
        commit_messages=[
            "feat(features): add causal rolling aggregations",
            "test(features): prove no future leakage",
        ],
        open_prs=[
            OpenPullRequest(
                number=11,
                head="feature/01-causal-pipeline",
                base="develop",
                title="feat(features): old title",
                body="old body",
            )
        ],
    )
    assert updated.action == "update"
    assert updated.existing_pr_number == 11
    assert updated.title == "feat: merge feature 01-causal-pipeline into develop"
    assert updated.auto_merge is False


def test_plan_develop_to_release_pr_creates_release_from_main_or_reuses_open_release() -> None:
    no_diff = plan_develop_to_release_pr(
        latest_main_tag="v0.1.0",
        commits_since_main=[],
        existing_release_branches=[],
        open_prs=[],
    )
    assert no_diff.action == "noop"

    fresh = plan_develop_to_release_pr(
        latest_main_tag=None,
        commits_since_main=["feat(gitflow): bootstrap governance"],
        existing_release_branches=[],
        open_prs=[],
    )
    assert fresh.action == "create"
    assert fresh.head == "develop"
    assert fresh.base == "release/v0.1.0"
    assert fresh.create_base_from_ref == "main"
    assert fresh.title == "chore(release): prepare release v0.1.0"
    assert fresh.auto_merge is False

    reused = plan_develop_to_release_pr(
        latest_main_tag="v0.1.0",
        commits_since_main=["feat(batch): add scorer", "fix(ci): fix check"],
        existing_release_branches=["release/v0.2.0"],
        open_prs=[],
    )
    assert reused.action == "create"
    assert reused.base == "release/v0.2.0"
    assert reused.create_base_from_ref is None
    assert reused.auto_merge is False


def test_plan_release_to_main_pr_validates_branch_and_is_idempotent() -> None:
    with pytest.raises(ValueError, match="release"):
        plan_release_to_main_pr("develop", open_prs=[])

    created = plan_release_to_main_pr("release/v0.2.0", open_prs=[])
    assert created.action == "create"
    assert created.head == "release/v0.2.0"
    assert created.base == "main"
    assert created.title == "chore(release): release v0.2.0"
    assert created.auto_merge is False

    unchanged = plan_release_to_main_pr(
        "release/v0.2.0",
        open_prs=[
            OpenPullRequest(
                number=24,
                head="release/v0.2.0",
                base="main",
                title=created.title,
                body=created.body,
            )
        ],
    )
    assert unchanged.action == "noop"
    assert unchanged.existing_pr_number == 24


def test_cli_validate_branch_and_commits_return_expected_exit_codes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from banking_fti.governance.cli import main

    assert main(["validate-branch", "--branch", "feature/00-gitflow"]) == 0
    assert main(["validate-branch", "--branch", "hotfix/urgent"]) == 1

    assert (
        main(
            [
                "validate-commits",
                "--pr-title",
                "feat(gitflow): add automated pr chain",
                "--message",
                "feat(gitflow): add branch validator",
            ]
        )
        == 0
    )
    assert main(["validate-commits", "--pr-title", "Bad title"]) == 1
    captured = capsys.readouterr()
    assert "Bad title" in captured.err


def test_cli_plan_promotion_outputs_json_action(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from banking_fti.governance.cli import main

    exit_code = main(
        [
            "plan-promotion",
            "--stage",
            "develop-to-release",
            "--latest-tag",
            "v0.1.0",
            "--message",
            "feat(batch): add portfolio scorer",
            "--existing-release-branches-json",
            "[]",
            "--open-prs-json",
            "[]",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "create"
    assert payload["head"] == "develop"
    assert payload["base"] == "release/v0.2.0"
    assert payload["create_base_from_ref"] == "main"
    assert payload["auto_merge"] is False


def test_plan_develop_to_release_pr_ignores_stale_released_branches() -> None:
    plan = plan_develop_to_release_pr(
        latest_main_tag="v0.1.0",
        commits_since_main=["feat(batch): add portfolio scorer"],
        existing_release_branches=["release/v0.1.0"],
        open_prs=[],
    )

    assert plan.action == "create"
    assert plan.base == "release/v0.2.0"
    assert plan.create_base_from_ref == "main"


def test_cli_plan_promotion_parses_github_rest_api_pr_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from banking_fti.governance.cli import main

    github_prs = json.dumps(
        [
            {
                "number": 42,
                "head": {"ref": "feature/01-init"},
                "base": {"ref": "develop"},
                "title": "feat(features): old title",
                "body": None,
            }
        ]
    )
    exit_code = main(
        [
            "plan-promotion",
            "--stage",
            "feature-to-develop",
            "--feature-branch",
            "feature/01-init",
            "--message",
            "feat(features): add causal pipeline",
            "--open-prs-json",
            github_prs,
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "update"
    assert payload["existing_pr_number"] == 42
    assert payload["auto_merge"] is False


def test_parse_conventional_commit_rejects_non_ascii_header() -> None:
    with pytest.raises(ValueError, match="ASCII"):
        parse_conventional_commit("feat(api): adicionar validação de fraude")


def test_plan_feature_to_develop_pr_returns_noop_when_no_commits_ahead() -> None:
    action = plan_feature_to_develop_pr(
        feature_branch="feature/01-empty",
        commit_messages=[],
        open_prs=[],
    )

    assert action.action == "noop"
    assert action.auto_merge is False


def test_plan_develop_to_release_pr_returns_noop_when_no_commits_ahead_of_open_release() -> None:
    action = plan_develop_to_release_pr(
        latest_main_tag=None,
        commits_since_main=["feat(gitflow): bootstrap governance"],
        existing_release_branches=["release/v0.1.0"],
        open_prs=[],
        commits_since_release=[],
    )

    assert action.action == "noop"
    assert action.auto_merge is False
