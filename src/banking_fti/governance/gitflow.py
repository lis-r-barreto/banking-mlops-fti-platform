"""Pure domain rules for Banking GitFlow, Conventional Branch/Commits, SemVer, and PR promotion."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

FEATURE_BRANCH_RE = re.compile(r"^feature/[a-z0-9]+(-[a-z0-9]+)*$")
SEMVER_RE = re.compile(
    r"^v(?P<major>0|[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)\.(?P<patch>0|[1-9][0-9]*)$"
)
RELEASE_BRANCH_RE = re.compile(r"^release/v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
PROTECTED_BRANCHES = frozenset({"main", "develop"})

ALLOWED_COMMIT_TYPES = frozenset(
    {
        "feat",
        "fix",
        "build",
        "chore",
        "ci",
        "docs",
        "perf",
        "refactor",
        "style",
        "test",
        "revert",
    }
)
MAX_HEADER_LENGTH = 72
COMMIT_HEADER_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[a-z0-9][a-z0-9._-]*)\))?(?P<bang>!)?:\s(?P<desc>\S.*)$"
)
FOOTER_LINE_RE = re.compile(r"^(?P<token>BREAKING CHANGE|[A-Za-z0-9-]+)(?::\s|\s#)(?P<val>.*)$")
MERGE_HEADER_RE = re.compile(
    r"^Merge (?:pull request #|branch '|remote-tracking branch '|[0-9a-f]{7,} into [0-9a-f]{7,})"
)
_RELEASE_BRANCH_REF_RE = re.compile(
    r"release/(v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))"
)
_RELEASE_TITLE_RE = re.compile(
    r"^chore\(release\):\s+release\s+(v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))$"
)


BranchType = Literal["feature", "release", "protected", "invalid"]


@dataclass(frozen=True)
class BranchValidationResult:
    """Result of validating a Git branch name against Banking GitFlow rules."""

    valid: bool
    branch: str
    branch_type: BranchType
    reason: str


def validate_branch_name(
    branch: str,
    *,
    allow_release: bool = False,
    allow_protected: bool = False,
) -> BranchValidationResult:
    """Validate a branch name against Conventional Branch 1.0.0 and feature-only dev rules."""
    if not branch:
        return BranchValidationResult(
            valid=False,
            branch=branch,
            branch_type="invalid",
            reason="Branch name must not be empty.",
        )

    if branch in PROTECTED_BRANCHES:
        if allow_protected:
            return BranchValidationResult(
                valid=True,
                branch=branch,
                branch_type="protected",
                reason="ok",
            )
        return BranchValidationResult(
            valid=False,
            branch=branch,
            branch_type="protected",
            reason=f"Direct work on protected branch '{branch}' is disallowed; use feature/*.",
        )

    if FEATURE_BRANCH_RE.fullmatch(branch):
        return BranchValidationResult(
            valid=True,
            branch=branch,
            branch_type="feature",
            reason="ok",
        )

    if RELEASE_BRANCH_RE.fullmatch(branch):
        if allow_release:
            return BranchValidationResult(
                valid=True,
                branch=branch,
                branch_type="release",
                reason="ok",
            )
        return BranchValidationResult(
            valid=False,
            branch=branch,
            branch_type="release",
            reason=f"Release branch '{branch}' is managed by promotion automation, not local dev.",
        )

    return BranchValidationResult(
        valid=False,
        branch=branch,
        branch_type="invalid",
        reason=(
            f"Branch '{branch}' violates Conventional Branch 1.0.0 / feature/*-only rule "
            "(expected lowercase kebab-case 'feature/<description>')."
        ),
    )


@dataclass(frozen=True)
class ConventionalCommit:
    """Parsed Conventional Commits 1.0.0 message."""

    commit_type: str
    scope: str | None
    description: str
    breaking: bool
    body: str
    footers: dict[str, str]


def _split_body_and_footers(rest_lines: Sequence[str]) -> tuple[str, dict[str, str]]:
    """Split optional body paragraphs and trailing git-trailer footers."""
    if not rest_lines:
        return "", {}

    # Strip leading blank lines (already verified at least one blank line after header).
    lines = list(rest_lines)
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return "", {}

    # Check if the trailing paragraph consists solely of valid footer lines.
    last_blank = -1
    for idx, line in enumerate(lines):
        if line == "":
            last_blank = idx

    candidate_start = last_blank + 1
    candidate_lines = lines[candidate_start:]
    footers: dict[str, str] = {}
    current_token: str | None = None
    is_footer_block = True

    for line in candidate_lines:
        match = FOOTER_LINE_RE.fullmatch(line)
        if match:
            current_token = match.group("token")
            footers[current_token] = match.group("val").strip()
        elif current_token is not None:
            footers[current_token] = f"{footers[current_token]}\n{line}".strip()
        else:
            is_footer_block = False
            break

    if is_footer_block and footers:
        body_lines = lines[:last_blank] if last_blank > 0 else []
        return "\n".join(body_lines).strip(), footers

    return "\n".join(lines).strip(), {}


def parse_conventional_commit(message: str) -> ConventionalCommit:
    """Parse and validate a commit message or PR title against Conventional Commits 1.0.0."""
    stripped = message.strip("\r\n")
    if not stripped:
        raise ValueError("Commit message must not be empty.")

    lines = stripped.splitlines()
    header = lines[0]
    if not header.isascii():
        raise ValueError(f"Commit header must use ASCII English characters only: {header!r}")
    if len(header) > MAX_HEADER_LENGTH:
        raise ValueError(
            f"Commit header exceeds {MAX_HEADER_LENGTH} characters ({len(header)}): {header!r}"
        )

    match = COMMIT_HEADER_RE.fullmatch(header)
    if not match:
        raise ValueError(f"Commit header violates Conventional Commits 1.0.0: {header!r}")

    commit_type = match.group("type")
    if commit_type not in ALLOWED_COMMIT_TYPES:
        raise ValueError(
            f"Unsupported commit type {commit_type!r}; allowed: {sorted(ALLOWED_COMMIT_TYPES)}"
        )

    description = match.group("desc")
    if description[0].isupper():
        raise ValueError(f"Commit description must start lowercase: {description!r}")
    if description.endswith("."):
        raise ValueError(f"Commit description must not end with a period: {description!r}")

    if len(lines) > 1 and lines[1] != "":
        raise ValueError("Conventional Commits 1.0.0 requires one blank line after the header.")

    body, footers = _split_body_and_footers(lines[1:])
    breaking = bool(match.group("bang")) or any(
        token in {"BREAKING CHANGE", "BREAKING-CHANGE"} for token in footers
    )

    return ConventionalCommit(
        commit_type=commit_type,
        scope=match.group("scope"),
        description=description,
        breaking=breaking,
        body=body,
        footers=footers,
    )


def is_merge_commit(message: str) -> bool:
    """Return True when the message is a standard Git/GitHub merge commit."""
    first_line = message.strip().splitlines()[0] if message.strip() else ""
    return bool(MERGE_HEADER_RE.match(first_line))


def validate_commit_messages(
    messages: Sequence[str],
    *,
    ignore_merges: bool = True,
) -> list[ConventionalCommit]:
    """Validate a sequence of commit messages and return parsed ConventionalCommit objects."""
    parsed: list[ConventionalCommit] = []
    for raw in messages:
        if ignore_merges and is_merge_commit(raw):
            continue
        parsed.append(parse_conventional_commit(raw))
    return parsed


@dataclass(frozen=True, order=True)
class SemVer:
    """Semantic Versioning 2.0.0 core triple."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, tag: str) -> SemVer:
        """Parse a `vX.Y.Z` tag string into a SemVer instance."""
        match = SEMVER_RE.fullmatch(tag.strip())
        if not match:
            raise ValueError(f"Invalid SemVer tag {tag!r}; expected 'vMAJOR.MINOR.PATCH'.")
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
        )

    def format(self) -> str:
        """Format as `vMAJOR.MINOR.PATCH`."""
        return f"v{self.major}.{self.minor}.{self.patch}"


def derive_next_semver(
    latest_tag: str | None,
    commit_messages: Sequence[str],
    *,
    initial_version: str = "v0.1.0",
) -> SemVer:
    """Derive the next SemVer release from Conventional Commits since `latest_tag`."""
    parsed = validate_commit_messages(commit_messages, ignore_merges=True)
    if not parsed:
        raise ValueError("Cannot derive next SemVer: no non-merge commits found.")

    if latest_tag is None:
        return SemVer.parse(initial_version)

    current = SemVer.parse(latest_tag)
    has_breaking = any(commit.breaking for commit in parsed)
    has_feat = any(commit.commit_type == "feat" for commit in parsed)

    if current.major == 0:
        if has_breaking or has_feat:
            return SemVer(0, current.minor + 1, 0)
        return SemVer(0, current.minor, current.patch + 1)

    if has_breaking:
        return SemVer(current.major + 1, 0, 0)
    if has_feat:
        return SemVer(current.major, current.minor + 1, 0)
    return SemVer(current.major, current.minor, current.patch + 1)


@dataclass(frozen=True)
class OpenPullRequest:
    """Representation of an existing open pull request."""

    number: int
    head: str
    base: str
    title: str
    body: str = ""


@dataclass(frozen=True)
class PromotionAction:
    """Idempotent GitHub PR promotion plan with manual merge enforced."""

    action: Literal["create", "update", "noop"]
    head: str
    base: str
    title: str
    body: str
    create_base_from_ref: str | None = None
    existing_pr_number: int | None = None
    auto_merge: bool = False


def _resolve_pr_action(
    *,
    head: str,
    base: str,
    title: str,
    body: str,
    open_prs: Sequence[OpenPullRequest],
    create_base_from_ref: str | None = None,
) -> PromotionAction:
    """Return an idempotent create/update/noop PromotionAction without enabling auto-merge."""
    for pr in open_prs:
        if pr.head == head and pr.base == base:
            action: Literal["noop", "update"] = (
                "noop" if (pr.title == title and pr.body == body) else "update"
            )
            return PromotionAction(
                action=action,
                head=head,
                base=base,
                title=title,
                body=body,
                create_base_from_ref=create_base_from_ref,
                existing_pr_number=pr.number,
                auto_merge=False,
            )

    return PromotionAction(
        action="create",
        head=head,
        base=base,
        title=title,
        body=body,
        create_base_from_ref=create_base_from_ref,
        existing_pr_number=None,
        auto_merge=False,
    )


def plan_feature_to_develop_pr(
    feature_branch: str,
    commit_messages: Sequence[str],
    open_prs: Sequence[OpenPullRequest],
) -> PromotionAction:
    """Plan an idempotent `feature/* -> develop` PR requiring manual merge."""
    branch_check = validate_branch_name(feature_branch)
    if not branch_check.valid:
        raise ValueError(branch_check.reason)

    parsed_commits = validate_commit_messages(commit_messages, ignore_merges=True)
    if not parsed_commits:
        return PromotionAction(
            action="noop",
            head=feature_branch,
            base="develop",
            title="",
            body=f"No new commits on `{feature_branch}` ahead of `develop`.",
            auto_merge=False,
        )

    slug = feature_branch.removeprefix("feature/")
    if len(parsed_commits) == 1:
        only = parsed_commits[0]
        scope_part = f"({only.scope})" if only.scope else ""
        bang = "!" if only.breaking else ""
        title = f"{only.commit_type}{scope_part}{bang}: {only.description}"
    else:
        has_feat = any(commit.commit_type == "feat" for commit in parsed_commits)
        pr_type = "feat" if has_feat else parsed_commits[0].commit_type
        title = f"{pr_type}: merge feature {slug} into develop"

    if len(title) > MAX_HEADER_LENGTH:
        title = title[:MAX_HEADER_LENGTH].rstrip(" .")

    body = (
        f"Automated promotion PR for `{feature_branch}` after CI passed.\n\n"
        "Manual review and manual merge into `develop` are mandatory (auto-merge is disabled)."
    )
    return _resolve_pr_action(
        head=feature_branch,
        base="develop",
        title=title,
        body=body,
        open_prs=open_prs,
    )


def extract_release_version_from_commit(message: str) -> str | None:
    """Extract a `vX.Y.Z` release tag from a release merge or squash commit message."""
    for raw_line in message.splitlines():
        line = raw_line.strip()
        branch_match = _RELEASE_BRANCH_REF_RE.search(line)
        if branch_match:
            return SemVer.parse(branch_match.group(1)).format()
        title_match = _RELEASE_TITLE_RE.fullmatch(line)
        if title_match:
            return SemVer.parse(title_match.group(1)).format()
    return None


def plan_develop_to_release_pr(
    latest_main_tag: str | None,
    commits_since_main: Sequence[str],
    existing_release_branches: Sequence[str],
    open_prs: Sequence[OpenPullRequest],
    *,
    commits_since_release: Sequence[str] | None = None,
    merged_release_branches: Sequence[str] = (),
) -> PromotionAction:
    """Plan an idempotent `develop -> release/vX.Y.Z` PR requiring manual merge."""
    parsed_commits = validate_commit_messages(commits_since_main, ignore_merges=True)
    if not parsed_commits:
        return PromotionAction(
            action="noop",
            head="develop",
            base="",
            title="",
            body="No unreleased commits on develop since main.",
            auto_merge=False,
        )

    merged_set = {b for b in merged_release_branches if RELEASE_BRANCH_RE.fullmatch(b)}
    released_versions: list[SemVer] = []
    if latest_main_tag:
        released_versions.append(SemVer.parse(latest_main_tag))
    for merged_branch in merged_set:
        released_versions.append(SemVer.parse(merged_branch.removeprefix("release/")))
    effective_released = max(released_versions) if released_versions else None

    valid_releases = sorted(
        (
            b
            for b in existing_release_branches
            if RELEASE_BRANCH_RE.fullmatch(b)
            and b not in merged_set
            and (
                effective_released is None
                or SemVer.parse(b.removeprefix("release/")) > effective_released
            )
        ),
        key=lambda b: SemVer.parse(b.removeprefix("release/")),
    )
    if valid_releases:
        release_branch = valid_releases[-1]
        if commits_since_release is not None and not validate_commit_messages(
            commits_since_release, ignore_merges=True
        ):
            return PromotionAction(
                action="noop",
                head="develop",
                base=release_branch,
                title="",
                body=f"No new commits on `develop` ahead of `{release_branch}`.",
                auto_merge=False,
            )
        version_tag = release_branch.removeprefix("release/")
        create_from: str | None = None
    else:
        baseline_tag = effective_released.format() if effective_released else None
        next_version = derive_next_semver(baseline_tag, commits_since_main)
        version_tag = next_version.format()
        release_branch = f"release/{version_tag}"
        create_from = "main"

    title = f"chore(release): prepare release {version_tag}"
    body = (
        f"Automated promotion PR from `develop` to `{release_branch}` "
        "after `dev` checks passed.\n\n"
        "Manual review and manual merge into the release branch are mandatory."
    )
    return _resolve_pr_action(
        head="develop",
        base=release_branch,
        title=title,
        body=body,
        open_prs=open_prs,
        create_base_from_ref=create_from,
    )


def plan_release_to_main_pr(
    release_branch: str,
    open_prs: Sequence[OpenPullRequest],
) -> PromotionAction:
    """Plan an idempotent `release/vX.Y.Z -> main` PR requiring manual merge."""
    branch_check = validate_branch_name(release_branch, allow_release=True)
    if not branch_check.valid or branch_check.branch_type != "release":
        raise ValueError(
            f"Expected a valid release branch 'release/vX.Y.Z', got {release_branch!r}."
        )

    version_tag = release_branch.removeprefix("release/")
    title = f"chore(release): release {version_tag}"
    body = (
        f"Automated promotion PR from `{release_branch}` to `main` after homologation passed.\n\n"
        "Manual review and manual merge into `main` are mandatory."
    )
    return _resolve_pr_action(
        head=release_branch,
        base="main",
        title=title,
        body=body,
        open_prs=open_prs,
    )
