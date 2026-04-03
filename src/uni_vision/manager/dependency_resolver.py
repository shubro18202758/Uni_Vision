"""Dependency Conflict Resolver — LLM-driven pip version conflict auto-resolution.

When the Manager Agent (Gemma 4 E2B) dynamically pulls and installs packages
from the open internet, transitive dependency conflicts can arise.  This
module detects those conflicts (via ``pip check``) and uses the LLM to
reason about compatible version pins, then applies the fix automatically.

The resolver is invoked **after every provisioning batch** and runs in a
retry loop until the environment is clean or max attempts are exhausted.

Design principles:
  * ``pip check`` is the single source of truth for conflict detection.
  * The LLM receives the raw ``pip check`` output + installed versions and
    returns a structured JSON with exact ``pip install`` commands to fix.
  * Each fix attempt is logged and recorded so the CompatibilityMatrix can
    learn from empirical data.
  * The resolver is fully async and safe to call from the ReAct loop.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ── Prompt template ──────────────────────────────────────────────

_DEPENDENCY_RESOLUTION_PROMPT = """\
You are the dependency resolver for Uni_Vision — a real-time computer vision
pipeline.  After dynamically installing packages, ``pip check`` reported the
following version conflicts:

{pip_check_output}

Packages that were just installed (triggered this check):
{recently_installed}

Currently installed versions of relevant packages:
{installed_versions}

Your task:
1. Identify the root cause of each conflict.
2. Determine compatible version pins that satisfy ALL packages.
3. Prefer UPGRADING to the latest compatible version over downgrading.
4. Do NOT remove packages that are required by the pipeline.
5. Minimise the number of reinstall commands.

Respond in JSON (no markdown, no commentary):
{{
  "analysis": "brief explanation of the conflict root cause",
  "fix_commands": [
    "package_spec_1",
    "package_spec_2"
  ],
  "confidence": 0.9,
  "reasoning": "why these specific versions resolve the conflict"
}}

Where each entry in ``fix_commands`` is a pip package specifier to install,
e.g. ``"numpy==1.24.4"`` or ``"protobuf>=3.20,<4"``.  These will be passed
directly to ``pip install``.
"""


# ── Data types ───────────────────────────────────────────────────


@dataclass
class ConflictCheckResult:
    """Result of a ``pip check`` invocation."""

    has_conflicts: bool = False
    raw_output: str = ""
    conflicts: list[str] = field(default_factory=list)


@dataclass
class ResolutionAttempt:
    """Record of a single fix attempt."""

    attempt_number: int
    conflicts_before: list[str]
    llm_analysis: str = ""
    fix_commands: list[str] = field(default_factory=list)
    success: bool = False
    conflicts_after: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class DependencyResolutionReport:
    """Full report of the conflict resolution process."""

    triggered_by_packages: list[str]
    initial_conflicts: list[str]
    attempts: list[ResolutionAttempt] = field(default_factory=list)
    resolved: bool = False
    final_conflicts: list[str] = field(default_factory=list)


# ── Resolver ─────────────────────────────────────────────────────


class DependencyConflictResolver:
    """Detect and auto-resolve pip dependency version conflicts using LLM reasoning.

    Parameters
    ----------
    llm_client:
        Async LLM client that exposes ``.generate(prompt)`` — the same
        Ollama client used by the Manager Agent.
    max_attempts:
        Maximum resolution retry loops before giving up.
    pip_timeout_s:
        Timeout for each individual ``pip install`` fix command.
    """

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        max_attempts: int = 3,
        pip_timeout_s: float = 120.0,
    ) -> None:
        self._llm = llm_client
        self._max_attempts = max_attempts
        self._pip_timeout = pip_timeout_s

    # ── Public API ────────────────────────────────────────────────

    async def check_and_resolve(
        self,
        recently_installed: list[str],
    ) -> DependencyResolutionReport:
        """Run ``pip check``, and if conflicts exist, resolve them via LLM.

        Parameters
        ----------
        recently_installed:
            Package specs that were just installed (for context in LLM prompt).

        Returns
        -------
        DependencyResolutionReport with full audit trail.
        """
        report = DependencyResolutionReport(
            triggered_by_packages=list(recently_installed),
            initial_conflicts=[],
        )

        # Initial conflict check
        check = await self._run_pip_check()
        report.initial_conflicts = list(check.conflicts)

        if not check.has_conflicts:
            report.resolved = True
            log.info("pip_check_clean", packages=recently_installed)
            return report

        log.warning(
            "pip_conflicts_detected",
            num_conflicts=len(check.conflicts),
            packages=recently_installed,
        )

        # Resolution loop
        current_conflicts = check
        for attempt_num in range(1, self._max_attempts + 1):
            attempt = ResolutionAttempt(
                attempt_number=attempt_num,
                conflicts_before=list(current_conflicts.conflicts),
            )

            # Ask the LLM for a fix
            fix_commands, analysis = await self._ask_llm_for_fix(
                current_conflicts,
                recently_installed,
            )
            attempt.llm_analysis = analysis
            attempt.fix_commands = fix_commands

            if not fix_commands:
                attempt.error = "LLM returned no fix commands"
                report.attempts.append(attempt)
                log.warning(
                    "dependency_resolve_no_fix",
                    attempt=attempt_num,
                )
                break

            # Apply the fixes
            apply_ok = await self._apply_fixes(fix_commands)
            if not apply_ok:
                attempt.error = "One or more fix commands failed"
                report.attempts.append(attempt)
                log.warning(
                    "dependency_fix_apply_failed",
                    attempt=attempt_num,
                    commands=fix_commands,
                )
                # Continue to next attempt — the partial fix may have
                # changed the conflict landscape
                current_conflicts = await self._run_pip_check()
                attempt.conflicts_after = list(current_conflicts.conflicts)
                attempt.success = not current_conflicts.has_conflicts
                continue

            # Re-check
            current_conflicts = await self._run_pip_check()
            attempt.conflicts_after = list(current_conflicts.conflicts)
            attempt.success = not current_conflicts.has_conflicts
            report.attempts.append(attempt)

            if not current_conflicts.has_conflicts:
                report.resolved = True
                log.info(
                    "dependency_conflicts_resolved",
                    attempts=attempt_num,
                    packages=recently_installed,
                )
                break

            log.info(
                "dependency_resolve_retry",
                attempt=attempt_num,
                remaining_conflicts=len(current_conflicts.conflicts),
            )

        report.final_conflicts = list(current_conflicts.conflicts)
        if not report.resolved:
            log.error(
                "dependency_conflicts_unresolved",
                attempts=len(report.attempts),
                remaining=report.final_conflicts,
            )

        return report

    async def check_only(self) -> ConflictCheckResult:
        """Run ``pip check`` without attempting resolution.

        Useful for health checks and monitoring.
        """
        return await self._run_pip_check()

    # ── Internal helpers ──────────────────────────────────────────

    async def _run_pip_check(self) -> ConflictCheckResult:
        """Execute ``pip check`` and parse the output."""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pip",
                "check",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30.0,
            )
            output = stdout.decode().strip()

            if proc.returncode == 0 and "No broken requirements" in output:
                return ConflictCheckResult(has_conflicts=False, raw_output=output)

            # Parse conflict lines (one per line)
            conflict_lines = [
                line.strip() for line in output.splitlines() if line.strip() and not line.startswith("WARNING")
            ]

            return ConflictCheckResult(
                has_conflicts=bool(conflict_lines),
                raw_output=output,
                conflicts=conflict_lines,
            )
        except asyncio.TimeoutError:
            log.error("pip_check_timeout")
            return ConflictCheckResult(
                has_conflicts=False,
                raw_output="pip check timed out",
            )
        except Exception as exc:
            log.error("pip_check_error", error=str(exc))
            return ConflictCheckResult(
                has_conflicts=False,
                raw_output=f"pip check error: {exc}",
            )

    async def _get_installed_versions(
        self,
        conflict_lines: list[str],
    ) -> str:
        """Get installed versions of packages mentioned in conflicts."""
        # Extract package names from conflict lines
        packages: set[str] = set()
        for line in conflict_lines:
            # pip check output format: "pkg X.Y has requirement dep>=Z, but you have dep A.B"
            words = line.split()
            for w in words:
                # Heuristic: package names are lowercase, may have hyphens
                clean = w.strip(".,()\"'")
                if (
                    clean
                    and clean[0].isalpha()
                    and not clean.startswith(("has", "but", "you", "have", "which", "requires", "not", "installed"))
                ):
                    packages.add(clean)

        if not packages:
            return "(unable to extract package names)"

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pip",
                "show",
                *list(packages)[:20],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            # Filter to just Name + Version lines for brevity
            lines = []
            for line in stdout.decode().splitlines():
                if line.startswith(("Name:", "Version:", "Requires:", "---")):
                    lines.append(line)
            return "\n".join(lines) if lines else stdout.decode()[:2000]
        except Exception:
            return "(unable to retrieve installed versions)"

    async def _ask_llm_for_fix(
        self,
        check_result: ConflictCheckResult,
        recently_installed: list[str],
    ) -> tuple[list[str], str]:
        """Ask the LLM for fix commands.

        Returns (fix_commands, analysis_text).
        Falls back to heuristic resolution if no LLM is available.
        """
        if self._llm is None:
            return self._heuristic_fix(check_result), "heuristic (no LLM available)"

        installed_versions = await self._get_installed_versions(
            check_result.conflicts,
        )

        prompt = _DEPENDENCY_RESOLUTION_PROMPT.format(
            pip_check_output=check_result.raw_output,
            recently_installed=", ".join(recently_installed),
            installed_versions=installed_versions,
        )

        try:
            response = await self._llm.generate(prompt)
            parsed = json.loads(response)

            fix_commands = parsed.get("fix_commands", [])
            analysis = parsed.get("analysis", "")

            # Validate: each command must be a plausible pip spec
            valid_commands = [cmd for cmd in fix_commands if isinstance(cmd, str) and cmd.strip() and len(cmd) < 200]

            if not valid_commands:
                log.warning(
                    "llm_returned_no_valid_fixes",
                    raw_response=response[:500],
                )
                return self._heuristic_fix(check_result), "heuristic fallback (LLM gave no valid commands)"

            log.info(
                "llm_dependency_fix_proposed",
                commands=valid_commands,
                confidence=parsed.get("confidence", "?"),
            )

            return valid_commands, analysis

        except json.JSONDecodeError:
            log.warning("llm_dependency_fix_json_error", raw=response[:300] if "response" in dir() else "")
            return self._heuristic_fix(check_result), "heuristic fallback (LLM JSON parse error)"
        except Exception as exc:
            log.warning("llm_dependency_fix_error", error=str(exc))
            return self._heuristic_fix(check_result), f"heuristic fallback (LLM error: {exc})"

    @staticmethod
    def _heuristic_fix(check_result: ConflictCheckResult) -> list[str]:
        """Best-effort heuristic when LLM is unavailable.

        Parses ``pip check`` output to find required version specs and
        generates ``pip install`` commands for the required versions.
        """
        fix_commands: list[str] = []
        for line in check_result.conflicts:
            # Format: "pkg X.Y has requirement dep>=Z, but you have dep A.B"
            # or:    "pkg X.Y requires dep>=Z, which is not installed"
            if "has requirement" in line:
                parts = line.split("has requirement")
                if len(parts) == 2:
                    req_part = parts[1].split(",")[0].strip()
                    if req_part:
                        fix_commands.append(req_part)
            elif "requires" in line and "not installed" in line:
                parts = line.split("requires")
                if len(parts) == 2:
                    req_part = parts[1].split(",")[0].strip()
                    if req_part:
                        fix_commands.append(req_part)
        return fix_commands

    async def _apply_fixes(self, fix_commands: list[str]) -> bool:
        """Run ``pip install`` for each fix command.

        Returns True if ALL commands succeeded.
        """
        all_ok = True
        for spec in fix_commands:
            log.info("dependency_fix_install", package=spec)
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    spec,
                    "--quiet",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._pip_timeout,
                )
                if proc.returncode != 0:
                    log.error(
                        "dependency_fix_failed",
                        package=spec,
                        stderr=stderr.decode()[:300],
                    )
                    all_ok = False
                else:
                    log.info("dependency_fix_installed", package=spec)
            except asyncio.TimeoutError:
                log.error("dependency_fix_timeout", package=spec)
                all_ok = False
            except Exception as exc:
                log.error("dependency_fix_error", package=spec, error=str(exc))
                all_ok = False
        return all_ok
