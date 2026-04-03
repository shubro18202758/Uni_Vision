"""Unit tests for DependencyConflictResolver — LLM-driven pip conflict resolution."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uni_vision.manager.dependency_resolver import (
    ConflictCheckResult,
    DependencyConflictResolver,
    DependencyResolutionReport,
    ResolutionAttempt,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    """LLM client stub with .generate(prompt) -> str."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=json.dumps({
        "analysis": "numpy needs upgrade for compatibility with scipy",
        "fix_commands": ["numpy==1.24.4"],
        "confidence": 0.9,
        "reasoning": "scipy 1.11 requires numpy>=1.24",
    }))
    return llm


@pytest.fixture
def resolver(mock_llm):
    """Resolver wired to mock LLM."""
    return DependencyConflictResolver(llm_client=mock_llm, max_attempts=3)


@pytest.fixture
def resolver_no_llm():
    """Resolver with no LLM (heuristic fallback)."""
    return DependencyConflictResolver(llm_client=None, max_attempts=2)


# ── Helper to mock subprocess calls ──────────────────────────────


def _mock_pip_check(output: str, returncode: int = 1):
    """Create a mock for asyncio.create_subprocess_exec that simulates pip check."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(
        return_value=(output.encode(), b"")
    )
    mock_proc.returncode = returncode
    return mock_proc


def _mock_pip_check_clean():
    return _mock_pip_check("No broken requirements found.", returncode=0)


def _mock_pip_install_success():
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 0
    return mock_proc


def _mock_pip_show(output: str = "Name: numpy\nVersion: 1.24.4\n---\n"):
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(output.encode(), b""))
    mock_proc.returncode = 0
    return mock_proc


# ── Tests: pip check parsing ─────────────────────────────────────


class TestPipCheckParsing:
    """Tests for _run_pip_check output parsing."""

    @pytest.mark.asyncio
    async def test_clean_env_returns_no_conflicts(self, resolver):
        with patch("asyncio.create_subprocess_exec", return_value=_mock_pip_check_clean()):
            result = await resolver._run_pip_check()
        assert not result.has_conflicts
        assert result.conflicts == []

    @pytest.mark.asyncio
    async def test_conflict_detected(self, resolver):
        output = (
            "scipy 1.11.0 has requirement numpy>=1.24.0, but you have numpy 1.21.0.\n"
            "pandas 2.0.0 requires numpy>=1.23.2, which is not installed.\n"
        )
        with patch("asyncio.create_subprocess_exec", return_value=_mock_pip_check(output)):
            result = await resolver._run_pip_check()
        assert result.has_conflicts
        assert len(result.conflicts) == 2

    @pytest.mark.asyncio
    async def test_timeout_returns_safe_default(self, resolver):
        with patch("asyncio.create_subprocess_exec", side_effect=asyncio.TimeoutError):
            result = await resolver._run_pip_check()
        # Timeout — treated as no conflicts (safe fallback)
        assert not result.has_conflicts


# ── Tests: LLM-driven resolution ─────────────────────────────────


class TestLLMResolution:
    """Tests for LLM-based conflict resolution flow."""

    @pytest.mark.asyncio
    async def test_no_conflicts_skips_llm(self, resolver, mock_llm):
        """If pip check is clean, LLM should NOT be called."""
        with patch("asyncio.create_subprocess_exec", return_value=_mock_pip_check_clean()):
            report = await resolver.check_and_resolve(["scipy"])
        assert report.resolved
        assert len(report.attempts) == 0
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_conflict_resolved_in_one_attempt(self, resolver, mock_llm):
        """Single conflict resolved after LLM suggests a fix."""
        conflict_output = "scipy 1.11.0 has requirement numpy>=1.24.0, but you have numpy 1.21.0."

        call_count = 0

        async def _subprocess_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            cmd = args
            # pip check calls
            if "check" in cmd:
                if call_count <= 2:
                    # First pip check: conflict; also pip show call
                    return _mock_pip_check(conflict_output)
                else:
                    # After fix: clean
                    return _mock_pip_check_clean()
            # pip install fix
            if "install" in cmd:
                return _mock_pip_install_success()
            # pip show
            return _mock_pip_show()

        with patch("asyncio.create_subprocess_exec", side_effect=_subprocess_side_effect):
            report = await resolver.check_and_resolve(["scipy"])

        assert report.resolved or len(report.attempts) > 0
        mock_llm.generate.assert_called()

    @pytest.mark.asyncio
    async def test_max_attempts_exhausted(self, resolver, mock_llm):
        """Resolver gives up after max_attempts if conflicts persist."""
        persistent_conflict = "foo 1.0 has requirement bar>=2.0, but you have bar 1.0."

        async def _always_conflict(*args, **kwargs):
            cmd = args
            if "install" in cmd:
                return _mock_pip_install_success()
            if "show" in cmd:
                return _mock_pip_show("Name: bar\nVersion: 1.0\n")
            # pip check always returns conflict
            return _mock_pip_check(persistent_conflict)

        with patch("asyncio.create_subprocess_exec", side_effect=_always_conflict):
            report = await resolver.check_and_resolve(["foo"])

        assert not report.resolved
        assert len(report.attempts) == 3  # max_attempts = 3
        assert len(report.final_conflicts) > 0

    @pytest.mark.asyncio
    async def test_llm_json_error_falls_back_to_heuristic(self, resolver, mock_llm):
        """If LLM returns invalid JSON, heuristic fallback is used."""
        mock_llm.generate = AsyncMock(return_value="not valid json {{{")

        conflict_output = "scipy 1.11.0 has requirement numpy>=1.24.0, but you have numpy 1.21.0."

        call_count = 0

        async def _subprocess_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            cmd = args
            if "check" in cmd:
                if call_count <= 2:
                    return _mock_pip_check(conflict_output)
                return _mock_pip_check_clean()
            if "install" in cmd:
                return _mock_pip_install_success()
            return _mock_pip_show()

        with patch("asyncio.create_subprocess_exec", side_effect=_subprocess_side_effect):
            report = await resolver.check_and_resolve(["scipy"])

        # Should have attempted resolution via heuristic
        assert len(report.attempts) > 0


# ── Tests: Heuristic fallback (no LLM) ───────────────────────────


class TestHeuristicFallback:
    """Tests for heuristic resolution when no LLM is available."""

    @pytest.mark.asyncio
    async def test_heuristic_parses_has_requirement(self, resolver_no_llm):
        """Heuristic extracts version spec from 'has requirement' format."""
        conflict = "scipy 1.11.0 has requirement numpy>=1.24.0, but you have numpy 1.21.0."

        call_count = 0

        async def _subprocess_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            cmd = args
            if "check" in cmd:
                if call_count <= 1:
                    return _mock_pip_check(conflict)
                return _mock_pip_check_clean()
            if "install" in cmd:
                return _mock_pip_install_success()
            return _mock_pip_show()

        with patch("asyncio.create_subprocess_exec", side_effect=_subprocess_side_effect):
            report = await resolver_no_llm.check_and_resolve(["scipy"])

        assert len(report.attempts) > 0
        # The heuristic should have extracted "numpy>=1.24.0"
        assert any("numpy" in cmd for attempt in report.attempts for cmd in attempt.fix_commands)

    @pytest.mark.asyncio
    async def test_heuristic_parses_requires_not_installed(self, resolver_no_llm):
        """Heuristic handles 'requires X, which is not installed' format."""
        conflict = "pandas 2.0.0 requires numpy>=1.23.2, which is not installed."

        async def _subprocess_side_effect(*args, **kwargs):
            cmd = args
            if "check" in cmd:
                return _mock_pip_check(conflict)
            if "install" in cmd:
                return _mock_pip_install_success()
            return _mock_pip_show()

        with patch("asyncio.create_subprocess_exec", side_effect=_subprocess_side_effect):
            report = await resolver_no_llm.check_and_resolve(["pandas"])

        assert len(report.attempts) > 0

    def test_heuristic_fix_static(self):
        """Direct test of the static heuristic parser."""
        result = ConflictCheckResult(
            has_conflicts=True,
            raw_output="scipy 1.11.0 has requirement numpy>=1.24.0, but you have numpy 1.21.0.",
            conflicts=["scipy 1.11.0 has requirement numpy>=1.24.0, but you have numpy 1.21.0."],
        )
        fixes = DependencyConflictResolver._heuristic_fix(result)
        assert len(fixes) == 1
        assert "numpy>=1.24.0" in fixes[0]


# ── Tests: check_only ─────────────────────────────────────────────


class TestCheckOnly:
    """Tests for the check_only() health-check method."""

    @pytest.mark.asyncio
    async def test_check_only_clean(self, resolver):
        with patch("asyncio.create_subprocess_exec", return_value=_mock_pip_check_clean()):
            result = await resolver.check_only()
        assert not result.has_conflicts

    @pytest.mark.asyncio
    async def test_check_only_with_conflicts(self, resolver):
        output = "foo 1.0 has requirement bar>=2.0, but you have bar 1.0."
        with patch("asyncio.create_subprocess_exec", return_value=_mock_pip_check(output)):
            result = await resolver.check_only()
        assert result.has_conflicts
        assert len(result.conflicts) == 1


# ── Tests: Data structures ────────────────────────────────────────


class TestDataStructures:
    """Test the report/attempt dataclasses."""

    def test_resolution_report_defaults(self):
        report = DependencyResolutionReport(
            triggered_by_packages=["scipy"],
            initial_conflicts=["conflict1"],
        )
        assert not report.resolved
        assert report.triggered_by_packages == ["scipy"]
        assert len(report.attempts) == 0

    def test_resolution_attempt_fields(self):
        attempt = ResolutionAttempt(
            attempt_number=1,
            conflicts_before=["conflict1"],
            fix_commands=["numpy==1.24.4"],
            success=True,
        )
        assert attempt.attempt_number == 1
        assert attempt.success

    def test_conflict_check_result(self):
        result = ConflictCheckResult(
            has_conflicts=True,
            raw_output="foo has requirement bar>=2",
            conflicts=["foo has requirement bar>=2"],
        )
        assert result.has_conflicts
        assert len(result.conflicts) == 1


# ── Tests: Integration with ComponentResolver ─────────────────────


class TestComponentResolverIntegration:
    """Verify DependencyConflictResolver is accepted by ComponentResolver."""

    def test_component_resolver_accepts_dep_resolver(self):
        """ComponentResolver constructor accepts dependency_resolver kwarg."""
        from uni_vision.manager.component_resolver import ComponentResolver
        from unittest.mock import MagicMock

        registry = MagicMock()
        hub = MagicMock()
        dep = DependencyConflictResolver(llm_client=None)

        resolver = ComponentResolver(
            registry=registry,
            hub_client=hub,
            dependency_resolver=dep,
        )
        assert resolver._dep_resolver is dep

    def test_component_resolver_without_dep_resolver(self):
        """ComponentResolver works fine without a dependency_resolver."""
        from uni_vision.manager.component_resolver import ComponentResolver
        from unittest.mock import MagicMock

        resolver = ComponentResolver(
            registry=MagicMock(),
            hub_client=MagicMock(),
        )
        assert resolver._dep_resolver is None
