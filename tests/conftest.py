from __future__ import annotations

from pathlib import Path

import pytest

from redactor import review_workspace


@pytest.fixture(autouse=True)
def forbid_implicit_frontend_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must use fixture assets or an explicitly prepared real frontend."""

    def reject_command(*args: object, **kwargs: object) -> None:
        pytest.fail("Tests must not implicitly install or build frontend dependencies")

    monkeypatch.setattr(review_workspace, "run_frontend_command", reject_command)


@pytest.fixture
def synthetic_frontend_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Exercise backend asset packaging without building or installing a frontend.

    Only backend modules explicitly requesting this fixture use the placeholder.
    Browser and synthetic-demo integration tests continue to use the real build.
    """
    frontend = tmp_path / "synthetic_frontend"
    assets = frontend / "dist" / "assets"
    assets.mkdir(parents=True)
    (frontend / "dist" / "index.html").write_text(
        "<!doctype html><title>Synthetic backend-test frontend</title>"
        '<script type="module" src="/app/assets/synthetic.js"></script>',
        encoding="utf-8",
    )
    (assets / "synthetic.js").write_text(
        "// Placeholder asset for backend packaging tests, not a browser application.\n",
        encoding="utf-8",
    )
    (frontend / "package.json").write_text(
        '{"name":"synthetic-backend-test-frontend","private":true}\n', encoding="utf-8"
    )
    (frontend / "package-lock.json").write_text('{"lockfileVersion":3}\n', encoding="utf-8")
    write_assets = review_workspace.write_react_workspace_assets

    def copy_assets(
        package_root: Path,
        *,
        build_frontend: bool = True,
        frontend_root: Path | None = None,
    ) -> None:
        assert not build_frontend, "Backend tests must never install or build frontend dependencies"
        write_assets(
            package_root,
            build_frontend=False,
            frontend_root=frontend_root if frontend_root is not None else frontend,
        )

    monkeypatch.setattr(review_workspace, "write_react_workspace_assets", copy_assets)
    return frontend
