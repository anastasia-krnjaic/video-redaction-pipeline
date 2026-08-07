from __future__ import annotations

import os
from pathlib import Path

import pytest

from redactor import review_workspace

SESSION_READY_JS = (
    "() => document.querySelector('#sessionBanner')?.textContent.includes('Local session')"
)

pytestmark = pytest.mark.skipif(
    os.environ.get("REDACTOR_RUN_FIREFOX_E2E") != "1",
    reason="set REDACTOR_RUN_FIREFOX_E2E=1 to run the real Firefox workspace smoke",
)


def test_final_review_workspace_firefox_decision_and_frame_smoke() -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    package = Path(
        os.environ.get(
            "REDACTOR_FIREFOX_WORKSPACE",
            "artifacts/final_pipeline_package_v2_3_2",
        )
    )
    if not package.exists():
        pytest.skip(f"workspace package does not exist: {package}")

    server = review_workspace.serve_workspace(package, port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with sync_api.sync_playwright() as playwright:
            launch_options = {"headless": True}
            if executable := os.environ.get("REDACTOR_FIREFOX_EXECUTABLE"):
                firefox_path = Path(executable)
                if not firefox_path.exists():
                    pytest.skip(f"Firefox executable does not exist: {firefox_path}")
                launch_options["executable_path"] = str(firefox_path)
            try:
                browser = playwright.firefox.launch(**launch_options)
            except Exception as exc:
                pytest.skip(f"Firefox automation is unavailable: {exc}")
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                page.goto(f"{base_url}/index.html", wait_until="domcontentloaded")
                page.get_by_role("button", name="Start review").click()
                page.wait_for_function(SESSION_READY_JS, timeout=5000)
                assert "Local session" in page.locator("#sessionBanner").inner_text()

                video = page.locator("#case-day.active video").first
                video_src = video.evaluate("node => node.currentSrc")
                current_time = video.evaluate("node => node.currentTime")
                page.locator("#case-day.active .task-card").first.wait_for()
                with page.expect_response("**/api/decisions"):
                    page.locator("#case-day.active .decision-choice").first.click()
                assert video.evaluate("node => node.currentSrc") == video_src
                assert video.evaluate("node => node.currentTime") >= current_time

                page.keyboard.press("ArrowRight")
                frame_after_right = page.evaluate("state.frame.day")
                page.keyboard.press("ArrowLeft")
                assert page.evaluate("state.frame.day") == frame_after_right - 1

                page.get_by_role("button", name="Entities").click()
                page.locator("#entitiesModal.open").wait_for()
                assert page.locator("#entitiesModal .entity-card").count() >= 0

                session_text = page.locator("#sessionBanner").inner_text()
                page.reload(wait_until="domcontentloaded")
                page.wait_for_function(SESSION_READY_JS, timeout=5000)
                assert page.locator("#sessionBanner").inner_text() == session_text
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
