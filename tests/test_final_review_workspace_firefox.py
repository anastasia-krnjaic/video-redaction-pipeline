from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from redactor import review_workspace

SESSION_READY_JS = (
    "() => document.querySelector('#sessionBanner')?.textContent.includes('Local session')"
)

pytestmark = pytest.mark.skipif(
    os.environ.get("REDACTOR_RUN_FIREFOX_E2E") != "1",
    reason="set REDACTOR_RUN_FIREFOX_E2E=1 to run the real Firefox workspace smoke",
)


def record_step(report: dict[str, Any], name: str, **details: Any) -> None:
    report.setdefault("steps", []).append({"details": details, "name": name, "ok": True})


def write_report(package: Path, report: dict[str, Any]) -> None:
    out = package / "browser_smoke" / "browser_smoke_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task_card_selector(task_id: str) -> str:
    return f'#case-day.active .task-card[data-task-id="{task_id}"]'


def marker_selector(task_id: str) -> str:
    return f'#case-day.active .marker[data-task-id="{task_id}"]'


def test_final_review_workspace_firefox_entity_timeline_smoke() -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    package = Path(
        os.environ.get(
            "REDACTOR_FIREFOX_WORKSPACE",
            "artifacts/final_pipeline_package_v2_3_3",
        )
    )
    if not package.exists():
        pytest.skip(f"workspace package does not exist: {package}")

    report: dict[str, Any] = {
        "browser": "firefox",
        "package": package.as_posix(),
        "passed": False,
        "schema_version": "final_review_workspace_browser_smoke.v1",
        "steps": [],
    }
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
                page.wait_for_function("() => Boolean(DATA?.cases?.day)", timeout=8000)
                record_step(report, "app_load")

                page.get_by_role("button", name="Start review").click()
                page.wait_for_function(SESSION_READY_JS, timeout=5000)
                assert "Local session" in page.locator("#sessionBanner").inner_text()
                record_step(report, "start_anonymous_review")

                page.get_by_role("button", name="Entities").click()
                page.locator("#entitiesModal.open").wait_for()
                entity_count = page.locator("#entitiesModal .entity-card").count()
                assert entity_count > 0
                assert "Unresolved evidence" in page.locator("#entitiesModal").inner_text()
                record_step(report, "entity_modal", entity_count=entity_count)

                representative_frame = page.locator("#entitiesModal .entity-card").first.evaluate(
                    "node => Number((findEntity('day', node.dataset.entityId) || {})"
                    ".representative_case_frame)"
                )
                page.locator("#entitiesModal .entity-card img").first.click()
                page.wait_for_function(
                    "(frame) => state.frame.day === frame",
                    arg=representative_frame,
                    timeout=3000,
                )
                record_step(report, "representative_thumbnail_jump", frame=representative_frame)

                overlap_pair = page.evaluate(
                    """() => {
                      const rows = effectiveEntities('day');
                      for (let i = 0; i < rows.length; i++) {
                        for (let j = i + 1; j < rows.length; j++) {
                          const sameCategory = rows[i].category === rows[j].category;
                          if (sameCategory && entitiesOverlap(rows[i], rows[j])) {
                            return [rows[i].entity_id, rows[j].entity_id];
                          }
                        }
                      }
                      return null;
                    }"""
                )
                if overlap_pair:
                    page.get_by_role("button", name="Entities").click()
                    page.once("dialog", lambda dialog: dialog.accept(overlap_pair[1]))
                    merge_selector = (
                        f'#entitiesModal .entity-card[data-entity-id="{overlap_pair[0]}"] '
                        '[data-action="entity-merge"]'
                    )
                    page.locator(merge_selector).click()
                    page.wait_for_function(
                        "() => document.querySelector('#sessionBanner')"
                        "?.textContent.includes('Merge rejected')",
                        timeout=3000,
                    )
                    record_step(report, "simultaneous_merge_rejection", pair=overlap_pair)

                page.keyboard.press("Escape")
                page.wait_for_function(
                    "() => !document.querySelector('#entitiesModal')?.classList.contains('open')",
                    timeout=3000,
                )

                first_marker = page.locator("#case-day.active .marker").first
                first_task_id = first_marker.get_attribute("data-task-id")
                first_task_frame = page.evaluate(
                    """(taskId) => {
                      const t = allTasks('day').find(x => x.review_task_id === taskId);
                      return t.source_frame_start - caseData('day').source_frame_start;
                    }""",
                    first_task_id,
                )
                first_marker.click()
                page.wait_for_function(
                    "(expected) => state.frame.day === expected",
                    arg=first_task_frame,
                    timeout=3000,
                )
                assert page.locator(task_card_selector(first_task_id)).count() == 1
                record_step(
                    report,
                    "timeline_marker_to_task",
                    task_id=first_task_id,
                    frame=first_task_frame,
                )

                page.locator(task_card_selector(first_task_id)).click()
                page.wait_for_function(
                    "(expected) => state.frame.day === expected",
                    arg=first_task_frame,
                    timeout=3000,
                )
                assert "active" in page.locator(marker_selector(first_task_id)).get_attribute(
                    "class"
                )
                record_step(report, "task_to_timeline_marker", task_id=first_task_id)

                video = page.locator("#case-day.active video").first
                video_src = video.evaluate("node => node.currentSrc")
                current_time = video.evaluate("node => node.currentTime")
                with page.expect_response("**/api/decisions"):
                    page.locator(
                        task_card_selector(first_task_id) + " .decision-choice"
                    ).first.click()
                assert video.evaluate("node => node.currentSrc") == video_src
                assert video.evaluate("node => node.currentTime") >= current_time
                page.wait_for_function(
                    "(taskId) => document.querySelector("
                    '`#case-day.active .marker[data-task-id="${taskId}"]`'
                    ")?.className.includes('resolved')",
                    arg=first_task_id,
                    timeout=3000,
                )
                assert "resolved" in page.locator(marker_selector(first_task_id)).get_attribute(
                    "class"
                )
                record_step(
                    report,
                    "correct_decision_green_marker_no_video_reload",
                    task_id=first_task_id,
                )

                needs_task_id = page.evaluate("() => firstOpenTask('day').review_task_id")
                with page.expect_response("**/api/decisions"):
                    page.locator(
                        task_card_selector(needs_task_id) + " .decision-needs-correction"
                    ).click()
                assert "blocking" in page.locator(marker_selector(needs_task_id)).get_attribute(
                    "class"
                )
                record_step(report, "needs_correction_bucket_red_marker", task_id=needs_task_id)

                unsure_task_id = page.evaluate("() => firstOpenTask('day').review_task_id")
                with page.expect_response("**/api/decisions"):
                    page.locator(task_card_selector(unsure_task_id) + " .decision-unsure").click()
                assert "partial" in page.locator(marker_selector(unsure_task_id)).get_attribute(
                    "class"
                )
                record_step(report, "unsure_bucket_orange_marker", task_id=unsure_task_id)

                page.locator(marker_selector(first_task_id)).click()
                assert page.evaluate("state.selectedTask.day") == first_task_id
                record_step(report, "completed_task_selection")

                page.locator("#case-day.active video").click()
                page.wait_for_function(
                    "() => !document.querySelector('#case-day.active video').paused", timeout=5000
                )
                page.locator("#case-day.active video").click()
                page.wait_for_function(
                    "() => document.querySelector('#case-day.active video').paused", timeout=5000
                )
                record_step(report, "video_click_play_pause")

                page.keyboard.press("ArrowRight")
                frame_after_right = page.evaluate("state.frame.day")
                page.keyboard.press("ArrowLeft")
                assert page.evaluate("state.frame.day") == frame_after_right - 1
                record_step(report, "single_frame_keys")

                page.locator("#case-day.active [data-action='fullscreen']").click()
                page.wait_for_function(
                    "() => document.fullscreenElement && "
                    "document.querySelector('#case-day.active .frame-hud')"
                    "?.offsetParent !== null",
                    timeout=5000,
                )
                assert "Source frame" in page.locator("#case-day.active .frame-hud").inner_text()
                page.keyboard.press("Escape")
                record_step(report, "fullscreen_hud")

                session_id = page.evaluate("state.session?.session_id")
                page.reload(wait_until="domcontentloaded")
                page.wait_for_function(SESSION_READY_JS, timeout=5000)
                assert page.evaluate("state.session?.session_id") == session_id
                assert session_id in page.locator("#sessionBanner").inner_text()
                assert page.evaluate("(taskId) => Boolean(state.decisions[taskId])", first_task_id)
                record_step(report, "reload_state_restoration")
                report["passed"] = True
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        write_report(package, report)
