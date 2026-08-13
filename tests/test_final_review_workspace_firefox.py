from __future__ import annotations

import json
import os
import re
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


def migration_session_id(package: Path) -> str | None:
    report_path = package / "review_state_migration_report.json"
    if not report_path.exists():
        report_path = package / "review_v2_final_migration_report.json"
    if not report_path.exists():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("migration_session_id", "session_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.startswith("session_"):
            return value
    migration = payload.get("migration")
    if isinstance(migration, dict):
        value = migration.get("session_id")
        if isinstance(value, str) and value.startswith("session_"):
            return value
    return None


def task_card_selector(case_id: str, task_id: str) -> str:
    return f'.case-panel[data-case-id="{case_id}"] .task-card[data-task-id="{task_id}"]'


def active_case_panel(page: Any, case_id: str) -> Any:
    panel = page.locator(f'.case-panel[data-case-id="{case_id}"]')
    panel.wait_for(state="visible", timeout=10_000)
    return panel


def click_bucket(page: Any, case_id: str, bucket: str) -> None:
    panel = active_case_panel(page, case_id)
    panel.locator(".filter-row").get_by_role(
        "button", name=re.compile(rf"^{re.escape(bucket)}\b")
    ).click()


def bucket_names(panel: Any) -> list[str]:
    names = []
    for raw in panel.locator(".filter-row button").all_inner_texts():
        names.append(re.sub(r"\s+\d+\s*$", "", raw.strip()).strip())
    return names


def test_final_review_workspace_firefox_entity_timeline_smoke() -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    package = Path(
        os.environ.get(
            "REDACTOR_FIREFOX_WORKSPACE",
            "artifacts/final_pipeline_package_v2_3_4",
        )
    )
    if not package.exists():
        pytest.skip(f"workspace package does not exist: {package}")
    session_id = migration_session_id(package)
    if not session_id:
        pytest.skip(f"workspace package does not include a migration session id: {package}")
    if not (package / "sessions" / session_id).exists():
        pytest.skip(f"workspace package does not include migration session state: {session_id}")

    report: dict[str, Any] = {
        "browser": "firefox",
        "package": package.as_posix(),
        "passed": False,
        "schema_version": "final_review_workspace_browser_smoke.v1",
        "session_id": session_id,
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
                page.goto(
                    f"{base_url}/app/day?session_id={session_id}",
                    wait_until="domcontentloaded",
                )
                day_panel = active_case_panel(page, "day")
                page.wait_for_function(SESSION_READY_JS, timeout=10_000)
                assert "Local session" in page.locator("#sessionBanner").inner_text()
                assert session_id in page.locator("#sessionBanner").inner_text()
                assert "Unsure" not in page.locator("body").inner_text()
                assert "Uncertain" not in page.locator("body").inner_text()
                record_step(report, "migration_session_app_load")

                assert bucket_names(day_panel) == [
                    "Open",
                    "Needs correction",
                    "Correct",
                    "Not sensitive",
                ]
                assert "Waiting" not in bucket_names(day_panel)
                assert "Completed" not in bucket_names(day_panel)
                record_step(report, "canonical_queue_buckets")

                duplicate_task_id = "license_plate_plate_review_0005"
                click_bucket(page, "day", "Correct")
                day_duplicate = page.locator(task_card_selector("day", duplicate_task_id))
                day_duplicate.wait_for(state="visible", timeout=5_000)
                assert (
                    day_duplicate.locator(".task-state-row .badge").first.inner_text()
                    == "Correct"
                )
                record_step(report, "day_duplicate_completed", task_id=duplicate_task_id)

                day_panel.get_by_role("button", name="Entities").click()
                entity_dialog = page.get_by_role("dialog", name="Review objects")
                entity_dialog.wait_for(state="visible", timeout=5_000)
                entity_dialog.get_by_role("button", name="All").click()
                entity_count = entity_dialog.locator(".entity-card").count()
                assert entity_count > 0
                assert entity_dialog.locator(".entity-card img").count() > 0
                assert entity_dialog.locator(".policy-segment").count() > 0
                record_step(report, "entity_modal_assets", entity_count=entity_count)
                entity_dialog.get_by_role("button", name="Close").last.click()

                page.get_by_role("button", name="Night").click()
                night_panel = active_case_panel(page, "night")
                assert "night" in night_panel.locator(".case-header").inner_text().lower()
                assert bucket_names(night_panel) == [
                    "Open",
                    "Needs correction",
                    "Correct",
                    "Not sensitive",
                ]
                click_bucket(page, "night", "Open")
                night_duplicate = page.locator(task_card_selector("night", duplicate_task_id))
                night_duplicate.wait_for(state="visible", timeout=5_000)
                assert (
                    night_duplicate.locator(".task-state-row .badge").first.inner_text()
                    == "Open"
                )
                action_labels = [
                    item.strip()
                    for item in night_duplicate.locator(".task-actions button").all_inner_texts()
                ]
                assert action_labels == [
                    "Correct",
                    "Needs correction",
                    "Not sensitive",
                ]
                assert "WHY THIS WAS FLAGGED" in night_duplicate.inner_text()
                assert "WHAT TO CHECK" in night_duplicate.inner_text()
                record_step(
                    report,
                    "night_duplicate_open_and_three_actions",
                    task_id=duplicate_task_id,
                )

                with page.expect_response("**/api/v1/decisions"):
                    night_duplicate.get_by_role("button", name=re.compile(r"^Correct$")).click()
                click_bucket(page, "night", "Correct")
                night_completed = page.locator(task_card_selector("night", duplicate_task_id))
                night_completed.wait_for(state="visible", timeout=5_000)
                assert (
                    night_completed.locator(".task-state-row .badge").first.inner_text()
                    == "Correct"
                )
                record_step(report, "night_duplicate_decision_isolated", task_id=duplicate_task_id)

                page.get_by_role("button", name="Day").click()
                active_case_panel(page, "day")
                click_bucket(page, "day", "Correct")
                day_duplicate = page.locator(task_card_selector("day", duplicate_task_id))
                day_duplicate.wait_for(state="visible", timeout=5_000)
                assert (
                    day_duplicate.locator(".task-state-row .badge").first.inner_text()
                    == "Correct"
                )
                record_step(report, "day_duplicate_still_completed", task_id=duplicate_task_id)

                page.get_by_role("button", name="Final Results").click()
                page.locator(".case-panel.active").get_by_role(
                    "heading", name="Final Results"
                ).wait_for(timeout=5_000)
                final_options = [
                    item.strip()
                    for item in page.locator(
                        ".case-panel.active .version-selector option"
                    ).all_inner_texts()
                ]
                assert "Original" in final_options
                assert "Automatic" in final_options
                assert "Latest reviewed" not in final_options
                assert "Final" not in final_options
                assert page.locator(".download-menu").count() > 0
                record_step(report, "final_results_hides_unavailable_roles")
                report["passed"] = True
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        write_report(package, report)
