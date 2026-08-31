"""Offline checks for the single-agent runner and its safety boundaries."""
import json
import os
import re
import sys
import tempfile
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mle_agent.harness.agent_main as agent_main_module
from mle_agent.harness.agent_main import (
    _converged,
    _log_row,
    _stability_across_seeds,
    render_stability_section,
)
import mle_agent.harness.agent_tools as agent_tools_module
from mle_agent.harness.agent_tools import (
    AgentToolRuntime,
    BootstrapState,
    execute_model,
    inspect_ml_environment,
    inspect_train_valid_data,
    install_python_dependencies,
)
from mle_agent.harness.config import load_config
from mle_agent.harness.console import RunConsole
from mle_agent.harness.data_view import (
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VALID_ROWS,
    classify_date,
)
from mle_agent.harness.hooks import PostFileSaveHook, run_post_file_save_hooks
from mle_agent.harness.evaluation import ScoredPredictions
from mle_agent.harness.logger import RunLogger
from mle_agent.harness.root_model import make_root_model_py
from mle_agent.harness.run_environment import (
    AUTO_INSTALL_ALLOWLIST,
    create_run_environment,
    snapshot_run_environment,
)
from mle_agent.harness.submission import write_hidden_submission
from mle_agent.harness.provider import (
    LLMClient,
    LLMResponse,
    OpenAICompatClient,
    ToolCall,
)
from mle_agent.harness.tools import exec_write_file
from mle_agent.harness.validator import scan_candidate_source, validate_row
from mle_agent.research_agent.agent import ResearchAgent, _console_reasoning_line
from mle_agent.research_agent.prompts import render_prompt


def _fake_trusted_scorer(path, *_args, **_kwargs):
    source = (Path(path).parent.parent / "model.py").read_text(encoding="utf-8")
    def metric(name: str, fallback: float) -> float:
        match = re.search(rf"['\"]{re.escape(name)}['\"]\s*:\s*([0-9.]+)", source)
        return float(match.group(1)) if match else fallback
    return ScoredPredictions(
        metrics={
            "GAUC": metric("GAUC", 0.61),
            "nDCG@5": metric("nDCG@5", 0.60),
            "primary": metric("primary", 0.605),
        },
        rows=1,
    )


def test_convergence_requires_three_completed_no_gain_experiments():
    assert not _converged([0.6016, 0.6016, 0.6016], 0.002, 3)
    assert _converged([0.6016, 0.6016, 0.6017, 0.6017], 0.002, 3)
    assert not _converged([0.6016, 0.6017, 0.6050, 0.6050], 0.002, 3)


def test_test_split_scanner_catches_aliases_and_computed_keys():
    assert scan_candidate_source("data = load(path)\nx = data['test']")
    assert scan_candidate_source("x = splits['te' + 'st']")
    assert scan_candidate_source("x = splits.get('test')")
    assert not scan_candidate_source("x = splits['train']\ny = splits['valid']")


def test_candidate_cannot_bypass_the_dependency_install_policy():
    assert scan_candidate_source("import subprocess\nsubprocess.run(['pip', 'install', 'x'])")
    assert scan_candidate_source("import pip\npip.main(['install', 'x'])")
    assert scan_candidate_source("import os\nos.system('python -m pip install x')")
    assert scan_candidate_source("from os import system\nsystem('pip install x')")
    assert scan_candidate_source("pip = __import__('pip')")
    assert not scan_candidate_source("import os\npath = os.path.join('train', 'features.csv')")


def test_candidate_write_cannot_escape_to_prefix_sibling():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "trial"
        root.mkdir()
        result = exec_write_file("../trial_evil/escaped.py", "bad", root)
        assert result.startswith("ERROR")
        assert not (Path(temp) / "trial_evil" / "escaped.py").exists()


def test_candidate_process_does_not_receive_google_api_key():
    config = load_config()
    secret = "candidate-must-not-see-this-key"
    previous = os.environ.get("GOOGLE_API_KEY")
    try:
        os.environ["GOOGLE_API_KEY"] = secret
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp)
            trial.joinpath("model.py").write_text(
                "import json, os\n"
                "import argparse\n"
                "ap=argparse.ArgumentParser(); ap.add_argument('--data_dir'); ap.add_argument('--seed'); "
                "ap.add_argument('--prediction-path'); ap.add_argument('--trial-config'); a=ap.parse_args()\n"
                "print(os.environ.get('GOOGLE_API_KEY', '<missing>'))\n"
                "open(a.prediction_path,'w').write('row_id,user_id,video_id,score\\n')\n"
                "print(json.dumps({'GAUC': 0.61, 'nDCG@5': 0.60, 'primary': 0.605}))\n",
                encoding="utf-8",
            )
            result = execute_model(trial, config)
    finally:
        if previous is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = previous

    assert result.success
    assert secret not in result.output
    assert "<missing>" in result.output


def test_execute_model_resource_usage_survives_without_resource_module():
    """On Windows the POSIX `resource` module is absent; execute_model must still run."""
    config = load_config()
    original_resource = agent_tools_module.resource
    agent_tools_module.resource = None
    try:
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp)
            trial.joinpath("model.py").write_text(
                "import json, argparse\n"
                "ap=argparse.ArgumentParser(); ap.add_argument('--data_dir'); ap.add_argument('--seed'); "
                "ap.add_argument('--prediction-path'); ap.add_argument('--trial-config'); a=ap.parse_args()\n"
                "open(a.prediction_path,'w').write('row_id,user_id,video_id,score\\n')\n"
                "print(json.dumps({'GAUC': 0.61, 'nDCG@5': 0.60, 'primary': 0.605}))\n",
                encoding="utf-8",
            )
            result = execute_model(trial, config)
    finally:
        agent_tools_module.resource = original_resource

    assert result.success
    assert result.resource_usage["peak_rss_platform_units"] is None
    assert "cpu_user_seconds" in result.resource_usage
    assert result.resource_usage["accelerator"] == "not_measured"


def test_general_post_file_save_hook_uses_path_matcher_and_file_placeholder():
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / "generated" / "nested" / "example.txt"
        target.parent.mkdir(parents=True)
        target.write_text("saved", encoding="utf-8")
        hook = PostFileSaveHook(
            name="read-saved-file",
            path_glob="generated/**/*.txt",
            command=(
                sys.executable,
                "-c",
                "import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())",
                "{file}",
            ),
        )

        results = run_post_file_save_hooks(target, (hook,))

        assert len(results) == 1
        assert results[0].success
        assert results[0].output == "saved"


def test_model_post_file_save_hook_reports_syntax_error_immediately():
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp) / "experiment_workspace" / "run_001" / "trial_001"
        trial.mkdir(parents=True)

        result = exec_write_file("model.py", "def broken(:\n    pass\n", trial)

        assert result.startswith("FAILED:")
        assert "PostFileSave model-py-compile" in result
        assert "SyntaxError" in result
        assert trial.joinpath("model.py").exists()


def test_model_post_file_save_hook_stays_silent_after_valid_save():
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp) / "experiment_workspace" / "run_001" / "trial_001"
        trial.mkdir(parents=True)

        result = exec_write_file("model.py", "value = 1\n", trial)

        assert result == "OK: wrote 10 bytes to model.py"


def test_run_model_skips_after_failed_save_until_model_is_repaired():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp) / "experiment_workspace" / "run_001" / "trial_001"
        trial.mkdir(parents=True)
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        runtime = AgentToolRuntime(trial, config, BootstrapState(required=False))
        execute_calls: list[Path] = []
        original_execute_model = agent_tools_module.execute_model

        def fake_execute_model(candidate_dir, _config, **_kwargs):
            execute_calls.append(candidate_dir)
            return agent_tools_module.ModelExecution(
                True,
                {"GAUC": 0.61, "nDCG@5": 0.60, "primary": 0.605},
                "",
                None,
                0.01,
            )

        agent_tools_module.execute_model = fake_execute_model
        try:
            failed_save = runtime.dispatch(
                "write_file", {"path": "model.py", "content": "def broken(:\n"}
            )
            skipped = json.loads(runtime.dispatch("run_model", {
                "hypothesis": "Exercise the failed-save gate",
                "reasoning": "A queued execution must not run invalid Python.",
                "target_component": "model",
            }))

            assert failed_save.startswith("FAILED:")
            assert not skipped["success"]
            assert skipped["skipped"]
            assert skipped["wall_seconds"] == 0.0
            assert skipped["error"].startswith("SKIPPED:")
            assert execute_calls == []

            repaired_save = runtime.dispatch(
                "write_file", {"path": "model.py", "content": "value = 2\n"}
            )
            executed = json.loads(runtime.dispatch("run_model", {
                "hypothesis": "Exercise the repaired-save gate",
                "reasoning": "A successful save must re-enable execution.",
                "target_component": "model",
            }))

            assert repaired_save.startswith("OK:")
            assert executed["success"]
            assert execute_calls == [trial]
        finally:
            agent_tools_module.execute_model = original_execute_model


def test_fixed_organizer_date_splits_and_counts_are_pinned():
    assert classify_date(20220408) == "train"
    assert classify_date(20220421) == "train"
    assert classify_date(20220422) == "valid"
    assert classify_date(20220428) == "valid"
    assert classify_date(20220429) == "test"
    assert classify_date(20220508) == "test"
    assert classify_date(20220509) == "outside"
    assert (EXPECTED_TRAIN_ROWS, EXPECTED_VALID_ROWS, EXPECTED_TEST_ROWS) == (
        1_141_112, 124_909, 170_588
    )


def test_root_candidate_preserves_trusted_finalization_contract():
    source = make_root_model_py(load_config())

    assert "--submission-path" in source
    assert "write_hidden_submission" in source
    assert scan_candidate_source(source) == []


def test_trusted_submission_writer_outputs_aligned_finite_scores():
    rows = [
        (20220429, "user-a", "video-1", "author", "tab", 1000.0, 0),
        (20220429, "user-a", "video-2", "author", "tab", 2000.0, 1),
    ]
    split_name = "".join(("te", "st"))
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "submission.csv"
        write_hidden_submission(
            path,
            {split_name: rows},
            {split_name: ([[1], [2]], [0, 1], ["user-a", "user-a"])},
            lambda features: [0.25, -1.5],
        )
        records = path.read_text(encoding="utf-8").splitlines()

    assert records == [
        "row_id,user_id,video_id,score",
        "0,user-a,video-1,0.25",
        "1,user-a,video-2,-1.5",
    ]


_CHANGED_SCORING_MODEL = """import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument('--data_dir')
ap.add_argument('--seed')
ap.add_argument('--prediction-path')
ap.add_argument('--trial-config')
a = ap.parse_args()
experiment_variant = 'semantic-change'
open(a.prediction_path, 'w').write('row_id,user_id,video_id,score\\n')
print(json.dumps({'GAUC': 0.61, 'nDCG@5': 0.60, 'primary': 0.605}))
"""


def _write_and_run_calls(call_prefix: str, hypothesis: str, reasoning: str):
    return [
        ToolCall(
            id=f"{call_prefix}_write",
            name="write_file",
            input={"path": "model.py", "content": _CHANGED_SCORING_MODEL},
        ),
        ToolCall(
            id=f"{call_prefix}_run",
            name="run_model",
            input={
                "hypothesis": hypothesis,
                "reasoning": reasoning,
                "target_component": "model",
            },
        ),
    ]


def test_run_model_rejects_unchanged_and_comment_only_candidates():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        initial = "value = 1\n"
        trial.joinpath("model.py").write_text(initial, encoding="utf-8")
        runtime = AgentToolRuntime(trial, config, BootstrapState(required=False))
        payload = {
            "hypothesis": "Change behavior",
            "reasoning": "Test the guard.",
            "target_component": "model",
        }

        unchanged = json.loads(runtime.dispatch("run_model", payload))
        assert not unchanged["success"]
        assert not unchanged["candidate_changed"]
        assert "semantically unchanged" in unchanged["error"]

        runtime.dispatch("write_file", {
            "path": "model.py",
            "content": "# only a comment changed\nvalue = 1\n",
        })
        comment_only = json.loads(runtime.dispatch("run_model", payload))
        assert not comment_only["success"]
        assert not comment_only["candidate_changed"]


def test_run_model_accepts_a_semantically_changed_candidate():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        runtime = AgentToolRuntime(trial, config, BootstrapState(required=False))
        runtime.dispatch("write_file", {
            "path": "model.py",
            "content": _CHANGED_SCORING_MODEL,
        })
        result = json.loads(runtime.dispatch("run_model", {
            "hypothesis": "Change executable model code",
            "reasoning": "A semantic edit must be measured.",
            "target_component": "model",
        }))
        assert result["success"]
        assert result["candidate_changed"]
        assert result["metrics"]["primary"] == 0.605


def _dead_end_runtime(temp: str):
    config = load_config()
    trial = Path(temp)
    trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
    runtime = AgentToolRuntime(trial, config, BootstrapState(required=False))
    runtime.dispatch("write_file", {"path": "model.py", "content": _CHANGED_SCORING_MODEL})
    return runtime


def test_run_model_rejects_repeating_the_measured_capacity_dead_end():
    with tempfile.TemporaryDirectory() as temp:
        runtime = _dead_end_runtime(temp)
        original_execute = agent_tools_module.execute_model
        calls: list = []
        agent_tools_module.execute_model = lambda *a, **k: calls.append(1)
        try:
            result = json.loads(runtime.dispatch("run_model", {
                "hypothesis": "Raise the FM embedding dimension k from 16 to 32 for more capacity.",
                "reasoning": "A larger latent space should capture more interactions.",
                "target_component": "model",
            }))
        finally:
            agent_tools_module.execute_model = original_execute
        assert not result["success"]
        assert result["metrics"] is None
        assert result["wall_seconds"] == 0.0
        assert "capacity is not the bottleneck" in result["error"]
        assert calls == []


def test_run_model_allows_a_legitimate_loss_change_after_the_dead_end_guard():
    with tempfile.TemporaryDirectory() as temp:
        runtime = _dead_end_runtime(temp)
        result = json.loads(runtime.dispatch("run_model", {
            "hypothesis": "Replace pointwise log loss with a per-user BPR pairwise ranking loss.",
            "reasoning": "Aligns the objective with the ranking metric.",
            "target_component": "loss",
        }))
        assert result["success"]
        assert result["metrics"]["primary"] == 0.605


def test_run_model_rejects_purely_user_side_first_order_features():
    with tempfile.TemporaryDirectory() as temp:
        runtime = _dead_end_runtime(temp)
        result = json.loads(runtime.dispatch("run_model", {
            "hypothesis": "Add a bucketed user_active_degree feature.",
            "reasoning": "Sparse users may benefit from a pooled representation.",
            "target_component": "features",
            "feature_sources": ["user_active_degree from user_features_pure"],
            "feature_transformations": ["quantile-bucket user_active_degree on train"],
            "leakage_controls": ["fit bucket edges on train rows only"],
        }))
        assert not result["success"]
        assert "within-user ranking" in result["error"]


def test_run_model_allows_user_side_feature_crossed_with_item_side():
    with tempfile.TemporaryDirectory() as temp:
        runtime = _dead_end_runtime(temp)
        result = json.loads(runtime.dispatch("run_model", {
            "hypothesis": "Cross user_active_degree with video_id popularity bucket.",
            "reasoning": "The cross varies within a user and can reorder the list.",
            "target_component": "features",
            "feature_sources": ["user_active_degree", "video_id"],
            "feature_transformations": ["cross user_active_degree x video_id popularity bucket"],
            "leakage_controls": ["fit on train only"],
        }))
        assert result["success"]


def test_record_task_context_requires_both_new_measured_dead_ends():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        model_path = trial / "model.py"
        model_path.write_text("value = 1\n", encoding="utf-8")
        resolved = str(model_path.resolve())
        state = BootstrapState(
            required=False,
            required_candidate_model_path=resolved,
            fully_read_paths={resolved},
        )
        runtime = AgentToolRuntime(trial, config, state)
        payload = _task_context_payload([resolved])
        payload["feature_engineering_context"]["measured_dead_ends"] = [
            "The organizer's 13 static fields produced no gain over the five-field baseline.",
        ]
        result = json.loads(runtime.dispatch("record_task_context", payload))
        assert result["error"] == "TASK_CONTEXT_INVALID"
        errors = " ".join(result["feature_context_errors"])
        assert "k=8/16/32" in errors
        assert "user-side first-order" in errors


def test_stability_across_seeds_runs_the_two_extra_seeds():
    config = load_config()
    seen_seeds: list[int] = []

    def fake_execute_model(candidate_dir, cfg, *, seed=None, **_kwargs):
        seen_seeds.append(seed)
        return SimpleNamespace(
            success=True,
            metrics={"GAUC": 0.61, "nDCG@5": 0.60, "primary": 0.605 + seed * 1e-7},
        )

    original = agent_main_module.execute_model
    agent_main_module.execute_model = fake_execute_model
    try:
        out = _stability_across_seeds(
            Path("."), config, {"GAUC": 0.61, "nDCG@5": 0.60, "primary": 0.605}
        )
    finally:
        agent_main_module.execute_model = original

    assert seen_seeds == [314, 2718]  # seed 42 metrics are passed in, not re-run
    assert set(out["per_seed"]) == {"42", "314", "2718"}
    assert out["primary_mean"] is not None
    assert out["primary_std"] >= 0.0
    section = render_stability_section(out, best_iteration=3)
    assert "Multi-seed stability" in section
    assert "trial_003" in section


def test_stability_section_handles_no_new_best():
    section = render_stability_section(None, best_iteration=0)
    assert "No candidate beat the reproduced baseline" in section


def test_log_row_carries_stability_and_still_validates():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        (trial / "model.py").write_text(
            _CHANGED_SCORING_MODEL.replace("semantic-change", "initial-candidate"),
            encoding="utf-8",
        )
        client = _FakeClient()
        agent = ResearchAgent(
            config, client=client, provider_retry_delay_s=0,
            bootstrap_state=BootstrapState(required=False),
        )
        result = agent.run_iteration(1, trial, 0.6016, 0.6016, max_turns=1)
        assert result.success
        stability = {
            "seeds": [42, 314, 2718],
            "primary_mean": 0.6051,
            "primary_std": 0.0004,
            "per_seed": {
                "42": {"GAUC": 0.61, "nDCG@5": 0.60, "primary": 0.605},
                "314": {"GAUC": 0.61, "nDCG@5": 0.60, "primary": 0.6055},
                "2718": {"GAUC": 0.61, "nDCG@5": 0.60, "primary": 0.6048},
            },
        }
        row = _log_row(
            1, 0, result, "success", True, 0.6016, 0.2468,
            "--- diff", trial / "model.py", stability,
        )
        assert validate_row(row) == []
        assert len(row["stability"]["per_seed"]) == 3


def test_run_model_rejects_a_previously_scored_semantic_candidate():
    config = load_config()
    state = BootstrapState(required=False)
    with tempfile.TemporaryDirectory() as temp:
        first_trial = Path(temp) / "trial_001"
        second_trial = Path(temp) / "trial_002"
        first_trial.mkdir()
        second_trial.mkdir()
        first_trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        second_trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")

        first = AgentToolRuntime(first_trial, config, state)
        first.dispatch("write_file", {"path": "model.py", "content": _CHANGED_SCORING_MODEL})
        scored = json.loads(first.dispatch("run_model", {
            "hypothesis": "First scoring of this candidate",
            "reasoning": "Establish the semantic fingerprint.",
            "target_component": "model",
        }))
        assert scored["success"]

        second = AgentToolRuntime(second_trial, config, state)
        second.dispatch("write_file", {"path": "model.py", "content": _CHANGED_SCORING_MODEL})
        duplicate = json.loads(second.dispatch("run_model", {
            "hypothesis": "Accidental duplicate",
            "reasoning": "This should be rejected before execution.",
            "target_component": "model",
        }))

        assert not duplicate["success"]
        assert "already scored successfully" in duplicate["error"]


def _task_context_payload(source_paths: list[str]) -> dict:
    return {
        "task_objective": "Improve within-user long-view ranking on KuaiRand.",
        "target_label": "long_view",
        "metrics": ["GAUC", "nDCG@5", "primary"],
        "data_splits": {
            "train": "20220408-20220421",
            "validation": "20220422-20220428",
            "test": "20220429-20220508; hidden and unavailable during development",
        },
        "baseline": "Official factorization-machine validation baseline.",
        "evaluation_protocol": ["Train on train and score validation only."],
        "hard_constraints": ["Never access the hidden test split."],
        "known_dead_ends": ["Comment-only edits."],
        "promising_directions": ["Evidence-backed ranking objectives."],
        "feature_engineering_context": {
            "baseline_fields": [
                "user_id", "video_id", "author_id", "tab", "dur_bucket",
            ],
            "measured_dead_ends": [
                "The organizer's 13 static fields produced no gain over the five-field baseline.",
                "Raising embedding dimension k=8/16/32 stayed flat (~0.589); capacity is not the bottleneck.",
                "Purely user-side first-order features contribute zero to within-user ranking; they help only through item-side interactions.",
            ],
            "promising_feature_families": [
                "train-history sequences", "temporal context", "user-item crosses",
            ],
            "leakage_controls": [
                "Fit every vocabulary, bucket, aggregate, and target statistic on train only."
            ],
            "implementation_boundary": (
                "Read immutable candidate_data and implement the complete feature pipeline in model.py."
            ),
        },
        "candidate_contract": ["Accept --data_dir and print JSON metrics."],
        "source_paths": source_paths,
    }


def test_bootstrap_gate_rejects_premature_context_edits_and_runs():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        state = BootstrapState()
        runtime = AgentToolRuntime(trial, config, state)

        early_context = json.loads(runtime.dispatch(
            "record_task_context", _task_context_payload([])
        ))
        assert early_context["error"] == "BOOTSTRAP_SOURCES_INCOMPLETE"

        early_write = json.loads(runtime.dispatch(
            "write_file", {"path": "model.py", "content": "value = 2\n"}
        ))
        assert early_write["error"] == "BOOTSTRAP_REQUIRED"
        assert trial.joinpath("model.py").read_text(encoding="utf-8") == "value = 1\n"

        early_run = json.loads(runtime.dispatch("run_model", {
            "hypothesis": "Premature change",
            "reasoning": "Exercise the gate.",
        }))
        assert early_run["error"] == "BOOTSTRAP_REQUIRED"
        assert runtime.executions == []
        assert len(state.rejected_actions) == 3


def test_bootstrap_requires_a_literature_query_with_retrieved_evidence():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        state = BootstrapState()
        runtime = AgentToolRuntime(trial, config, state)

        no_match = json.loads(runtime.dispatch("search_ml_literature", {
            "query": "zzzz_nonexistent_method_zzzz",
        }))
        assert not no_match["bootstrap_counted"]
        assert state.literature_queries == []

        matched = json.loads(runtime.dispatch("search_ml_literature", {
            "query": "within-user pairwise ranking loss",
        }))
        assert matched["bootstrap_counted"]
        assert matched["results"]
        assert state.literature_queries == ["within-user pairwise ranking loss"]


def test_task_docs_are_discovered_and_long_reads_are_explicitly_paginated():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        starter = root / "starter"
        trial = root / "trial"
        starter.mkdir()
        trial.mkdir()
        readme_text = "0123456789" * 5
        starter.joinpath("README.md").write_text(readme_text, encoding="utf-8")
        starter.joinpath("evaluate.py").write_text("metric = 'primary'\n", encoding="utf-8")
        starter.joinpath("data.py").write_text("SPLITS = {}\n", encoding="utf-8")
        starter.joinpath("ablation_features.py").write_text(
            "RESULT = '13 static fields: no gain'\n", encoding="utf-8"
        )
        root.joinpath("README-outside.md").write_text("not discoverable", encoding="utf-8")
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        config.BASELINE_ROOT = starter
        state = BootstrapState()
        runtime = AgentToolRuntime(trial, config, state)

        discovery = json.loads(runtime.dispatch("discover_task_docs", {}))
        assert discovery["primary_readme_path"] == str(starter.joinpath("README.md").resolve())
        discovered = {item["relative_path"] for item in discovery["documents"]}
        assert discovered == {
            "README.md", "evaluate.py", "data.py", "ablation_features.py",
        }

        first = json.loads(runtime.dispatch("read_file", {
            "path": discovery["primary_readme_path"], "offset": 0, "max_chars": 17,
        }))
        assert first["content"] == readme_text[:17]
        assert not first["complete"]
        assert first["next_offset"] == 17

        offset = first["next_offset"]
        while offset is not None:
            page = json.loads(runtime.dispatch("read_file", {
                "path": discovery["primary_readme_path"],
                "offset": offset,
                "max_chars": 17,
            }))
            offset = page["next_offset"]
        assert str(starter.joinpath("README.md").resolve()) in state.fully_read_paths


def test_completed_task_context_persists_across_iteration_runtimes():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        starter = root / "starter"
        first_trial = root / "trial_001"
        second_trial = root / "trial_002"
        starter.mkdir()
        first_trial.mkdir()
        second_trial.mkdir()
        starter.joinpath("README.md").write_text("Task documentation.\n", encoding="utf-8")
        starter.joinpath("evaluate.py").write_text("metric = 'primary'\n", encoding="utf-8")
        starter.joinpath("baseline.py").write_text("MODEL = 'fm'\n", encoding="utf-8")
        data_source = starter / "data.py"
        data_source.write_text("FIELDS = ['user_id']\n", encoding="utf-8")
        feature_ablation = starter / "ablation_features.py"
        feature_ablation.write_text("RESULT = '13 static fields: no gain'\n", encoding="utf-8")
        unread_source = starter / "baseline_scores.json"
        unread_source.write_text("{}\n", encoding="utf-8")
        first_trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        second_trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        config.BASELINE_ROOT = starter
        state = BootstrapState()
        first_runtime = AgentToolRuntime(first_trial, config, state)

        discovery = json.loads(first_runtime.dispatch("discover_task_docs", {}))
        required_paths = [
            discovery["primary_readme_path"],
            str(starter.joinpath("baseline.py").resolve()),
            str(starter.joinpath("evaluate.py").resolve()),
            str(data_source.resolve()),
            str(feature_ablation.resolve()),
            str(first_trial.joinpath("model.py").resolve()),
        ]
        for path in required_paths:
            response = json.loads(first_runtime.dispatch("read_file", {"path": path}))
            assert response["complete"]
        state.data_inspected = True
        state.environment_inspected = True
        state.environment_inventory = {"packages": {"numpy": {"installed": True}}}
        state.literature_queries.append("ranking losses for recommender systems")
        state.baseline_reproduced = True
        state.baseline_metrics = {"GAUC": 0.6674, "nDCG@5": 0.5358, "primary": 0.6016}

        rejected_context = json.loads(first_runtime.dispatch(
            "record_task_context",
            _task_context_payload(required_paths + [str(unread_source.resolve())]),
        ))
        assert rejected_context["error"] == "TASK_CONTEXT_INVALID"
        assert rejected_context["cited_sources_not_fully_read"] == [
            str(unread_source.resolve())
        ]

        recorded = json.loads(first_runtime.dispatch(
            "record_task_context", _task_context_payload(required_paths)
        ))
        assert recorded["success"]
        assert state.complete

        second_runtime = AgentToolRuntime(second_trial, config, state)
        write_result = second_runtime.dispatch(
            "write_file", {"path": "model.py", "content": "value = 2\n"}
        )
        assert write_result.startswith("OK")
        assert second_runtime.bootstrap_state.task_context == state.task_context
        evidence = state.evidence()
        assert evidence["complete"]
        assert evidence["task_context"]["target_label"] == "long_view"


def test_openai_compat_round_trips_gemini_thought_signature_opaquely():
    signature = "encrypted-provider-value"
    native_tool_call = SimpleNamespace(
        id="function-call-1",
        function=SimpleNamespace(name="inspect_data", arguments="{}"),
        extra_content={"google": {"thought_signature": signature}},
    )
    native_message = SimpleNamespace(
        content="Inspect the fixed splits.",
        tool_calls=[native_tool_call],
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=native_message, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    completions = SimpleNamespace(create=lambda **kwargs: completion)
    client = object.__new__(OpenAICompatClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._model = "gemini-3.5-flash-lite"

    response = client.complete([], tools=[{
        "name": "inspect_data",
        "description": "Inspect data.",
        "parameters": {"type": "object", "properties": {}},
    }])
    assert response.tool_calls[0].provider_metadata == {
        "extra_content": {"google": {"thought_signature": signature}}
    }

    history = []
    client.add_response_to_history(history, response)
    client.add_tool_results_to_history(history, response.tool_calls, ["result"])
    assert history[0]["tool_calls"][0]["extra_content"]["google"] == {
        "thought_signature": signature
    }
    assert history[1] == {
        "role": "tool",
        "tool_call_id": "function-call-1",
        "content": "result",
        "name": "inspect_data",
    }


def test_console_reasoning_prefers_summary_parses_reflection_and_has_fallback():
    summarized = LLMResponse(
        text=None,
        tool_calls=[ToolCall(id="1", name="inspect_data", input={})],
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        reasoning_summary="Inspect fixed splits before modeling.",
    )
    assert _console_reasoning_line(summarized) == "Inspect fixed splits before modeling."

    reflected = LLMResponse(
        text='{"reflection":"Validation improved enough to retain the candidate."}',
        tool_calls=[],
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
    )
    assert _console_reasoning_line(reflected) == (
        "Validation improved enough to retain the candidate."
    )

    fallback = LLMResponse(
        text=None,
        tool_calls=[ToolCall(id="2", name="run_model", input={})],
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
    )
    assert _console_reasoning_line(fallback) == "Calling local tool: run_model."


def test_structured_console_separates_agent_and_harness_without_dumping_code():
    stream = StringIO()
    run_console = RunConsole(stream=stream, use_color=False)
    run_console.harness("Run setup", provider="gemini / test-model")
    run_console.agent_reasoning(
        "Testing a ranking-aware loss because pointwise FM under-optimizes ordering.",
        phase="Experiment 1, turn 1/10",
        progress="Execution attempts=0",
        response_text="line one\nline two\nline three\nline four\nline five\nline six hidden",
        tool_names=["write_file"],
        stop_reason="tool_use",
        input_tokens=120,
        output_tokens=40,
        call_number=3,
        response_event_id="llm_000003",
        provider="gemini / test-model",
        latency_seconds=1.25,
        tool_summaries=["write_file(path=model.py, chars=35)"],
    )
    secret_code = "API_KEY_SHOULD_NOT_APPEAR = 'value'"
    run_console.agent_tool_call(
        "write_file", {"path": "model.py", "content": secret_code}
    )
    run_console.agent_tool_result("write_file", "OK: wrote 35 bytes to model.py")
    rendered = stream.getvalue()

    assert "--- Harness ---" in rendered
    assert "--- Agent ---" in rendered
    assert "Reasoning: Testing a ranking-aware loss" in rendered
    assert "LLM Call: #3 | llm_000003 | gemini / test-model | 1.25s" in rendered
    assert "LLM Response: line one" in rendered
    assert "Model Event: stop=tool_use; tokens in=120, out=40" in rendered
    assert "Actions Requested: write_file(path=model.py, chars=35)" in rendered
    assert "line five … [response truncated]" in rendered
    assert "line six hidden" not in rendered
    assert f"Tool Calling: write_file(path=model.py, chars={len(secret_code)})" in rendered
    assert "Tool Called: write_file" in rendered
    assert "Returned Result: OK: wrote 35 bytes to model.py" in rendered
    assert secret_code not in rendered
    assert "\033[" not in rendered

    colour_stream = StringIO()
    RunConsole(stream=colour_stream, use_color=True).harness("Colour check")
    assert "\033[" in colour_stream.getvalue()


class _FakeClient(LLMClient):
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools=None, max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text=None,
                tool_calls=_write_and_run_calls(
                    "call_1",
                    "Smoke-test the unified execution loop",
                    "Verify plumbing without a provider call.",
                ),
                stop_reason="tool_use",
                input_tokens=10,
                output_tokens=5,
            )
        return LLMResponse(
            text='{"reflection":"The unified loop returned metrics.","hypothesis_supported":true,"suggested_next":"real model"}',
            tool_calls=[],
            stop_reason="end_turn",
            input_tokens=12,
            output_tokens=7,
        )

    def add_response_to_history(self, messages, response):
        messages.append({"role": "assistant", "content": response.text or ""})

    def add_tool_results_to_history(self, messages, tool_calls, outputs):
        messages.append({"role": "user", "content": "\n".join(outputs)})


class _BootstrapThenExecuteClient(_FakeClient):
    def __init__(
        self,
        readme: Path,
        baseline: Path,
        evaluate: Path,
        data: Path,
        feature_ablation: Path,
        model: Path,
    ):
        super().__init__()
        self.readme = str(readme.resolve())
        self.baseline = str(baseline.resolve())
        self.evaluate = str(evaluate.resolve())
        self.data = str(data.resolve())
        self.feature_ablation = str(feature_ablation.resolve())
        self.model = str(model.resolve())

    def complete(self, messages, tools=None, max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="Discovering and reading the task sources.",
                tool_calls=[
                    ToolCall(id="discover", name="discover_task_docs", input={}),
                    ToolCall(
                        id="read_readme_1", name="read_file",
                        input={"path": self.readme, "offset": 0},
                    ),
                    ToolCall(
                        id="read_baseline", name="read_file",
                        input={"path": self.baseline, "offset": 0},
                    ),
                    ToolCall(
                        id="read_evaluate", name="read_file",
                        input={"path": self.evaluate, "offset": 0},
                    ),
                    ToolCall(
                        id="read_data", name="read_file",
                        input={"path": self.data, "offset": 0},
                    ),
                    ToolCall(
                        id="read_feature_ablation", name="read_file",
                        input={"path": self.feature_ablation, "offset": 0},
                    ),
                    ToolCall(
                        id="read_model", name="read_file",
                        input={"path": self.model, "offset": 0},
                    ),
                    ToolCall(id="inspect", name="inspect_data", input={}),
                    ToolCall(id="inspect_environment", name="inspect_environment", input={}),
                    ToolCall(
                        id="literature", name="search_ml_literature",
                        input={"query": "within-user pairwise ranking loss"},
                    ),
                ],
                stop_reason="tool_use", input_tokens=10, output_tokens=4,
            )
        if self.calls == 2:
            return LLMResponse(
                text="Reading the remaining README page.",
                tool_calls=[ToolCall(
                        id="read_readme_2", name="read_file",
                    input={"path": self.readme, "offset": 200},
                )],
                stop_reason="tool_use", input_tokens=10, output_tokens=3,
            )
        if self.calls == 3:
            return LLMResponse(
                text="Reproducing the unchanged official baseline.",
                tool_calls=[ToolCall(
                    id="reproduce_baseline", name="reproduce_baseline", input={},
                )],
                stop_reason="tool_use", input_tokens=10, output_tokens=3,
            )
        if self.calls == 4:
            return LLMResponse(
                text="Recording the retained task context after baseline reproduction.",
                tool_calls=[ToolCall(
                    id="record_context", name="record_task_context",
                    input=_task_context_payload([
                        self.readme, self.baseline, self.evaluate, self.data,
                        self.feature_ablation, self.model,
                    ]),
                )],
                stop_reason="tool_use", input_tokens=10, output_tokens=4,
            )
        if self.calls == 5:
            return LLMResponse(
                text="Implementing and measuring the first hypothesis.",
                tool_calls=_write_and_run_calls(
                    "bootstrap_run",
                    "Test a semantic model change after bootstrap",
                    "The retained task context and ranking literature justify measuring it.",
                ),
                stop_reason="tool_use", input_tokens=10, output_tokens=5,
            )
        return LLMResponse(
            text='{"reflection":"Bootstrap and execution completed.","hypothesis_supported":true,"suggested_next":"continue"}',
            tool_calls=[], stop_reason="end_turn", input_tokens=8, output_tokens=4,
        )


def test_research_agent_completes_and_retains_the_bootstrap_before_running():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        starter = root / "starter"
        trial = root / "trial"
        starter.mkdir()
        trial.mkdir()
        readme = starter / "README.md"
        baseline = starter / "baseline.py"
        evaluate = starter / "evaluate.py"
        data = starter / "data.py"
        feature_ablation = starter / "ablation_features.py"
        model = trial / "model.py"
        readme.write_text("A" * 300, encoding="utf-8")
        baseline.write_text("MODEL = 'fm'\n", encoding="utf-8")
        evaluate.write_text("metric = 'primary'\n", encoding="utf-8")
        data.write_text(
            "FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']\n",
            encoding="utf-8",
        )
        feature_ablation.write_text(
            "RESULT = '13 static fields: no gain'\n", encoding="utf-8"
        )
        model.write_text(
            """import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument('--data_dir')
ap.add_argument('--seed')
ap.add_argument('--prediction-path')
ap.add_argument('--trial-config')
a = ap.parse_args()
open(a.prediction_path, 'w').write('row_id,user_id,video_id,score\\n')
print(json.dumps({'GAUC': 0.6674, 'nDCG@5': 0.5358, 'primary': 0.6016}))
""",
            encoding="utf-8",
        )
        config.BASELINE_ROOT = starter
        config.AGENT_READ_MAX_CHARS = 400
        client = _BootstrapThenExecuteClient(
            readme, baseline, evaluate, data, feature_ablation, model
        )
        original_inspector = agent_tools_module.inspect_train_valid_data
        agent_tools_module.inspect_train_valid_data = lambda _config: {
            "policy": "train and validation only",
            "train": {"rows": 1_141_112},
            "valid": {"rows": 124_909},
        }
        try:
            agent = ResearchAgent(
                config, client=client, provider_retry_delay_s=0,
                bootstrap_state=BootstrapState(),
            )
            bootstrap = agent.run_bootstrap(trial, max_turns=4)
            result = agent.run_iteration(1, trial, 0.6016, 0.6016, max_turns=1)
        finally:
            agent_tools_module.inspect_train_valid_data = original_inspector

        assert bootstrap.success
        assert bootstrap.metrics["primary"] == 0.6016
        assert result.success
        assert result.metrics["primary"] == 0.605
        assert agent.bootstrap_evidence["complete"]
        assert agent.bootstrap_evidence["task_context"]["target_label"] == "long_view"
        assert any(
            "TASK_CONTEXT_RECORDED_AND_RETAINED" in str(message.get("content"))
            for message in agent.messages
        )
        assert client.calls == 6


def test_single_agent_owns_a_persistent_tool_loop():
    config = load_config()
    source = _CHANGED_SCORING_MODEL.replace("semantic-change", "initial-candidate")
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        (trial / "model.py").write_text(source, encoding="utf-8")
        client = _FakeClient()
        agent = ResearchAgent(
            config,
            client=client,
            provider_retry_delay_s=0,
            bootstrap_state=BootstrapState(required=False),
        )
        result = agent.run_iteration(1, trial, 0.6016, 0.6016, max_turns=1)
        assert result.success
        assert result.metrics["primary"] == 0.605
        assert result.hypothesis == "Smoke-test the unified execution loop"
        assert result.token_counts == {"input": 22, "output": 12}
        assert client.calls == 2
        assert result.reflection == "The unified loop returned metrics."
        assert {item["name"] for item in agent.prompt_evidence} == {
            "agent.md", "iteration.md"
        }
        row = _log_row(
            1, 0, result, "success", True, 0.6016, 0.2468,
            "--- diff", trial / "model.py",
        )
        assert validate_row(row) == []

        invalid = dict(row)
        invalid["code_diff"] = ""
        invalid["code_diff_reason"] = None
        assert any(
            "substantive code diff" in error for error in validate_row(invalid)
        )

        with tempfile.TemporaryDirectory() as log_temp:
            logger = RunLogger(Path(log_temp), "strict", strict_validation=True)
            try:
                try:
                    logger.write(invalid)
                except ValueError as exc:
                    assert "Invalid official run row" in str(exc)
                else:
                    raise AssertionError("strict logger accepted invalid v2 evidence")
            finally:
                logger.close()


def test_llm_trace_records_provider_responses_and_safe_tool_results():
    config = load_config()
    source = _CHANGED_SCORING_MODEL.replace("semantic-change", "initial-candidate")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        trial = root / "trial"
        trial.mkdir()
        (trial / "model.py").write_text(source, encoding="utf-8")
        logger = RunLogger(root, "trace-test")
        client = _FakeClient()
        agent = ResearchAgent(
            config,
            client=client,
            provider_retry_delay_s=0,
            bootstrap_state=BootstrapState(required=False),
            event_writer=logger.write_llm_event,
            provider_label="fake / test-model",
        )
        result = agent.run_iteration(1, trial, 0.6016, 0.6016, max_turns=1)
        totals = logger.running_totals()["llm_trace"]
        trace_path = logger.llm_events_path
        logger.close()

        assert result.success
        rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        responses = [row for row in rows if row["event_type"] == "llm_response"]
        tool_results = [row for row in rows if row["event_type"] == "tool_result"]
        assert len(responses) == 2
        assert len(tool_results) == 2
        assert totals["llm_response"] == 2
        assert totals["tool_result"] == 2
        assert responses[0]["phase"] == "experiment_1_turn_1"
        assert responses[0]["stop_reason"] == "tool_use"
        assert responses[0]["provider"] == "fake / test-model"
        assert responses[1]["phase"] == "experiment_1_closing_reflection"
        assert "unified loop returned metrics" in responses[1]["response_text"]
        write_call = next(
            call for call in responses[0]["tool_calls"] if call["name"] == "write_file"
        )
        assert "content" not in write_call["input"]
        assert write_call["input"]["content_chars"] == len(_CHANGED_SCORING_MODEL)
        assert len(write_call["input"]["content_sha256"]) == 64
        assert _CHANGED_SCORING_MODEL not in trace_path.read_text(encoding="utf-8")
        assert all(row["llm_response_id"] == "llm_000001" for row in tool_results)


class _EndEarlyThenExecuteClient(_FakeClient):
    def complete(self, messages, tools=None, max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="I am done.", tool_calls=[], stop_reason="end_turn",
                input_tokens=3, output_tokens=2,
            )
        if self.calls == 2:
            return LLMResponse(
                text=None,
                tool_calls=_write_and_run_calls(
                    "call_2", "Execute after correction", "Required."
                ),
                stop_reason="tool_use", input_tokens=4, output_tokens=2,
            )
        return LLMResponse(
            text='{"reflection":"Execution happened after protocol recovery.","hypothesis_supported":true,"suggested_next":"continue"}',
            tool_calls=[], stop_reason="end_turn", input_tokens=5, output_tokens=3,
        )


def _write_fake_scoring_model(trial: Path) -> None:
    trial.joinpath("model.py").write_text(
        _CHANGED_SCORING_MODEL.replace("semantic-change", "initial-candidate"),
        encoding="utf-8",
    )


def test_agent_cannot_end_before_run_model_when_turns_remain():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        _write_fake_scoring_model(trial)
        client = _EndEarlyThenExecuteClient()
        agent = ResearchAgent(
            config,
            client=client,
            provider_retry_delay_s=0,
            bootstrap_state=BootstrapState(required=False),
        )
        result = agent.run_iteration(1, trial, 0.6016, 0.6016, max_turns=2)
        assert result.success
        assert client.calls == 3
        assert any(
            event["action"] == "continue_with_execution_requirement"
            for event in result.recovery_events
        )


class _RetryOnceClient(_FakeClient):
    def complete(self, messages, tools=None, max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("temporary provider failure")
        if self.calls == 2:
            return LLMResponse(
                text=None,
                tool_calls=_write_and_run_calls(
                    "call_retry", "Retry provider once", "Transient failure."
                ),
                stop_reason="tool_use", input_tokens=6, output_tokens=3,
            )
        return LLMResponse(
            text='{"reflection":"The single retry recovered.","hypothesis_supported":true,"suggested_next":"continue"}',
            tool_calls=[], stop_reason="end_turn", input_tokens=7, output_tokens=4,
        )


def test_provider_errors_receive_exactly_one_retry():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        _write_fake_scoring_model(trial)
        client = _RetryOnceClient()
        trace_events = []
        agent = ResearchAgent(
            config,
            client=client,
            provider_retry_delay_s=0,
            bootstrap_state=BootstrapState(required=False),
            event_writer=trace_events.append,
            provider_label="fake / retry-model",
        )
        result = agent.run_iteration(1, trial, 0.6016, 0.6016, max_turns=1)
        assert result.success
        assert client.calls == 3  # failed call + retry + closing reflection
        provider_events = [e for e in result.recovery_events if e["type"] == "provider_error"]
        assert len(provider_events) == 1
        assert provider_events[0]["action"] == "retry_once"
        trace_errors = [e for e in trace_events if e["event_type"] == "provider_error"]
        trace_responses = [e for e in trace_events if e["event_type"] == "llm_response"]
        assert len(trace_errors) == 1
        assert trace_errors[0]["will_retry"]
        assert len(trace_responses) == 2
        assert trace_responses[0]["provider_call_attempt"] == 2


def test_prompt_templates_render_and_have_a_stable_hash():
    first = render_prompt(
        "iteration.md",
        iteration=1,
        candidate_dir="trial_001",
        parent_primary="0.601600",
        best_primary="0.601600",
        max_turns=8,
        stage_instruction="Perform EDA.",
        experiment_ledger="[]",
    )
    second = render_prompt(
        "iteration.md",
        iteration=2,
        candidate_dir="trial_002",
        parent_primary="0.602000",
        best_primary="0.602000",
        max_turns=10,
        stage_instruction="Use prior evidence.",
        experiment_ledger='[{"iteration":1,"primary":0.602}]',
    )
    assert "experiment `1`" in first.content
    assert first.template_sha256 == second.template_sha256
    assert len(first.template_sha256) == 64


def test_inspect_data_exposes_candidate_feature_sources_without_hidden_rows():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        data_dir = Path(temp)
        log_header = (
            "user_id,video_id,date,is_click,is_like,is_follow,is_comment,is_forward,"
            "long_view,duration_ms,tab\n"
        )
        data_dir.joinpath("log_standard_4_08_to_4_21_pure.csv").write_text(
            log_header + "u1,v1,20220408,1,0,0,0,0,1,1000,1\n",
            encoding="utf-8",
        )
        data_dir.joinpath("log_standard_4_22_to_5_08_pure.csv").write_text(
            log_header + "u1,v2,20220422,0,1,0,0,0,0,2000,0\n",
            encoding="utf-8",
        )
        data_dir.joinpath("user_features_pure.csv").write_text(
            "user_id,user_active_degree,register_days\nu1,full_active,100\n",
            encoding="utf-8",
        )
        data_dir.joinpath("video_features_basic_pure.csv").write_text(
            "video_id,author_id,music_id,video_type\nv1,a1,m1,NORMAL\n",
            encoding="utf-8",
        )
        data_dir.joinpath("video_features_statistic_pure.csv").write_text(
            "video_id,show_cnt,play_progress\nv1,10,0.5\n",
            encoding="utf-8",
        )
        config.DATA_DIR = data_dir

        summary = inspect_train_valid_data(config)

        inventory = summary["candidate_data"]
        assert inventory["raw_files_are_immutable_inputs"]
        assert inventory["baseline_fields"] == [
            "user_id", "video_id", "author_id", "tab", "dur_bucket",
        ]
        assert "music_id" in inventory["files"]["video_features_basic_pure.csv"]["columns"]
        assert "register_days" in inventory["files"]["user_features_pure.csv"]["columns"]
        assert "model.py" in inventory["implementation_boundary"]
        assert set(summary) >= {"train", "valid", "candidate_data"}
        assert "test" not in summary


def test_task_context_requires_the_measured_static_feature_dead_end():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        model_path = trial / "model.py"
        model_path.write_text("value = 1\n", encoding="utf-8")
        resolved_model = str(model_path.resolve())
        state = BootstrapState(
            required=False,
            required_candidate_model_path=resolved_model,
            fully_read_paths={resolved_model},
        )
        runtime = AgentToolRuntime(trial, config, state)
        payload = _task_context_payload([resolved_model])
        payload["feature_engineering_context"]["measured_dead_ends"] = [
            "Try arbitrary extra columns."
        ]

        result = json.loads(runtime.dispatch("record_task_context", payload))

        assert result["error"] == "TASK_CONTEXT_INVALID"
        assert any("13 static fields" in error for error in result["feature_context_errors"])


def test_feature_experiment_requires_sources_transforms_and_leakage_controls():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        runtime = AgentToolRuntime(trial, config, BootstrapState(required=False))
        runtime.dispatch("write_file", {
            "path": "model.py",
            "content": _CHANGED_SCORING_MODEL,
        })
        payload = {
            "hypothesis": "Add train-history item popularity as a feature.",
            "reasoning": "It varies within user and is computed from training impressions.",
            "target_component": "features",
        }

        rejected = json.loads(runtime.dispatch("run_model", payload))
        assert not rejected["success"]
        assert "FEATURE_EVIDENCE_REQUIRED" in rejected["error"]

        payload.update({
            "feature_sources": ["training user_id/video_id impression history"],
            "feature_transformations": ["log1p train-only video impression count"],
            "leakage_controls": [
                "Fit counts on splits['train'] only and apply the frozen mapping to validation."
            ],
        })
        scored = json.loads(runtime.dispatch("run_model", payload))
        assert scored["success"]
        assert scored["proposal"]["feature_sources"] == payload["feature_sources"]


def test_environment_inventory_reports_framework_availability_without_importing():
    inventory = inspect_ml_environment(load_config())

    assert inventory["python_version"]
    assert inventory["python_executable"]
    assert inventory["packages"]["numpy"]["installed"]
    assert set(inventory["packages"]) >= {
        "pandas", "scikit-learn", "torch", "recbole", "lightgbm",
    }
    assert "open-source" in inventory["policy"]


def test_run_environment_is_dedicated_and_writes_a_resolved_lock():
    config = load_config()
    base_python = config.PYTHON_EXE
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        workspace = root / "experiment_workspace" / "run_001"
        artifact_run = root / "artifacts" / "runs" / "run_001"
        workspace.mkdir(parents=True)

        created = create_run_environment(config, workspace, artifact_run)
        completed = snapshot_run_environment(config, phase="test_complete")

        assert created["success"]
        assert completed["success"]
        assert config.PYTHON_EXE != base_python
        assert Path(config.PYTHON_EXE).is_relative_to((workspace / ".venv").resolve())
        assert Path(str(completed["requirements_lock"])).is_file()
        manifest = json.loads(
            (artifact_run / "environment" / "manifest.json").read_text()
        )
        assert manifest["binary_only_installs"] is True
        assert manifest["auto_install_allowlist"] == sorted(AUTO_INSTALL_ALLOWLIST)
        assert [item["phase"] for item in manifest["snapshots"]] == [
            "created", "test_complete",
        ]


def test_allowlisted_dependency_auto_installs_without_user_intervention():
    config = load_config()
    installs: list[list[str]] = []
    original_probe = agent_tools_module._probe_python_environment
    agent_tools_module._probe_python_environment = lambda _config, names: {
        "success": True,
        "packages": {
            name: {"installed": False, "version": None} for name in names
        },
    }
    try:
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp)
            trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
            runtime = AgentToolRuntime(
                trial,
                config,
                BootstrapState(required=False),
                dependency_approver=lambda _requirements, _justification: (
                    (_ for _ in ()).throw(AssertionError("approval must not be requested"))
                ),
                dependency_installer=lambda _config, requirements: {
                    "success": not installs.append(requirements),
                    "installed_versions": {"lightgbm": "4.6.0"},
                    "output_tail": "installed binary wheel",
                },
            )

            result = json.loads(runtime.dispatch("request_dependency_install", {
                "packages": ["lightgbm>=4"],
                "justification": "Evaluate LambdaRank on within-user impression groups.",
            }))

            assert result["success"]
            assert installs == [["lightgbm>=4"]]
            operation = result["operations"][0]
            assert operation["authorization"] == "auto_allowlist"
            assert operation["approval_required"] is False
            assert operation["human_intervention"] is False
            assert operation["binary_only"] is True
    finally:
        agent_tools_module._probe_python_environment = original_probe


def test_already_available_dependency_request_is_still_logged():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        runtime = AgentToolRuntime(
            trial, config, BootstrapState(required=False)
        )

        result = json.loads(runtime.dispatch("request_dependency_install", {
            "packages": ["numpy"],
            "justification": "Use the already available numerical runtime.",
        }))

        assert result["status"] == "ALREADY_AVAILABLE"
        assert runtime.dependency_events[0]["outcome"] == "already_available"
        assert runtime.dependency_events[0]["installed_versions"]["numpy"]


def test_installer_targets_only_the_run_venv_with_binary_wheels():
    config = load_config()
    original_run = agent_tools_module.subprocess.run
    original_snapshot = agent_tools_module.snapshot_run_environment
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "-c" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({
                    "success": True,
                    "packages": {
                        "lightgbm": {"installed": True, "version": "4.6.0"},
                    },
                }),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="installed", stderr="")

    try:
        with tempfile.TemporaryDirectory() as temp:
            config.RUN_ENV_DIR = Path(temp) / ".venv"
            config.PYTHON_EXE = str(
                config.RUN_ENV_DIR
                / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            )
            agent_tools_module.subprocess.run = fake_run
            agent_tools_module.snapshot_run_environment = (
                lambda _config, phase: {"success": True, "phase": phase}
            )

            result = install_python_dependencies(config, ["lightgbm>=4"])

            install_command, install_kwargs = calls[0]
            assert result["success"]
            assert install_command[0] == config.PYTHON_EXE
            assert "--isolated" in install_command
            assert "--only-binary=:all:" in install_command
            assert "--no-input" in install_command
            assert "--user" not in install_command
            assert install_kwargs["env"]["PIP_REQUIRE_VIRTUALENV"] == "true"
            assert install_kwargs["env"]["PYTHONNOUSERSITE"] == "1"
    finally:
        agent_tools_module.subprocess.run = original_run
        agent_tools_module.snapshot_run_environment = original_snapshot


def test_dependency_install_requires_approval_and_records_the_intervention():
    config = load_config()
    approvals: list[tuple[list[str], str]] = []
    installs: list[list[str]] = []

    def approve(requirements: list[str], justification: str) -> bool:
        approvals.append((requirements, justification))
        return True

    def install(_config, requirements: list[str]) -> dict[str, object]:
        installs.append(requirements)
        return {
            "success": True,
            "requirements": requirements,
            "installed_versions": {"hackathon-ml-framework": "1.2.3"},
            "output_tail": "installed",
        }

    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        runtime = AgentToolRuntime(
            trial,
            config,
            BootstrapState(required=False),
            dependency_approver=approve,
            dependency_installer=install,
        )

        result = json.loads(runtime.dispatch("request_dependency_install", {
            "packages": ["hackathon-ml-framework==1.2.3"],
            "justification": "Faithfully evaluate a framework-backed ranking model.",
        }))

        assert result["success"]
        assert result["operations"][0]["approved"]
        assert approvals == [(
            ["hackathon-ml-framework==1.2.3"],
            "Faithfully evaluate a framework-backed ranking model.",
        )]
        assert installs == [["hackathon-ml-framework==1.2.3"]]
        event = runtime.dependency_events[0]
        assert event["authorization"] == "user_approved_off_allowlist"
        assert event["approval_required"] is True
        assert event["human_intervention"] is True
        assert event["outcome"] == "installed"
        assert event["installed_versions"] == {"hackathon-ml-framework": "1.2.3"}


def test_dependency_request_rejects_urls_and_respects_user_refusal():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        installs: list[list[str]] = []
        runtime = AgentToolRuntime(
            trial,
            config,
            BootstrapState(required=False),
            dependency_approver=lambda _requirements, _justification: False,
            dependency_installer=lambda _config, requirements: installs.append(requirements),
        )

        invalid = json.loads(runtime.dispatch("request_dependency_install", {
            "packages": ["framework @ https://example.com/package.whl"],
            "justification": "Use a remote wheel.",
        }))
        invalid_extras = json.loads(runtime.dispatch("request_dependency_install", {
            "packages": ["torch[cuda]"],
            "justification": "Request an unconstrained extra.",
        }))
        declined = json.loads(runtime.dispatch("request_dependency_install", {
            "packages": ["hackathon-declined-framework"],
            "justification": "Test an optional model family.",
        }))

        assert invalid["error"] == "INVALID_DEPENDENCY_REQUEST"
        assert invalid_extras["error"] == "INVALID_DEPENDENCY_REQUEST"
        assert declined["error"] == "DEPENDENCY_INSTALL_DECLINED"
        assert declined["operations"][0]["human_intervention"]
        assert installs == []
        assert runtime.dependency_events[0]["outcome"] == "invalid_request"
        assert runtime.dependency_events[1]["outcome"] == "invalid_request"
        assert runtime.dependency_events[2]["outcome"] == "user_declined"


# --- trusted scoring, bounded queries, and execution budgets ---------------------

def _write_validation_fixture(directory: Path, rows) -> None:
    """Write a minimal validation log the trusted scorer can derive labels from."""
    from mle_agent.harness.data_view import VALID_TEST_LOG

    lines = ["date,user_id,video_id,long_view,tab,duration_ms,play_time_ms"]
    for user, video, label in rows:
        lines.append(f"20220422,{user},{video},{label},1,10000,5000")
    (directory / VALID_TEST_LOG).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prediction_rows(rows, scores):
    header = "row_id,user_id,video_id,score"
    body = [
        f"{i},{user},{video},{score!r}"
        for i, ((user, video, _label), score) in enumerate(zip(rows, scores))
    ]
    return "\n".join([header, *body]) + "\n"


def test_trusted_scoring_rejects_validation_labels_copied_into_the_score_column():
    from mle_agent.harness.evaluation import (
        MAX_PLAUSIBLE_VALIDATION_GAUC,
        score_validation_predictions,
    )

    config = load_config()
    rows = [("u1", f"v{i}", i % 2) for i in range(20)]
    with tempfile.TemporaryDirectory() as temp:
        data_dir = Path(temp)
        _write_validation_fixture(data_dir, rows)
        leaked = data_dir / "leaked.csv"
        leaked.write_text(
            _prediction_rows(rows, [float(label) for _, _, label in rows]),
            encoding="utf-8",
        )
        try:
            score_validation_predictions(
                leaked, data_dir, config.BASELINE_ROOT / "evaluate.py"
            )
            raise AssertionError("label-copying predictions must be rejected")
        except ValueError as exc:
            assert "IMPLAUSIBLE_VALIDATION_GAUC" in str(exc)
            assert str(MAX_PLAUSIBLE_VALIDATION_GAUC) in str(exc)


def test_trusted_scoring_accepts_an_ordinary_model_score():
    from mle_agent.harness.evaluation import score_validation_predictions

    config = load_config()
    rows = [("u1", f"v{i}", i % 2) for i in range(20)]
    with tempfile.TemporaryDirectory() as temp:
        data_dir = Path(temp)
        _write_validation_fixture(data_dir, rows)
        honest = data_dir / "honest.csv"
        honest.write_text(_prediction_rows(rows, [0.5] * len(rows)), encoding="utf-8")
        scored = score_validation_predictions(
            honest, data_dir, config.BASELINE_ROOT / "evaluate.py"
        )
    assert scored.rows == len(rows)
    assert scored.metrics["GAUC"] < 0.95


def test_root_candidate_never_imports_the_trusted_harness():
    source = make_root_model_py(load_config())
    assert "from mle_agent" not in source
    assert "import mle_agent" not in source
    assert "def write_validation_predictions" in source
    assert "def write_hidden_submission" in source


def _write_query_fixture(directory: Path, valid_rows) -> None:
    from mle_agent.harness.data_view import TRAIN_LOG, VALID_TEST_LOG

    header = "date,user_id,video_id,long_view,tab,duration_ms,play_time_ms,is_rand"
    (directory / TRAIN_LOG).write_text(
        header + "\n20220408,u1,v1,1,1,1000,500,0\n", encoding="utf-8"
    )
    lines = [header]
    for user, video, label in valid_rows:
        lines.append(f"20220422,{user},{video},{label},1,1000,500,0")
    (directory / VALID_TEST_LOG).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_query_data_suppresses_groups_too_small_to_average_labels():
    from mle_agent.harness.eda import MIN_GROUP_ROWS, query_aggregates

    # One row per (user, video) pair: target_rate would be that row's label verbatim.
    rows = [(f"u{i}", f"v{i}", i % 2) for i in range(30)]
    with tempfile.TemporaryDirectory() as temp:
        data_dir = Path(temp)
        _write_query_fixture(data_dir, rows)
        result = query_aggregates(data_dir, {
            "split": "validation",
            "group_by": ["user_id", "video_id"],
            "metrics": ["rows", "target_rate"],
        })
    assert result["rows"] == []
    assert result["suppressed_small_groups"] == len(rows)
    assert result["suppressed_small_group_rows"] == len(rows)
    assert str(MIN_GROUP_ROWS) in result["policy"]


def test_query_data_returns_groups_large_enough_to_be_aggregates():
    from mle_agent.harness.eda import MIN_GROUP_ROWS, query_aggregates

    rows = [("u1", f"v{i}", i % 2) for i in range(MIN_GROUP_ROWS * 2)]
    with tempfile.TemporaryDirectory() as temp:
        data_dir = Path(temp)
        _write_query_fixture(data_dir, rows)
        result = query_aggregates(data_dir, {
            "split": "validation",
            "group_by": ["user_id"],
            "metrics": ["rows", "target_rate"],
        })
    assert len(result["rows"]) == 1
    assert result["rows"][0]["rows"] == MIN_GROUP_ROWS * 2
    assert 0.0 < result["rows"][0]["target_rate"] < 1.0
    assert result["suppressed_small_groups"] == 0


def test_query_data_refuses_to_build_more_groups_than_the_cardinality_cap():
    import mle_agent.harness.eda as eda_module

    rows = [(f"u{i}", f"v{i}", i % 2) for i in range(10)]
    original = eda_module.MAX_QUERY_GROUPS_SCANNED
    eda_module.MAX_QUERY_GROUPS_SCANNED = 3
    try:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            _write_query_fixture(data_dir, rows)
            try:
                eda_module.query_aggregates(data_dir, {
                    "split": "validation",
                    "group_by": ["user_id"],
                    "metrics": ["rows"],
                })
                raise AssertionError("unbounded grouping must be refused")
            except ValueError as exc:
                assert "more than 3 groups" in str(exc)
    finally:
        eda_module.MAX_QUERY_GROUPS_SCANNED = original


def _runtime_with_deadline(temp: Path, deadline):
    trial = temp / "trial"
    trial.mkdir()
    trial.joinpath("model.py").write_text("pass\n", encoding="utf-8")
    return AgentToolRuntime(trial, load_config(), BootstrapState(), run_deadline=deadline)


def test_harness_owns_the_execution_timeout_rather_than_the_agent_request():
    import time

    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        # No deadline: each class maps to its own ceiling, unknown classes downgrade.
        runtime = _runtime_with_deadline(Path(temp), None)
        assert runtime.grant_execution_timeout("quick")["timeout_seconds"] == (
            config.AGENT_QUICK_EXECUTION_TIMEOUT_S
        )
        assert runtime.grant_execution_timeout("substantial")["timeout_seconds"] == (
            config.AGENT_SUBSTANTIAL_EXECUTION_TIMEOUT_S
        )
        downgraded = runtime.grant_execution_timeout("unlimited")
        assert downgraded["granted_class"] == "normal"
        assert downgraded["timeout_seconds"] == config.AGENT_NORMAL_EXECUTION_TIMEOUT_S


def test_execution_timeout_is_clamped_to_the_remaining_wall_budget():
    import time

    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        # Far more budget than any class needs: nothing is clamped.
        roomy = _runtime_with_deadline(Path(temp), time.time() + 100_000)
        assert roomy.grant_execution_timeout("substantial")["granted_class"] == "substantial"

    with tempfile.TemporaryDirectory() as temp:
        # Enough for a quick run but far less than a substantial one.
        tight_budget = config.AGENT_WALL_RESERVE_S + config.AGENT_QUICK_EXECUTION_TIMEOUT_S * 3
        tight = _runtime_with_deadline(Path(temp), time.time() + tight_budget)
        grant = tight.grant_execution_timeout("substantial")
        assert grant["granted"]
        assert grant["granted_class"] == "clamped"
        assert grant["timeout_seconds"] < config.AGENT_SUBSTANTIAL_EXECUTION_TIMEOUT_S

    with tempfile.TemporaryDirectory() as temp:
        # Budget already spent: refuse rather than start a run that cannot finish.
        spent = _runtime_with_deadline(Path(temp), time.time() + 1)
        refused = spent.grant_execution_timeout("normal")
        assert not refused["granted"]
        assert "WALL_BUDGET_EXHAUSTED" in refused["reason"]


def test_iteration_aborted_before_a_proposal_still_writes_a_valid_log_row():
    """A provider outage on the first model call must not crash the run logger.

    The strict schema requires non-empty hypothesis and reasoning. An aborted
    iteration has neither, so the harness records what actually happened rather
    than raising while writing its own log row.
    """
    from dataclasses import replace

    from mle_agent.research_agent.adk_agent import AgentIterationResult

    aborted = AgentIterationResult(
        success=False,
        hypothesis="",
        reasoning="",
        reflection="",
        metrics=None,
        executions=[],
        recovery_events=[{"action": "user_declined_resume", "human_intervention": True}],
        token_counts={"input": 31485, "output": 2046},
        wall_seconds=10.6,
        error="provider quota exhausted and resume declined",
        final_code=None,
    )
    row = _log_row(1, 0, aborted, "failed", False, 0.6016, 0.2468, "", Path("trial_001/model.py"))

    assert row["aborted_before_proposal"] is True
    assert "provider quota exhausted" in row["reasoning"]
    assert row["hypothesis"].strip()
    assert validate_row(row) == []

    # The model may emit one field and not the other; whatever it said is kept.
    partial = replace(aborted, hypothesis="Add item popularity as a feature.")
    partial_row = _log_row(
        1, 0, partial, "failed", False, 0.6016, 0.2468, "", Path("trial_001/model.py")
    )
    assert partial_row["hypothesis"] == "Add item popularity as a feature."
    assert partial_row["reasoning"].strip()
    assert validate_row(partial_row) == []


def test_a_real_experiment_still_requires_genuine_reasoning():
    """The abort path must not become a way to log an experiment without reasoning."""
    from mle_agent.research_agent.adk_agent import AgentIterationResult

    ran_but_silent = AgentIterationResult(
        success=False,
        hypothesis="",
        reasoning="",
        reflection="",
        metrics=None,
        executions=[{"success": False, "proposal": {}}],
        recovery_events=[],
        token_counts={"input": 10, "output": 10},
        wall_seconds=1.0,
        error="candidate crashed",
        final_code=None,
    )
    row = _log_row(1, 0, ran_but_silent, "failed", False, 0.6016, 0.2468, "", Path("m.py"))

    assert row["aborted_before_proposal"] is False
    assert any("reasoning must be non-empty" in error for error in validate_row(row))


def _write_run_metrics(artifacts_root: Path, run_id: str, metrics: dict) -> None:
    results = artifacts_root / "runs" / run_id / "results"
    results.mkdir(parents=True)
    (results / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_finalization_refuses_runs_it_must_not_promote():
    """The guards that run before anything is generated or overwritten."""
    from mle_agent.harness import finalize as finalize_module

    original_artifacts = finalize_module.Config.ARTIFACTS_DIR
    with tempfile.TemporaryDirectory() as temp:
        artifacts_root = Path(temp) / "artifacts"
        finalize_module.Config.ARTIFACTS_DIR = artifacts_root
        data_dir = Path(temp) / "data"
        data_dir.mkdir()
        try:
            # Without explicit confirmation nothing is promoted at all.
            try:
                finalize_module.finalize_run(
                    "any", data_dir=data_dir, task_definition_confirmed=False
                )
                raise AssertionError("unconfirmed task definition must be refused")
            except ValueError as exc:
                assert "confirmation" in str(exc)

            # A run that never recorded confirmation is not promotable either.
            _write_run_metrics(artifacts_root, "unconfirmed", {
                "task_definition_confirmed": False, "converged": True,
            })
            try:
                finalize_module.finalize_run(
                    "unconfirmed", data_dir=data_dir, task_definition_confirmed=True
                )
                raise AssertionError("run without recorded confirmation must be refused")
            except ValueError as exc:
                assert "did not record task-definition confirmation" in str(exc)

            # Not converged, and no explicit budget-stop allowance.
            _write_run_metrics(artifacts_root, "unconverged", {
                "task_definition_confirmed": True,
                "converged": False,
                "stop_reason": "max_iterations",
            })
            try:
                finalize_module.finalize_run(
                    "unconverged", data_dir=data_dir, task_definition_confirmed=True
                )
                raise AssertionError("unconverged run must be refused by default")
            except ValueError as exc:
                assert "not converged" in str(exc)

            # A missing run is reported as missing, not as a crash.
            try:
                finalize_module.finalize_run(
                    "no_such_run", data_dir=data_dir, task_definition_confirmed=True
                )
                raise AssertionError("missing run must be refused")
            except FileNotFoundError as exc:
                assert "run metrics not found" in str(exc)
        finally:
            finalize_module.Config.ARTIFACTS_DIR = original_artifacts


def main() -> None:
    original_scorer = agent_tools_module.score_validation_predictions
    agent_tools_module.score_validation_predictions = _fake_trusted_scorer
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    try:
        for test in tests:
            test()
            print(f"  PASS  {test.__name__}")
    finally:
        agent_tools_module.score_validation_predictions = original_scorer
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
