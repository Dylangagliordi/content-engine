"""Tests for run_batch's pre-publish scoring wiring — written FIRST per
this project's TDD discipline, before the real function exists.

Real gap this closes (surfaced when the user asked "is it fully built
and merged and using every model" and pushed to "fix those gaps"):
learn.prediction.predict_performance_for_annotation existed and was
fully tested/verified, but run_batch.py — the actual script that
generates and persists real drafts — never called it. The model's own
module docstring says its whole purpose is "predicting a new draft's
likely performance before publish"; that step never happened in the
real pipeline. These tests cover the new score_variant_if_model_available
helper and insert_content_item's new niche/predicted_score persistence,
using a real isolated temp model directory (same pattern as
test_model_persistence.py), not mocks — a real trained model, saved,
then loaded and scored against a real ScriptVariant/DesignOutput shape.
"""
import shutil
import uuid

import pytest

from qualume.db.connection import get_connection
from qualume.learn.model_persistence import save_predictor
from qualume.learn.prediction import train_predictor
from qualume.models.contracts import CreativeDNA, DesignOutput, ScriptVariant
from qualume.run_batch import insert_content_item, score_variant_if_model_available


@pytest.fixture(autouse=True)
def _isolated_models_dir(tmp_path, monkeypatch):
    import qualume.learn.model_persistence as mp
    monkeypatch.setattr(mp, "MODELS_DIR", tmp_path / "models")
    yield
    if (tmp_path / "models").exists():
        shutil.rmtree(tmp_path / "models", ignore_errors=True)


def _real_training_rows(n=25):
    def _ann(i):
        return {
            "hook_type": "question" if i % 2 == 0 else "statement",
            "format_tag": "talking_head", "energy_level": i % 5 + 1,
            "shot_breakdown": [], "full_transcript": [], "video_duration_seconds": None,
            "hook_text": f"real hook number {i}",
        }
    return [(_ann(i), float(i % 5 + 1)) for i in range(n)]


def _script_variant(hook_text="a real hook about the product"):
    return ScriptVariant(
        hook_text=hook_text, body_text="real body copy", cta_text="shop now",
        tags=["real"], playbook_version="v1", prompt_version="v1", explore_or_exploit="exploit",
    )


def test_score_variant_returns_none_when_no_model_saved_for_niche():
    """Real, expected state — a niche nobody has trained a model for
    yet. None, not an exception, not a fabricated score — same refuse-
    rather-than-guess discipline as every other learn.* module."""
    variant = _script_variant()
    score = score_variant_if_model_available(variant, niche="zz_test_no_model_niche")
    assert score is None


def test_score_variant_returns_none_when_niche_is_none():
    """A batch run with no --niche argument is a real, valid case (e.g.
    local testing) — must not try to score against a non-existent
    'None' niche's model, just skip scoring entirely."""
    variant = _script_variant()
    score = score_variant_if_model_available(variant, niche=None)
    assert score is None


def test_score_variant_returns_a_real_float_when_a_model_exists():
    trained = train_predictor(_real_training_rows(), min_sample_size=20, use_text_embeddings=False)
    save_predictor(trained, niche="zz_test_scoring_niche", n_training_rows=25)

    variant = _script_variant(hook_text="a real hook number 3")
    score = score_variant_if_model_available(variant, niche="zz_test_scoring_niche")
    assert isinstance(score, float)


def test_score_variant_uses_the_variant_hook_text_not_a_fabricated_one():
    """Real correctness check: two variants with different hook_text
    should generally score differently when the model's text_embedder
    is actually active — proving the real variant content flows into
    the score, not a hardcoded placeholder."""
    trained = train_predictor(_real_training_rows(30), min_sample_size=20, use_text_embeddings=True)
    if trained["text_embedder"] is None:
        pytest.skip("sentence-transformers unavailable in this environment")
    save_predictor(trained, niche="zz_test_scoring_hooktext", n_training_rows=30)

    score_a = score_variant_if_model_available(
        _script_variant(hook_text="a completely unique hook about cooking recipes"),
        niche="zz_test_scoring_hooktext",
    )
    score_b = score_variant_if_model_available(
        _script_variant(hook_text="a totally different hook about weightlifting"),
        niche="zz_test_scoring_hooktext",
    )
    assert score_a != score_b


def _cleanup_content_item(content_item_id: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM creative_dna WHERE content_item_id = %s", (content_item_id,))
            cur.execute("DELETE FROM content_items WHERE id = %s", (content_item_id,))
        conn.commit()
    finally:
        conn.close()


def test_insert_content_item_persists_niche_and_predicted_score():
    """Real DB round-trip: insert_content_item must actually WRITE the
    predicted_score/niche columns (migration 020), not just accept them
    as unused parameters."""
    design_output = DesignOutput(
        asset_path="/fake/path.png", workflow_type="ig_static",
        creative_dna=CreativeDNA(
            content_item_id=uuid.uuid4(), hook_text="real hook", format_tag="carousel",
        ),
    )
    content_item_id = insert_content_item(
        design_output, batch_id=str(uuid.uuid4()), playbook_version="v1",
        write_prompt_version="v1", niche="zz_test_persist_niche", predicted_score=1.75,
    )
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT niche, predicted_score FROM content_items WHERE id = %s",
                    (content_item_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        assert row[0] == "zz_test_persist_niche"
        assert row[1] == 1.75
    finally:
        _cleanup_content_item(content_item_id)


def test_insert_content_item_allows_null_niche_and_score_when_not_scored():
    """Backward-compat/honest-gap case: a batch run without a --niche
    argument, or a niche with no saved model, must persist NULL, not a
    fabricated 0.0 or empty-string placeholder."""
    design_output = DesignOutput(
        asset_path="/fake/path.png", workflow_type="ig_static",
        creative_dna=CreativeDNA(content_item_id=uuid.uuid4(), hook_text="real hook"),
    )
    content_item_id = insert_content_item(
        design_output, batch_id=str(uuid.uuid4()), playbook_version="v1", write_prompt_version="v1",
    )
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT niche, predicted_score FROM content_items WHERE id = %s",
                    (content_item_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        assert row[0] is None
        assert row[1] is None
    finally:
        _cleanup_content_item(content_item_id)
