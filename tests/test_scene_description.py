"""Tests for v6 presenter/avatar description — written FIRST per this
project's TDD discipline.

The user's own target example: "this vid was a mom in her kitchen
passionately talking about night sweats". derive_scene_description is
the pure function that builds exactly that sentence shape from
structured presenter + setting + topic fields, so the database can
answer "describe this video" in plain language without a human
re-watching it.

Deliberately scoped to OBSERVABLE PRESENTATION only — apparent age
bucket and apparent gender presentation from how a person visibly
presents on camera, never identity inference. Same purpose as real
"avatar testing" in paid media buying (deciding who to CAST in future
content), not profiling the real person in someone else's video. See
PresenterDescription's docstring in contracts.py.
"""
from qualume.annotate.gemini_watcher import derive_scene_description
from qualume.models.contracts import PresenterDescription


def _mom_presenter(**overrides):
    data = {
        "apparent_age_range": "30s",
        "apparent_gender_presentation": "female",
        "relatable_archetype": "mom",
        "wardrobe_style": "casual loungewear",
        "delivery_affect": "passionately talking",
    }
    data.update(overrides)
    return PresenterDescription(**data)


def test_derive_scene_description_matches_the_users_own_target_example():
    """The exact shape the user asked for: 'this vid was a mom in her
    kitchen passionately talking about night sweats'."""
    sentence = derive_scene_description(
        presenters=[_mom_presenter()],
        setting="kitchen",
        video_topic_summary="night sweats and hot flashes during menopause",
    )
    assert sentence == (
        "A mom in a kitchen, passionately talking about night sweats and "
        "hot flashes during menopause."
    )


def test_derive_scene_description_no_presenter():
    """Product-only shots/voiceover-over-broll with nobody visibly on
    screen - must still produce a sensible sentence, not crash or
    silently omit the video entirely."""
    sentence = derive_scene_description(
        presenters=[],
        setting="kitchen counter",
        video_topic_summary="a 3-ingredient overnight oats recipe",
    )
    assert sentence == "Product/visual-only footage in a kitchen counter, about a 3-ingredient overnight oats recipe."


def test_derive_scene_description_multiple_presenters():
    """Two people in conversation - both should be reflected, not just
    the first one silently dropped."""
    presenters = [
        _mom_presenter(relatable_archetype="nurse", wardrobe_style="medical scrubs",
                       delivery_affect="calmly explaining"),
        _mom_presenter(apparent_age_range="20s", relatable_archetype="patient",
                       delivery_affect="nervously asking"),
    ]
    sentence = derive_scene_description(
        presenters=presenters, setting="clinic exam room",
        video_topic_summary="what to expect at a first prenatal visit",
    )
    assert "nurse" in sentence
    assert "patient" in sentence
    assert "clinic exam room" in sentence
    assert "what to expect at a first prenatal visit" in sentence


def test_derive_scene_description_no_setting_known():
    """shot_breakdown can be empty/unavailable (degraded-path annotation)
    - must degrade gracefully, not crash on a missing setting."""
    sentence = derive_scene_description(
        presenters=[_mom_presenter()], setting=None,
        video_topic_summary="a morning skincare routine",
    )
    assert sentence == "A mom, passionately talking about a morning skincare routine."


def test_derive_scene_description_no_topic_known():
    sentence = derive_scene_description(
        presenters=[_mom_presenter()], setting="bathroom", video_topic_summary=None,
    )
    assert sentence == "A mom in a bathroom, passionately talking."
