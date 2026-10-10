import pytest
from pydantic import ValidationError
from qualume.models.contracts import CampaignBrief, ScriptVariant


def test_campaign_brief_requires_playbook_version():
    with pytest.raises(ValidationError):
        CampaignBrief(product_or_topic="x", offer_or_goal="y", constraints=[])


def test_campaign_brief_rejects_out_of_range_ratio():
    with pytest.raises(ValidationError):
        CampaignBrief(
            product_or_topic="x", offer_or_goal="y",
            playbook_version="v0", explore_exploit_ratio=1.5
        )


def test_script_variant_rejects_invalid_explore_flag():
    with pytest.raises(ValidationError):
        ScriptVariant(
            hook_text="h", body_text="b", cta_text="c",
            playbook_version="v0", prompt_version="v0",
            explore_or_exploit="maybe"
        )
