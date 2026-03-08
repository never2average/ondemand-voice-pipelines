import pytest

from app.agents.intent_schema import IntentSchemaAgent
from app.agents.base import PipelineBuildContext
from app.schemas.pipeline import OptimizationObjective


@pytest.mark.asyncio
async def test_intent_schema_agent_parses_multiple_intents():
    agent = IntentSchemaAgent()
    result = await agent.run(
        PipelineBuildContext(
            pipeline_id="pipe-1",
            pipeline_name="test",
            pipeline_description="desc",
            intent_prompt="check balance, transfer funds, dispute charge",
            asr_provider="whisper",
            optimization_objective=OptimizationObjective(),
        )
    )
    assert len(result.artifact.intents) == 3
    assert result.artifact.intents[0].intent_name == "check_balance"
    assert result.artifact.fallback_intent == "unknown"


@pytest.mark.asyncio
async def test_intent_schema_agent_parses_nontechnical_prompt_into_clean_intents():
    agent = IntentSchemaAgent()
    result = await agent.run(
        PipelineBuildContext(
            pipeline_id="pipe-2",
            pipeline_name="banking-router",
            pipeline_description="desc",
            intent_prompt="\n".join(
                [
                    "I'm setting up a phone support line for a retail bank.",
                    "Customers usually say things like:",
                    '- "I want to check my balance"',
                    '- "I need to transfer money between accounts"',
                    '- "I need to dispute a charge on my card"',
                    '- "I need a replacement card because mine is lost"',
                    "If it's something else, send it to unknown.",
                ]
            ),
            asr_provider="whisper",
            optimization_objective=OptimizationObjective(),
        )
    )

    assert [intent.intent_name for intent in result.artifact.intents] == [
        "check_balance",
        "transfer_money_between_accounts",
        "dispute_charge_card",
        "replacement_card_lost",
    ]
    assert result.artifact.fallback_intent == "unknown"
