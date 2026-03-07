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
