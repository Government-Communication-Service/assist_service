from app.smart_targets.service import SmartTargetsService


class TestSmartTargetsService:
    async def test_no_metric_selected(self, mock_get_available_metrics, mock_llm_smart_targets_choice, mock_messages):
        service = SmartTargetsService()
        available_metrics = await service.get_available_metrics()
        r = await service._select_metrics(messages=mock_messages, available_metrics=available_metrics)
        assert len(r) == 0, f"Expected no metrics to be selected, got {len(r)}"
