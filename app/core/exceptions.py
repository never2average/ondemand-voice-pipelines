class PipelineNotFoundError(Exception):
    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        super().__init__(f"Pipeline {pipeline_id} not found")


class PipelineNotReadyError(Exception):
    def __init__(self, pipeline_id: str, status: str):
        self.pipeline_id = pipeline_id
        self.status = status
        super().__init__(f"Pipeline {pipeline_id} is not ready (status={status})")


class ASRProviderError(Exception):
    pass


class IntentExtractionError(Exception):
    pass


class AgentError(Exception):
    pass
