
from typing import Dict, Any, List
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


class IntermediateValuesCallback(BaseCallbackHandler):

    def __init__(self) -> None:
        super().__init__()
        self.intermediate_values = {}

    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
        """
        Add name of run into intermediate dict
        """
        #self.outputs[run_id] = None
        return super().on_chain_start(serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)

    def on_chain_end(
            self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        """
        Add output of run as value for name of run with no output
        """
        print('here')
        self.intermediate_values.update({str(run_id): outputs})
        return super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    