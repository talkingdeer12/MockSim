from .event import Event
from typing import Any, Callable, Tuple


def basic_stage_template(stage_idx: int) -> Callable[["PipelineModule", Any], Tuple[Any, int, bool]]:
    """Create a pipeline stage function skeleton.

    Parameters
    ----------
    stage_idx : int
        Index of the stage being defined.

    Returns
    -------
    Callable[[PipelineModule, Any], Tuple[Any, int, bool]]
        A function with the signature ``func(module, data)`` that processes one
        cycle of work and returns ``(data, next_stage, do_stall)``.

    The returned function demonstrates common operations such as sending or
    receiving events. Replace the placeholder logic with module specific code.

    Example
    -------
    >>> from sim_core.module import PipelineModule
    >>> from sim_core.stage_templates import basic_stage_template
    >>> class DummyPipe(PipelineModule):
    ...     def __init__(self, engine):
    ...         super().__init__(engine, "Dummy", {}, 2)
    ...         funcs = [basic_stage_template(i) for i in range(2)]
    ...         self.set_stage_funcs(funcs)
    ...     def handle_pipeline_output(self, data):
    ...         print(f"output={data}")
    >>> module = DummyPipe(engine)
    >>> module.add_data({"value": 1})
    >>> engine.run_until_idle()
    output={'value': 1}
    """

    def stage_func(module, data):
        """Template for a single pipeline stage."""
        # Example: send an event to another module
        # evt = Event(
        #     src=module,
        #     dst=some_other_module,
        #     cycle=module.engine.current_cycle + 1,
        #     event_type="MY_EVENT",
        #     payload={}
        # )
        # module.send_event(evt)

        # Example: check conditions and optionally stall
        # if not module.can_accept_event():
        #     return data, stage_idx, True

        # TODO: modify ``data`` or compute results here
        output = data

        # Move to the next stage by default
        next_stage = stage_idx + 1

        # Return ``True`` as the last value to stall this stage
        return output, next_stage, False

    return stage_func
