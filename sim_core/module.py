from .event import Event


class PipelineStage:
    """Template for a single pipeline stage."""

    def __init__(self, module, stage_idx, latency=1):
        self.module = module
        self.stage_idx = stage_idx
        self.latency = latency
        self.next_stage = None

    def process(self, data):
        """Process data for this stage.

        The default implementation simply forwards the data to the next stage
        after the configured latency. Subclasses should override this method for
        custom behaviour.
        """
        if self.module.stalled:
            # re-schedule the same stage for the next cycle when stalled
            self.module.schedule_stage(self.stage_idx, data, self.latency)
            return

        if self.next_stage is not None:
            self.module.schedule_stage(self.next_stage.stage_idx, data,
                                      self.next_stage.latency)
        else:
            self.module.pipeline_output(data)


class HardwareModule:
    def __init__(self, engine, name, mesh_info, num_stages=0):
        self.engine = engine
        self.name = name
        self.mesh_info = mesh_info

        self.stalled = False
        self.pipeline = [PipelineStage(self, i) for i in range(num_stages)]
        for i in range(len(self.pipeline) - 1):
            self.pipeline[i].next_stage = self.pipeline[i + 1]

    def schedule_stage(self, stage_idx, data, delay=1):
        event = Event(
            src=self,
            dst=self,
            cycle=self.engine.current_cycle + delay,
            event_type="PIPELINE_STAGE",
            payload={"stage_idx": stage_idx, "data": data},
            priority=stage_idx,
        )
        self.send_event(event)

    def pipeline_output(self, data):
        """Called when the last pipeline stage finishes processing."""
        pass

    def handle_event(self, event):
        if event.event_type == "PIPELINE_STAGE":
            stage_idx = event.payload["stage_idx"]
            data = event.payload["data"]
            if 0 <= stage_idx < len(self.pipeline):
                self.pipeline[stage_idx].process(data)
            return

    def send_event(self, event):
        self.engine.push_event(event)
