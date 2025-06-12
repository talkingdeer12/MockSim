from .event import Event


class PipelineStage:
    """Simple event-driven pipeline stage."""

    def __init__(self, module, name, latency=1, depth=1, next_stage=None):
        self.module = module
        self.name = name
        self.latency = latency
        self.depth = depth
        self.next_stage = next_stage
        self.queue = []

    def is_full(self):
        return len(self.queue) >= self.depth

    def push(self, item, latency=None):
        if self.is_full():
            return False
        eff_latency = latency if latency is not None else self.latency
        self.queue.append(item)
        ev = Event(
            src=self.module,
            dst=self.module,
            cycle=self.module.engine.current_cycle + eff_latency,
            event_type="PIPELINE_ADVANCE",
            payload={"stage": self.name},
        )
        self.module.send_event(ev)
        return True

    def advance(self):
        if not self.queue:
            return
        item = self.queue[0]
        if self.next_stage:
            if self.next_stage.is_full():
                # Stall and retry next cycle
                ev = Event(
                    src=self.module,
                    dst=self.module,
                    cycle=self.module.engine.current_cycle + 1,
                    event_type="PIPELINE_ADVANCE",
                    payload={"stage": self.name},
                )
                self.module.send_event(ev)
                return
            self.queue.pop(0)
            self.next_stage.push(item)
        else:
            self.queue.pop(0)
            self.module.on_pipeline_complete(self.name, item)


class HardwareModule:
    def __init__(self, engine, name, mesh_info):
        self.engine = engine
        self.name = name
        self.mesh_info = mesh_info
        self.pipeline_stages = {}

    def add_stage(self, stage):
        self.pipeline_stages[stage.name] = stage

    def handle_event(self, event):
        if event.event_type == "PIPELINE_ADVANCE":
            stage_name = event.payload["stage"]
            stage = self.pipeline_stages.get(stage_name)
            if stage:
                stage.advance()
            return
        self.handle_event_module(event)

    def handle_event_module(self, event):
        """Subclasses should override to handle external events."""
        pass

    def on_pipeline_complete(self, stage_name, item):
        """Called when the final stage completes."""
        pass

    def send_event(self, event):
        self.engine.push_event(event)
