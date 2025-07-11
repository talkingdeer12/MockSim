from .event import Event


class HardwareModule:
    def __init__(self, engine, name, mesh_info, buffer_capacity=4, frequency=1000):
        self.engine = engine
        self.name = name
        self.mesh_info = mesh_info
        self.buffer_capacity = buffer_capacity
        self.buffer_occupancy = 0
        self.frequency = frequency  # MHz

    # Credit based buffer bookkeeping
    def _reserve_slot(self, event=None):
        if self.buffer_occupancy >= self.buffer_capacity:
            return False
        self.buffer_occupancy += 1
        return True

    def _release_slot(self, event=None):
        if self.buffer_occupancy > 0:
            self.buffer_occupancy -= 1

    def can_accept_event(self, event=None):
        return self.buffer_occupancy < self.buffer_capacity

    def _process_event(self, event):
        try:
            if self.engine.logger:
                stage = (
                    event.payload.get("stage_idx", 0)
                    if isinstance(event.payload, dict)
                    else 0
                )
                evt_type = event.event_type
                if isinstance(event.payload, dict) and event.payload.get("op_type"):
                    evt_type = f"{evt_type}-{event.payload['op_type']}"
                self.engine.logger.log_event(
                    self.engine.current_cycle, self.name, stage, evt_type
                )
            self.handle_event(event)
        finally:
            self._release_slot(event)

    def handle_event(self, event):
        if event.event_type == "RETRY_SEND":
            retry_evt = event.payload["event"]
            self.send_event(retry_evt)

    def send_event(self, event):
        if not event.dst._reserve_slot(event):
            # Destination buffer full; stall and retry next cycle
            if hasattr(self, "set_stall"):
                self.set_stall(1)
            retry = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle + 1,
                event_type="RETRY_SEND",
                payload={"event": event},
            )
            self.engine.push_event(retry)
        else:
            self.engine.push_event(event)


class PipelineModule(HardwareModule):
    """Base class for modules with event driven pipelined execution."""

    def __init__(self, engine, name, mesh_info, num_stages, buffer_capacity=4, frequency=1000):
        super().__init__(engine, name, mesh_info, buffer_capacity, frequency)
        self.num_stages = num_stages
        self.stage_funcs = [lambda m, d: (d, i + 1, False) for i in range(num_stages)]
        self.stage_queues = [list() for _ in range(num_stages)]
        self.stage_scheduled = [False for _ in range(num_stages)]
        self.stage_capacity = buffer_capacity

    def set_stage_funcs(self, funcs):
        if len(funcs) != self.num_stages:
            raise ValueError("stage_funcs length must match num_stages")
        self.stage_funcs = funcs

    def add_data(self, data, stage_idx=0):
        if stage_idx >= self.num_stages:
            raise ValueError("stage_idx out of range")
        self.stage_queues[stage_idx].append(data)
        self._schedule_stage(stage_idx)

    def _schedule_stage(self, idx):
        if not self.stage_scheduled[idx]:
            evt = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle + 1,
                event_type="PIPE_STAGE",
                payload={"stage_idx": idx},
                priority=-idx,
            )
            self.send_event(evt)
            self.stage_scheduled[idx] = True

    def handle_event(self, event):
        if event.event_type == "PIPE_STAGE":
            idx = event.payload["stage_idx"]
            self.stage_scheduled[idx] = False
            self._on_stage_execute(idx)
            self._execute_stage(idx)
        else:
            super().handle_event(event)

    def set_stall(self, cycles):
        # Backwards compatibility for send_event based stalling.
        pass

    def _on_stage_execute(self, idx):
        """Hook called before processing a stage."""
        pass

    def _execute_stage(self, idx):
        if not self.stage_queues[idx]:
            return
        data = self.stage_queues[idx][0]
        func = self.stage_funcs[idx]
        out_data, next_stage, do_stall = func(self, data)
        if do_stall:
            self._schedule_stage(idx)
            return

        if (
            next_stage < self.num_stages
            and len(self.stage_queues[next_stage]) >= self.stage_capacity
        ):
            # downstream stage full - retry later
            self._schedule_stage(idx)
            return

        self.stage_queues[idx].pop(0)

        if next_stage >= self.num_stages:
            self.handle_pipeline_output(out_data)
        else:
            self.stage_queues[next_stage].append(out_data)
            self._schedule_stage(next_stage)

        if self.stage_queues[idx]:
            self._schedule_stage(idx)

    def handle_pipeline_output(self, data):
        pass


