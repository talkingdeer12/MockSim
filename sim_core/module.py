from .event import Event


class HardwareModule:
    def __init__(self, engine, name, mesh_info):
        self.engine = engine
        self.name = name
        self.mesh_info = mesh_info
    
    def handle_event(self, event):
        pass

    def send_event(self, event):
        self.engine.push_event(event)


class PipelineModule(HardwareModule):
    """Base class for modules with simple pipelined execution."""

    def __init__(self, engine, name, mesh_info, num_stages):
        super().__init__(engine, name, mesh_info)
        self.num_stages = num_stages
        self.stage_funcs = [lambda m, d: (d, i + 1, False)
                           for i in range(num_stages)]
        self.stage_queues = [list() for _ in range(num_stages)]
        self.output_queue = []
        self.stall = False
        self.stall_cycles = 0
        self.tick_scheduled = False

    def set_stage_funcs(self, funcs):
        if len(funcs) != self.num_stages:
            raise ValueError("stage_funcs length must match num_stages")
        self.stage_funcs = funcs

    def add_data(self, data, stage_idx=0):
        if stage_idx >= self.num_stages:
            raise ValueError("stage_idx out of range")
        self.stage_queues[stage_idx].append(data)
        self._schedule_tick()

    def _schedule_tick(self):
        if not self.tick_scheduled:
            evt = Event(src=self, dst=self, cycle=self.engine.current_cycle + 1,
                        event_type="PIPELINE_TICK", payload={})
            self.send_event(evt)
            self.tick_scheduled = True

    def handle_event(self, event):
        if event.event_type == "PIPELINE_TICK":
            self.tick_scheduled = False
            self._pipeline_step()
        else:
            super().handle_event(event)

    def set_stall(self, cycles):
        self.stall = True
        self.stall_cycles = max(self.stall_cycles, cycles)

    def _pipeline_step(self):
        if self.stall:
            if self.stall_cycles > 0:
                self.stall_cycles -= 1
            if self.stall_cycles == 0:
                self.stall = False
            self._schedule_tick()
            return

        for idx in reversed(range(self.num_stages)):
            if not self.stage_queues[idx]:
                continue
            data = self.stage_queues[idx][0]
            func = self.stage_funcs[idx]
            out_data, next_stage, do_stall = func(self, data)
            if do_stall:
                self.set_stall(1)
                continue
            self.stage_queues[idx].pop(0)
            if next_stage >= self.num_stages:
                self.output_queue.append(out_data)
            else:
                self.stage_queues[next_stage].append(out_data)

        while self.output_queue:
            item = self.output_queue.pop(0)
            self.handle_pipeline_output(item)

        # if there is more data to process or stalled, schedule next tick
        active = any(self.stage_queues) or self.stall
        if active:
            self._schedule_tick()

    def handle_pipeline_output(self, data):
        pass
