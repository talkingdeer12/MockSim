from sim_core.module import PipelineModule
from sim_core.event import Event

class NPU(PipelineModule):
    def __init__(self, engine, name, mesh_info, pipeline_stages=5, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, pipeline_stages, buffer_capacity)
        self.expected_dma_reads = {}
        self.received_dma_reads = {}
        self.expected_dma_writes = {}
        self.received_dma_writes = {}
        self.task_cycles_remaining = 0
        self.task_total_cycles = 0
        self.task_identifier = None
        self.cp_name = None
        funcs = [self._make_stage_func(i) for i in range(pipeline_stages)]
        self.set_stage_funcs(funcs)

    def _make_stage_func(self, idx):
        def func(mod, data):
            return data, idx + 1, False
        return func

    def _on_stage_execute(self, idx):
        if idx == 0 and self.task_cycles_remaining > 0 and len(self.stage_queues[0]) < self.stage_capacity:
            self.stage_queues[0].append({})
            self.task_cycles_remaining -= 1
        if idx == 0 and (self.task_cycles_remaining > 0 or self.stage_queues[0]):
            self._schedule_stage(0)

    def handle_pipeline_output(self, data):
        if self.task_total_cycles and self.task_cycles_remaining == 0 and not any(self.stage_queues):
            evt = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle,
                data_size=4,
                identifier=self.task_identifier,
                event_type="NPU_CMD_DONE",
                payload={
                    "dst_coords": self.mesh_info["cp_coords"][self.cp_name],
                    "npu_name": self.name,
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(evt)
            self.task_total_cycles = 0
            self.task_identifier = None
            self.cp_name = None

    def handle_event(self, event):
        if event.event_type == "NPU_DMA_IN":
            total = event.payload["data_size"] // 4
            self.expected_dma_reads[event.identifier] = total
            self.received_dma_reads[event.identifier] = 0
            self.cp_name = event.payload["src_name"]
            dram_coords = list(self.mesh_info["dram_coords"].values())[0]
            for i in range(total):
                read_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle + i,
                    data_size=4,
                    identifier=event.identifier,
                    event_type="DMA_READ",
                    payload={
                        "dst_coords": dram_coords,
                        "src_name": self.name,
                        "need_reply": True,
                        "task_cycles": event.payload.get("task_cycles", 5),
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(read_evt)
        elif event.event_type == "DMA_READ_REPLY":
            self.received_dma_reads[event.identifier] += 1
            if self.received_dma_reads[event.identifier] >= self.expected_dma_reads[event.identifier]:
                done_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=4,
                    identifier=event.identifier,
                    event_type="NPU_DMA_IN_DONE",
                    payload={
                        "dst_coords": self.mesh_info["cp_coords"][self.cp_name],
                        "npu_name": self.name,
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(done_evt)
                del self.expected_dma_reads[event.identifier]
                del self.received_dma_reads[event.identifier]
        elif event.event_type == "NPU_CMD":
            cycles = event.payload["task_cycles"]
            self.task_cycles_remaining = cycles
            self.task_total_cycles = cycles
            self.task_identifier = event.identifier
            self.cp_name = event.payload["src_name"]
            self._schedule_stage(0)
        elif event.event_type == "NPU_DMA_OUT":
            total = event.payload["data_size"] // 4
            self.expected_dma_writes[event.identifier] = total
            self.received_dma_writes[event.identifier] = 0
            self.cp_name = event.payload["src_name"]
            dram_coords = list(self.mesh_info["dram_coords"].values())[0]
            for i in range(total):
                wr_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle + i,
                    data_size=4,
                    identifier=event.identifier,
                    event_type="DMA_WRITE",
                    payload={
                        "dst_coords": dram_coords,
                        "src_name": self.name,
                        "need_reply": True,
                        "task_cycles": event.payload.get("task_cycles", 5),
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(wr_evt)
        elif event.event_type == "WRITE_REPLY":
            self.received_dma_writes[event.identifier] += 1
            if self.received_dma_writes[event.identifier] >= self.expected_dma_writes[event.identifier]:
                done_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=4,
                    identifier=event.identifier,
                    event_type="NPU_DMA_OUT_DONE",
                    payload={
                        "dst_coords": self.mesh_info["cp_coords"][self.cp_name],
                        "npu_name": self.name,
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(done_evt)
                del self.expected_dma_writes[event.identifier]
                del self.received_dma_writes[event.identifier]
        else:
            super().handle_event(event)

    def get_my_router(self):
        coords = self.mesh_info["npu_coords"][self.name]
        return self.mesh_info["router_map"][coords]
