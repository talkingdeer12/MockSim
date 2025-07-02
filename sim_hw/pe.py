from sim_core.module import PipelineModule
from sim_core.event import Event
import random

class PE(PipelineModule):
    def __init__(self, engine, name, mesh_info, mac_units=32, mac_width=32, pipeline_stages=5, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, pipeline_stages, buffer_capacity)
        self.mac_units = mac_units
        self.mac_width = mac_width
        self.expected_dma_reads = {}
        self.received_dma_reads = {}
        self.expected_dma_writes = {}
        self.received_dma_writes = {}
        self.gemm_cycles_remaining = 0
        self.gemm_total_cycles = 0
        self.gemm_program = None
        self.cp_name = None
        funcs = [self._make_stage_func(i) for i in range(pipeline_stages)]
        self.set_stage_funcs(funcs)

    def _make_stage_func(self, idx):
        def func(mod, data):
            if random.random() < 0.2:
                mod.set_stall(1)
                return data, idx, True
            return data, idx + 1, False
        return func

    def _on_stage_execute(self, idx):
        if idx == 0 and self.gemm_cycles_remaining > 0 and len(self.stage_queues[0]) < self.stage_capacity:
            self.stage_queues[0].append({})
            self.gemm_cycles_remaining -= 1
        if idx == 0 and (self.gemm_cycles_remaining > 0 or self.stage_queues[0]):
            self._schedule_stage(0)

    def handle_pipeline_output(self, data):
        if self.gemm_total_cycles and self.gemm_cycles_remaining == 0 and not any(self.stage_queues):
            evt = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle,
                data_size=4,
                program=self.gemm_program,
                event_type="PE_GEMM_DONE",
                payload={
                    "dst_coords": self.mesh_info["cp_coords"][self.cp_name],
                    "pe_name": self.name,
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(evt)
            self.gemm_total_cycles = 0
            self.gemm_program = None
            self.cp_name = None

    def handle_event(self, event):
        if event.event_type == "PE_DMA_IN":
            total = event.payload["data_size"] // 4
            self.expected_dma_reads[event.program] = total
            self.received_dma_reads[event.program] = 0
            self.cp_name = event.payload["src_name"]
            dram_coords = list(self.mesh_info["dram_coords"].values())[0]
            for i in range(total):
                read_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle + i,
                    data_size=4,
                    program=event.program,
                    event_type="DMA_READ",
                    payload={
                        "dst_coords": dram_coords,
                        "src_name": self.name,
                        "need_reply": True,
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(read_evt)
        elif event.event_type == "DMA_READ_REPLY":
            self.received_dma_reads[event.program] += 1
            if self.received_dma_reads[event.program] >= self.expected_dma_reads[event.program]:
                done_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=4,
                    program=event.program,
                    event_type="PE_DMA_IN_DONE",
                    payload={
                        "dst_coords": self.mesh_info["cp_coords"][self.cp_name],
                        "pe_name": self.name,
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(done_evt)
                del self.expected_dma_reads[event.program]
                del self.received_dma_reads[event.program]
        elif event.event_type == "PE_GEMM":
            M, N, K = event.payload["gemm_shape"]
            cycles = (M * N * K + self.mac_units - 1) // self.mac_units
            self.gemm_cycles_remaining = cycles
            self.gemm_total_cycles = cycles
            self.gemm_program = event.program
            self.cp_name = event.payload["src_name"]
            self._schedule_stage(0)
        elif event.event_type == "PE_DMA_OUT":
            total = event.payload["data_size"] // 4
            self.expected_dma_writes[event.program] = total
            self.received_dma_writes[event.program] = 0
            self.cp_name = event.payload["src_name"]
            dram_coords = list(self.mesh_info["dram_coords"].values())[0]
            for i in range(total):
                wr_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle + i,
                    data_size=4,
                    program=event.program,
                    event_type="DMA_WRITE",
                    payload={
                        "dst_coords": dram_coords,
                        "src_name": self.name,
                        "need_reply": True,
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(wr_evt)
        elif event.event_type == "WRITE_REPLY":
            self.received_dma_writes[event.program] += 1
            if self.received_dma_writes[event.program] >= self.expected_dma_writes[event.program]:
                done_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=4,
                    program=event.program,
                    event_type="PE_DMA_OUT_DONE",
                    payload={
                        "dst_coords": self.mesh_info["cp_coords"][self.cp_name],
                        "pe_name": self.name,
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(done_evt)
                del self.expected_dma_writes[event.program]
                del self.received_dma_writes[event.program]
        else:
            super().handle_event(event)

    def get_my_router(self):
        coords = self.mesh_info["pe_coords"][self.name]
        return self.mesh_info["router_map"][coords]
