from sim_core.module import PipelineModule
from sim_core.event import Event

class NPU(PipelineModule):
    def __init__(self, engine, name, mesh_info, pipeline_stages=5, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, pipeline_stages, buffer_capacity)
        self.expected_dma_reads = {}
        self.received_dma_reads = {}
        self.expected_dma_writes = {}
        self.received_dma_writes = {}
        self.opcode_cycles_remaining = 0
        self.opcode_total_cycles = 0
        self.program_identifier = None
        # Track the module name that issued the current command
        self.requester_name = None
        # Map program identifiers to the module that last interacted with them
        self.requester_name_by_prog = {}
        self.cmd_need_reply = False
        funcs = [self._make_stage_func(i) for i in range(pipeline_stages)]
        self.set_stage_funcs(funcs)

    def _make_stage_func(self, idx):
        def func(mod, data):
            return data, idx + 1, False
        return func

    def _on_stage_execute(self, idx):
        """Hook called before executing a pipeline stage."""
        pass

    def handle_pipeline_output(self, data):
        if not self.opcode_total_cycles:
            return

        self.opcode_cycles_remaining -= 1
        if self.opcode_cycles_remaining == 0:
            dst_name = data.get(
                "dst_name",
                self.requester_name_by_prog.get(
                    self.program_identifier, self.requester_name
                ),
            )
            coords = (
                self.mesh_info.get("cp_coords", {}).get(dst_name)
                or self.mesh_info.get("pe_coords", {}).get(dst_name)
                or self.mesh_info.get("npu_coords", {}).get(dst_name)
                or self.mesh_info.get("dram_coords", {}).get(dst_name)
            )
            evt = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle,
                data_size=4,
                program=self.program_identifier,
                event_type="NPU_CMD_DONE",
                payload={
                    "dst_coords": coords,
                    "npu_name": self.name,
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(evt)
            self.opcode_total_cycles = 0
            self.program_identifier = None
            self.requester_name = None
            self.cmd_need_reply = False

    def handle_event(self, event):
        if event.event_type == "NPU_DMA_IN":
            total = event.payload["data_size"] // 4
            self.expected_dma_reads[event.program] = total
            self.received_dma_reads[event.program] = 0
            self.requester_name = event.payload["src_name"]
            self.requester_name_by_prog[event.program] = self.requester_name
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
                        "opcode_cycles": event.payload.get("opcode_cycles", 5),
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(read_evt)
        elif event.event_type == "DMA_READ_REPLY":
            self.received_dma_reads[event.program] += 1
            if self.received_dma_reads[event.program] >= self.expected_dma_reads[event.program]:
                dst_name = self.requester_name_by_prog.get(event.program, self.requester_name)
                coords = (
                    self.mesh_info.get("cp_coords", {}).get(dst_name)
                    or self.mesh_info.get("pe_coords", {}).get(dst_name)
                    or self.mesh_info.get("npu_coords", {}).get(dst_name)
                    or self.mesh_info.get("dram_coords", {}).get(dst_name)
                )
                done_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=4,
                    program=event.program,
                    event_type="NPU_DMA_IN_DONE",
                    payload={
                        "dst_coords": coords,
                        "npu_name": self.name,
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(done_evt)
                del self.expected_dma_reads[event.program]
                del self.received_dma_reads[event.program]
        elif event.event_type == "NPU_CMD":
            cycles = event.payload["opcode_cycles"]
            self.opcode_cycles_remaining = cycles
            self.opcode_total_cycles = cycles
            self.program_identifier = event.program
            self.requester_name = event.payload["src_name"]
            self.requester_name_by_prog[event.program] = self.requester_name
            need_reply = event.payload.get("need_reply", False)
            for _ in range(cycles):
                token = {}
                if need_reply:
                    token["dst_name"] = self.requester_name
                self.add_data(token, stage_idx=0)
            self.cmd_need_reply = need_reply
        elif event.event_type == "NPU_DMA_OUT":
            total = event.payload["data_size"] // 4
            self.expected_dma_writes[event.program] = total
            self.received_dma_writes[event.program] = 0
            self.requester_name = event.payload["src_name"]
            self.requester_name_by_prog[event.program] = self.requester_name
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
                        "opcode_cycles": event.payload.get("opcode_cycles", 5),
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(wr_evt)
        elif event.event_type == "WRITE_REPLY":
            self.received_dma_writes[event.program] += 1
            if self.received_dma_writes[event.program] >= self.expected_dma_writes[event.program]:
                dst_name = self.requester_name_by_prog.get(event.program, self.requester_name)
                coords = (
                    self.mesh_info.get("cp_coords", {}).get(dst_name)
                    or self.mesh_info.get("pe_coords", {}).get(dst_name)
                    or self.mesh_info.get("npu_coords", {}).get(dst_name)
                    or self.mesh_info.get("dram_coords", {}).get(dst_name)
                )
                done_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=4,
                    program=event.program,
                    event_type="NPU_DMA_OUT_DONE",
                    payload={
                        "dst_coords": coords,
                        "npu_name": self.name,
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(done_evt)
                del self.expected_dma_writes[event.program]
                del self.received_dma_writes[event.program]
                # Do not remove mapping so that future operations can use it
        else:
            super().handle_event(event)

    def get_my_router(self):
        coords = self.mesh_info["npu_coords"][self.name]
        return self.mesh_info["router_map"][coords]
