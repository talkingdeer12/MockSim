from sim_core.module import PipelineModule
from sim_core.event import Event

class NPU(PipelineModule):
    def __init__(self, engine, name, mesh_info, pipeline_stages=5, buffer_capacity=4, txn_bytes=128):
        super().__init__(engine, name, mesh_info, pipeline_stages, buffer_capacity)
        # Track per-task DMA activity
        self.expected_dma_reads = {}
        self.received_dma_reads = {}
        self.expected_dma_writes = {}
        self.received_dma_writes = {}

        # Command execution bookkeeping. ``cmd_queue`` holds pending commands
        # while ``current_cmd`` tracks the command currently flowing through the
        # pipeline.
        self.cmd_queue = []
        self.current_cmd = None

        # Map (program, stream_id) identifiers to the requester module so DONE
        # events can be routed correctly.
        self.requester_name_by_prog = {}
        funcs = [self._make_stage_func(i) for i in range(pipeline_stages)]
        self.set_stage_funcs(funcs)

        # Event handler dispatch table. This mirrors the CP style so new
        # operations can be added without editing ``handle_event``.
        self.event_handlers = {}
        self._register_default_handlers()
        # Maximum size of each memory transaction sent to the IOD.
        self.txn_bytes = txn_bytes

    def _make_stage_func(self, idx):
        def func(mod, data):
            return data, idx + 1, False
        return func

    def _on_stage_execute(self, idx):
        """Hook called before executing a pipeline stage."""
        pass

    def _start_next_cmd(self):
        """Issue the next command in ``cmd_queue`` if the pipeline is idle."""
        if self.current_cmd or not self.cmd_queue:
            return

        info = self.cmd_queue.pop(0)
        self.current_cmd = {"info": info, "remaining": info["cycles"]}
        for _ in range(info["cycles"]):
            self.add_data({}, stage_idx=0)

    def register_handler(self, evt_type, fn):
        """Register an event handler function for ``evt_type``."""
        self.event_handlers[evt_type] = fn

    def _register_default_handlers(self):
        self.register_handler("NPU_DMA_IN", self._handle_npu_dma_in)
        self.register_handler("DMA_READ_REPLY", self._handle_dma_read_reply)
        self.register_handler("NPU_CMD", self._handle_npu_cmd)
        self.register_handler("NPU_DMA_OUT", self._handle_npu_dma_out)
        self.register_handler("WRITE_REPLY", self._handle_write_reply)

    def handle_pipeline_output(self, data):
        """Called when a pipeline token exits the final stage."""
        if not self.current_cmd:
            return

        self.current_cmd["remaining"] -= 1
        if self.current_cmd["remaining"] == 0:
            info = self.current_cmd["info"]
            dst_name = info["dst_name"]
            coords = (
                self.mesh_info.get("cp_coords", {}).get(dst_name)
                or self.mesh_info.get("npu_coords", {}).get(dst_name)
                or self.mesh_info.get("iod_coords", {}).get(dst_name)
            )
            evt = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle,
                data_size=4,
                program=info["program"],
                event_type="NPU_CMD_DONE",
                payload={
                    "dst_coords": coords,
                    "npu_name": self.name,
                    "stream_id": info.get("stream_id"),
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(evt)
            self.current_cmd = None
            self._start_next_cmd()

    def handle_event(self, event):
        handler = self.event_handlers.get(event.event_type)
        if handler:
            handler(event)
        else:
            super().handle_event(event)

    # ------------------------------------------------------------------
    # Individual event handlers

    def _handle_npu_dma_in(self, event):
        key = (event.program, event.payload.get("stream_id"))
        total_bytes = event.payload["data_size"]
        self.expected_dma_reads[key] = total_bytes
        self.received_dma_reads[key] = 0
        self.requester_name_by_prog[key] = event.payload["src_name"]
        iod_coords = list(self.mesh_info.get("iod_coords", {}).values())[0]

        txn = self.txn_bytes
        num_txn = (total_bytes + txn - 1) // txn
        for i in range(num_txn):
            size = min(txn, total_bytes - i * txn)
            read_evt = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle + i,
                data_size=size,
                program=event.program,
                event_type="DMA_READ",
                payload={
                    "dst_coords": iod_coords,
                    "src_name": self.name,
                    "need_reply": True,
                    "opcode_cycles": event.payload.get("opcode_cycles", 5),
                    "stream_id": event.payload.get("stream_id"),
                    "eaddr": event.payload.get("eaddr", 0) + i * txn,
                    "iaddr": event.payload.get("iaddr", 0) + i * txn,
                    "data_size": size,
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(read_evt)

    def _handle_dma_read_reply(self, event):
        key = (event.program, event.payload.get("stream_id"))
        size = event.payload.get("data_size", 0)
        self.received_dma_reads[key] += size
        if self.received_dma_reads[key] >= self.expected_dma_reads[key]:
            dst_name = self.requester_name_by_prog.get(key)
            coords = (
                self.mesh_info.get("cp_coords", {}).get(dst_name)
                or self.mesh_info.get("npu_coords", {}).get(dst_name)
                or self.mesh_info.get("iod_coords", {}).get(dst_name)
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
                    "stream_id": event.payload.get("stream_id"),
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(done_evt)
            del self.expected_dma_reads[key]
            del self.received_dma_reads[key]

    def _handle_npu_cmd(self, event):
        cmd = {
            "program": event.program,
            "stream_id": event.payload.get("stream_id"),
            "cycles": event.payload["opcode_cycles"],
            "dst_name": event.payload["src_name"],
        }
        self.cmd_queue.append(cmd)
        self.requester_name_by_prog[(event.program, cmd["stream_id"])] = cmd["dst_name"]
        self._start_next_cmd()

    def _handle_npu_dma_out(self, event):
        key = (event.program, event.payload.get("stream_id"))
        total_bytes = event.payload["data_size"]
        self.expected_dma_writes[key] = total_bytes
        self.received_dma_writes[key] = 0
        self.requester_name_by_prog[key] = event.payload["src_name"]
        iod_coords = list(self.mesh_info.get("iod_coords", {}).values())[0]
        txn = self.txn_bytes
        num_txn = (total_bytes + txn - 1) // txn
        for i in range(num_txn):
            size = min(txn, total_bytes - i * txn)
            wr_evt = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle + i,
                data_size=size,
                program=event.program,
                event_type="DMA_WRITE",
                payload={
                    "dst_coords": iod_coords,
                    "src_name": self.name,
                    "need_reply": True,
                    "opcode_cycles": event.payload.get("opcode_cycles", 5),
                    "stream_id": event.payload.get("stream_id"),
                    "eaddr": event.payload.get("eaddr", 0) + i * txn,
                    "iaddr": event.payload.get("iaddr", 0) + i * txn,
                    "data_size": size,
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(wr_evt)

    def _handle_write_reply(self, event):
        key = (event.program, event.payload.get("stream_id"))
        size = event.payload.get("data_size", 0)
        self.received_dma_writes[key] += size
        if self.received_dma_writes[key] >= self.expected_dma_writes[key]:
            dst_name = self.requester_name_by_prog.get(key)
            coords = (
                self.mesh_info.get("cp_coords", {}).get(dst_name)
                or self.mesh_info.get("npu_coords", {}).get(dst_name)
                or self.mesh_info.get("iod_coords", {}).get(dst_name)
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
                    "stream_id": event.payload.get("stream_id"),
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(done_evt)
            del self.expected_dma_writes[key]
            del self.received_dma_writes[key]

    def get_my_router(self):
        coords = self.mesh_info["npu_coords"][self.name]
        return self.mesh_info["router_map"][coords]
