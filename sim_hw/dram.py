from sim_core.module import PipelineModule
from sim_core.event import Event

class DRAM(PipelineModule):
    """DRAM model with multiple independent channels."""

    def __init__(
        self,
        engine,
        name,
        mesh_info,
        pipeline_latency=5,
        num_channels=1,
        buffer_capacity=4,
    ):
        super().__init__(engine, name, mesh_info, 1, buffer_capacity)
        self.pipeline_latency = pipeline_latency
        self.num_channels = num_channels
        self.channel_queues = [[] for _ in range(num_channels)]
        self.channel_scheduled = [False] * num_channels
        self.last_channel = -1
        self.set_stage_funcs([self._stage_func])

        # Event dispatch table to make adding new opcodes easy.
        self.event_handlers = {}
        self._register_default_handlers()

    def _stage_func(self, mod, data):
        data["remaining"] -= 1
        if data["remaining"] > 0:
            return data, 0, True
        return data, 1, False

    def _select_channel(self, op):
        """Choose a channel for the incoming request."""
        self.last_channel = (self.last_channel + 1) % self.num_channels
        return self.last_channel

    def _schedule_channel(self, ch):
        if not self.channel_scheduled[ch]:
            payload = {"channel_id": ch}
            if self.channel_queues[ch]:
                payload["op_type"] = self.channel_queues[ch][0]["type"]
            evt = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle + 1,
                event_type="DRAM_CHANNEL",
                payload=payload,
            )
            self.send_event(evt)
            self.channel_scheduled[ch] = True

    def register_handler(self, evt_type, fn):
        """Register handler ``fn`` for ``evt_type``."""
        self.event_handlers[evt_type] = fn

    def _register_default_handlers(self):
        self.register_handler("DMA_WRITE", self._handle_dma_access)
        self.register_handler("DMA_READ", self._handle_dma_access)
        self.register_handler("DRAM_CHANNEL", self._handle_dram_channel)

    def handle_event(self, event):
        handler = self.event_handlers.get(event.event_type)
        if handler:
            handler(event)
        else:
            super().handle_event(event)

    # ------------------------------------------------------------------
    # Individual handlers

    def _handle_dma_access(self, event):
        op = {
            "type": event.event_type,
            "program": event.program,
            "src_name": event.payload["src_name"],
            "remaining": event.payload.get("opcode_cycles", self.pipeline_latency),
            "stream_id": event.payload.get("stream_id"),
        }
        if event.payload.get("need_reply"):
            op["dst_name"] = event.payload["src_name"]
        ch = self._select_channel(op)
        print(
            f"[DRAM] enqueue {op['type']} stream={op.get('stream_id')} ch={ch} cycle={self.engine.current_cycle}"
        )
        op["channel_id"] = ch
        self.channel_queues[ch].append(op)
        self._schedule_channel(ch)

    def _handle_dram_channel(self, event):
        ch = event.payload["channel_id"]
        self.channel_scheduled[ch] = False
        if not self.channel_queues[ch]:
            return
        op = self.channel_queues[ch][0]
        op["remaining"] -= 1
        if op["remaining"] > 0:
            self._schedule_channel(ch)
            return
        self.channel_queues[ch].pop(0)
        self.handle_pipeline_output(op)
        if self.channel_queues[ch]:
            self._schedule_channel(ch)

    def handle_pipeline_output(self, op):
        if op["type"] == "DMA_WRITE":
            evt_type = "WRITE_REPLY"
        else:
            evt_type = "DMA_READ_REPLY"
        if "dst_name" in op:
            dst_name = op["dst_name"]
            coords = (self.mesh_info.get("pe_coords", {}).get(dst_name)
                      or self.mesh_info.get("npu_coords", {}).get(dst_name)
                      or self.mesh_info.get("cp_coords", {}).get(dst_name))
            print(
                f"[DRAM] complete {op['type']} ch={op.get('channel_id')} stream={op.get('stream_id')} cycle={self.engine.current_cycle}"
            )
            reply_event = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle,
                data_size=4,
                program=op["program"],
                event_type=evt_type,
                payload={
                    "dst_coords": coords,
                    "input_port": 0,
                    "vc": 0,
                    "stream_id": op.get("stream_id"),
                    "channel_id": op.get("channel_id"),
                },
            )
            self.send_event(reply_event)

    def get_my_router(self):
        coords = self.mesh_info["dram_coords"][self.name]
        return self.mesh_info["router_map"][coords]
