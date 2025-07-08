from sim_core.module import PipelineModule
from sim_core.event import Event


class DRAMChannel(PipelineModule):
    """Pipeline for a single DRAM channel.

    Parameters
    ----------
    latency : int
        Number of cycles each operation requires.
    capacity : int
        Maximum number of in-flight operations.
    dram : DRAM
        Parent DRAM instance used to dispatch completion events.
    channel_id : int
        Logical identifier for this channel.
    """

    def __init__(self, engine, name, mesh_info, latency, capacity, dram, channel_id):
        super().__init__(engine, name, mesh_info, 1, buffer_capacity=capacity)
        self.latency = latency
        self.dram = dram
        self.channel_id = channel_id
        self.set_stage_funcs([self._stage])

    def add_op(self, op):
        op["remaining"] = self.latency
        self.add_data(op)

    def _stage(self, mod, op):
        op["remaining"] -= 1
        if op["remaining"] > 0:
            return op, 0, True
        return op, 1, False

    def handle_pipeline_output(self, op):
        self.dram.channel_complete(self.channel_id, op)

class DRAM(PipelineModule):
    """DRAM model supporting multiple independent channels.

    Each :class:`DRAMChannel` has its own request queue and executes DMA
    operations in a simple one-stage pipeline so up to ``num_channels`` requests
    can progress simultaneously.  Completed replies include the ``channel_id``
    responsible for the request so higher level modules can correlate results.
    """

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
        self.channels = [
            DRAMChannel(
                engine,
                f"{name}_ch{i}",
                mesh_info,
                pipeline_latency,
                buffer_capacity,
                self,
                i,
            )
            for i in range(num_channels)
        ]
        # Track last channel used per program to distribute load in round-robin
        # fashion.  Falls back to global round-robin if no program info.
        self.last_channel = -1
        self._last_chan_by_prog = {}
        # Unused pipeline stage for compatibility with PipelineModule
        self.set_stage_funcs([lambda m, d: (d, 1, False)])

        # Event dispatch table to make adding new opcodes easy.
        self.event_handlers = {}
        self._register_default_handlers()

    def _select_channel(self, op):
        """Choose a channel for the incoming request.

        Uses per-program round-robin scheduling so concurrent programs spread
        traffic evenly across channels.  If no program is specified, falls back
        to a global round-robin counter."""
        prog = op.get("program")
        if prog is not None:
            last = self._last_chan_by_prog.get(prog, -1)
            ch = (last + 1) % self.num_channels
            self._last_chan_by_prog[prog] = ch
            return ch

        self.last_channel = (self.last_channel + 1) % self.num_channels
        return self.last_channel


    def register_handler(self, evt_type, fn):
        """Register handler ``fn`` for ``evt_type``."""
        self.event_handlers[evt_type] = fn

    def _register_default_handlers(self):
        self.register_handler("DMA_WRITE", self._handle_dma_access)
        self.register_handler("DMA_READ", self._handle_dma_access)

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
            "task_id": event.payload.get("task_id"),
        }
        if event.payload.get("need_reply"):
            op["dst_name"] = event.payload["src_name"]
        ch = self._select_channel(op)
        print(
            f"[DRAM] enqueue {op['type']} task={op.get('task_id')} ch={ch} cycle={self.engine.current_cycle}"
        )
        op["channel_id"] = ch
        self.channels[ch].add_op(op)

    def channel_complete(self, ch, op):
        """Callback from :class:`DRAMChannel` when an operation finishes."""
        self.handle_pipeline_output(op)


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
                f"[DRAM] complete {op['type']} ch={op.get('channel_id')} task={op.get('task_id')} cycle={self.engine.current_cycle}"
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
                    "task_id": op.get("task_id"),
                    "channel_id": op.get("channel_id"),
                },
            )
            self.send_event(reply_event)

    def get_my_router(self):
        coords = self.mesh_info["dram_coords"][self.name]
        return self.mesh_info["router_map"][coords]
