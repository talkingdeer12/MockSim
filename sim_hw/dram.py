from sim_core.module import PipelineModule
from sim_core.event import Event

class DRAM(PipelineModule):
    """Simple DRAM model with independent read/write channels."""

    def __init__(self, engine, name, mesh_info, pipeline_latency=5, buffer_capacity=4):
        # Two stages used as independent pipelines for read and write channels
        super().__init__(engine, name, mesh_info, 2, buffer_capacity)
        self.pipeline_latency = pipeline_latency
        funcs = [self._make_stage_func(i) for i in range(2)]
        self.set_stage_funcs(funcs)

    def _make_stage_func(self, idx):
        def func(mod, data):
            data["remaining"] -= 1
            if data["remaining"] > 0:
                return data, idx, True
            return data, self.num_stages, False
        return func

    def handle_event(self, event):
        if event.event_type in ("DMA_WRITE", "DMA_READ"):
            op = {
                "type": event.event_type,
                "program": event.program,
                "src_name": event.payload["src_name"],
                "remaining": event.payload.get("opcode_cycles", self.pipeline_latency),
            }
            if event.payload.get("need_reply"):
                op["dst_name"] = event.payload["src_name"]
            stage = 0 if event.event_type == "DMA_READ" else 1
            self.add_data(op, stage_idx=stage)
        else:
            super().handle_event(event)

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
                },
            )
            self.send_event(reply_event)

    def get_my_router(self):
        coords = self.mesh_info["dram_coords"][self.name]
        return self.mesh_info["router_map"][coords]
