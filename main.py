from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.logger import EventLogger
from sim_hw.cp import ControlProcessor
from sim_hw.npu import NPU
from sim_hw.iod import IOD
from sim_ml.llama3_decoder import FakeLlama3DecoderBlock
from sim_ml.llama3_sim_hook import linear_gemm_hook
import torch

def main():
    engine = SimulatorEngine()
    logger = EventLogger()
    engine.set_logger(logger)
    x_size, y_size = 3, 2

    mesh_info = {
        "mesh_size": (x_size, y_size),
        "router_map": None,
        "npu_coords": {},
        "cp_coords": {},
        "iod_coords": {},
    }

    mesh = create_mesh(engine, x_size, y_size, mesh_info)
    mesh_info["router_map"] = mesh

    npu_coords_list = [(0,0)]
    cp_coords_dict = {"CP": (1,1)}
    iod_coords_dict = {"IOD": (0,1)}

    mesh_info["cp_coords"] = cp_coords_dict
    mesh_info["iod_coords"] = iod_coords_dict

    # Instantiate a single NPU
    npu = NPU(engine, "NPU_0", mesh_info)
    mesh_info["npu_coords"]["NPU_0"] = npu_coords_list[0]
    mesh[npu_coords_list[0]].attach_module(npu)
    engine.register_module(npu)

    iod = IOD(engine, "IOD", mesh_info)
    mesh[iod_coords_dict["IOD"]].attach_module(iod)
    engine.register_module(iod)

    cp = ControlProcessor(engine, "CP", mesh_info, npus=[npu])
    mesh[cp_coords_dict["CP"]].attach_module(cp)
    engine.register_module(cp)

    hidden_size = 32
    block = FakeLlama3DecoderBlock(hidden_size, layer_idx=0)
    for i, (name, module) in enumerate(block.named_modules()):
        if isinstance(module, torch.nn.Linear):
            module.sim_layer_idx = i
            module.register_forward_hook(linear_gemm_hook(cp, mesh_info))

    batch = 2
    seq = 8
    x = torch.randn(batch * seq, hidden_size)
    print("===== PyTorch Llama3 Decoder Forward (w/ 하드웨어 시뮬레이터) =====")
    y = block(x)
    engine.run_until_idle()
    logger.save_html('timeline.html')

if __name__ == "__main__":
    main()
