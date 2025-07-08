from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.logger import EventLogger
from sim_hw.cp import ControlProcessor
from sim_hw.pe import PE
from sim_hw.dram import DRAM
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
        "pe_coords": {},
        "cp_coords": {},
        "dram_coords": {},
    }

    mesh = create_mesh(engine, x_size, y_size, mesh_info)
    mesh_info["router_map"] = mesh

    pe_coords_list = [(0,0), (1,0), (2,0), (0,1)]
    cp_coords_dict = {"CP": (2,1)}
    dram_coords_dict = {"DRAM": (1,1)}

    mesh_info["cp_coords"] = cp_coords_dict
    mesh_info["dram_coords"] = dram_coords_dict

    pes = []
    for i, coords in enumerate(pe_coords_list):
        pe_name = f"PE_{i}"
        mesh_info["pe_coords"][pe_name] = coords
        pe = PE(engine, pe_name, mesh_info)
        mesh[coords].attach_module(pe)
        engine.register_module(pe)
        pes.append(pe)

    # Use four independent channels in the DRAM model
    dram = DRAM(engine, "DRAM", mesh_info, num_channels=4)
    mesh[dram_coords_dict["DRAM"]].attach_module(dram)
    engine.register_module(dram)

    cp = ControlProcessor(engine, "CP", mesh_info, pes, dram)
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
