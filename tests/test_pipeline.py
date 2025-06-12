from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_hw.cp import ControlProcessor
from sim_hw.pe import PE
from sim_hw.dram import DRAM
from sim_core.event import Event


def run_test():
    engine = SimulatorEngine()
    mesh_info = {
        "mesh_size": (3, 1),
        "router_map": None,
        "pe_coords": {},
        "cp_coords": {},
        "dram_coords": {},
    }
    mesh = create_mesh(engine, 3, 1, mesh_info)
    mesh_info["router_map"] = mesh

    pe = PE(engine, "PE_0", mesh_info)
    mesh_info["pe_coords"]["PE_0"] = (0, 0)
    mesh[(0, 0)].attached_module = pe
    engine.register_module(pe)

    dram = DRAM(engine, "DRAM", mesh_info)
    mesh_info["dram_coords"]["DRAM"] = (1, 0)
    mesh[(1, 0)].attached_module = dram
    engine.register_module(dram)

    cp = ControlProcessor(engine, "CP", mesh_info, [pe], dram)
    mesh_info["cp_coords"]["CP"] = (2, 0)
    mesh[(2, 0)].attached_module = cp
    engine.register_module(cp)

    gemm_event = Event(
        src=None,
        dst=cp,
        cycle=0,
        identifier="GEMM_TEST",
        event_type="GEMM",
        payload={"gemm_shape": (2, 2, 2), "weights_size": 16, "act_size": 16},
    )
    cp.send_event(gemm_event)

    engine.run_until_idle()


if __name__ == "__main__":
    run_test()
