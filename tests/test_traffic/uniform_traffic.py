from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from .traffic_gen import TrafficGenerator


def run_uniform_traffic(x=16, y=16, packets_per_node=20, max_tick=10000):
    engine = SimulatorEngine()
    mesh_info = {
        "mesh_size": (x, y),
        "router_map": None,
    }
    mesh = create_mesh(engine, x, y, mesh_info)
    mesh_info["router_map"] = mesh

    gens = []
    for cx in range(x):
        for cy in range(y):
            name = f"TG_{cx}_{cy}"
            tg = TrafficGenerator(engine, name, mesh_info, (cx, cy), packets_per_node)
            mesh[(cx, cy)].attach_module(tg)
            engine.register_module(tg)
            tg.start()
            gens.append(tg)

    engine.run_until_idle(max_tick=max_tick)

    latencies = [lat for g in gens for lat in g.latencies]
    avg = sum(latencies) / len(latencies) if latencies else 0
    print(f"Generated packets: {sum(g.sent for g in gens)}")
    print(f"Received packets: {sum(g.received for g in gens)}")
    print(f"Average waiting time: {avg:.2f} cycles")
    return avg


if __name__ == "__main__":
    run_uniform_traffic()
