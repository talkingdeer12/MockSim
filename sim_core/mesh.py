from .router import Router

def create_mesh(engine, x_size, y_size, mesh_info):
    mesh = {}
    for x in range(x_size):
        for y in range(y_size):
            name = f"Router_{x}_{y}"
            router = Router(engine, name, x, y, mesh_info)
            mesh[(x, y)] = router
            engine.register_module(router)
    for x in range(x_size):
        for y in range(y_size):
            router = mesh[(x, y)]
            neighbors = {}
            if x > 0: neighbors['W'] = mesh[(x-1, y)]
            if x < x_size-1: neighbors['E'] = mesh[(x+1, y)]
            if y > 0: neighbors['N'] = mesh[(x, y-1)]
            if y < y_size-1: neighbors['S'] = mesh[(x, y+1)]
            router.set_neighbors(neighbors)
    return mesh
