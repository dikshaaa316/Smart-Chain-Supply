import sys
sys.path.insert(0, '.')
import app.services.route_optimizer as ro
from datetime import datetime, timedelta

tasks = [
    ro.ShipmentTask("SHP001", "Mumbai", "Delhi", 500, datetime.now(), datetime.now() + timedelta(days=3), 5),
]
vehicles = [
    ro.Vehicle("VEH001", "CARRIER1", "Mumbai", 2000, datetime.now()),
    ro.Vehicle("VEH002", "CARRIER2", "Bengaluru", 1500, datetime.now()),
]

def test_fn():
    tasks_ = tasks
    vehicles_ = vehicles
    
    unique_cities = list(set(["Mumbai", "Delhi", "Bengaluru"]))
    dist_matrix_cities = {c1: {c2: 10.0 for c2 in unique_cities} for c1 in unique_cities}
    dur_matrix_cities = {c1: {c2: 1.0 for c2 in unique_cities} for c1 in unique_cities}

    nodes = ["Mumbai", "Mumbai", "Delhi"]
    starts = []
    ends = []
    for v in vehicles_:
        nodes.append(v.current_location)
        starts.append(len(nodes) - 1)
        nodes.append("Mumbai")
        ends.append(len(nodes) - 1)
    
    manager = ro.pywrapcp.RoutingIndexManager(len(nodes), len(vehicles_), starts, ends)
    routing = ro.pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return 100
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        return 0
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, [2000, 1500], True, 'Capacity')

    pickup_node = 1
    delivery_node = 2
    routing.AddPickupAndDelivery(pickup_node, delivery_node)

    search_parameters = ro.pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 1
    
    print("Solving...")
    solution = routing.SolveWithParameters(search_parameters)
    print("Solved:", solution is not None)

test_fn()
