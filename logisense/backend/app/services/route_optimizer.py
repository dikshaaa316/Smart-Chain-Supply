"""
Route Optimizer Service using Google OR-Tools.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import networkx as nx

from ortools.constraint_solver import routing_enums_pb2, pywrapcp

from app.services import route_graph

class NoFeasibleRouteError(Exception):
    """Exception raised when no feasible route can be found."""
    pass

@dataclass
class ShipmentTask:
    shipment_id: str
    origin: str
    destination: str
    weight_kg: float
    time_window_start: datetime
    time_window_end: datetime
    priority: int  # 1-5

@dataclass
class Vehicle:
    vehicle_id: str
    carrier_id: str
    current_location: str
    capacity_kg: float
    available_from: datetime

@dataclass
class OptimizationResult:
    vehicle_id: str
    shipment_ids: List[str]
    route: List[str]
    total_distance_km: float
    total_duration_hrs: float
    estimated_cost: float

def optimize_routes(
    tasks: List[ShipmentTask], 
    vehicles: List[Vehicle], 
    disrupted_edges: List[Tuple[str, str]]
) -> List[OptimizationResult]:
    """
    Optimizes routes for a set of vehicles and shipment tasks using OR-Tools VRP Solver.
    """
    if not tasks or not vehicles:
        return []

    # Initialize graph
    graph = route_graph.build_route_graph()
    for u, v in disrupted_edges:
        route_graph.apply_disruption_to_graph(graph, {"route_id": f"{u}-{v}", "severity": 1.0})

    # Get unique cities involved
    unique_cities = set(["Mumbai"])  # Default depot
    for task in tasks:
        unique_cities.add(task.origin)
        unique_cities.add(task.destination)
    for v in vehicles:
        unique_cities.add(v.current_location)
    unique_cities = list(unique_cities)

    # a. Build a distance matrix between all cities involved
    dist_matrix_cities = {}
    dur_matrix_cities = {}
    for c1 in unique_cities:
        dist_matrix_cities[c1] = {}
        dur_matrix_cities[c1] = {}
        for c2 in unique_cities:
            if c1 == c2:
                dist_matrix_cities[c1][c2] = 0.0
                dur_matrix_cities[c1][c2] = 0.0
            else:
                path_info = route_graph.find_shortest_path(graph, c1, c2, weight='duration')
                if path_info:
                    dist_matrix_cities[c1][c2] = path_info['total_distance_km']
                    dur_matrix_cities[c1][c2] = path_info['total_duration_hrs']
                else:
                    dist_matrix_cities[c1][c2] = 999999.0  # High penalty for unreachable
                    dur_matrix_cities[c1][c2] = 999999.0

    # b. Set up OR-Tools RoutingModel
    # Nodes: 0 (Depot - Mumbai), 1..2N (Pickups & Deliveries), 2N+1..2N+2V (Vehicle Starts & Ends)
    nodes = ["Mumbai"]
    for task in tasks:
        nodes.append(task.origin)      # Pickup
        nodes.append(task.destination) # Delivery

    starts = []
    ends = []
    for v in vehicles:
        # Start node for vehicle
        nodes.append(v.current_location)
        starts.append(len(nodes) - 1)
        
        # End node for vehicle (unique index, maps to Mumbai)
        nodes.append("Mumbai")
        ends.append(len(nodes) - 1)

    manager = pywrapcp.RoutingIndexManager(len(nodes), len(vehicles), starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    # Transit callbacks
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        c1 = nodes[from_node]
        c2 = nodes[to_node]
        return int(dist_matrix_cities[c1][c2])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        c1 = nodes[from_node]
        c2 = nodes[to_node]
        return int(dur_matrix_cities[c1][c2] * 60)

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    # Time dimension
    base_time = min((task.time_window_start for task in tasks), default=datetime.now())
    base_time = min(base_time, min((v.available_from for v in vehicles), default=base_time))

    routing.AddDimension(
        time_callback_index,
        100000,  # allow waiting time (large value)
        100000,  # maximum time per vehicle
        False,   # Don't force start cumul to zero
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')

    # Capacity dimension
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        if 0 < from_node <= 2 * len(tasks):
            task_idx = (from_node - 1) // 2
            is_pickup = (from_node - 1) % 2 == 0
            weight = tasks[task_idx].weight_kg
            return int(weight) if is_pickup else int(-weight)
        return 0

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [int(v.capacity_kg) for v in vehicles],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity'
    )

    # Time windows & Pickups/Deliveries
    for i, task in enumerate(tasks):
        pickup_node = 1 + 2 * i
        delivery_node = 2 + 2 * i
        
        pickup_index = manager.NodeToIndex(pickup_node)
        delivery_index = manager.NodeToIndex(delivery_node)

        # Time windows constraints
        start_mins = max(0, int((task.time_window_start - base_time).total_seconds() / 60))
        end_mins = max(0, int((task.time_window_end - base_time).total_seconds() / 60))
        
        time_dimension.CumulVar(pickup_index).SetRange(start_mins, end_mins)
        time_dimension.CumulVar(delivery_index).SetRange(start_mins, end_mins)

        # Add Pickup and Delivery
        routing.AddPickupAndDelivery(pickup_node, delivery_node)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
        )
        routing.solver().Add(
            time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index)
        )

        # Penalty for unassigned shipments
        routing.AddDisjunction([pickup_index], 10000)
        routing.AddDisjunction([delivery_index], 10000)

    # Vehicle availability
    for i, v in enumerate(vehicles):
        start_index = routing.Start(i)
        avail_mins = max(0, int((v.available_from - base_time).total_seconds() / 60))
        time_dimension.CumulVar(start_index).SetRange(avail_mins, 100000)

    # c. Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30

    # d. Solve
    solution = routing.SolveWithParameters(search_parameters)

    # e. Extract routes
    if not solution:
        raise NoFeasibleRouteError("OR-Tools could not find a feasible route. Try relaxing constraints.")

    results = []
    for vehicle_id_idx in range(len(vehicles)):
        index = routing.Start(vehicle_id_idx)
        route_nodes = []
        route_shipments = []
        route_distance = 0.0
        route_duration = 0.0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            city = nodes[node_index]
            if not route_nodes or route_nodes[-1] != city:
                route_nodes.append(city)
                
            if 0 < node_index <= 2 * len(tasks):
                task_idx = (node_index - 1) // 2
                is_pickup = (node_index - 1) % 2 == 0
                shipment_id = tasks[task_idx].shipment_id
                if is_pickup:
                    route_shipments.append(shipment_id)
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            if not routing.IsEnd(index):
                from_city = nodes[manager.IndexToNode(previous_index)]
                to_city = nodes[manager.IndexToNode(index)]
                route_distance += dist_matrix_cities[from_city][to_city]
                route_duration += dur_matrix_cities[from_city][to_city]

        # Add end node
        end_node_index = manager.IndexToNode(index)
        end_city = nodes[end_node_index]
        if not route_nodes or route_nodes[-1] != end_city:
            route_nodes.append(end_city)
            
        if route_shipments:
            results.append(OptimizationResult(
                vehicle_id=vehicles[vehicle_id_idx].vehicle_id,
                shipment_ids=route_shipments,
                route=route_nodes,
                total_distance_km=route_distance,
                total_duration_hrs=route_duration,
                estimated_cost=route_distance * 10.0  # simple heuristic cost
            ))

    return results

def suggest_reroute(shipment_id: str, current_route: List[str], disrupted_segment: Tuple[str, str], graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Given a disrupted segment (edge), finds the best alternative route.
    """
    u, v = disrupted_segment
    
    # Calculate original duration
    original_duration = 0.0
    for i in range(len(current_route) - 1):
        if graph.has_edge(current_route[i], current_route[i+1]):
            original_duration += route_graph.get_effective_duration(graph, current_route[i], current_route[i+1])
    
    # Temporarily remove disrupted edge
    edge_data = None
    if graph.has_edge(u, v):
        edge_data = graph.edges[u, v]
        graph.remove_edge(u, v)
        
    origin = current_route[0]
    destination = current_route[-1]
    
    try:
        paths = route_graph.find_k_shortest_paths(graph, origin, destination, k=1)
        if paths:
            best_path = paths[0]
            new_route = best_path['path']
            new_duration = best_path['total_duration_hrs']
            time_saved_mins = (original_duration - new_duration) * 60
            
            return {
                "original_route": current_route,
                "recommended_route": new_route,
                "original_duration_hrs": original_duration,
                "new_duration_hrs": new_duration,
                "time_saved_mins": time_saved_mins,
                "reason": f"Avoided disrupted segment {u}-{v}",
                "confidence": 0.85
            }
        else:
            return {
                "original_route": current_route,
                "recommended_route": current_route,
                "original_duration_hrs": original_duration,
                "new_duration_hrs": original_duration,
                "time_saved_mins": 0.0,
                "reason": "No feasible alternative route found",
                "confidence": 0.0
            }
    finally:
        # Restore edge
        if edge_data:
            graph.add_edge(u, v, **edge_data)

if __name__ == "__main__":
    print("Running Route Optimizer Demo...")
    
    # 5 sample tasks
    tasks = [
        ShipmentTask("SHP001", "Mumbai", "Delhi", 500, datetime.now(), datetime.now() + timedelta(days=3), 5),
        ShipmentTask("SHP002", "Pune", "Ahmedabad", 300, datetime.now(), datetime.now() + timedelta(days=2), 4),
        ShipmentTask("SHP003", "Bengaluru", "Chennai", 400, datetime.now(), datetime.now() + timedelta(days=2), 3),
        ShipmentTask("SHP004", "Hyderabad", "Pune", 200, datetime.now(), datetime.now() + timedelta(days=4), 2),
        ShipmentTask("SHP005", "Delhi", "Lucknow", 600, datetime.now(), datetime.now() + timedelta(days=3), 1),
    ]
    
    # 2 vehicles
    vehicles = [
        Vehicle("VEH001", "CARRIER1", "Mumbai", 2000, datetime.now()),
        Vehicle("VEH002", "CARRIER2", "Bengaluru", 1500, datetime.now()),
    ]
    
    print("Tasks:")
    for t in tasks:
        print(f"  {t.shipment_id}: {t.origin} -> {t.destination} ({t.weight_kg}kg)")
        
    print("\nVehicles:")
    for v in vehicles:
        print(f"  {v.vehicle_id}: {v.current_location} ({v.capacity_kg}kg capacity)")
    
    try:
        print("\nOptimizing routes (30s time limit)...")
        results = optimize_routes(tasks, vehicles, [])
        for res in results:
            print(f"\nVehicle {res.vehicle_id} takes shipments {res.shipment_ids}")
            print(f"  Route: {' -> '.join(res.route)}")
            print(f"  Distance: {res.total_distance_km:.2f} km, Duration: {res.total_duration_hrs:.2f} hrs")
            print(f"  Estimated Cost: {res.estimated_cost:.2f}")
    except NoFeasibleRouteError as e:
        print(f"Error: {e}")

    print("\nTesting suggest_reroute:")
    graph = route_graph.build_route_graph()
    sample_route = ["Mumbai", "Surat", "Ahmedabad", "Jaipur", "Delhi"]
    disrupted_edge = ("Surat", "Ahmedabad")
    print(f"Original Route: {' -> '.join(sample_route)}")
    print(f"Disrupted Edge: {disrupted_edge}")
    suggestion = suggest_reroute("SHP001", sample_route, disrupted_edge, graph)
    print(f"Recommended Route: {' -> '.join(suggestion['recommended_route'])}")
    print(f"Time Saved: {suggestion['time_saved_mins']:.2f} mins")
    print(f"Reason: {suggestion['reason']}")
