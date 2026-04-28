import networkx as nx
from typing import List, Dict, Any

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat",
    "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal",
    "Visakhapatnam", "Patna", "Vadodara", "Ludhiana", "Agra"
]

def build_route_graph() -> nx.DiGraph:
    """Builds the base route graph with nodes (cities) and edges (routes)."""
    graph = nx.DiGraph()
    for city in CITIES:
        graph.add_node(city)

    # Plausible edges for the base graph
    edges_data = [
        ("Mumbai", "Delhi", 1400, ["truck", "rail", "air"]),
        ("Delhi", "Mumbai", 1400, ["truck", "rail", "air"]),
        ("Mumbai", "Pune", 150, ["truck", "rail"]),
        ("Pune", "Mumbai", 150, ["truck", "rail"]),
        ("Mumbai", "Ahmedabad", 530, ["truck", "rail", "air"]),
        ("Ahmedabad", "Mumbai", 530, ["truck", "rail", "air"]),
        ("Mumbai", "Bengaluru", 980, ["truck", "rail", "air"]),
        ("Bengaluru", "Mumbai", 980, ["truck", "rail", "air"]),

        ("Delhi", "Jaipur", 280, ["truck", "rail"]),
        ("Jaipur", "Delhi", 280, ["truck", "rail"]),
        ("Delhi", "Lucknow", 550, ["truck", "rail", "air"]),
        ("Lucknow", "Delhi", 550, ["truck", "rail", "air"]),
        ("Delhi", "Ludhiana", 310, ["truck", "rail"]),
        ("Ludhiana", "Delhi", 310, ["truck", "rail"]),
        ("Delhi", "Agra", 230, ["truck", "rail"]),
        ("Agra", "Delhi", 230, ["truck", "rail"]),

        ("Bengaluru", "Chennai", 350, ["truck", "rail", "air"]),
        ("Chennai", "Bengaluru", 350, ["truck", "rail", "air"]),
        ("Bengaluru", "Hyderabad", 570, ["truck", "rail", "air"]),
        ("Hyderabad", "Bengaluru", 570, ["truck", "rail", "air"]),

        ("Hyderabad", "Chennai", 630, ["truck", "rail", "air"]),
        ("Chennai", "Hyderabad", 630, ["truck", "rail", "air"]),
        ("Hyderabad", "Pune", 590, ["truck", "rail", "air"]),
        ("Pune", "Hyderabad", 590, ["truck", "rail", "air"]),

        ("Chennai", "Visakhapatnam", 800, ["truck", "rail"]),
        ("Visakhapatnam", "Chennai", 800, ["truck", "rail"]),
        ("Visakhapatnam", "Kolkata", 880, ["truck", "rail"]),
        ("Kolkata", "Visakhapatnam", 880, ["truck", "rail"]),

        ("Kolkata", "Patna", 580, ["truck", "rail"]),
        ("Patna", "Kolkata", 580, ["truck", "rail"]),
        ("Kolkata", "Lucknow", 980, ["truck", "rail", "air"]),
        ("Lucknow", "Kolkata", 980, ["truck", "rail", "air"]),

        ("Ahmedabad", "Surat", 260, ["truck", "rail"]),
        ("Surat", "Ahmedabad", 260, ["truck", "rail"]),
        ("Surat", "Mumbai", 280, ["truck", "rail"]),
        ("Mumbai", "Surat", 280, ["truck", "rail"]),

        ("Ahmedabad", "Jaipur", 680, ["truck", "rail"]),
        ("Jaipur", "Ahmedabad", 680, ["truck", "rail"]),

        ("Jaipur", "Agra", 240, ["truck", "rail"]),
        ("Agra", "Jaipur", 240, ["truck", "rail"]),

        ("Lucknow", "Kanpur", 90, ["truck", "rail"]),
        ("Kanpur", "Lucknow", 90, ["truck", "rail"]),
        ("Kanpur", "Agra", 280, ["truck", "rail"]),
        ("Agra", "Kanpur", 280, ["truck", "rail"]),

        ("Kanpur", "Bhopal", 530, ["truck", "rail"]),
        ("Bhopal", "Kanpur", 530, ["truck", "rail"]),
        ("Bhopal", "Indore", 190, ["truck", "rail"]),
        ("Indore", "Bhopal", 190, ["truck", "rail"]),

        ("Indore", "Ahmedabad", 390, ["truck", "rail"]),
        ("Ahmedabad", "Indore", 390, ["truck", "rail"]),
        ("Indore", "Nagpur", 410, ["truck", "rail"]),
        ("Nagpur", "Indore", 410, ["truck", "rail"]),

        ("Nagpur", "Hyderabad", 500, ["truck", "rail", "air"]),
        ("Hyderabad", "Nagpur", 500, ["truck", "rail", "air"]),

        ("Nagpur", "Pune", 710, ["truck", "rail", "air"]),
        ("Pune", "Nagpur", 710, ["truck", "rail", "air"]),

        ("Vadodara", "Ahmedabad", 110, ["truck", "rail"]),
        ("Ahmedabad", "Vadodara", 110, ["truck", "rail"]),
        ("Vadodara", "Surat", 150, ["truck", "rail"]),
        ("Surat", "Vadodara", 150, ["truck", "rail"]),
    ]

    for u, v, dist, modes in edges_data:
        if u in graph.nodes and v in graph.nodes:
            base_duration = dist / 60.0  # 60 km/h average speed
            route_id = f"{u}-{v}"
            graph.add_edge(
                u, v,
                route_id=route_id,
                distance_km=float(dist),
                base_duration_hrs=base_duration,
                transport_modes=modes,
                current_risk_score=0.0,
                congestion_factor=1.0
            )

    return graph

def update_edge_risk(graph: nx.DiGraph, origin: str, destination: str, risk_score: float, congestion_factor: float) -> None:
    """Updates a specific edge's risk and congestion in place."""
    if graph.has_edge(origin, destination):
        graph[origin][destination]['current_risk_score'] = min(max(risk_score, 0.0), 1.0)
        graph[origin][destination]['congestion_factor'] = min(max(congestion_factor, 1.0), 3.0)

def get_effective_duration(graph: nx.DiGraph, origin: str, destination: str) -> float:
    """Returns base_duration_hrs * congestion_factor for an edge."""
    if graph.has_edge(origin, destination):
        edge_data = graph[origin][destination]
        return edge_data['base_duration_hrs'] * edge_data['congestion_factor']
    return float('inf')

def find_shortest_path(graph: nx.DiGraph, origin: str, destination: str, weight: str = 'duration') -> Dict[str, Any]:
    """Finds the shortest path between origin and destination based on the given weight."""
    
    def weight_func(u, v, d):
        if weight == 'duration':
            return d['base_duration_hrs'] * d['congestion_factor']
        elif weight == 'distance':
            return d['distance_km']
        elif weight == 'risk':
            return d['current_risk_score']
        return 1.0

    try:
        path = nx.shortest_path(graph, source=origin, target=destination, weight=weight_func)
    except nx.NetworkXNoPath:
        return {}

    total_distance = 0.0
    total_duration = 0.0
    total_risk = 0.0
    edges_count = len(path) - 1
    
    for i in range(edges_count):
        u, v = path[i], path[i+1]
        edge_data = graph[u][v]
        total_distance += edge_data['distance_km']
        total_duration += edge_data['base_duration_hrs'] * edge_data['congestion_factor']
        total_risk += edge_data['current_risk_score']
        
    avg_risk = total_risk / edges_count if edges_count > 0 else 0.0
    
    return {
        "path": path,
        "total_distance_km": total_distance,
        "total_duration_hrs": total_duration,
        "risk_score": avg_risk
    }

def find_k_shortest_paths(graph: nx.DiGraph, origin: str, destination: str, k: int = 3) -> List[Dict[str, Any]]:
    """Returns top-k paths sorted by effective duration."""
    from networkx.algorithms.simple_paths import shortest_simple_paths
    
    def weight_func(u, v, d):
        return d['base_duration_hrs'] * d['congestion_factor']

    try:
        paths_gen = shortest_simple_paths(graph, source=origin, target=destination, weight=weight_func)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    results = []
    import itertools
    for path in itertools.islice(paths_gen, k):
        total_distance = 0.0
        total_duration = 0.0
        total_risk = 0.0
        edges_count = len(path) - 1
        
        for i in range(edges_count):
            u, v = path[i], path[i+1]
            edge_data = graph[u][v]
            total_distance += edge_data['distance_km']
            total_duration += edge_data['base_duration_hrs'] * edge_data['congestion_factor']
            total_risk += edge_data['current_risk_score']
            
        avg_risk = total_risk / edges_count if edges_count > 0 else 0.0
        
        results.append({
            "path": path,
            "total_distance_km": total_distance,
            "total_duration_hrs": total_duration,
            "risk_score": avg_risk
        })
        
    return results

def apply_disruption_to_graph(graph: nx.DiGraph, disruption_dict: Dict[str, Any]) -> nx.DiGraph:
    """
    Given a disruption dict (route_id, severity, disruption_type), 
    increases congestion_factor and risk_score on affected edges proportional to severity.
    """
    route_id = disruption_dict.get('route_id')
    severity = disruption_dict.get('severity', 0.0)
    
    for u, v, d in graph.edges(data=True):
        if d.get('route_id') == route_id:
            # Increase congestion factor up to 3.0
            new_congestion = min(3.0, d['congestion_factor'] + severity * 2.0)
            d['congestion_factor'] = new_congestion
            
            # Increase risk score up to 1.0
            new_risk = min(1.0, d['current_risk_score'] + severity)
            d['current_risk_score'] = new_risk
            break
            
    return graph

if __name__ == "__main__":
    print("Building logistics route graph...")
    G = build_route_graph()
    print(f"Graph initialized with {G.number_of_nodes()} cities and {G.number_of_edges()} routes.")
    
    # Example: applying a disruption
    disruption = {"route_id": "Mumbai-Delhi", "severity": 0.8, "disruption_type": "weather"}
    print(f"\nApplying disruption: {disruption}")
    apply_disruption_to_graph(G, disruption)
    
    print("\nFinding 3 shortest paths from Mumbai to Delhi:")
    paths = find_k_shortest_paths(G, origin="Mumbai", destination="Delhi", k=3)
    
    for i, path_info in enumerate(paths, 1):
        print(f"\nPath {i}:")
        print(f"  Route: {' -> '.join(path_info['path'])}")
        print(f"  Duration (hrs): {path_info['total_duration_hrs']:.2f}")
        print(f"  Distance (km): {path_info['total_distance_km']:.2f}")
        print(f"  Risk Score: {path_info['risk_score']:.2f}")
