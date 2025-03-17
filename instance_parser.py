import xml.etree.ElementTree as ET

def parse_instance(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    nodes = {}
    charging_stations = set()
    customers = set()
    depot = None

    # Parse nodes and classify them by type:
    network = root.find('network')
    nodes_xml = network.find('nodes')
    for node in nodes_xml:
        node_id = int(node.attrib['id'])
        cx = float(node.find('cx').text)
        cy = float(node.find('cy').text)
        nodes[node_id] = (cx, cy)
        node_type = node.attrib['type']
        if node_type == "0":
            depot = node_id
        elif node_type == "1":
            customers.add(node_id)
        elif node_type == "2":
            charging_stations.add(node_id)

    # Build cost and travel time matrices.
    cost_matrix = {}
    travel_time_matrix = {}
    links_xml = network.find('links')
    for link in links_xml:
        head = int(link.attrib['head'])
        tail = int(link.attrib['tail'])
        energy_consumption = float(link.find('energy_consumption').text)
        travel_time = float(link.find('travel_time').text)
        cost_matrix[(head, tail)] = energy_consumption
        cost_matrix[(tail, head)] = energy_consumption  # symmetric
        travel_time_matrix[(head, tail)] = travel_time
        travel_time_matrix[(tail, head)] = travel_time

    # Get fleet parameters.
    fleet = root.find('fleet')
    vehicle_profile = fleet.find('vehicle_profile')
    battery_capacity = float(vehicle_profile.find('custom').find('battery_capacity').text)
    num_vehicles = int(vehicle_profile.attrib['number'])
    vehicle_capacity = float(vehicle_profile.find('capacity').text)
    max_travel_time = float(vehicle_profile.find('max_travel_time').text)

    # Parse requests (for demand/service time).
    requests = {}
    requests_xml = root.find('requests')
    for req in requests_xml:
        req_node = int(req.attrib['node'])
        quantity = float(req.find('quantity').text)
        service_time = float(req.find('service_time').text)
        requests[req_node] = {'quantity': quantity, 'service_time': service_time}

    print(f"Customers Parsed: {customers}")
    print(f"Charging Stations Parsed: {charging_stations}")
    print(f"Depot: {depot}")

    return (nodes, charging_stations, depot, customers, cost_matrix, travel_time_matrix,
            battery_capacity, num_vehicles, vehicle_capacity, max_travel_time, requests)
