from informed_search import UndirectedGraph

cities = {
            'Northridge': [('Westwood', 25), ('Downtown', 30), ('Hillside', 15)],
            'Westwood': [('Northridge', 25), ('Beverly', 8), ('Santa Monica', 12), ('Downtown', 20)],
            'Santa Monica': [('Westwood', 12), ('Venice', 5), ('LAX', 10)],
            'Venice': [('Santa Monica', 5), ('Marina', 7), ('Culver', 9)],
            'Marina': [('Venice', 7), ('LAX', 8), ('Playa', 6)],
            'LAX': [('Santa Monica', 10), ('Marina', 8), ('Inglewood', 5), ('Hawthorne', 7)],
            'Beverly': [('Westwood', 8), ('Hollywood', 6), ('Century City', 4)],
            'Hollywood': [('Beverly', 6), ('Downtown', 12), ('Silver Lake', 8)],
            'Downtown': [('Northridge', 30), ('Westwood', 20), ('Hollywood', 12),
                         ('East LA', 8), ('Commerce', 10)],
            'Century City': [('Beverly', 4), ('West LA', 7), ('Culver', 9)],
            'Culver': [('Century City', 9), ('Venice', 9), ('Inglewood', 6)],
            'Inglewood': [('Culver', 6), ('LAX', 5), ('Hawthorne', 4), ('Gardena', 8)],
            'Hawthorne': [('LAX', 7), ('Inglewood', 4), ('Gardena', 5), ('Torrance', 11)],
            'Gardena': [('Inglewood', 8), ('Hawthorne', 5), ('Torrance', 7), ('Long Beach', 15)],
            'Torrance': [('Hawthorne', 11), ('Gardena', 7), ('Long Beach', 10)],
            'Long Beach': [('Gardena', 15), ('Torrance', 10)],
            'East LA': [('Downtown', 8), ('Commerce', 6)],
            'Commerce': [('Downtown', 10), ('East LA', 6)],
            'Hillside': [('Northridge', 15)],
            'West LA': [('Century City', 7), ('Santa Monica', 9)],
            'Silver Lake': [('Hollywood', 8), ('Downtown', 9)],
            'Playa': [('Marina', 6), ('LAX', 9)]
         }

distances_to_silver_lake = {
                              'Silver Lake':     0.0,
                              'Northridge':     18.11,
                              'Westwood':       10.16,
                              'Santa Monica':   13.48,
                              'Venice':         13.4,
                              'Marina':         12.73,
                              'LAX':            12.78,
                              'Beverly':        7.51,
                              'Hollywood':      3.37,
                              'Downtown':       2.83,
                              'Century City':   8.66,
                              'Culver':         8.49,
                              'Inglewood':      9.86,
                              'Hawthorne':      12.97,
                              'Gardena':        13.88,
                              'Torrance':       17.8,
                              'Long Beach':     22.31,
                              'East LA':        7.09,
                              'Commerce':       8.68,
                              'Hillside':       7.23,
                              'West LA':        10.15,
                              'Playa':          12.98,
                           }

cities_graph = UndirectedGraph()

for city in cities.keys():
    cities_graph.add_vertex(city)

for city1 in cities.keys():
    for city2, distance in cities[city1]:
        cities_graph.add_edge(city1, city2, distance)

if __name__ == "__main__":

    path, distance = cities_graph.greedy(city, 'Silver Lake', distances_to_silver_lake)
    print('Shortest route:\n ', ' -> '.join(path))
    print('Total distance: ', distance)

