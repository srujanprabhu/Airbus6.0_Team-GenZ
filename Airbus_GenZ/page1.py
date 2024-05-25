import requests
import heapq
import math
import streamlit as st


class Graph:

    def __init__(self):
        self.edges = {}
        self.coordinates = {}
        self.API_KEY = '9c65640240e37a0967c73db1363d83a0'


    def add_edge(self, from_node, to_node, from_coords, to_coords):
        if from_node not in self.edges:
            self.edges[from_node] = []
        if to_node not in self.edges:
            self.edges[to_node] = []

        # Calculate distance using Haversine formula
        distance = self.calculate_distance(from_coords, to_coords)

        self.edges[from_node].append((to_node, distance))
        self.edges[to_node].append((from_node, distance))

        # Store coordinates for each node
        self.coordinates[from_node] = from_coords
        self.coordinates[to_node] = to_coords

    def calculate_distance(self, coords1, coords2):
        """
        Calculate the great circle distance between two points
        on the Earth's surface given their latitude and longitude
        in degrees using the Haversine formula.
        """
        lat1, lon1 = math.radians(coords1[0]), math.radians(coords1[1])
        lat2, lon2 = math.radians(coords2[0]), math.radians(coords2[1])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = 6371 * c  # Earth radius in kilometers

        return distance

    def get_weather_data(self, lat, lon):
        try:
            url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.API_KEY}'
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            weather_data = response.json()
            return weather_data
        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None

    def is_bad_weather(self, weather_data):
        try:
            weather_id = weather_data['weather'][0]['id']
            return weather_id < 700  # Example: weather_id < 700 indicates bad weather (rain, snow, etc.)
        except (KeyError, TypeError, IndexError):
            print("Unexpected weather data format")
            return False

    def astar_with_weather(self, start, end):
        # Priority queue to store nodes to be processed
        open_set = [(0, start)]  # (f_score, node)
        heapq.heapify(open_set)

        # Cost from start along best known path
        g_scores = {node: float('inf') for node in self.edges}
        g_scores[start] = 0

        # Estimated total cost from start to goal through y
        f_scores = {node: float('inf') for node in self.edges}
        f_scores[start] = self.calculate_heuristic(start, end)

        # Dictionary to track shortest path
        came_from = {node: None for node in self.edges}

        while open_set:
            current_f_score, current_node = heapq.heappop(open_set)

            if current_node == end:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = came_from[current_node]
                return path[::-1], g_scores[end]

            for neighbor, weight in self.edges[current_node]:
                tentative_g_score = g_scores[current_node] + weight
                if tentative_g_score < g_scores[neighbor]:
                    came_from[neighbor] = current_node
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + self.calculate_heuristic(neighbor, end)

                    # Check weather for the edge (current_node, neighbor)
                    from_coords = self.coordinates[current_node]
                    to_coords = self.coordinates[neighbor]
                    weather_data_from = self.get_weather_data(from_coords[0], from_coords[1])
                    weather_data_to = self.get_weather_data(to_coords[0], to_coords[1])

                    if (self.is_bad_weather(weather_data_from) or self.is_bad_weather(weather_data_to)):
                        # Skip this edge if bad weather detected
                        continue

                    heapq.heappush(open_set, (f_scores[neighbor], neighbor))

        # If end node is not reachable
        return [], float('inf')

    def calculate_heuristic(self, node, end):
        """
        A heuristic function to estimate the straight-line distance (Euclidean distance)
        between two points represented by the nodes on Earth's surface.
        """
        if node in self.coordinates and end in self.coordinates:
            return self.calculate_distance(self.coordinates[node], self.coordinates[end])
        else:
            return float('inf')