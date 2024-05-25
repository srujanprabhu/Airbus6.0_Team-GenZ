import streamlit as st
import numpy as np
import pandas as pd
import json
import math
import networkx as nx
import random
import requests
import heapq
import page1 as wa
def main():
    st.sidebar.title("GEN-Z Airbus Control Center")

    # Select box for Pilot Access
    pilot_access = st.sidebar.selectbox("Features",
                                        ["Select an option", "Flight Details", "Optimal Path Finder", "Emergency", "Weather Analysis", "Current weather status", "GPS Failure", "Flight Location", "Power Failure","User Guide"])




    # Display welcome message if no selection is made
    if pilot_access == "Select an option":
        st.title("Welcome to AIRBUS")
        st.write("Please select an option from the sidebar.")
    else:
        # Display content based on the selection
        if pilot_access == "Flight Details":
            flight_details()
        elif pilot_access == "Optimal Path Finder":
            optimal_path_finder()
        elif pilot_access == "Emergency":
            emergency()
        elif pilot_access == "Current weather status":
            current_weather_status()
        elif pilot_access == "Weather Analysis":
            weather_check()
        elif pilot_access == "GPS Failure":
            gps_failure()
        if pilot_access == "Flight Location":
            flight_location()
        elif pilot_access == "Power Failure":
            launch_rat()
        elif pilot_access == "User Guide":
            user_guide()


def flight_details():
    st.title("Flight Details")

    def generate_flight_data(num_samples=20):
        flight_data = {
            'Flight ID': np.arange(1, num_samples + 1),
            'Flight Name': [f'Flight {i}' for i in range(1, num_samples + 1)],
            'Airline Name': [f'Airline {i}' for i in range(1, num_samples + 1)],
            'Flight Age': np.random.randint(1, 20, size=num_samples),  # Random age between 1 and 20 years
            'Flight Capacity': np.random.randint(100, 300, size=num_samples),
            # Random capacity between 100 and 300 passengers
            'Fuel Capacity': np.random.randint(5000, 10000, size=num_samples),
            # Random fuel capacity between 5000 and 10000 gallons
            'Top Speed': np.random.randint(500, 700, size=num_samples),  # Random top speed between 500 and 700 mph
            'Air Pressure': np.random.randint(800, 1000, size=num_samples),
            # Random air pressure between 800 and 1000 psi
            'Engine Status': np.random.choice(['OK', 'Warning', 'Error'], size=num_samples),  # Random engine status
            'Electricity Percentage': np.random.randint(50, 100, size=num_samples),
            # Random electricity percentage between 50% and 100%
            'Range': np.random.randint(1000, 5000, size=num_samples)  # Random range between 1000 and 5000 miles
        }
        return pd.DataFrame(flight_data)

    def flight_details(flight_id):
        flight_data = generate_flight_data()
        flight_info = flight_data[flight_data['Flight ID'] == flight_id]
        if not flight_info.empty:
            st.subheader(f"Flight Details for Flight ID: {flight_id}")
            st.subheader("-----------------------------------------")
            st.write(f"Flight Name: {flight_info['Flight Name'].values[0]}")
            st.write(f"Airline Name: {flight_info['Airline Name'].values[0]}")
            st.write(f"Flight Age: {flight_info['Flight Age'].values[0]} years")
            st.write(f"Flight Capacity: {flight_info['Flight Capacity'].values[0]} passengers")
            st.write(f"Fuel Capacity: {flight_info['Fuel Capacity'].values[0]} gallons")
            st.write(f"Top Speed: {flight_info['Top Speed'].values[0]} mph")
            st.write(f"Air Pressure: {flight_info['Air Pressure'].values[0]} psi")
            st.write(f"Engine Status: {flight_info['Engine Status'].values[0]}")
            st.write(f"Electricity Percentage: {flight_info['Electricity Percentage'].values[0]}%")
            st.write(f"Range: {flight_info['Range'].values[0]} miles")
        else:
            st.error(f"No flight found with Flight ID: {flight_id}")

    flight_id = st.number_input("Enter Flight ID:", min_value=1, max_value=30, step=1)
    if st.button("Get Flight Details"):
        flight_details(flight_id)

def optimal_path_finder():

    # Function to load airport data
    def load_airport_data():
        with open('airports.json.json', 'r') as file:
            data = json.load(file)
        return data['results']

    # Function to calculate the great circle distance
    def great_circle_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Radius of the Earth in kilometers
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    # Function to create a graph of airports with random connections
    def create_airport_graph(airports, num_connections=3):
        G = nx.Graph()
        for airport in airports:
            G.add_node(airport['column_1'], data=airport)

        for airport in airports:
            connections = random.sample(airports, num_connections)
            for connection in connections:
                if airport != connection:
                    distance = great_circle_distance(airport['latitude'], airport['longitude'], connection['latitude'],
                                                     connection['longitude'])
                    G.add_edge(airport['column_1'], connection['column_1'], weight=distance)

        return G

    # Function to find the optimal path using A* algorithm
    def find_optimal_path(graph, source, destination):
        def heuristic(node1, node2):
            lat1, lon1 = graph.nodes[node1]['data']['latitude'], graph.nodes[node1]['data']['longitude']
            lat2, lon2 = graph.nodes[node2]['data']['latitude'], graph.nodes[node2]['data']['longitude']
            return great_circle_distance(lat1, lon1, lat2, lon2)

        return nx.astar_path(graph, source, destination, heuristic=heuristic)

    # Function to calculate the total distance of the path
    def calculate_total_distance(graph, path):
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += graph[path[i]][path[i + 1]]['weight']
        return total_distance

    # Main function to run the Streamlit app
    def path():
        st.title("Optimal Flight Path Finder")

        airports = load_airport_data()
        airport_options = {airport['column_1']: f"{airport['airport_name']} ({airport['city_name']})" for airport in
                           airports}

        source = st.selectbox("Select Source Airport", [""] + list(airport_options.keys()),
                              format_func=lambda x: airport_options.get(x, "Select an airport"))
        destination = st.selectbox("Select Destination Airport", [""] + list(airport_options.keys()),
                                   format_func=lambda x: airport_options.get(x, "Select an airport"))

        if st.button("Find Path"):
            if source and destination:
                graph = create_airport_graph(airports)
                optimal_path = find_optimal_path(graph, source, destination)
                total_distance = calculate_total_distance(graph, optimal_path)

                st.write("Optimal Path:")
                for airport_code in optimal_path:
                    st.write(airport_options[airport_code] + "->")

                st.write(f"Total Distance: {total_distance:.2f} km")
            else:
                st.write("Please select both source and destination airports.")

    if __name__ == "__main__":
        path()

def emergency():
    # Function to load airport data
    def load_airport_data():
        with open('airports.json.json', 'r') as file:
            data = json.load(file)
        return data['results']

    # Function to calculate the great circle distance
    def great_circle_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Radius of the Earth in kilometers
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    # Function to find the nearest airport within the fuel range or gliding range
    def find_nearest_airport(current_lat, current_lon, current_fuel_percentage, airports, max_distance):
        nearest_airport = None
        shortest_distance = float('inf')

        for airport in airports:
            distance = great_circle_distance(current_lat, current_lon, airport['latitude'], airport['longitude'])
            if distance <= max_distance and distance < shortest_distance:
                shortest_distance = distance
                nearest_airport = airport

        return nearest_airport, shortest_distance

    def emergency():
        st.title("Emergency Landing Finder")

        airports = load_airport_data()
        st.subheader('Enter the current geographical location')
        current_lat = st.number_input("Enter Current Latitude", format="%.6f")
        current_lon = st.number_input("Enter Current Longitude", format="%.6f")
        current_fuel_percentage = st.slider("Current Fuel Percentage", 0, 100, 50)

        # Assuming the maximum range of the aircraft is 4000 km when fully fueled
        max_range = 4000 * (current_fuel_percentage / 100)

        nearest_airport, distance = find_nearest_airport(current_lat, current_lon, current_fuel_percentage, airports,
                                                         max_range)

        prepare_landing = st.button("Prepare Emergency Landing")

        if prepare_landing:
            if nearest_airport:
                st.write(f"Nearest Airport within range ({max_range:.2f} km):")
                st.write(f"Airport Name: {nearest_airport['airport_name']}")
                st.write(f"City: {nearest_airport['city_name']}")
                st.write(f"Country: {nearest_airport['country_name']}")
                st.write(f"Distance: {distance:.2f} km")
            else:
                st.error(
                    "No airport found within the specified range. Please provide altitude to calculate gliding range.")
                current_altitude = st.number_input("Enter Current Altitude", format="%.2f")
                gliding_range = max_range * (current_altitude / 10000)  # Assuming maximum altitude of 10,000 meters
                nearest_airport, distance = find_nearest_airport(current_lat, current_lon, current_fuel_percentage,
                                                                 airports, gliding_range)
                st.write(f"Nearest Airport within gliding range ({gliding_range:.2f} km):")
                glide_landing = st.button("Emergency Landing", key=32)

                if nearest_airport and glide_landing:
                    st.write(f"Nearest Airport within gliding range ({gliding_range:.2f} km):")
                    st.write(f"Airport Name: {nearest_airport['airport_name']}")
                    st.write(f"City: {nearest_airport['city_name']}")
                    st.write(f"Country: {nearest_airport['country_name']}")
                    st.write(f"Distance: {distance:.2f} km")
                elif nearest_airport:
                    st.error("No airport found within the gliding range.")
                if st.button("BACK", key=42):
                    st.experimental_rerun()
            # while True:
            #     pass

    if __name__ == "__main__":
        emergency()

def weather_check():
    st.title("Weather Analysis")

    # Example usage
    graph = wa.Graph()
    graph.add_edge('Vijayawada', 'Hyderabad', (16.5062, 80.6480), (17.3850, 78.4867))
    graph.add_edge('Hyderabad', 'Chennai', (17.3850, 78.4867), (13.0827, 80.2707))
    graph.add_edge('Chennai', 'Bengaluru', (13.0827, 80.2707), (12.9716, 77.5946))
    graph.add_edge('Vijayawada', 'Chennai', (16.5062, 80.6480), (13.0827, 80.2707))
    graph.add_edge('Vijayawada', 'Bengaluru', (16.5062, 80.6480), (12.9716, 77.5946))
    graph.add_edge('Hyderabad', 'Bengaluru', (17.3850, 78.4867), (12.9716, 77.5946))
    graph.add_edge('Bengaluru', 'Thanjavur', (12.9716, 77.5946), (10.7852, 79.1378))
    graph.add_edge('Chennai', 'Coimbatore', (13.0827, 80.2707), (11.0168, 76.9558))
    graph.add_edge('Bengaluru', 'Kolkata', (12.9716, 77.5946), (22.5726, 88.3639))
    graph.add_edge('Delhi', 'Mumbai', (28.6139, 77.2090), (19.0760, 72.8777))
    graph.add_edge('Mumbai', 'Chennai', (19.0760, 72.8777), (13.0827, 80.2707))
    graph.add_edge('Hyderabad', 'Jaipur', (17.3850, 78.4867), (26.9124, 75.7873))
    graph.add_edge('Jaipur', 'Ahmedabad', (26.9124, 75.7873), (23.0225, 72.5714))
    graph.add_edge('Bengaluru', 'Goa', (12.9716, 77.5946), (15.2993, 74.1240))
    graph.add_edge('Delhi', 'Amritsar', (28.6139, 77.2090), (31.6340, 74.8737))
    graph.add_edge('Mumbai', 'Pune', (19.0760, 72.8777), (18.5204, 73.8567))
    graph.add_edge('Kolkata', 'Patna', (22.5726, 88.3639), (25.5941, 85.1376))
    graph.add_edge('Bengaluru', 'Hyderabad', (12.9716, 77.5946), (17.3850, 78.4867))
    graph.add_edge('Mumbai', 'Hyderabad', (19.0760, 72.8777), (17.3850, 78.4867))
    graph.add_edge('Chennai', 'Hyderabad', (13.0827, 80.2707), (17.3850, 78.4867))

    # Get unique nodes from the graph
    all_cities = list(graph.edges.keys())

    # Streamlit app

    st.title("Flight Path Finder")

    # Select source and destination airports
    source_airport = st.selectbox("Select Source Airport:", ['Select source airport'] + all_cities)
    destination_airport = st.selectbox("Select Destination Airport:", ['Select destination airport']+all_cities)

    if st.button("Find Path"):
        # Perform A* search with weather conditions
        final_path, final_distance = graph.astar_with_weather(source_airport, destination_airport)

        if final_path:
            st.success("Weather is ok!")
            st.write(f"Final path from {source_airport} to {destination_airport}: {' -> '.join(final_path)}")
            st.write(f"Final distance: {final_distance:.2f} km")
        else:
            st.error("No path found")
def current_weather_status():
    def fetch_weather_data(lat, lon, api_key):
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Error fetching weather data.")
            return None

    def display_airline_safety_info(data):
        if data:
            st.subheader("Weather Conditions:")
            st.write(f"Temperature: {data['main']['temp']} °C")
            st.write(f"Wind Speed: {data['wind']['speed']} m/s")
            st.write(f"Visibility: {data['visibility']} m")
            st.write(f"Cloud Cover: {data['weather'][0]['description']}")

            st.subheader("Aircraft Information:")
            st.write(f"Latitude: {data['coord']['lat']}")
            st.write(f"Longitude: {data['coord']['lon']}")
            st.write(f"Altitude: {data['main']['pressure']} hPa")

            st.subheader("Air Traffic Control:")
            st.write(f"Sunrise Time: {data['sys']['sunrise']}")
            st.write(f"Sunset Time: {data['sys']['sunset']}")

            st.subheader("Other Information:")
            st.write(f"Location Name: {data['name']}")
            st.write(f"Country: {data['sys']['country']}")
        else:
            st.error("No data available.")

    st.title("Airline Safety Information")

    latitude = st.number_input("Enter Latitude:")
    longitude = st.number_input("Enter Longitude:")
    api_key = "9c65640240e37a0967c73db1363d83a0"  # Replace with your OpenWeatherMap API key

    fetch_data = st.button("Fetch Data")

    if fetch_data:
        weather_data = fetch_weather_data(latitude, longitude, api_key)
        display_airline_safety_info(weather_data)

def gps_failure():
    st.title("GPS Failure")

    # Streamlit app title
    st.title("Inertial Navigation System (INS) Simulation")
    st.write("Inertial Navigation System (INS) is a navigation technology that estimates an object's position, "
             "velocity, and orientation using onboard sensors, such as accelerometers and gyroscopes. By continuously "
             "integrating acceleration to calculate velocity and then integrating velocity to calculate position, "
             "INS provides autonomous navigation capabilities independent of external references like GPS, "
             "making it essential for aircraft, spacecraft, and submarines.")
    # Function to simulate INS integration over time
    def simulate_ins(initial_latitude, initial_longitude):
        # Initialize estimated position
        estimated_latitude = initial_latitude
        estimated_longitude = initial_longitude

        # Simulate movement and position estimation over time
        for _ in range(10):  # Simulate for 10 time steps (seconds)
            # Simulate acceleration and gyroscope readings (example values)
            acceleration_x = random.uniform(-1.0, 1.0)  # m/s^2
            acceleration_y = random.uniform(-1.0, 1.0)  # m/s^2
            acceleration_z = random.uniform(-1.0, 1.0)  # m/s^2

            gyro_x = random.uniform(-0.1, 0.1)  # rad/s
            gyro_y = random.uniform(-0.1, 0.1)  # rad/s
            gyro_z = random.uniform(-0.1, 0.1)  # rad/s

            # Simulate time step (example: 1 second)
            time_step = 1.0  # seconds

            # Integrate accelerometer data to estimate velocity
            velocity_x = acceleration_x * time_step
            velocity_y = acceleration_y * time_step
            velocity_z = acceleration_z * time_step

            # Integrate velocity to estimate position changes
            estimated_latitude += velocity_x * time_step * 1e-5  # Simplified integration with scale factor
            estimated_longitude += velocity_y * time_step * 1e-5  # Simplified integration with scale factor

            # Introduce small random drift (error simulation)
            estimated_latitude += random.uniform(-0.0001, 0.0001)
            estimated_longitude += random.uniform(-0.0001, 0.0001)

            # Display estimated position
            st.write(
                f"Estimated Latitude: {estimated_latitude:.6f} degrees, Estimated Longitude: {estimated_longitude:.6f} degrees")

    # User input for initial latitude and longitude
    initial_latitude = st.number_input("Enter Initial Latitude (degrees)", value=30.0, step=0.000001)
    initial_longitude = st.number_input("Enter Initial Longitude (degrees)", value=-90.0, step=0.000001)

    # Button to start simulation
    if st.button("Start Simulation"):
        # Call the function to simulate INS integration over time
        simulate_ins(initial_latitude, initial_longitude)

def flight_location():
    # Sample real flight IDs and airline names
    flight_ids = ["AI101", "6E342", "SG876", "UK235", "AI567", "G8-420", "AI780", "6E511", "UK921", "SG324"]
    airline_names = ["Air India", "IndiGo", "SpiceJet", "Vistara", "GoAir", "AirAsia India"]

    # Function to generate random flight data
    def generate_flight_data(num_flights):
        flight_data = []
        for i in range(num_flights):
            flight_id = random.choice(flight_ids)
            flight_name = f"Flight-{i + 1}"
            flight_airline = random.choice(airline_names)
            flight_latitude = random.uniform(8, 37)  # Latitude range for India
            flight_longitude = random.uniform(68, 98)  # Longitude range for India
            fuel_percentage = random.randint(0, 100)
            max_range = random.randint(2000, 5000)  # Maximum flight range in km
            flight_data.append({
                'flight_id': flight_id,
                'flight_name': flight_name,
                'flight_airline': flight_airline,
                'flight_latitude': flight_latitude,
                'flight_longitude': flight_longitude,
                'fuel_percentage': fuel_percentage,
                'max_range': max_range
            })
        return flight_data

    # Function for the control center
    def control_center(flight_data):
        st.title("Control Center")

        airline = st.selectbox("Select Airline:", airline_names)
        filtered_flights = [flight for flight in flight_data if flight['flight_airline'] == airline]
        flight_ids_filtered = [flight['flight_id'] for flight in filtered_flights]

        if len(flight_ids_filtered) > 0:
            flight_id = st.selectbox("Select Flight ID:", flight_ids_filtered)
            selected_flight = [flight for flight in filtered_flights if flight['flight_id'] == flight_id][0]

            st.subheader("Basic Information:")
            st.write(f"Flight ID: {selected_flight['flight_id']}")
            st.write(f"Flight Name: {selected_flight['flight_name']}")
            st.write(f"Airline: {selected_flight['flight_airline']}")
            st.write(f"Latitude: {selected_flight['flight_latitude']}")
            st.write(f"Longitude: {selected_flight['flight_longitude']}")
            st.write(f"Fuel Percentage: {selected_flight['fuel_percentage']}%")
            st.write(f"Max Range: {selected_flight['max_range']} km")

            if st.button("Send Message"):
                # Assume sending message to flight with ID 'flight_id'
                st.success("Message sent successfully.")
        else:
            st.warning("No flights available for selected airline.")

    # Generate random flight data for 10 flights
    num_flights = 100
    flight_data = generate_flight_data(num_flights)

    # Run the control center function
    control_center(flight_data)


def user_guide():
    st.title("USER GUIDE")

def launch_rat():
    st.title("Ram Air Turbine")
    st.write("Airbus Launched the most Innovative Idea that is RAT, it’s generally a small turbine which converts the wind when its flying with high speed and converts into Electrical Energy, similar to Windmills.")
    if st.button("Launch RAT"):
        st.success("Launched...")
        st.success("Got power supply")

if __name__ == "__main__":
    main()
