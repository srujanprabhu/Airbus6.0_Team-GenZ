# Enhancing Flight Navigation Mechanism for Optimal Route Planning and Risk Mitigation - AIrbus Aerothon 6.0

## Project by GEN-Z Team

This project aims to provide an enhanced flight navigation mechanism that uses real-time weather data to plan optimal flight routes while mitigating risks associated with bad weather conditions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/flight-navigation.git
    cd flight-navigation
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the project, navigate to the project directory and run the following command:

```sh
streamlit run main.py
```

## Project Structure
flight-navigation
├── main.py
├── page1.py
├── airports.json.
├── flight.csv.
├── requirements.txt.
└── README.md.


## Theme of the Project
Enhancing Flight Navigation Mechanism for Optimal Route Planning and Risk Mitigation

We created an User Friendly prototype which is helpful in terms of AirBus Control System. We designed, developed, and implemented a robust software solution that leverages existing algorithms to identify optimal flight paths. 
Excecute the prototype in terminal by typing the command - streamlit run main.py
Excecution - 
Main page Consists of Title, Gen-Z Airbus control System
There will be Option to select the Features, features in our project are - 

1) Flight Details - This feature helps to fetch flight details, by entering the single input that is Airplane ID, after entering the ID of the Plane, we'll get every information about that Plane.
   
2) Optimal Path Finder - The main challenge in this project is Optimal Path Finder, Two find the distance between two places or some given co-ordinates, Great Circle route Algorithm is used. The great circle route is the shortest path between two points on the surface of a sphere, such as the Earth. But it is not optimal. To find the OPTIMAL DISTANCE between two points, Dijkstra's Algorithm can be used, but it also has it’s own limitations. Dijkstra, Can be less efficient for large graphs if the goal is to find a path to a specific target because it explores all possible paths. We used A* Search Algorithm to find the Best Optimal path between two Places, it is most efficient compared to Dijkstra and it can find optimal path efficiently. A* Can be more efficient if the heuristic function is well-designed, reducing the number of nodes explored. Time complexity Depends on the heuristic, but in the worst case, it can be similar to Dijkstra's algorithm. So A* is the best Algorithm to find the Optimal Flight Paths.
Select Source Airport and Destination Airport, it will give the best path in best time.

3)Emergency Landing - This is one of the most important feature in this project, by giving the input as current co-ordiantes and fuel percentage, it will estimate and give the best nearest places to land to ensure safety of passengers and million dollar Airplane.

4)Real time Weather Analysis - We implemented Real time weather analysis by fetching the real time data from OpenWeather API - (API key is available, just for evaluation purposes/Please dont misuse), by giving the source and destination airport, it will give analysis and by using previous Optimal Path finder, it will give the best Path to reach the destination or if the weather is too bad at destination or source, it says No path Found, because of safety purposes. We did this by fetching data real time and giving a threshold to make optimal decisions

5)Current weather status - It will also fetch data from API and give the details of weather at specific coordinates independent of Paths

6)GPS Failure - We used INS (Inertial Navigation System), this is not real data, but it's assumed to be real, Inertial Navigation System (INS) is a navigation technology that estimates an object's position, velocity, and orientation using onboard sensors, such as accelerometers and gyroscopes. By continuously integrating acceleration to calculate velocity and then integrating velocity to calculate position, INS provides autonomous navigation capabilities independent of external references like GPS, making it essential for aircraft, spacecraft, and submarines.
By giving the initial Latitude and Longitude, it will caluclate Present Co-ordinates without any GPS, by taking inputs of different sensors present in the Aircraft

7)Sharing Message to Flight Location - We will share the current Flight location to the Airport control centre, to communicate and prepare for the emergency Landings

8)Electronic Power Failure - We can click to launch Ram Air Turbine(RAT) to generate electricy from Turbines

9)User Guide - Consists of Pilot manuals, commands, precautions everything.


 

