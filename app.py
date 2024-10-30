from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import random
import os
import csv
import io
from flask import send_file


app = Flask(__name__)
app.config['SECRET_KEY'] = 'a3d2e9b1c2f3d8e7f4b6a9c1b8f2e5d1'

geolocator = Nominatim(user_agent="tsp_solver")

# Function to calculate haversine distance
def distance(coord1, coord2):
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # Earth radius in kilometers

# Function to calculate total distance of a route
def total_distance(route, coords):
    return sum(distance(coords[route[i]], coords[route[(i + 1) % len(route)]]) for i in range(len(route)))

# Hill Climbing algorithm for TSP
def hill_climbing(coords):
    n = len(coords)
    current_solution = list(range(n))
    random.shuffle(current_solution)

    def solution_distance(solution):
        return total_distance(solution, coords)

    current_cost = solution_distance(current_solution)

    for _ in range(1000):
        new_solution = current_solution[:]
        i, j = random.sample(range(n), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        new_cost = solution_distance(new_solution)
        if new_cost < current_cost:
            current_solution = new_solution
            current_cost = new_cost

    return current_solution

# Function to plot the route
def plot_route(coords, cities, route):
    plt.figure(figsize=(10, 6))
    for i in range(len(route)):
        start, end = route[i], route[(i + 1) % len(route)]
        plt.plot([coords[start][1], coords[end][1]], [coords[start][0], coords[end][0]], 'bo-')
    for i, city in enumerate(cities):
        plt.text(coords[i][1], coords[i][0], city, fontsize=12, ha='right')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig('static/route.png')
    plt.close()

# Function to estimate travel time
def estimate_travel_time(route, coords, mode):
    # Average speeds in km/h
    speeds = {
        "car": 80,    # average speed of a car
        "train": 100,  # average speed of a train
        "flight": 600  # average speed of a flight
    }
    total_dist = total_distance(route, coords)
    travel_time = total_dist / speeds[mode]
    return travel_time

# Route to download the route as a CSV file
@app.route("/download", methods=["POST"])
def download():
    cities = request.form.getlist("cities")
    route = request.form.getlist("route")
    total_distance = request.form["total_distance"]
    travel_time = request.form["travel_time"]
    mode = request.form["mode"]

    # Create a CSV file in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Write CSV header
    writer.writerow(["City", "Order in Route"])
    for i, city in enumerate(route):
        writer.writerow([city, i + 1])

    writer.writerow([])
    writer.writerow(["Total Distance (km)", total_distance])
    writer.writerow(["Travel Time (hours)", travel_time])
    writer.writerow(["Transport Mode", mode])

    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype="text/csv", as_attachment=True, download_name="tsp_route.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        cities = request.form["cities"].replace("\n", ",").split(",")
        cities = [city.strip() for city in cities if city.strip()]

        if len(cities) < 2:
            return render_template("index.html", error="Please enter at least 2 cities.")

        coords = []
        for city in cities:
            location = geolocator.geocode(city)
            if location:
                coords.append((location.latitude, location.longitude))
            else:
                return render_template("index.html", error=f"City '{city}' not found.")

        # Solve TSP using Hill Climbing
        route = hill_climbing(coords)

        # Calculate total distance
        total_dist = total_distance(route, coords)

        # Get selected mode of transport
        mode = request.form.get("mode", "car")

        # Estimate travel time
        travel_time = estimate_travel_time(route, coords, mode)

        # Generate the route plot
        plot_route(coords, cities, route)

        ordered_cities = [cities[i] for i in route]

        return render_template("index.html", cities=cities, image="route.png", route=ordered_cities, 
                               total_distance=total_dist, travel_time=travel_time, mode=mode)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
