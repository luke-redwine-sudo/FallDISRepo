from flask import Flask, render_template, request, jsonify, send_file
import folium
from shapely.geometry import Polygon, Point, mapping
import pandas as pd
import io
import os

app = Flask(__name__)

# Global list to store submitted polygons
polygons = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    """
    Endpoint to receive the drawn polygons from the client.
    """
    data = request.json
    for feature in data['features']:
        # Retrieve coordinates in GeoJSON format
        coordinates = feature['geometry']['coordinates'][0]
        # Create a shapely Polygon (using lon, lat)
        coords = [(lon, lat) for lon, lat in coordinates]  # Keep lon, lat for GeoJSON
        label = feature['properties'].get('label', 'Unnamed Area')
        color = feature['properties'].get('color', '#FF5733')  # Default color if not provided

        # Create a shapely Polygon
        polygon = Polygon(coords)
        polygons.append({'polygon': polygon, 'label': label, 'color': color})

    return jsonify({'message': 'Polygons submitted successfully!'})

@app.route('/get_polygons', methods=['GET'])
def get_polygons():
    """
    Endpoint to retrieve all stored polygons as GeoJSON.
    """
    geojson_features = []
    for poly_data in polygons:
        # Convert shapely Polygon to GeoJSON format
        geojson_feature = {
            'type': 'Feature',
            'geometry': mapping(poly_data['polygon']),
            'properties': {
                'label': poly_data['label'],
                'color': poly_data['color']
            }
        }
        geojson_features.append(geojson_feature)

    return jsonify({'type': 'FeatureCollection', 'features': geojson_features})
# Global list to store submitted polygons
polygons = []

# Directory to store processed files
PROCESSED_DIR = 'processed_files'
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.route('/process_csv', methods=['POST'])
def process_csv():
    if 'csvFile' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['csvFile']
    df = pd.read_csv(file)

    # Ensure the 'Crop' column is added
    df['Crop'] = None

    for i, row in df.iterrows():
        point = Point(row['Lng'], row['Lat'])  # Use 'lon' and 'lat' for consistency
        for poly in polygons:
            if poly['polygon'].contains(point):
                df.at[i, 'Crop'] = poly['label']
                break

    # Save the updated DataFrame to a new CSV file
    processed_filename = os.path.join(PROCESSED_DIR, 'processed_data.csv')
    df.to_csv(processed_filename, index=False)

    # Return the download URL and csv data
    csv_data = df.to_dict(orient='records')  # Convert DataFrame to a list of dicts
    return jsonify({
        'download_url': f'/download/{os.path.basename(processed_filename)}',
        'csvData': csv_data  # Include csvData in the response
    })


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='text/csv', as_attachment=True, download_name=filename)
    else:
        return "File not found.", 404

@app.route('/check_point', methods=['GET'])
def check_point():
    """
    Endpoint to check if a point is inside any stored polygon.
    """
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))
    point = Point(lon, lat)  # Create point with lon, lat

    for poly_data in polygons:
        if poly_data['polygon'].contains(point):
            return jsonify({'inside': True, 'label': poly_data['label']})
    return jsonify({'inside': False, 'label': None})

if __name__ == '__main__':
    app.run(debug=True)
