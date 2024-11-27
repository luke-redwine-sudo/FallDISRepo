import folium
from shapely.geometry import Polygon, Point
from folium.plugins import Draw

class PolygonDrawer:
    def __init__(self, map_location=[0, 0], zoom_start=2):
        # Initialize a folium map
        self.map = folium.Map(location=map_location, zoom_start=zoom_start)
        self.polygons = []  # List to store labeled polygons

        # Add drawing tools to the map
        draw = Draw(
            draw_options={
                'polyline': False,
                'rectangle': True,
                'circle': False,
                'circlemarker': False,
                'marker': False,
                'polygon': True,
            },
            edit_options={'edit': True, 'remove': True}
        )
        draw.add_to(self.map)

        # Add a submit button using a custom HTML button
        submit_button = folium.Html("""
            <button onclick="submitPolygons()" style="position:absolute;top:10px;left:10px;z-index:9999;">Submit</button>
            <script>
                function submitPolygons() {
                    var drawnItems = window.L.featureGroup();  // Use drawn items on map
                    window.map.eachLayer(function(layer) {
                        if (layer instanceof L.Polygon || layer instanceof L.Rectangle) {
                            drawnItems.addLayer(layer);
                        }
                    });
                    alert('Polygons submitted: ' + drawnItems.getLayers().length);
                }
            </script>
        """, script=True)
        submit_popup = folium.Popup(submit_button, max_width=2650)
        folium.Marker(location=map_location, popup=submit_popup).add_to(self.map)

    def add_polygon(self, coordinates, label):
        """
        Adds a labeled polygon to the map and stores it in the list of polygons.
        :param coordinates: List of coordinate tuples [(lat, lon), (lat, lon), ...]
        :param label: Label for the polygon
        """
        # Create a shapely polygon
        polygon = Polygon(coordinates)
        self.polygons.append({'polygon': polygon, 'label': label})

        # Add the polygon to the map for visualization
        folium.Polygon(
            locations=coordinates,
            color='blue',
            fill=True,
            fill_opacity=0.4,
            tooltip=label
        ).add_to(self.map)

    def is_point_in_polygon(self, lat, lon):
        """
        Checks if a given point is inside any of the polygons.
        :param lat: Latitude of the point
        :param lon: Longitude of the point
        :return: A tuple (bool, label) indicating if the point is inside a polygon, and its label
        """
        point = Point(lon, lat)
        for poly_data in self.polygons:
            if poly_data['polygon'].contains(point):
                return True, poly_data['label']
        return False, None

    def render(self):
        """
        Renders the map with all polygons and tools.
        """
        return self.map
