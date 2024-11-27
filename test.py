import PolygonDrawer

# Create an instance of PolygonDrawer
polygon_drawer = PolygonDrawer.PolygonDrawer(map_location=[32.7767, -96.7970], zoom_start=10)

# Manually add some polygons
polygon_drawer.add_polygon([(32.78, -96.8), (32.79, -96.81), (32.77, -96.82)], "Area 1")
polygon_drawer.add_polygon([(32.75, -96.75), (32.76, -96.76), (32.74, -96.77)], "Area 2")

# Check if a point is inside any of the polygons
point_in_area, label = polygon_drawer.is_point_in_polygon(32.78, -96.8)
print(f"Point is inside a labeled area: {point_in_area}, Label: {label}")

# Render the map
polygon_drawer.render()
