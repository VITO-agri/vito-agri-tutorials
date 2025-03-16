from ipyleaflet import (
    DrawControl,
    LayersControl,
    Map,
    SearchControl,
    basemaps,
    basemap_to_tiles,
)
from IPython.display import display
from ipywidgets import HTML, widgets
import geopandas as gpd
from shapely.geometry import Polygon, Point, shape
import matplotlib.pyplot as plt
from pathlib import Path

from vito_agri_tutorials.utils.geo import poly_latlon_to_utm


class ui_map:
    def __init__(
        self,
        geometry_type: str = "polygon",
    ):
        """
        Initializes an ipyleaflet map with a draw control to draw points or polygons.

        Parameters
        ----------
        geometry_type : str, optional
            The geometry type to draw on the map. By default 'polygon' is selected.
            Possible values are 'polygon' or 'point'.
        """

        self.geometry_type = geometry_type
        # Initialize the objects dictionary
        self.objects = gpd.GeoDataFrame(columns=["id", "geometry"])

        # Construct description widget
        message = f"Use the draw control to draw a {self.geometry_type} on the map. When you want to save the object, provide a unique object ID in the field below the map and hit the submit button. This will automatically store the last drawn object."
        description = widgets.HTML(value=f"<p>{message}</p>")

        # Construct base layers
        osm = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
        osm.base = True
        osm.name = "Open street map"

        img = basemap_to_tiles(basemaps.Esri.WorldImagery)
        img.base = True
        img.name = "Satellite imagery"

        # Construct map
        self.map = Map(
            center=(51.1872, 5.1154), zoom=2, layers=[img, osm], scroll_wheel_zoom=True
        )
        self.map.add_control(LayersControl())

        self.draw_control = DrawControl(edit=False)

        if geometry_type == "polygon":

            self.draw_control.polygon = {
                "shapeOptions": {
                    "fillColor": "#6be5c3",
                    "color": "#00F",
                    "fillOpacity": 0.3,
                },
                "drawError": {"color": "#dd253b", "message": "Oups!"},
                "allowIntersection": False,
                "metric": ["km"],
            }
            self.draw_control.rectangle = {
                "shapeOptions": {
                    "fillColor": "#6be5c3",
                    "color": "#00F",
                    "fillOpacity": 0.3,
                },
                "drawError": {"color": "#dd253b", "message": "Oups!"},
                "allowIntersection": False,
                "metric": ["km"],
            }

        elif geometry_type == "point":
            self.draw_control.marker = {
                "shapeOptions": {
                    "color": "red",
                },
            }
            self.draw_control.polygon = {}
            self.draw_control.rectangle = {}

        else:
            raise ValueError(
                f"Unknown geometry type: {geometry_type}, choose either 'polygon' or 'point'"
            )

        self.draw_control.circle = {}
        self.draw_control.polyline = {}
        self.draw_control.circlemarker = {}

        # # Attach the event listener to the draw control
        # self.draw_control.on_draw(self.handle_draw)
        self.map.add_control(self.draw_control)

        search = SearchControl(
            position="topleft",
            url="https://nominatim.openstreetmap.org/search?format=json&q={s}",
            zoom=20,
        )
        self.map.add_control(search)

        # Create an input field and submit button
        self.inputfield = widgets.Text(
            placeholder="Enter a unique ID for this object",
            description="Object ID",
        )

        self.submit_button = widgets.Button(
            description="Submit", button_style="success"
        )
        self.submit_button.on_click(self.handle_submit)

        self.input = widgets.HBox([self.inputfield, self.submit_button])

        # Define output field
        self.output = widgets.Output()

        self.widget = widgets.VBox(
            [description, self.map, self.input, self.output],
            layout={"height": "600px"},
        )

        return display(self.widget)

    def handle_submit(self, event):
        with self.output:

            self.output.clear_output()

            # Get last drawn object
            obj = self.draw_control.last_draw

            if obj.get("geometry") is None:
                message = "Error: First draw an object on the map!"
                display(HTML(f'<span style="color: red;"><b>{message}</b></span>'))
                return

            object_id = self.inputfield.value
            if object_id in self.objects.keys():
                message = "Error: ID already exists. Please choose a different one."
                display(HTML(f'<span style="color: red;"><b>{message}</b></span>'))
                return

            geometry_type = obj["geometry"]["type"]

            if geometry_type == "Point":
                lon, lat = obj["geometry"]["coordinates"]
                geometry = Point(lon, lat)

                display(HTML(f"<b>Point location:</b> Lat:{lat}, Lon:{lon}"))

            else:
                geometry = Polygon(shape(obj.get("geometry")))
                bbox = geometry.bounds
                display(HTML(f"<b>Bounding box:</b> {bbox}"))

                # We convert our polygon to local UTM projection
                # for area computation
                poly_utm, epsg = poly_latlon_to_utm(geometry)
                area = poly_utm.area / 1e6
                display(HTML(f"<b>Area of polygon:</b> {area:.2f} km²"))

            new_entry = gpd.GeoDataFrame(
                [{"id": object_id, "geometry": geometry}],
                geometry="geometry",
                crs="EPSG:4326",
            )
            self.objects = gpd.pd.concat([self.objects, new_entry], ignore_index=True)

            # Display "success" message
            message = f"Object {object_id} added to the map."
            display(HTML(f'<span style="color: green;"><b>{message}</b></span>'))

            # Reset input field
            self.inputfield.value = ""

    def get_objects(self):

        if self.objects.empty:
            raise ValueError(
                "Please first draw something on the map before proceeding."
            )

        return self.objects


class RegionPicker:
    def __init__(self, max_level: int = 1):

        indir = Path(__file__).resolve().parents[1] / "resources"

        self.adm0_df = gpd.read_file(indir / "GAUL_0.gpkg")
        self.adm1_df = gpd.read_file(indir / "GAUL_1.gpkg")
        self.adm2_df = gpd.read_file(indir / "GAUL_2.gpkg")

        self.max_level = max_level

        self.selections = []

        self.dropdown_adm0 = widgets.Dropdown(
            options=["Select Country"]
            + sorted(list(self.adm0_df["ADM0_NAME"].unique())),
            description="Country:",
            style={"description_width": "initial"},
        )

        self.dropdown_adm1 = widgets.Dropdown(
            options=["Select State"],
            description="State:",
            style={"description_width": "initial"},
        )

        self.dropdown_adm2 = widgets.Dropdown(
            options=["Select District"],
            description="District:",
            style={"description_width": "initial"},
        )

        self.submit_button = widgets.Button(
            description="Add to selection", button_style="success"
        )
        self.reset_button = widgets.Button(
            description="Reset selection", button_style="danger"
        )

        self.output = widgets.Output()
        self.plot_output = widgets.Output()

        self.dropdown_adm0.observe(self.update_adm1, names="value")
        self.dropdown_adm1.observe(self.update_adm2, names="value")
        self.submit_button.on_click(self.submit_selection)
        self.reset_button.on_click(self.reset_selection)

        display(widgets.VBox(self._get_widgets()))

    def _get_widgets(self):

        levels = range(self.max_level + 1)
        dropdowns = [self.dropdown_adm0, self.dropdown_adm1, self.dropdown_adm2]

        # Hide all dropdowns initially
        for dropdown in dropdowns:
            dropdown.layout.visibility = "hidden"

        # Show only the relevant dropdowns based on max_level
        for i in levels:
            dropdowns[i].layout.visibility = "visible"

        # Return the list properly by using + instead of extend()
        return dropdowns[: self.max_level + 1] + [
            self.submit_button,
            self.reset_button,
            self.output,
            self.plot_output,
        ]

    def update_adm1(self, *args):

        if self.max_level == 0:
            return

        selected_country = self.dropdown_adm0.value

        with self.plot_output:
            self.plot_output.clear_output()

        if selected_country == "Select Country":
            self.dropdown_adm1.options = ["Select State"]
        else:
            filtered_states = self.adm1_df[
                self.adm1_df["ADM0_NAME"] == selected_country
            ]
            self.dropdown_adm1.options = ["Select State"] + sorted(
                list(filtered_states["ADM1_NAME"].unique())
            )
            self.plot_states(filtered_states)
        self.dropdown_adm1.value = "Select State"
        self.dropdown_adm2.options = ["Select District"]
        self.dropdown_adm2.value = "Select District"

    def update_adm2(self, *args):

        if self.max_level <= 1:
            return

        selected_state = self.dropdown_adm1.value
        with self.plot_output:
            self.plot_output.clear_output()

        if selected_state == "Select State":
            self.dropdown_adm2.options = ["Select District"]
        else:
            filtered_districts = self.adm2_df[
                self.adm2_df["ADM1_NAME"] == selected_state
            ]
            self.dropdown_adm2.options = ["Select District"] + sorted(
                list(filtered_districts["ADM2_NAME"].unique())
            )
            self.plot_districts(filtered_districts)
        self.dropdown_adm2.value = "Select District"

    def plot_districts(self, districts_gdf):
        with self.plot_output:
            self.plot_output.clear_output()
            fig, ax = plt.subplots(figsize=(16, 16))
            districts_gdf.plot(ax=ax, edgecolor="black", alpha=0.5)
            for _, row in districts_gdf.iterrows():
                if row.geometry:
                    centroid = row.geometry.centroid
                    ax.text(
                        centroid.x,
                        centroid.y,
                        row["ADM2_NAME"],
                        fontsize=10,
                        ha="center",
                        color="red",
                    )
            plt.title("Selected State - Districts")
            plt.show()

    def plot_states(self, states_gdf):
        with self.plot_output:
            self.plot_output.clear_output()
            fig, ax = plt.subplots(figsize=(16, 16))
            states_gdf.plot(ax=ax, edgecolor="black", alpha=0.5)
            for _, row in states_gdf.iterrows():
                if row.geometry:  # Ensure the geometry is valid
                    centroid = row.geometry.centroid
                    ax.text(
                        centroid.x,
                        centroid.y,
                        row["ADM1_NAME"],
                        fontsize=10,
                        ha="center",
                        color="red",
                    )
            plt.title("Selected Country - States")
            plt.show()

    def submit_selection(self, b):
        with self.output:
            self.output.clear_output()
            if self.dropdown_adm0.value == "Select Country":
                print("Please select at least a country.")

            if self.dropdown_adm2.value != "Select District":
                result = {
                    "L0": self.dropdown_adm0.value,
                    "L1": self.dropdown_adm1.value,
                    "L2": self.dropdown_adm2.value,
                }
                self.dropdown_adm2.value = "Select District"
                filtered_districts = self.adm2_df[
                    self.adm2_df["ADM1_NAME"] == self.dropdown_adm1.value
                ]
                self.plot_districts(filtered_districts)
            elif self.dropdown_adm1.value != "Select State":
                result = {
                    "L0": self.dropdown_adm0.value,
                    "L1": self.dropdown_adm1.value,
                    "L2": None,
                }
                self.dropdown_adm1.value = "Select State"
                filtered_states = self.adm1_df[
                    self.adm1_df["ADM0_NAME"] == self.dropdown_adm0.value
                ]
                self.plot_states(filtered_states)
            else:
                result = {"L0": self.dropdown_adm0.value, "L1": None, "L2": None}
                self.dropdown_adm0.value = "Select Country"
            self.selections.append(result)
            print("Current Selections:")
            for item in self.selections:
                print(item)
            # self.plot_output.clear_output()

    def reset_selection(self, b):
        with self.output:
            self.output.clear_output()
            self.selections = []
            print("Selections reset.")
            self.dropdown_adm0.value = "Select Country"
            self.dropdown_adm1.options = ["Select State"]
            self.dropdown_adm2.options = ["Select District"]
            self.plot_output.clear_output

    def selection_to_gdf(self, level: int) -> gpd.GeoDataFrame:
        """
        Get the selected regions as a GeoDataFrame with geometries.
        """

        if self.selections == []:
            raise ValueError("No regions selected yet.")

        if level not in [0, 1, 2]:
            raise ValueError("Selected GAUL level should be one of 0, 1, 2")

        # Get the corresponding GeoDataFrame
        if level == 0:
            gdf = self.adm0_df.copy()
        elif level == 1:
            gdf = self.adm1_df.copy()
        else:
            gdf = self.adm2_df.copy()

        # Select regions which have the desired level
        regions = [
            region for region in self.selections if region[f"L{level}"] is not None
        ]
        if len(regions) == 0:
            raise ValueError(f"No regions with level {level} found in the list.")

        # Get the corresponding geometries
        for level in range(level + 1):
            # Get attribute name
            attr_name = f"ADM{level}_NAME"
            # Filter on region names
            region_names_lvl = [region[f"L{level}"] for region in regions]
            gdf = gdf[gdf[attr_name].isin(region_names_lvl)]

        gdf.rename(columns={attr_name: "id"}, inplace=True)

        return gdf
