import geojson
import geopandas as gpd
from shapely import geometry


def gdf_to_geojson(gdf):
    """Convert a geopandas dataframe to a geojson object,
    which can be used as spatial_extent in OpenEO requests."""
    json = gdf.to_json()
    return geojson.loads(json)


def bbox_latlon_to_utm(bbox):
    """This function converts a bounding box defined in lat/lon
    to local UTM coordinates.
    It returns the bounding box in UTM and the epsg code
    of the resulting UTM projection."""

    # convert bounding box to geodataframe
    bbox_poly = geometry.box(*bbox)
    bbox_poly_utm, epsg = poly_latlon_to_utm(bbox_poly)
    bbox_utm = bbox_poly_utm.bounds

    return bbox_utm, epsg


def poly_latlon_to_utm(poly):

    # convert polygon to geodataframe
    poly_gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    # estimate best UTM zone
    crs = poly_gdf.estimate_utm_crs()
    epsg = int(crs.to_epsg())

    # convert to UTM
    poly_utm = poly_gdf.to_crs(crs).geometry[0]

    return poly_utm, epsg
