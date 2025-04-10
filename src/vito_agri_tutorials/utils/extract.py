import xarray as xr
import geopandas as gpd
from pyproj import Transformer
from rasterio import CRS


def extract_points(
    inarr: xr.DataArray, sampled_points: gpd.GeoDataFrame, epsg: int = 32736
) -> xr.DataArray:
    points_x, points_y = [], []

    for points in sampled_points:
        for point in points.geoms:
            points_x.append(point.x)
            points_y.append(point.y)

    # Transform the coordinates of the sampled points into the CRS of the
    # input sensor array
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_epsg(epsg),
        always_xy=True
    )

    points_data = []
    for pt_x, pt_y in zip(points_x, points_y):
        pt_x, pt_y = transformer.transform(pt_x, pt_y)
        point_data = inarr.sel(x=pt_x, y=pt_y, method='nearest')
        points_data.append(point_data)

    training_data = xr.concat(
        points_data, dim='sample'
    ).assign_coords(
        {'sample': list(range(0, len(points_x)))}
    ).drop_vars(['x', 'y'])

    return training_data
