import logging
import os

import numpy as np

from datetime import timedelta
from glob import glob
from netCDF4 import Dataset

from scipy import ndimage

from google.cloud import bigquery

from src.bq_connector import BigQueryClient

PROJECT = 'modelagem-169213'
schemas = [{"name": "station_id", "type": "STRING", "mode": "REQUIRED"},
           {"name": "station_code", "type": "STRING", "mode": "REQUIRED"},
           {"name": "station_owner", "type": "STRING", "mode": "NULLABLE"},
           {"name": "station_provider", "type": "STRING", "mode": "REQUIRED"},
           {"name": "public", "type": "BOOL", "mode": "REQUIRED"},
           {"name": "geometry", "type": "GEOGRAPHY", "mode": "REQUIRED"},
           {"name": "longitude", "type": "FLOAT", "mode": "REQUIRED"},
           {"name": "latitude", "type": "FLOAT", "mode": "REQUIRED"},
           {"name": "altitude", "type": "FLOAT", "mode": "REQUIRED"},
           {"name": "city", "type": "STRING", "mode": "REQUIRED"},
           {"name": "state", "type": "STRING", "mode": "REQUIRED"},
           {"name": "country", "type": "STRING", "mode": "REQUIRED"},
           {"name": "datetime", "type": "TIMESTAMP", "mode": "REQUIRED"},
           {"name": "data", "type": "FLOAT", "mode": "REQUIRED"}]


def write_df_csv(df_now, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_now = df_now.set_index('datetime')
    df_now = df_now.sort_index()
    df_now.to_csv(out_path)
    logging.info("src.datahub.write_df_csv >> done writing intermediaries CSVs")


def _geometry_to_wkt(value, lon=None, lat=None):
    """Convert various geometry representations to a WKT string."""
    if hasattr(value, "wkt"):
        return value.wkt
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return f"POINT({value[0]} {value[1]})"
    if lon is not None and lat is not None:
        return f"POINT({lon} {lat})"
    raise TypeError(f"Unsupported geometry type: {type(value)!r}")


def write_df_bq(df_now, table, dataset_id='stations_verified'):
    df_now = df_now.copy()
    if 'geometry' not in df_now.columns:
        df_now['geometry'] = df_now.apply(
            lambda row: _geometry_to_wkt(None, lon=row.longitude, lat=row.latitude),
            axis=1)
    else:
        df_now['geometry'] = df_now.apply(
            lambda row: _geometry_to_wkt(row.geometry, lon=row.longitude, lat=row.latitude),
            axis=1)

    bq = BigQueryClient(PROJECT)

    schema = []
    for sch in schemas:
        schema.append(
            bigquery.SchemaField(
                sch["name"],
                sch["type"],
                sch["mode"]))

    if not bq.dataset_exists(dataset_id):
        bq.create_dataset(dataset_id)
    if not bq.table_exists(dataset_id, table):
        bq.create_table(dataset_id, table, schema)

    bq.client.insert_rows_from_dataframe(f'{dataset_id}.{table}',
                                         df_now,
                                         selected_fields=schema,
                                         skip_invalid_rows=True,
                                         ignore_unknown_values=True)

    logging.info("src.datahub.write_df_bq >> done inserting data into BigQuery")


def draw_distance(array, flags):

    points_events = np.where(np.isin(array, flags))
    array_dist = np.ones(array.shape)
    array_dist[points_events] = 0
    x = ndimage.distance_transform_edt(array_dist)

    return x


def get_array_skymask(date, parameters, force_clear_sky=False):

    files_mask = glob(
        parameters['acmf_info']['path'].format(
            procdate=date -
            timedelta(
                hours=1)))
    files_mask.sort()

    if not files_mask:
        if force_clear_sky:
            raise FileNotFoundError(f"clear sky mask files not found @ {parameters['acmf_info']['path']}")

        return None, None, None

    for count, file in enumerate(files_mask):

        logging.info(f'src.datahub.get_array_skymask >> reading clear sky mask file >> {file}')

        # leitura do arquivo
        nc = Dataset(file)
        acmf_array = nc.variables[parameters['acmf_info']['varname']][0][:][:]
        latitude = nc.variables[parameters['acmf_info']['lat_name']][:]
        longitude = nc.variables[parameters['acmf_info']['lon_name']][:]

        if latitude[0] < latitude[-1]:
            acmf_array = np.flipud(acmf_array)
            latitude = np.flipud(latitude)

        # Threshold mask
        x = draw_distance(
            array=acmf_array,
            flags=parameters['acmf_info']['cloud_flags'])
        resolution = np.abs(latitude[0] - latitude[1])
        dist_in_degree = x * resolution

        # colocando um limiar
        buffer = parameters['buffer_distance_in_degree']
        dist_in_degree[np.where(dist_in_degree <= buffer)] = 0

        if count == 0:
            array_stack_in_degree = dist_in_degree[:, :, np.newaxis]
            continue

        array_stack_in_degree = np.concatenate(
            (array_stack_in_degree, dist_in_degree[
                :, :, np.newaxis]), axis=2)

    # this array is the min distance between the pixel and the clear sky
    return np.nanmin(array_stack_in_degree, axis=2), latitude, longitude
