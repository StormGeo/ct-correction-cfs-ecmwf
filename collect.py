#!/airflow/models/ct-near/venv/bin/python3

import argparse
import datetime
import logging as log
import numpy as np
import os
import pandas as pd
import yaml

from distutils.util import strtobool
from functools import partial

import netCDF4 as nc

from ncBuilder import ncHelper

from src.bq_connector import BigQueryClient
from src.commons import PROVIDER_LIST, PROVIDER_TO_VALID
from src.commons import config_log
from src.interpol_methods import get_indexes

PROJECT = 'modelagem-169213'
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))) + '/ct-near'

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"] = '/airflow/base/credentials/gcp-bq.json'


def fix_hgt(row, topo=None, tlats=None, tlons=None, vname=None):
    ix, iy = get_indexes(tlats, tlons, row['latitude'], row['longitude'])
    hgt = topo[ix, iy]
    if hgt < 0:
        hgt = 0

    if row['altitude'] is None:
        row['altitude'] = hgt

    elif row['altitude'] < 0:
        row['altitude'] = hgt

    elif row['altitude'] > 5000:
        row['altitude'] = hgt

    else:
        if np.abs(hgt - row['altitude']) > 100:

            # if vname in ['2m_air_temperature']:
            #
            #     # do not use abs, the subtraction signal will adjust the value
            #     row['data'] -= 6.5 * (hgt - row['altitude'])/1000

            row['altitude'] = hgt
    return row

def get_args():
    parser = argparse.ArgumentParser(
        description='Data Collection from BigQuery')
    parser.add_argument(
        '-v',
        dest='vname',
        required=True,
        help='Variable to be processed',
        choices=['2m_air_temperature', 'total_precipitation', '10m_wind', '10m_wind_direct',
                 '2m_relative_humidity', 'msl_pressure', '10m_wind_gust'],
        action='store')
    parser.add_argument(
        '-d',
        dest='dtnow',
        required=False,
        help='Target Date ISO format',
        default=datetime.datetime.utcnow().isoformat(),
        action='store')
    parser.add_argument(
        '-l',
        dest='lag',
        required=False,
        help='Lag in hours',
        default=97,
        action='store')
    parser.add_argument(
        '-a',
        dest='domain',
        required=False,
        help='Domain Area [lat0, lat1, lon0, lon1]',
        default=[-53.0, 22.0, -88.7, -31.4],
        nargs=4,
        action='store')
    parser.add_argument(
        '-o',
        dest='out_path',
        required=True,
        help='Output Parquet file path',
        action='store')
    parser.add_argument(
        '--inmet',
        dest='inmet',
        required=False,
        help='Use only inmet stations',
        default=False,
        action='store_true')

    parser.add_argument(
        '--oficial-only',
        dest="oficial_only",
        default=False,
        choices=[True, False],
        action='store',
        type=lambda x: bool(strtobool(x)),
        help='Use this parameter to use only oficial providers, check in src.commons.PROVIDER_LIST'
    )

    parser.add_argument(
        '--to-validate',
        dest="to_validate",
        default=False,
        choices=[True, False],
        action='store',
        type=lambda x: bool(strtobool(x)),
        help='Use this parameter to run in validate, the src.commons.PROVIDER_TO_VALID providers will not be considered'
    )

    parser.add_argument(
        '--use-bl',
        dest="use_bl",
        default=True,
        choices=[True, False],
        action='store',
        type=lambda x: bool(strtobool(x)),
        help='Use this parameter to include black listed weather stations'
    )


    return parser.parse_args()


def get_table(vname):
    return {'2m_air_temperature': 'air_temperature',
            'total_precipitation': 'precipitation_amount',
            '2m_relative_humidity': 'relative_humidity',
            '10m_u_component_of_wind': 'wind_speed_u',
            '10m_v_component_of_wind': 'wind_speed_v',
            '10m_wind_speed': 'wind_speed',
            '10m_wind_direction': 'wind_direction',
            '10m_wind_gust': 'wind_speed_of_gust',
            'msl_pressure': 'msl_pressure'}.get(vname, None)


def load_config(vname):
    path = f'{BASE_DIR}/configs/{vname}.yml'
    with open(path, 'r') as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise e
        return data


def direction(U, V):
    rp2 = np.divide(45.0, np.arctan(1.0))
    direction = np.add(np.multiply(np.arctan2(U, V), rp2), 180)
    direction[np.where(direction < 0)] = 0
    return direction


def wind2components(wspd, wdir):
    wdir[np.isnan(wdir)] = 45
    mdir = 270 - wdir
    mdir[mdir <= 0] += 360
    U = wspd * np.cos(np.radians(mdir))
    V = wspd * np.sin(np.radians(mdir))
    return U, V


def abs_max(col):
    try:
        return col[np.nanargmax(np.abs(col))]
    except:
        return np.nan


def main():
    config_log()
    args = get_args()
    dtNow = pd.Timestamp(datetime.datetime.fromisoformat(
        args.dtnow)).floor('05T').tz_localize(0).to_pydatetime()
    dtBefore = dtNow - datetime.timedelta(hours=int(args.lag))
    lat0, lat1, lon0, lon1 = args.domain

    black_list = load_config('black_list')
    black_list = [f"'{i}'" for i in black_list["station_id"]]

    bq = BigQueryClient(PROJECT)
    use_black_list = args.use_bl

    if args.vname == '10m_wind':
        # Query wind direction
        query = f"""
                SELECT  *
                    FROM `{PROJECT}.stations.wind_direction`
                    WHERE datetime <= '{dtNow.isoformat()}'
                          AND datetime > '{dtBefore.isoformat()}'
                          AND ST_WITHIN(geometry,
                          ST_GeogFromText('POLYGON(({lon0} {lat0}, {lon1} {lat0}, {lon1} {lat1}, {lon0} {lat1}, {lon0} {lat0}))'))
                          AND station_id not in ({', '.join(black_list)})
                          AND data <= 360
                          AND data >= 0
                """

        if args.oficial_only:
            provider_list_string = ', '.join([f"'{value}'" for value in PROVIDER_LIST + ['ana']])

            query += f' AND station_provider in ({provider_list_string})'

        df_wdir = bq.client.query(query).to_dataframe(
            progress_bar_type='tqdm', bqstorage_client=bq.stclient)

        df_wdir = df_wdir.rename(columns={'data': 'wdir'})
        df_wdir = df_wdir[df_wdir['wdir'].notnull()]
        uniq_ids = np.unique(df_wdir['station_id'].values)
        uniq_ids = [f"'{i}'" for i in uniq_ids]

        # Query wind speed
        query = f"""
                SELECT  *
                    FROM `{PROJECT}.stations.wind_speed`
                    WHERE datetime <= '{dtNow.isoformat()}'
                          AND datetime > '{dtBefore.isoformat()}'
                          AND station_id in ({', '.join(uniq_ids)})
                """
        df_wspd = bq.client.query(query).to_dataframe(
            progress_bar_type='tqdm', bqstorage_client=bq.stclient)
        df_wspd = df_wspd.rename(columns={'data': 'wspd'})
        df_wspd = df_wspd[df_wspd['wspd'].notnull()]

        df = pd.merge(df_wspd, df_wdir[['datetime', 'station_id', 'wdir']], on=[
                      'datetime', 'station_id'])
        df = df.dropna()

        df['U'], df['V'] = wind2components(
            df['wspd'].values, df['wdir'].values)

        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.index = pd.DatetimeIndex(df.index).floor('T')
        df.sort_index(inplace=True)

        df = df.groupby(['station_id']).resample('1H', closed='right', label='right').agg({
            'latitude': 'first',
            'longitude': 'first',
            'altitude': 'first',
            'geometry': 'first',
            'station_code': 'first',
            'station_owner': 'first',
            'station_provider': 'first',
            'public': 'first',
            'city': 'first',
            'state': 'first',
            'country': 'first',
            'wdir': np.nanmedian,
            'wspd': np.nanmax,
            'U': abs_max,
            'V': abs_max
        }).dropna().reset_index()

    else:
        table = get_table(args.vname) if args.vname not in [
            '10m_wind_direct'] else 'wind_speed'

        if use_black_list:
            query = f"""
                    SELECT *
                    FROM `{PROJECT}.stations.{table}`
                        WHERE datetime <= '{dtNow.isoformat()}'
                              AND datetime > '{dtBefore.isoformat()}'
                              AND ST_WITHIN(geometry,
                                  ST_GeogFromText('POLYGON(({lon0} {lat0}, {lon1} {lat0}, {lon1} {lat1}, {lon0} {lat1}, {lon0} {lat0}))'))
                              AND station_id not in ({', '.join(black_list)})
                    """
        else:
            query = f"""
                                SELECT *
                                FROM `{PROJECT}.stations.{table}`
                                    WHERE datetime <= '{dtNow.isoformat()}'
                                          AND datetime > '{dtBefore.isoformat()}'
                                          AND ST_WITHIN(geometry,
                                              ST_GeogFromText('POLYGON(({lon0} {lat0}, {lon1} {lat0}, {lon1} {lat1}, {lon0} {lat1}, {lon0} {lat0}))'))
                                """
        if args.vname == 'total_precipitation':
            query += ' AND data != 0.2'
            query += ' AND data >= 0'

        if args.vname == '2m_air_temperature':
            query += ' AND data < 50'
            query += ' AND data > -20'

        if args.vname == '2m_relative_humidity':
            query += ' AND data <= 100'
            query += ' AND data > 0'

        if args.oficial_only:
            provider_list_string = ', '.join([f"'{value}'" for value in PROVIDER_LIST + ['ana']])
            query += f' AND station_provider in ({provider_list_string})'

        df = bq.client.query(query).to_dataframe(
            progress_bar_type='tqdm', bqstorage_client=bq.stclient)

        df['datetime'] = pd.to_datetime(df['datetime'])

        df.set_index('datetime', inplace=True)
        df.index = pd.DatetimeIndex(df.index).floor('T')
        df.sort_index(inplace=True)

        if args.vname in ['total_precipitation']:
            df = df.groupby(['station_id']).resample('1H', closed='right', label='right').agg({
                'latitude': 'first',
                'longitude': 'first',
                'altitude': 'first',
                'geometry': 'first',
                'station_code': 'first',
                'station_owner': 'first',
                'station_provider': 'first',
                'public': 'first',
                'city': 'first',
                'state': 'first',
                'country': 'first',
                'data': np.nansum
            }).dropna().reset_index()
        else:
            df = df.groupby(['station_id']).resample('1H', closed='right', label='right').agg({
                'latitude': 'first',
                'longitude': 'first',
                'altitude': 'first',
                'geometry': 'first',
                'station_code': 'first',
                'station_owner': 'first',
                'station_provider': 'first',
                'public': 'first',
                'city': 'first',
                'state': 'first',
                'country': 'first',
                'data': np.nanmean if args.vname not in ['10m_wind_direct'] else np.nanmax
            }).dropna().reset_index()

    if args.inmet:
        df = df[df['station_owner'] == 'inmet']

    if args.to_validate:
        df = df.loc[~df['station_provider'].isin(PROVIDER_TO_VALID)]

    df.set_index('datetime', inplace=True)
    uniq_idx = df.index.unique().sort_values()

    idx = np.argmin(np.abs((uniq_idx - dtNow).total_seconds()))

    if uniq_idx[idx] > dtNow:
        idx = -2

    log.info(f"Current hour been processed is: {uniq_idx[idx]}, {idx}")

    df_now = df.loc[uniq_idx[idx]].reset_index()
    df = df[df['station_id'].isin(df_now['station_id'])]
    df = df.loc[df.index <= uniq_idx[idx]]

    log.info(f"Total number of collected stations: {len(df_now['latitude'])}")

    uniq_providers = np.unique(df['station_provider'].values)
    for up in uniq_providers:
        log.info(f"Total number of {up} stations present in current hour: {len(np.unique(df[df['station_provider'] == up]['station_id'].values))}")

    # fixing altitudes
    log.info("Starting altitudes correction")
    topo_file = nc.Dataset(f'{BASE_DIR}/data/etopo_0p01.nc', 'r')
    topo = topo_file.variables['z'][:]
    tlats, tlons = ncHelper.get_lats_lons(topo_file)

    df = df.apply(partial(fix_hgt, topo=topo, tlats=tlats, tlons=tlons, vname=args.vname),
                  axis=1)

    topo_file.close()
    df['altitude'] = df['altitude'].round(1)

    os.makedirs(os.path.split(args.out_path)[0], exist_ok=True)

    df.reset_index().to_parquet(args.out_path)

if __name__ == "__main__":
    main()
