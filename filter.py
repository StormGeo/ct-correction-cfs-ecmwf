#!/airflow/models/ct-near/venv/bin/python3
import argparse
import concurrent.futures
import os

import logging as log

import shapely
import yaml

import numpy as np
import pandas as pd

from distutils.util import strtobool

from src.commons import config_log
from src.datahub import get_array_skymask
from src.erlang import RIQ_filter
from src.erlang import accumulated_filter
from src.erlang import aceleration_filter
from src.erlang import clear_sky_filter
from src.erlang import clusterizer_filter
from src.erlang import is_within_distance
from src.erlang import monotony_filter
from src.erlang import monotony_filter_sequence
from src.erlang import threshold_filter
from src.erlang import zscore_filter

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))) + '/ct-near'

OFICIAL_PROVIDERS = ['redemet', 'ogimet', 'metos',
                     'cgesp', 'ciram', 'vale',
                     'omega', 'inema', 'simepar',
                     'funceme', 'inmet']


def get_args():
    parser = argparse.ArgumentParser(description='Data Filtering')
    parser.add_argument(
        '-i',
        dest='input_path',
        required=True,
        help='Input Parquet file path',
        action='store')
    parser.add_argument(
        '-o',
        dest='out_path',
        required=True,
        help='Output Parquet file path',
        action='store')
    parser.add_argument(
        '-v',
        dest='vname',
        required=True,
        help='Variable to be processed',
        action='store')

    parser.add_argument(
        '-r', dest='res',
        required=False,
        help='domain Resolution, used to radius filter',
        default=0.125,
        action='store',
    )
    parser.add_argument(
        '--full-filter',
        dest="full_filter",
        default=True,
        choices=[True, False],
        action='store',
        type=lambda x: bool(strtobool(x)),
        help='Use this parameter to set the filter'
    )

    parser.add_argument(
        '--verbose',
        dest="verbose",
        default=False,
        choices=[True, False],
        action='store',
        type=lambda x: bool(strtobool(x)),
        help='Use this parameter to set the filter'
    )

    return parser.parse_args()


def load_config(vname):
    path = f'{BASE_DIR}/configs/{vname}.yml'
    with open(path, 'r') as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise e
        return data


def process_filters(row, row_historic, array, vname, latitude,
                    longitude, filter_type, parameters):

    # conditions to don't apply filters
    if row['station_provider'] in ['redemet', 'ogimet', 'inmet']:
        return row

    else:

        if filter_type == 'clear_sky':
            r = clear_sky_filter(row,
                                 array,
                                 latitude,
                                 longitude,
                                 vname,
                                 parameters)
        elif filter_type == 'RIQ':
            r = RIQ_filter(row,
                           row_historic,
                           vname,
                           parameters)
        elif filter_type == 'monotony':
            r = monotony_filter(row,
                                row_historic,
                                vname,
                                row['datetime'],
                                parameters)
        elif filter_type == 'monotony_sequence':
            r = monotony_filter_sequence(row,
                                         row_historic,
                                         vname,
                                         row['datetime'],
                                         parameters)
        elif filter_type == 'aceleration':
            r = aceleration_filter(row,
                                   row_historic,
                                   vname,
                                   parameters)
        elif filter_type == 'zscore':
            r = zscore_filter(row,
                              row_historic,
                              vname,
                              parameters)
        elif filter_type == 'threshold':
            r = threshold_filter(row,
                                 vname,
                                 parameters)
        elif filter_type == 'accumulated':
            r = accumulated_filter(row,
                                   row_historic,
                                   vname,
                                   parameters)
        else:
            raise ValueError(f"filter not defined @ {filter_type}")

        row['data'] = r['data']
        return row


def apply_filter(df, df_historic, vname, filter_type, parameters,
                 array=False, latitude=False, longitude=False):
    if vname in ['2m_air_temperature', 'total_precipitation',
                 '2m_relative_humidity', 'msl_pressure', '10m_wind_gust', '10m_wind_direct']:
        vname = 'data'
    if filter_type == 'temporal_cluster':
        df_out = clusterizer_filter(df_historic, vname, parameters)
    elif filter_type == 'cluster':
        df_out = clusterizer_filter(df, vname, parameters)
    elif filter_type == 'threshold':
        df_out = threshold_filter(df, vname, parameters)
    else:
        df_out = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for _, row in df.iterrows():
                if row.empty:
                    continue

                if filter_type in ['clear_sky', 'threshold', 'magic_number']:
                    future = executor.submit(
                        process_filters, *(row, False, array, vname, latitude, longitude, filter_type, parameters))
                else:
                    row_historic = df_historic[
                        df_historic['station_id'] == row['station_id']]

                    if row_historic.empty:
                        continue
                    if len(row_historic) < 10:
                        continue
                    future = executor.submit(
                        process_filters, *(row, row_historic, array, vname, latitude, longitude, filter_type, parameters))

                df_out.append(future)

            concurrent.futures.wait(df_out)
        df_out = pd.concat([pd.DataFrame(d.result()).T for d in df_out])
    return df_out


def main():
    config_log()
    args = get_args()

    df = pd.read_parquet(args.input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values("datetime")
    df.set_index("datetime", inplace=True)

    df_historic = df.loc[df.index != df.index[df.index.argmax()]].reset_index()
    df = df.loc[df.index == df.index[df.index.argmax()]].reset_index()
    uniq_providers = np.unique(df['station_provider'].values)

    for provider in OFICIAL_PROVIDERS:
        if provider in uniq_providers:
            log.info(f"Filtering {provider.upper()} neighbourhoods")
            npixels = 15 if args.vname not in ['total_precipitation'] else 7
            dis = args.res * 112 * npixels

            try:
                df = is_within_distance(df, provider, distance=dis * 1000,
                                        oficial_providers=OFICIAL_PROVIDERS, verbose=args.verbose)
            except Exception as e:
                log.warning(f"Radius filtering failed @ {provider.upper()} {e}")
                continue

    config = load_config(args.vname)

    if args.full_filter:
        for filter_type in config['filters']:
            log.info(f"Applying {filter_type} filter")
            if args.vname == '10m_wind':
                for wind_component in ['U', 'V']:
                    df_now_component = df.copy()
                    df_now_component['data'] = df[wind_component]
                    out_component = apply_filter(
                        df_now_component,
                        df_historic,
                        'data',
                        filter_type,
                        config[filter_type]
                    )
                    df[wind_component] = out_component['data']

                df = df.dropna(subset=['U', 'V'])
            else:
                if filter_type == 'clear_sky':
                    array_min_distance, latitude, longitude = get_array_skymask(
                        date=df['datetime'].max(), parameters=config[filter_type])

                    if latitude is None:
                        log.warning(f"filter.main >> could not apply clear sky filter")
                        continue
                    out = apply_filter(
                        df,
                        df_historic,
                        args.vname,
                        filter_type,
                        config[filter_type],
                        array=array_min_distance,
                        latitude=latitude,
                        longitude=longitude)
                else:
                    out = apply_filter(
                        df,
                        df_historic,
                        args.vname,
                        filter_type,
                        config[filter_type])

                df = out.loc[~out['data'].isna(), :]

    try:
        df['geometry'] = df['geometry'].apply(shapely.wkt.loads)

    except TypeError:
        df = df.drop('geometry', axis=1)

    df.to_parquet(args.out_path)

if __name__ == "__main__":
    main()
