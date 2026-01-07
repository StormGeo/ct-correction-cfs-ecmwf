#!/airflow-dev/models/ct-observed/venv/bin/python3
import argparse
import multiprocessing
import os
import yaml
from datetime import datetime

import logging as log

import numpy as np
import pandas as pd
import xarray as xr

from distutils.util import strtobool

from src.commons import config_log
from src.datahub import get_array_skymask
from src.datahub import write_df_bq
from src.datahub import write_df_csv
from src.interpol_methods import interpol_data, interpol_data_try_except

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))) + '/ct-near'

INTERPOL_CHOICES = [
            'cressman',
            'barnes',
            'idw',
            'rbf',
            'spline',
            'linear',
            # 'bilinear',
            'nearest',
            'cubic',
            'tps',
            'kriging',
            'ukriging',
            'circles',
            'metpy',
            'samet',
            'merge']

def get_args():
    parser = argparse.ArgumentParser(
        description='Interpolation and BigQuery Insertion')
    parser.add_argument(
        '-i', '--input',
        dest='input_path',
        required=True,
        help='Input Parquet file path',
        action='store')
    parser.add_argument(
        '-o', '--output',
        dest='out_path',
        required=True,
        help='Output file path',
        action='store')
    parser.add_argument(
        '-v', '--variable',
        dest='vname',
        required=True,
        help='Variable to be processed',
        action='store')
    parser.add_argument(
        '-r', '--resolution',
        dest='res',
        required=False,
        help='domain Resolution',
        default=0.125,
        action='store',
    )
    parser.add_argument(
        '-a', '--area',
        dest='domain',
        required=False,
        help='domain Area [lat0, lat1, lon0, lon1]',
        default=[-53.0, 22.0, -88.7, -31.4],
        nargs=4,
        action='store',
    )
    parser.add_argument(
        '-m', '--merge',
        dest='merge_path',
        required=False,
        help='netCDF4 file to be merged',
        default=None,
        action='store',
    )
    parser.add_argument(
        '-d', '--datetime',
        dest='datetime_str',
        required=False,
        help='Reference datetime (YYYYMMDDHH) used to select model time step',
        action='store',
        type=str,
    )
    parser.add_argument(
        '-fc', '--force-clear-sky',
        dest="force_clear_sky",
        default=False,
        choices=[True, False],
        action='store',
        type=lambda x: bool(strtobool(x)),
        help='Use to raise error when clear sky files'
    )
    parser.add_argument(
        '-ut', '--use-topography',
        dest='use_topo',
        default=True,
        choices=[True, False],
        action='store',
        type=lambda x: bool(strtobool(x)),
        help='Use this to set if topography will be available or not'
    )
    parser.add_argument(
        '-b',
        dest='will_update_bq',
        required=False,
        type=lambda x: bool(strtobool(x)),
        help='Enable or disable BigQuery insertion',
        default=True, choices=[True, False], action='store')
    parser.add_argument(
        '-im', '--interpolation-method',
        dest="interpol_method",
        default='cressman',
        choices=INTERPOL_CHOICES,
        required=False,
        action='store',
        type=str,
        help='Set the interpolation method'
    )
    parser.add_argument(
        '-ir', '--interpolation-radius',
        dest='interpolation_radius',
        default=500,
        required=False,
        type=float,
        help='Interpolation radius in km')

    parser.add_argument(
        '-sc', '--spread-correction',
        dest="spread_correction",
        default=True,
        choices=[True, False],
        action='store',
        type=lambda x: bool(strtobool(x)),
        help='Use to raise error when clear sky files'
    )

    parser.add_argument("-mf", "--method-function",
                        dest='method_function',
                        default=None,
                        action='store',
                        type=str,
                        help='Define which method choose according with interpolation method options')

    parser.add_argument("-rp", '--reinsert-point',
                        dest="reinsert_point",
                        default=True,
                        choices=[True, False],
                        action='store',
                        type=lambda x: bool(strtobool(x)),
                        help='Use to reinsert station points on interpolated grid')

    parser.add_argument("-cm", '--composed-method',
                        dest='composed_method',
                        default=False,
                        choices=[True, False],
                        action='store',
                        type=lambda x: bool(strtobool(x)),
                        help="Set two interpolation methods based on altitude threshold, can be more expensive")

    parser.add_argument("-im1", '--interpolation-method1',
                        dest="interpol_method1",
                        default='cressman',
                        choices=INTERPOL_CHOICES,
                        required=False,
                        action='store',
                        type=str,
                        help='Set the interpolation method lower equal altitude threshold  in composed interpolation'
                        )
    parser.add_argument("-im2", '--interpolation-method2',
                        dest="interpol_method2",
                        default='cressman',
                        choices=INTERPOL_CHOICES,
                        required=False,
                        action='store',
                        type=str,
                        help='Set the interpolation method higher altitude threshold  in composed interpolation'
                        )

    parser.add_argument("-at", "--altitude-threshold",
                        dest='altitude_threshold',
                        default=500,
                        action='store',
                        type=float,
                        help='Define altitude threshold to composed interpolation'
                        )

    parser.add_argument("-mf1", "--method-function1",
                        dest='method_function1',
                        default=None,
                        action='store',
                        type=str,
                        help='Define which method use lower equal altitude threshold in composed interpolation')
    parser.add_argument("-mf2", "--method-function2",
                        dest='method_function2',
                        default=None,
                        action='store',
                        type=str,
                        help='Define which method use higher altitude threshold in composed interpolation')

    return parser.parse_args()


def load_config(vname):
    path = f'{BASE_DIR}/configs/{vname}.yml'
    with open(path, 'r') as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise e
        return data


def main():
    config_log()
    args = get_args()
    df = pd.read_parquet(args.input_path)

    # Writing data to BigQuery and CSV
    if args.will_update_bq:
        p1 = multiprocessing.Process(target=write_df_bq, args=(df, args.vname))
        p1.start()
        p1.join()

    # Interpolation process
    log.info("interpol.main >> starting interpolation process")
    p3s, p2s = [], []
    mask_ds = None
    its_a_wind_process = False
    if args.vname == '10m_wind':

        _out_path = args.out_path
        for wind_component in ['U', 'V']:
            out_path = _out_path.replace('10m_wind', f"10m_{wind_component.lower()}_component_of_wind")
            args.out_path = out_path

            df_component = df.copy()
            df_component['data'] = df[wind_component]

            args.vname = f"10m_{wind_component.lower()}_component_of_wind"

            p3 = multiprocessing.Process(
                target=interpol_data_try_except, args=(
                    df_component, args))
            p3.start()
            p3s.append(p3)

            p2 = multiprocessing.Process(
                target=write_df_csv, args=(
                    df_component, out_path.replace(
                        'nc', 'csv')))

            p2.start()
            p2s.append(p2)

            log.info(f"interpol.main >> netcdf created @ {args.out_path}")
            its_a_wind_process = True

    elif args.vname == 'total_precipitation':
        if args.interpol_method == 'merge':
            log.info("interpol.main >> skipping clear sky mask for merge interpolation")
            p3 = multiprocessing.Process(target=interpol_data_try_except,
                                         args=(df, args))
            p3.start()
            p3s.append(p3)
        else:
            dtNow = pd.to_datetime(df['datetime'].values[0]).to_pydatetime()
            config = load_config(args.vname)

            array_min_distance, latitude, longitude = get_array_skymask(date=dtNow,
                                                                        parameters=config[
                                                                            'clear_sky'],
                                                                        force_clear_sky=args.force_clear_sky)

            if array_min_distance is None or latitude is None or longitude is None:
                if args.force_clear_sky:
                    log.error("interpol.main >> clear sky mask not found")
                    raise FileNotFoundError("Clear sky mask not found")
                else:
                    p3 = multiprocessing.Process(target=interpol_data_try_except,
                                                 args=(df, args))
                    p3.start()
                    p3s.append(p3)

            else:
                array_categorical = np.where(
                    array_min_distance > config['clear_sky'].get(
                        "buffer_distance_in_degree", .5), 0, 1)

                mask_ds = xr.Dataset(data_vars={'clear_sky_mask': (['latitude', 'longitude'], array_categorical)},
                                     coords={'latitude': (['latitude'], latitude),
                                             'longitude': (['longitude'], longitude)})

                p3 = multiprocessing.Process(target=interpol_data_try_except,
                                             args=(df, args),
                                             kwargs={"clear_sky_mask": mask_ds, 'use_topo': args.use_topo})
                p3.start()
                p3s.append(p3)

        p2 = multiprocessing.Process(
            target=write_df_csv, args=(
                df, args.out_path.replace(
                    'nc', 'csv')))

        p2.start()
        p2s.append(p2)

    else:
        p3 = multiprocessing.Process(target=interpol_data_try_except, args=(df, args))
        p3.start()
        p3s.append(p3)
        p2 = multiprocessing.Process(
            target=write_df_csv, args=(
                df, args.out_path.replace(
                    'nc', 'csv')))

        p2.start()
        p2s.append(p2)

    for p2 in p2s:
        p2.join()
        if p2.exitcode != 0:
            raise Exception(f"error in write_df_csv")
    for p3 in p3s:
        p3.join()

        if p3.exitcode != 0:

            log.warning(f"interpol.main >> something went wrong during interpolation @"
                        f" using metpy.cressman to ensure the result")

            # try ensure interpolation
            args.composed_method = False
            args.interpol_method = 'cressman'
            # args.method_function = 'cressman'

            if its_a_wind_process:
                args.vname = '10m_wind'

            interpol_data(df, args, clear_sky_mask=mask_ds)

    if not os.path.exists(args.out_path):
        for pp in p2s + p3s:
            pp.kill()  # try avoid zombie processes

        raise FileNotFoundError(f"expected output file was not created @ {args.out_path}")

    if not its_a_wind_process:
        log.info(f"interpol.main >> netcdf created @ {args.out_path}")


if __name__ == "__main__":
    start = datetime.now()
    log.info("interpol >> starting process ")
    main()
    end = datetime.now()
    log.info(f"interpol >> process done @ {(end-start).total_seconds()} seconds")
