#!/airflow-dev/models/ct-observed/venv/bin/python3
# -----------------------------------------------------------------------------
# Geração de total_precipitation horário a partir de multisensor_rainfall_rate
#
# Este script:
#   1. Recebe uma data/hora alvo via argumento -d no formato YYYYMMDDHH
#      (ex: 2025111800 -> 2025-11-18 00 UTC).
#   2. Lê 6 arquivos de taxa de precipitação (multisensor_rainfall_rate)
#      em passos de 10 minutos da HORA ANTERIOR ao horário alvo
#      (ex: 23:00, 23:10, ..., 23:50).
#   3. Cada arquivo é lido no diretório:
#         /data/observed/mergestorm_as/multisensor_rainfall_rate/YYYY/DDD/
#      onde DDD é o dia juliano do horário de cada passo.
#   4. Converte a taxa (mm/h) para acumulado de 10 min, soma para obter o
#      acumulado horário (mm) e interpola para uma grade fixa de 0.125°.
#   5. Salva o resultado na variável total_precipitation no arquivo:
#         /data/observed/mergestorm_as/total_precipitation/YYYY/DDD/
#         mergestorm_as_total_precipitation_YYYYMMDDHH.nc
#      onde DDD é o dia juliano do horário alvo.
#
# Exemplo de uso:
#   python script.py -d 2025111800
# -----------------------------------------------------------------------------

import argparse
import os
from datetime import datetime, timedelta

import xarray as xr
import numpy as np

# Nome da variável de entrada e de saída
VAR_IN = "multisensor_rainfall_rate"
VAR_OUT = "total_precipitation"

# Caminhos base
BASE_IN = "/data/observed/mergestorm_as/multisensor_rainfall_rate"
BASE_OUT = "/data/observed/mergestorm_as/total_precipitation"

# Grade fixa (0.125°)
LAT_MIN, LAT_MAX, DLAT = -53.0, 22.875, 0.125
LON_MIN, LON_MAX, DLON = -88.7, -30.45, 0.125


def get_year_and_julian(dt: datetime):
    """Retorna ano e dia juliano (003, 120, etc.)."""
    year = dt.strftime("%Y")
    julian = f"{dt.timetuple().tm_yday:03d}"
    return year, julian


def build_file_list(target_dt: datetime):
    """
    Para um horário alvo HH, monta os 6 arquivos de 10 em 10 min da hora anterior:
    HH-60, HH-50, ..., HH-10.

    O diretório é baseado no dia juliano de CADA tempo t.
    Exemplo: -d 2025111800 -> arquivos em 2025/321 (dia anterior).
    """
    times = [target_dt - timedelta(minutes=m) for m in range(60, 0, -10)]
    files = []

    for t in times:
        year, julian = get_year_and_julian(t)
        in_dir = os.path.join(BASE_IN, year, julian)
        fname = f"mergestorm_as_multisensor_rainfall_rate_{t.strftime('%Y%m%d%H%M')}.nc"
        fpath = os.path.join(in_dir, fname)
        files.append(fpath)

    return files


def compute_accum(target_dt: datetime) -> xr.Dataset:
    """
    Calcula o acumulado horário (mm) entre target_dt-1h e target_dt
    e interpola para a grade fixa 0.125°.
    """
    files = build_file_list(target_dt)

    accum = None
    first_ds = None

    for path in files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        ds = xr.open_dataset(path)
        if VAR_IN not in ds:
            raise ValueError(f"Variável '{VAR_IN}' não encontrada em {path}")

        # Assume taxa em mm/h num único tempo
        var = ds[VAR_IN].isel(time=0)

        # Converte 10 min de taxa mm/h -> mm
        step_mm = var * (10.0 / 60.0)

        accum = step_mm if accum is None else accum + step_mm

        if first_ds is None:
            first_ds = ds
        else:
            ds.close()

    # Tempo de referência da rodada (fim do acumulado)
    reftime = target_dt.replace(tzinfo=None)

    # Cria eixo de tempo em horas desde a rodada (1 passo: 0 h)
    time_vals = np.array([0.0], dtype="float64")

    # Adiciona dimensão tempo (numérica)
    accum = accum.expand_dims("time")
    accum = accum.assign_coords(time=("time", time_vals))

    # Renomeia variável para a de saída
    accum.name = VAR_OUT

    # Copia atributos da variável original (se tiver)
    if first_ds is not None:
        accum.attrs.update(first_ds[VAR_IN].attrs)

    # Ajusta atributos da variável acumulada
    start_dt = target_dt - timedelta(hours=1)
    accum.attrs["units"] = "mm"
    accum.attrs["long_name"] = (
        f"Total precipitation between "
        f"{start_dt.strftime('%Y-%m-%d %H:%M')} and "
        f"{target_dt.strftime('%Y-%m-%d %H:%M')} UTC"
    )

    # === Interpolação para grade fixa 0.125° ===
    lat_reg = np.arange(LAT_MIN, LAT_MAX + DLAT / 2.0, DLAT)
    lon_reg = np.arange(LON_MIN, LON_MAX + DLON / 2.0, DLON)

    accum_interp = accum.interp(latitude=lat_reg, longitude=lon_reg)
    accum_interp.name = VAR_OUT
    accum_interp.attrs.update(accum.attrs)

    # Dataset final com dims (time, latitude, longitude)
    out_ds = xr.Dataset(
        {VAR_OUT: (("time", "latitude", "longitude"), accum_interp.data)},
        coords={
            "time": ("time", time_vals),
            "latitude": ("latitude", lat_reg),
            "longitude": ("longitude", lon_reg),
        },
    )

    # Atributos da coordenada tempo no estilo "hours since RefTime"
    units_str = f"hours since {reftime:%Y-%m-%d %H:%M:%S}"
    out_ds["time"].attrs.update(
        {
            "standard_name": "time",
            "long_name": "Time coordinate",
            "units": units_str,
            "calendar": "standard",
        }
    )
    out_ds["time"].encoding.update(
        {
            "dtype": "float64",
            "units": units_str,
            "calendar": "standard",
        }
    )

    # Atributos globais
    if first_ds is not None:
        out_ds.attrs.update(first_ds.attrs)
        first_ds.close()

    # Guarda também o RefTime como atributo global
    out_ds.attrs["RefTime"] = reftime.strftime("%Y-%m-%d %H:%M:%S")

    return out_ds


def main():
    parser = argparse.ArgumentParser(
        description="Gera total_precipitation horário a partir de multisensor_rainfall_rate."
    )
    parser.add_argument(
        "-d",
        "--date",
        required=True,
        help="Data alvo no formato YYYYMMDDHH (ex: 2025110308).",
    )
    args = parser.parse_args()

    # Exemplo: 2025110308 -> 2025-11-03 08:00 UTC
    target_dt = datetime.strptime(args.date, "%Y%m%d%H")

    # Ano e DJ do horário alvo (para saída)
    year_out, julian_out = get_year_and_julian(target_dt)

    # Diretório de saída:
    # /data/observed/mergestorm_as/total_precipitation/YYYY/DDD/
    out_dir = os.path.join(BASE_OUT, year_out, julian_out)
    os.makedirs(out_dir, exist_ok=True)

    # Nome do arquivo de saída:
    # mergestorm_as_total_precipitation_YYYYMMDDHH.nc
    outfile = os.path.join(
        out_dir, f"mergestorm_as_total_precipitation_{args.date}.nc"
    )

    out_ds = compute_accum(target_dt)

    comp = {VAR_OUT: {"zlib": True, "complevel": 4}}
    out_ds.to_netcdf(outfile, format="NETCDF4", encoding=comp)

    print(f"✔ Acumulado salvo em grade fixa 0.125°: {outfile}")


if __name__ == "__main__":
    main()
