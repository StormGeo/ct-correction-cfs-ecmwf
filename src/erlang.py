import logging as log

from datetime import timedelta

import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.stats import zscore
from shapely import wkt
from sklearn.cluster import DBSCAN

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


def RIQ_filter(df_input, df_historic, column_name, parameters):
    # construindo os parametros
    q1 = df_historic[column_name].quantile(parameters['lower'] / 100)
    q3 = df_historic[column_name].quantile(parameters['upper'] / 100)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    if df_input[column_name] > upper_bound or df_input[
            column_name] < lower_bound:
        df_input[column_name] = np.nan

    return df_input


def monotony_filter(df_input, df_historic, column_name, date, parameters):
    # get info last_hours
    if df_input[column_name] == parameters[
            'allowed_value']:  # valores permitidos, ex: 0 da chuva nao eh invalido, e sai da funcao
        return df_input

    # procura no df historico, periodo deve ser menor ou igual ao tempo puxado
    # do historico (deve ser em poucas horas)
    df_hours_late = df_historic[
        df_historic['datetime'] >= date -
        timedelta(
            hours=parameters['period_in_hours'])]

    # minimum data values | nao filtra com muito NaN na serie (ex: nas ultimas
    # 6 horas)
    if sum(df_hours_late[column_name].isna()) > parameters[
            'min_number_of_invalid_window'] or df_hours_late.empty is True:
        return df_input

    # count number o repeated values | traz o numero de repeticoes do horario
    # atual
    count_values = df_hours_late[column_name].value_counts()
    repeated_value = count_values[count_values.index == df_input[column_name]]

    if repeated_value.empty:  # nao ha valores repetido > sai da funcao
        return df_input

    repeated_value = repeated_value.values[0]
    if repeated_value >= parameters[
            'min_repeated_data']:  # correcao da frequencia minima
        df_input[column_name] = np.nan

    return df_input

# def filter_isolated(df_input, df_historic, column_name, date, parameters):
#     df_h = df_historic.sort_values(by=['datetime']).iloc[-2:, :]
#     df['prev'] = df['data'].shift(1)
#     df['next'] = df['data'].shift(-1)
#
#     filtered_df = df[~((df['data'] == magic_value) & (df['prev'] != magic_value) & (df['next'] != magic_value))]
#     filtered_df = filtered_df.drop(columns=['prev', 'next'])
#
#     return filtered_df


def monotony_filter_sequence(
        df_input, df_historic, column_name, date, parameters):

    # get info last_hours
    # semelhante ao monotony POREM essa funcao pega a sequencia repetida ( ex:
    # temperaturas constantes )
    df_hours_late = df_historic[
        df_historic['datetime'] >= date -
        timedelta(
            hours=parameters['period_in_hours'])]
    df_hours_late = df_hours_late.sort_values(by=['datetime'])

    # minimum data values
    if sum(df_hours_late[column_name].isna()) > parameters[
            'min_number_of_invalid_window'] or df_hours_late.empty:
        return df_input

    # count number o repeated and sequenced values
    # faz a diferença 1 a 1 para ver os deltas
    diff_values = np.abs(np.diff(df_hours_late[column_name], prepend=np.nan))

    min_repeated_data = parameters['min_repeated_data']
    # soma os ultimos valores da series, se o delta for 0 significa que
    # repetiu os dados, ou seja, sequencia monotona
    sum_last_values = np.nansum(diff_values[-min_repeated_data:])

    # se o ultimo valor da serie historica foi diferente do atual  remove
    # tambem
    if sum_last_values == 0:
        df_input[column_name] = np.nan

    return df_input


def aceleration_filter(df_input, df_historic, column_name, parameters):
    df_historic.sort_values(by=['datetime'])
    df_historic['acceleration'] = np.abs(
        np.diff(df_historic[column_name], prepend=np.nan))
    mean_accel = df_historic['acceleration'].mean()
    std_accel = df_historic['acceleration'].std()
    # cria uma coluna com a diferenca do registro atual com a serie historica
    df_historic['diff'] = abs(df_historic[column_name] - df_input[column_name])

    # faz um loop para ver se a diferenca eh maior que os 3 ultimos registros
    # do historico e se a diff eh maior que o zscore nos configs
    for last_idx in range(1, len(df_historic['diff']), 1):
        if np.isnan(df_historic['diff'].iloc[-last_idx]) is False and np.abs(
                (df_historic['diff'].iloc[-last_idx] - mean_accel)) > parameters['zscore'] * std_accel:
            df_input[column_name] = np.nan

    return df_input


def zscore_filter(df_input, df_historic, column_name, parameters):
    # zscore basicao
    mean_accel = df_historic[column_name].mean()
    std_accel = df_historic[column_name].std()
    if np.abs(df_input[column_name] -
              mean_accel) > parameters['threshold'] * std_accel:
        df_input[column_name] = np.nan
    return df_input


def clusterizer_filter(df_input, column_name, parameters):
    # Use DBSCAN to cluster data points
    df_input = df_input.loc[~df_input['data'].isna(), :]

    # Use DBSCAN to cluster data points
    dbscan = DBSCAN(
        eps=parameters['horizontal_dist_in_degree'],
        min_samples=parameters['n_samples'])  # monta o cluster

    # monta um dataframe com as info de lat, lon, alt, date e a coluna da
    # variavel
    if 'altitude' in df_input:
        df = pd.DataFrame({'latitude': df_input['latitude'],
                           'longitude': df_input['longitude'],
                           'altitude': df_input['altitude'],
                           'date': df_input['datetime'],
                           'station_id': df_input['station_id'],
                           'station_code': df_input['station_code'],
                           'station_owner': df_input['station_owner'],
                           'station_provider': df_input['station_provider'],
                           'public': df_input['public'],
                           'geometry': df_input['geometry'],
                           'city': df_input['city'],
                           'state': df_input['state'],
                           'country': df_input['country'],
                           column_name: df_input[column_name]})
    else:
        df = pd.DataFrame({'latitude': df_input['latitude'],
                           'longitude': df_input['longitude'],
                           'date': df_input['datetime'],
                           'date': df_input['datetime'],
                           'station_id': df_input['station_id'],
                           'station_code': df_input['station_code'],
                           'station_owner': df_input['station_owner'],
                           'station_provider': df_input['station_provider'],
                           'public': df_input['public'],
                           'geometry': df_input['geometry'],
                           'city': df_input['city'],
                           'state': df_input['state'],
                           'country': df_input['country'],
                           column_name: df_input[column_name]})
        # cria uma coluna fake da altitude
        df['altitude'] = np.nan

    if parameters["vertical_parameter"]["mode"] == "on":
        # faz a normalizacao da altura de acordo com o espeficicado no json,
        # nesse caso os valores de lat e lon serao n mesma escala de busca
        df['altitude_standart'] = df['altitude'] * \
            (parameters['horizontal_dist_in_degree'] /
             parameters['vertical_parameter']['value'])
        df['cluster'] = dbscan.fit_predict(
            df[['latitude', 'longitude', 'altitude_standart']].values)  # faz o fit da serie

    else:
        df['cluster'] = dbscan.fit_predict(
            df[['latitude', 'longitude']].values)  # faz o fit da serie

    # salva as estaocoes nao clusterizadas ( bad clusters )
    df_ignored = df[df['cluster'] == -1].copy(deep=True)
    df = df[df['cluster'] != -1]

    # Compute cluster statistics
    cluster_stats = df.groupby('cluster').agg(num_elements=(column_name, 'count'),
                                              values=(column_name, list),
                                              latitude=('latitude', list),
                                              longitude=('longitude', list),
                                              altitude=('altitude', list),
                                              date=('date', list),
                                              station_id=('station_id', list),
                                              station_code=(
                                                  'station_code', list),
                                              station_owner=(
                                                  'station_owner', list),
                                              station_provider=(
                                                  'station_provider', list),
                                              public=('public', list),
                                              geometry=('geometry', list),
                                              city=('city', list),
                                              state=('state', list),
                                              country=('country', list)
                                              )

    # Separe clusters with less than 3 elements
    # since is impossible to determine outiliers
    clusters_low_elements = cluster_stats[
        cluster_stats['num_elements'] < parameters['min_estations_per_cluster']].copy(
        deep=True)  # salva os clusters com poucas estacoes
    cluster_stats = cluster_stats[cluster_stats['num_elements'] >= parameters[
        'min_estations_per_cluster']]  # filtras clusters com poucas estacoes

    # Remove outliers from each cluster based on z-score
    for idx, row in cluster_stats.iterrows():
        values = row['values']
        latitude = row['latitude']
        longitude = row['longitude']
        altitude = row['altitude']
        date = row['date']
        station_id = row['station_id']
        station_code = row['station_code']
        station_owner = row['station_owner']
        station_provider = row['station_provider']
        public = row['public']
        geometry = row['geometry']
        city = row['city']
        state = row['state']
        country = row['country']

        if row['station_provider'] in ['inmet', 'redemet', 'ogimet']:
            mask = slice(None, None)

        else:
            if parameters['RIQ_parameter']['mode'].lower() == 'on':
                riq_input = pd.DataFrame(data={'values': values})
                before = []
                for _, value in riq_input.iterrows():
                    out = RIQ_filter(df_input=value,
                                     df_historic=riq_input,
                                     column_name='values',
                                     parameters=parameters['RIQ_parameter'])
                    before.append(out['values'])
                mask = np.where(~np.isnan(before))
            else:
                z_scores = zscore(values)
                mask = np.abs(z_scores) <= parameters['zscore_soft']

        latitude = np.array(latitude)[mask]
        longitude = np.array(longitude)[mask]
        values = np.array(values)[mask]
        altitude = np.array(altitude)[mask]
        date = np.array(date)[mask]
        station_id = np.array(station_id)[mask]
        station_code = np.array(station_code)[mask]
        station_owner = np.array(station_owner)[mask]
        station_provider = np.array(station_provider)[mask]
        public = np.array(public)[mask]
        geometry = np.array(geometry)[mask]
        city = np.array(city)[mask]
        state = np.array(state)[mask]
        country = np.array(country)[mask]

        cluster_stats.at[idx, 'altitude'] = altitude
        cluster_stats.at[idx, 'date'] = date
        cluster_stats.at[idx, 'values'] = values
        cluster_stats.at[idx, 'latitude'] = latitude
        cluster_stats.at[idx, 'longitude'] = longitude
        cluster_stats.at[idx, 'num_elements'] = len(values)
        cluster_stats.at[idx, 'station_id'] = station_id
        cluster_stats.at[idx, 'station_code'] = station_code
        cluster_stats.at[idx, 'station_owner'] = station_owner
        cluster_stats.at[idx, 'station_provider'] = station_provider
        cluster_stats.at[idx, 'public'] = public
        cluster_stats.at[idx, 'geometry'] = geometry
        cluster_stats.at[idx, 'city'] = city
        cluster_stats.at[idx, 'state'] = state
        cluster_stats.at[idx, 'country'] = country

        if row['station_provider'] in ['inmet', 'redemet', 'ogimet']:
            mask = slice(None, None)

        else:
            if parameters['RIQ_parameter']['mode'].lower() == 'on':
                riq_input = pd.DataFrame(data={'values': values})
                before = []
                for _, value in riq_input.iterrows():
                    out = RIQ_filter(df_input=value,
                                     df_historic=riq_input,
                                     column_name='values',
                                     parameters=parameters['RIQ_parameter'])
                    before.append(out['values'])
                mask = np.where(~np.isnan(before))
            else:
                z_scores = zscore(values)
                mask = np.abs(z_scores) <= parameters['zscore_hard']

        latitude = np.array(latitude)[mask]
        longitude = np.array(longitude)[mask]
        values = np.array(values)[mask]
        altitude = np.array(altitude)[mask]
        date = np.array(date)[mask]
        station_id = np.array(station_id)[mask]
        station_code = np.array(station_code)[mask]
        station_owner = np.array(station_owner)[mask]
        station_provider = np.array(station_provider)[mask]
        public = np.array(public)[mask]
        geometry = np.array(geometry)[mask]
        city = np.array(city)[mask]
        state = np.array(state)[mask]
        country = np.array(country)[mask]

        cluster_stats.at[idx, 'altitude'] = altitude
        cluster_stats.at[idx, 'date'] = date
        cluster_stats.at[idx, 'values'] = values
        cluster_stats.at[idx, 'latitude'] = latitude
        cluster_stats.at[idx, 'longitude'] = longitude
        cluster_stats.at[idx, 'num_elements'] = len(values)
        cluster_stats.at[idx, 'station_id'] = station_id
        cluster_stats.at[idx, 'station_code'] = station_code
        cluster_stats.at[idx, 'station_owner'] = station_owner
        cluster_stats.at[idx, 'station_provider'] = station_provider
        cluster_stats.at[idx, 'public'] = public
        cluster_stats.at[idx, 'geometry'] = geometry
        cluster_stats.at[idx, 'city'] = city
        cluster_stats.at[idx, 'state'] = state
        cluster_stats.at[idx, 'country'] = country

    cluster_stats = cluster_stats[cluster_stats['num_elements'] > 0]
    # concatena os clusters com poucas  estacoes
    cluster_stats = pd.concat([cluster_stats, clusters_low_elements], axis=0)

    list_latitude = []
    list_longitude = []
    list_altitude = []
    list_date = []
    list_values = []
    list_station_id = []
    list_station_code = []
    list_station_owner = []
    list_station_provider = []
    list_public = []
    list_geometry = []
    list_city = []
    list_state = []
    list_country = []

    for idx, row in cluster_stats.iterrows(
    ):  # retorna os clusters em dataframe novamete para a saida
        list_latitude.extend(list(row['latitude']))
        list_longitude.extend(list(row['longitude']))
        list_altitude.extend(list(row['altitude']))
        list_date.extend(list(row['date']))
        list_values.extend(list(row['values']))
        list_station_id.extend(list(row['station_id']))
        list_station_code.extend(list(row['station_code']))
        list_station_owner.extend(list(row['station_owner']))
        list_station_provider.extend(list(row['station_provider']))
        list_public.extend(list(row['public']))
        list_geometry.extend(list(row['geometry']))
        list_city.extend(list(row['city']))
        list_state.extend(list(row['state']))
        list_country.extend(list(row['country']))

    for idx, row in clusters_low_elements.iterrows(
    ):  # retorna os clusters em dataframe novamete para a saida
        list_latitude.extend(list(row['latitude']))
        list_longitude.extend(list(row['longitude']))
        list_altitude.extend(list(row['altitude']))
        list_date.extend(list(row['date']))
        list_values.extend(list(row['values']))
        list_station_id.extend(list(row['station_id']))
        list_station_code.extend(list(row['station_code']))
        list_station_owner.extend(list(row['station_owner']))
        list_station_provider.extend(list(row['station_provider']))
        list_public.extend(list(row['public']))
        list_geometry.extend(list(row['geometry']))
        list_city.extend(list(row['city']))
        list_state.extend(list(row['state']))
        list_country.extend(list(row['country']))

    for idx, row in df_ignored.iterrows():  # salfa as esacoes com bad cluster
        list_latitude.append(row['latitude'])
        list_longitude.append(row['longitude'])
        list_altitude.append(row['altitude'])
        list_date.append(row['date'])
        list_values.append(row[column_name])
        list_station_id.append(row['station_id'])
        list_station_code.append(row['station_code'])
        list_station_owner.append(row['station_owner'])
        list_station_provider.append(row['station_provider'])
        list_public.append(row['public'])
        list_geometry.append(row['geometry'])
        list_city.append(row['city'])
        list_state.append(row['state'])
        list_country.append(row['country'])

    # dataframe de saida
    df_out = pd.DataFrame({'latitude': list_latitude,
                           'longitude': list_longitude,
                           'altitude': list_altitude,
                           'datetime': list_date,
                           column_name: list_values,
                           'station_id': list_station_id,
                           'station_code': list_station_code,
                           'station_owner': list_station_owner,
                           'station_provider': list_station_provider,
                           'public': list_public,
                           'geometry': list_geometry,
                           'city': list_city,
                           'state': list_state,
                           'country': list_country,
                           })

    return df_out


def threshold_filter(df_input, column_name, parameters):
    # Apply upper threshold
    if parameters['value_upper'] != 'NaN':
        df_input[column_name] = np.where(
            df_input[column_name] > parameters['upper'],
            parameters['value_upper'],
            df_input[column_name]
        )
    else:
        df_input[column_name] = np.where(
            df_input[column_name] > parameters['upper'],
            np.nan,
            df_input[column_name]
        )

    # Apply lower threshold
    if parameters['value_lower'] != 'NaN':
        df_input[column_name] = np.where(
            df_input[column_name] < parameters['lower'],
            parameters['value_lower'],
            df_input[column_name]
        )
    else:
        df_input[column_name] = np.where(
            df_input[column_name] < parameters['lower'],
            np.nan,
            df_input[column_name]
        )

    return df_input


def accumulated_filter(df_input, df_historic, column_name, parameters):
    df_historic = df_historic.sort_values(by=['datetime'])

    # faz o diff linha a linha do historico com valor anterior | prepend para
    # ter nan no index 0
    diff_column = np.diff(df_historic[column_name], prepend=np.nan)
    # pega valores diferente de 0 para nao considerar o 0 da chuva
    diff_column = diff_column[df_historic[column_name] != 0]
    # ve a quantidade de valores positivos
    len_diff = len(diff_column[diff_column >= 0])

    # filtra caso encontre a quantidade minima de valores considerados
    # acumulados
    if len_diff >= (parameters['perc_true_condiction'] /
                    100) * len(df_historic[column_name]):
        df_input[column_name] = np.nan

    return df_input


def clear_sky_filter(df_input, clear_sky_array, latitude,
                     longitude, column_name, parameters, verbose=False):

    # traz o ponto do mapa onde a estacao esta localizada
    lat_row, lon_row = df_input['latitude'], df_input['longitude']
    lat_index = (abs(latitude - lat_row)).argmin()
    lon_index = (abs(longitude - lon_row)).argmin()
    value_at_point = clear_sky_array[lat_index, lon_index]

    count_droped = 0
    droped_vals = []
    # se o ponto em todos os campos onde foi acumulado foi maior que 0, ou
    # seja, sempre foi longe de uma nuvem
    if value_at_point > 0 and df_input[
            column_name] > parameters['threhold_precip_min']:

        if verbose:
            log.info('src.erlang.clear_sky_filter >> data filtered by clear_sky_mask')
            log.info(df_input[['station_owner', 'station_provider',
                               'datetime', 'longitude', 'latitude', 'data']])
            log.info(f'src.erlang.clear_sky_filter >> '
                     f'closest point (in degree °) from any cloud mask >> {value_at_point}')
        droped_vals.append(df_input[column_name])
        df_input[column_name] = np.nan
        count_droped += 1

    if len(droped_vals) > 0:
        max_dropped, min_dropped = np.nanmax(droped_vals), np.nanmin(droped_vals)
        log.info(f"src.erlang.clear_sky_filter >> done, "
                 f"{count_droped} droped with max/min: {max_dropped}/{min_dropped}")
    else:
        log.info(f"src.erlang.clear_sky_filter >> done, no stations to drop")

    return df_input


def is_within_distance(df, provider, distance=20 * 1000, oficial_providers=[], verbose=False):

    distance /= (112 * 1000)

    try:
        df['geometry'] = df['geometry'].apply(wkt.loads)

    except TypeError:  # it can happen if geometry column is already a Point type
        pass

    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    tgt_gdf = gdf[gdf['station_provider'] == provider]
    other_gdf = gdf[gdf['station_provider'] != provider]

    filter_gdf = []
    count = 0

    def process_row(idx, row):
        distances = tgt_gdf.geometry.distance(row.geometry)

        # print((np.nanmin(distances), np.nanmax(distances), distance))

        if np.nanmin(distances) >= distance:
            return row, True
        else:
            if row['station_provider'] in oficial_providers:
                if verbose:
                    log.info(f'Station {row["station_code"]} from {row["station_provider"]} is within {distance} degrees '
                             f'from {provider} station, keeping')

                return row, True
        return row, False

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_row,
                idx,
                row): idx for idx,
            row in other_gdf.iterrows()}
        for future in as_completed(futures):
            row, keep = future.result()
            if keep:
                filter_gdf.append(row)
            else:
                count += 1

    log.info(f'src.erlang.is_within_distance >> number of stations within {distance*112} km from {provider} >> {count}, dropped')

    filter_gdf = pd.DataFrame(filter_gdf)
    if not filter_gdf.empty:
        filter_gdf = gpd.GeoDataFrame(filter_gdf, geometry='geometry')
        filter_gdf = pd.concat([filter_gdf, tgt_gdf])
        return filter_gdf
    else:
        return df
