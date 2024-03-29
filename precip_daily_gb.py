"""
Created by Mike Kittridge on 2021-10-01.

"""
import os
import numpy as np
import pandas as pd
from scipy import log, exp, mean, stats, special
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, RandomForestClassifier
import xarray as xr
from tethysts import Tethys, utils
import yaml
from shapely import wkb
from shapely.geometry import mapping, Point
import geopandas as gpd
import copy
from hdf5tools import File, utils, core
import booklet
from distfit import distfit

pd.options.display.max_columns = 10

#####################################
### Parameters

base_path = os.path.realpath(os.path.dirname(__file__))

with open(os.path.join(base_path, 'parameters.yml')) as param1:
    param = yaml.safe_load(param1)

source = param['source']
datasets = source['datasets'].copy()
public_url = source['public_url']
processing_code = source['processing_code']
local_tz = 'Etc/GMT-12'
version = source['version']
run_date = source['run_date']

# with open(os.path.join(base_path, 'remote.yml')) as param2:
#     remote = yaml.safe_load(param2)

# s3_remote = remote['remote']['s3'].copy()

stn_precip_ts_path = '/media/data01/data/pmp/stn_precip/tethys_stn_precip.blt'
stn_data_ts_path = '/media/data01/data/pmp/stn_precip/tethys_stn_data.csv'
precip_datasets_path = '/media/data01/data/pmp/stn_precip/tethys_precip_datasets.csv'

min_year_range = 10

islands_gpkg = 'islands.gpkg'

stn_attrs = {'NAE': {'long_name': 'normalised absolute error', 'description': 'The absolute error normalised to the sample mean.'},
             'bias': {'long_name': 'bias', 'description': 'The bias is the error normalised to the sample mean.'}}

dataset_params_export = ['dataset_id', 'feature', 'parameter', 'method', 'product_code', 'owner', 'aggregation_statistic', 'frequency_interval', 'units',  'utc_offset', 'license', 'attribution', 'precision']

# stn_encoding = {'NAE': }

#####################################
### Functions


# def create_shifted_df(series, from_range, to_range, freq_code, agg_fun, ref_name, include_0=False, discrete=False, **kwargs):
#     """

#     """
#     if not isinstance(series, pd.Series):
#         raise TypeError('series must be a pandas Series.')
#     if not isinstance(series.index, pd.DatetimeIndex):
#         raise TypeError('The series index must be a pandas DatetimeIndex.')

#     df = series.reset_index()
#     data_col = df.columns[1]
#     ts_col = df.columns[0]
#     s2 = tu.grp_ts_agg(df, None, ts_col, freq_code, agg_fun, discrete, **kwargs)[data_col]

#     if include_0:
#         f_hours = list(range(from_range-1, to_range+1))
#         f_hours[0] = 0
#     else:
#         f_hours = list(range(from_range, to_range+1))

#     df_list = []
#     for d in f_hours:
#         n1 = s2.shift(d, 'H')
#         n1.name = ref_name + '_' + str(d)
#         df_list.append(n1)
#     data = pd.concat(df_list, axis=1).dropna()

#     return data


####################################
### Get data

islands = gpd.read_file(os.path.join(base_path, islands_gpkg))

# TODO: Remove filter below to add in the north island
# islands = islands[islands.island == 'south'].copy()

## Datasets
tethys1 = Tethys()

all_datasets = tethys1.datasets.copy()

p_datasets1 = [ds for ds in all_datasets if (ds['feature'] == 'atmosphere') and (ds['parameter'] == 'precipitation') and (ds['product_code'] == 'quality_controlled_data') and (ds['frequency_interval'] in ('H', '1H'))]

p_datasets2 = [ds for ds in all_datasets if (ds['feature'] == 'atmosphere') and (ds['parameter'] == 'precipitation') and (ds['product_code'] == 'raw_data') and (ds['frequency_interval'] == '1H') and (ds['owner'] == 'FENZ')]

p_datasets3 = [ds for ds in all_datasets if (ds['feature'] == 'atmosphere') and (ds['parameter'] == 'precipitation') and (ds['product_code'] == 'quality_controlled_data') and (ds['frequency_interval'] == '24H') and (ds['owner'] == 'NIWA')]

p_datasets = p_datasets1 + p_datasets2 + p_datasets3

era5_dataset = [ds for ds in all_datasets if (ds['feature'] == 'atmosphere') and (ds['parameter'] == 'precipitation') and (ds['product_code'] == 'reanalysis-era5-land') and (ds['frequency_interval'] == 'H')][0]

p_datasets_dict = {ds['dataset_id']: ds for ds in p_datasets}

## Stations
p_stns = []

for d in p_datasets:
    for island in islands.island:
        poly = islands[islands.island == island].geometry.iloc[0]
        poly_geo = mapping(poly)

        p_stns1 = tethys1.get_stations(d['dataset_id'], geometry=poly_geo)
        [s.update({'island': island}) for s in p_stns1]
        p_stns.extend(p_stns1)

era5_stns = []
for island in islands.island:
    poly = islands[islands.island == island].geometry.iloc[0]
    poly_geo = mapping(poly)

    p_stns1 = tethys1.get_stations(era5_dataset['dataset_id'], geometry=poly_geo)
    [s.update({'island': island}) for s in p_stns1]
    era5_stns.extend(p_stns1)

era5_stn_ids = [s['station_id'] for s in era5_stns]

## Filters

count_limits = {'H': min_year_range*364*24,
                '24H': min_year_range*364
                }

# Date range and duplicate filter
stns2 = {}
for stn in copy.deepcopy(p_stns):
    stn_id = stn['station_id']
    ds = p_datasets_dict[stn['dataset_id']]
    freq = ds['frequency_interval']
    if freq == '1H':
        freq = 'H'

    count_limit = count_limits[freq]

    if 'content_length' in stn:
        count = stn['content_length']
    elif 'dimensions' in stn:
        count = stn['dimensions']['time']

    if count > count_limit:
        stn['count'] = count

        if stn_id in stns2:
            if count > stns2[stn_id]['count']:
                stns2[stn_id] = stn
        else:
            stns2[stn_id] = stn


################################################
### Save data

# h5_file = File(stn_precip_ts_path, mode='w')

stn_ids = np.array(list(stns2.keys()))

# stn_id_coord = h5_file.create_coordinate('station_id', stn_ids)

# stn_index_coord = h5_file.create_coordinate('station_index', shape=(1,), dtype='int8', maxshape=(None,), chunks=(100000, ))

# stn_index_coord = h5_file.create_coordinate('ignore_index', np.array([0], dtype='int8'), maxshape=(None,), chunks=(100000, ))

# stn_id_ds = h5_file.create_data_variable('station_id', ('ignore_index', ), np.array([stn_ids[0]]), maxshape=(None,), chunks=(500000, ))
# time_ds = h5_file.create_data_variable('time', ('ignore_index', ), np.array(['2000-01-01'], dtype='datetime64[s]'), maxshape=(None,), chunks=(500000, ))
# precip_ds = h5_file.create_data_variable('precipitation', ('ignore_index', ), np.array([0], dtype='float32'), dtype='int16', maxshape=(None,), chunks=(500000, ), scale_factor=0.1, missing_value=-99, fillvalue=-99)


# stn_index_coord.reshape((0, ))

with booklet.open(stn_precip_ts_path, 'n', value_serializer='pd_zstd', key_serializer='str', n_buckets=10007) as f:
    stn_df_list = []
    for i, stn_id in enumerate(stn_ids):
        print(stn_id)
        stn = stns2[stn_id]

        p_data1 = tethys1.get_results(stn['dataset_id'], stn['station_id']).squeeze('height', drop=True)
        p_data2 = p_data1[['station_id', 'precipitation']].squeeze(drop=True).to_dataframe().dropna().reset_index()
        stn_data = p_data1[['station_id', 'ref', 'name', 'lon', 'lat']].to_dataframe().reset_index(drop=True)

        stn_data['dataset_id'] = stn['dataset_id']
        stn_data['from_date'] = stn['time_range']['from_date']
        stn_data['to_date'] = stn['time_range']['to_date']
        stn_data['island'] = stn['island']
        stn_data['data_count'] = stn['count']

        stn_data = stn_data.set_index('station_id')

        stn_df_list.append(stn_data)

        # p_data2.to_hdf(stn_precip_ts_path, 'stn_precip', mode='a', complevel=1, complib='blosc:zstd', append=True, format='table', index=False)

        f[stn_id] = p_data2

stn_df = pd.concat(stn_df_list)
stn_df.to_csv(stn_data_ts_path)

dataset_list = []
for ds_, ds in p_datasets_dict.items():
    data = [ds[key] for key in dataset_params_export]
    data.extend([ds['time_range']['from_date'], ds['time_range']['to_date']])
    dataset_list.append(data)

cols = copy.deepcopy(dataset_params_export)
cols.extend(['from_date', 'to_date'])

ds_df = pd.DataFrame(dataset_list, columns=cols)

ds_df.to_csv(precip_datasets_path, index=False)


##################################################
### Prep data for modelling

with booklet.open(stn_precip_ts_path) as f:
    for stn_id, data in f.items():
        print(stn_id)

        data2 = data.drop('station_id', axis=1).set_index('time')['precipitation']

        stn = stn_df.loc[stn_id]
        ds_id = stn['dataset_id']
        ds = ds_df[ds_df.dataset_id == ds_id].iloc[0]
        freq = ds['frequency_interval']

        if freq != '24H':

            ## Correct for data that is not hourly...
            r1 = data2.rolling(5, center=True)

            r2 = [pd.infer_freq(r.index) for r in r1]

            r3 = pd.Series(r2, index=data2.index)
            r3.loc[r3.isnull()] = 'Y'
            r3.loc[r3.str.contains('H')] = 'H'
            r3.loc[~(r3.str.contains('H') | r3.str.contains('Y'))] = 'D'
            r3.loc[r3.str.contains('Y')] = np.nan
            r3 = r3.fillna('ffill')
            r4 = r3 == 'H'

            data3 = data2[r4].copy()

            if len(data3) < (8*364*24):
                continue

            offset = '-3H'
            data4 = data3.iloc[1:].resample('24H', offset=offset).sum()
            # data4.index = data4.index.round('D')
        else:
            time1 = data2.index[10]

            hour = time1.hour
            if hour > 12:
                hour = 24 - hour

            offset = f'{hour}H'

            data4a = data2.resample('24H', offset=offset).sum()
            data4 = data4a.resample('2W').max()

            dfit = distfit(distr=['lognorm', 'genextreme'])
            dfit.fit_transform(data4.values)
            dfit.plot(n_top=2)

            # data4.index = data4.index.round('D')

        ## Geometry buffer
        geo1 = Point(stn['lon'], stn['lat'])
        poly_geo = mapping(geo1.buffer(0.15))

        ## Get ERA5-Land data
        era_stns1 = tethys1.get_stations(era5_dataset['dataset_id'], geometry=poly_geo)
        era_stn_ids = [s1['station_id'] for s1 in era_stns1]

        era5_data0 = tethys1.get_results(era5_dataset['dataset_id'], era_stn_ids)

        era5_data1 = era5_data0.squeeze(drop=True)
        dims = era5_data1.dims
        lat_len = dims['lat']
        lon_len = dims['lon']
        era5_data1 = era5_data1.assign(stn_id=(('lat', 'lon'), np.arange(lon_len*lat_len).reshape(lat_len, lon_len)))

        era5_data2 = era5_data1.to_dataframe().reset_index().drop(['lat', 'lon'], axis=1).set_index(['stn_id', 'time'])['precipitation'].unstack(0).dropna()

        era5_data3a = era5_data2.resample('24H', offset=offset).sum()
        era5_data3b = era5_data3a.loc[data4a.index[0]:]
        era5_data3 = era5_data3b.resample('2W').max()

        ## Prep data for modelling
        era5_data4 = era5_data3[era5_data3.index.isin(data4.index)].copy()
        data5 = data4[data4.index.isin(era5_data4.index)].copy()

        # era5_data4[era5_data4 <= 0.5] = 0
        # data5[data5 <= 0.5] = 0

        # Shift era5 data
        # shift = [-1, 0, 1]

        # ## Shift times in era5
        # df_list = []
        # for c in era5_data4:
        #     s2 = era5_data4[c]
        #     for d in shift:
        #         n1 = s2.shift(d, 'W')
        #         n1.name = str(c) + '_' + str(d)
        #         df_list.append(n1)
        # era5_data6 = pd.concat(df_list, axis=1).dropna()
        era5_data6 = era5_data4

        data6 = data5[data5.index.isin(era5_data6.index)].copy()

        index_len = len(data6)
        train_len = int(index_len*0.75)
        test_len = index_len - train_len

        test_features_df = era5_data6[train_len:]
        test_features = np.array(test_features_df)

        test_labels_df = data6[train_len:]
        test_labels = np.array(test_labels_df)

        train_features_df = era5_data6[:train_len]
        train_features = np.array(train_features_df)
        train_labels_df = data6[:train_len]
        train_labels = np.array(train_labels_df)

        model = HistGradientBoostingRegressor(loss='squared_error', max_iter=100, learning_rate=0.05, early_stopping=False)
        # gbp = HistGradientBoostingRegressor(loss='poisson', max_iter=100, learning_rate=0.1)
        model = RandomForestRegressor(n_estimators = 200, n_jobs=7)
        # rfc = RandomForestClassifier(n_estimators = 200, n_jobs=4)

        # model_dict = {'gbsq': gbsq, 'gbp': gbp, 'rfr': rfr}
        # model_dict = {'rfr': rfr, 'rfc': rfc}

        model.fit(train_features, train_labels)

        ## Make the predictions and combine with the actuals
        predictions1 = model.predict(test_features)

        predict1 = pd.Series(predictions1, index=test_features_df.index, name='predicted')
        # predict1.loc[predict1 <= 0.5] = 0
        predict1.loc[predict1 <= 0] = 0

        combo1 = pd.merge(test_labels_df.reset_index(), predict1.reset_index(), on='time', how='left').set_index('time')

        combo1['error'] = combo1['predicted'] - combo1['precipitation']
        combo1['AE'] = combo1['error'].abs()
        mean_actual = combo1['precipitation'].mean()
        mean_ae = combo1['AE'].mean()
        nae = round(mean_ae/mean_actual, 3)
        mean_error = combo1['error'].mean()
        bias = round(mean_error/mean_actual, 3)

        ## Prepare the ts data
        combo2 = pd.merge(train_labels_df.reset_index(), predict1.reset_index(), on='time', how='right')
        combo2['residuals'] = combo2['predicted'] - combo2['precipitation']

        combo3 = combo2.drop('precipitation', axis=1).rename(columns={'predicted': 'precipitation'}).set_index(['geometry', 'time', 'height']).to_xarray()

        ## Prepare stn data
        # stn2 = copy.deepcopy(stn)
        stn2 = {k: v for k, v in stn.items() if k in ['station_id', 'ref', 'name', 'altitude']}
        stn2['NAE'] = nae
        stn2['bias'] = bias
        stn2['geometry'] = g1

        stn3 = pd.DataFrame([stn2]).set_index('geometry').to_xarray()

        ## Combo
        combo4 = xr.combine_by_coords([combo3, stn3], data_vars='minimal')

        ## Full train
        all_features = np.array(era5_data6)
        all_labels = np.array(data6)
        model.fit(all_features, all_labels)

        predict2 = model.predict(all_features)
        predict3 = pd.Series(predict2, index=era5_data6.index, name='predicted')











































################################################
### Run through all stations

parent_datasets = [era5_dataset['dataset_id']]
parent_datasets.extend([s['dataset_id'] for s in p_datasets])

datasets['precipitation'][0].update({'parent_datasets': parent_datasets})

try:
    ### Initialize
    # run_date = pd.Timestamp.today(tz='utc').round('s').tz_localize(None)
    # to_date = (run_date - pd.tseries.offsets.MonthEnd(1)).floor('D')

    titan = tu.titan.Titan()

    ### Create dataset_ids, check if datasets.json exist on remote, and if not add it
    titan.load_dataset_metadata(datasets)

    titan.load_connection_params(s3_remote['connection_config'], s3_remote['bucket'], public_url, version=version)

    titan.load_run_date(processing_code, run_date)


    for stn in stns2:
        print(stn)
        p_data1 = tethys1.get_results(stn['dataset_id'], stn['station_id'], squeeze_dims=True)
        p_data2 = p_data1['precipitation'].drop(['geometry', 'height']).to_dataframe().dropna()

        ## Correct for data that is not hourly...
        r1 = p_data2.rolling(5, center=True)

        r2 = [pd.infer_freq(r.index) for r in r1]

        r3 = pd.Series(r2, index=p_data2.index)
        r3.loc[r3.isnull()] = 'Y'
        r3.loc[r3.str.contains('H')] = 'H'
        r3.loc[~(r3.str.contains('H') | r3.str.contains('Y'))] = 'D'
        r3.loc[r3.str.contains('Y')] = np.nan
        r3 = r3.fillna('ffill')
        r4 = r3 == 'H'

        p_data3 = p_data2[r4].copy()

        if not p_data3.empty:
            g1 = str(p_data1.geometry.values)
            geo = wkb.loads(g1, hex=True)
            poly_geo = mapping(geo.buffer(0.15))

            era_stns1 = tethys1.get_stations(era5_dataset['dataset_id'], geometry=poly_geo)
            era_stn_ids = [s1['station_id'] for s1 in era_stns1]

            era1 = tethys1.get_bulk_results(era5_dataset['dataset_id'], era_stn_ids, squeeze_dims=True).dropna('time')

            ## Adjust times to UTC+12
            p_data3.index = p_data3.index + pd.DateOffset(hours=12)
            era1['time'] = pd.to_datetime(era1['time']) + pd.DateOffset(hours=12)

            ## Additional processing
            era2 = era1.drop('height').to_dataframe().reset_index().drop(['lat', 'lon'], axis=1)
            era3 = era2.set_index(['station_id', 'time'])['precipitation'].unstack(0)

            era4 = era3[era3.index.isin(p_data3.index)].copy()
            p_data3 = p_data3[p_data3.index.isin(era4.index)].copy()

            if not p_data3.empty:
                p_data4 = p_data3.resample('D').sum()
                era5 = era3.resample('D').sum()

                shift = [-1, 0, 1]

                ## Shift times in era5
                df_list = []
                for c in era5:
                    s2 = era5[c]
                    for d in shift:
                        n1 = s2.shift(d, 'D')
                        n1.name = c + '_' + str(d)
                        df_list.append(n1)
                era6 = pd.concat(df_list, axis=1).dropna()

                p_data5 = p_data4[p_data4.index.isin(era6.index)].copy()
                p_data5[p_data5.precipitation <= 0.5] = 0

                # from_date = p_data5.index[0]
                # to_date = p_data5.index[-1]

                # time_range = (to_date - from_date).days
                # year_range = int(time_range/365)
                n_years = len(p_data5)/365

                ## Package up for analysis
                if n_years >= min_year_range:

                    test_features_df = era6
                    test_features = np.array(test_features_df)

                    test_labels_df = p_data5['precipitation']
                    test_labels = np.array(test_labels_df)

                    train_features_df = era6[era6.index.isin(p_data5.index)]
                    train_features = np.array(train_features_df)
                    train_labels_df = p_data5['precipitation']
                    train_labels = np.array(train_labels_df)

                    # gbsq = HistGradientBoostingRegressor(loss='squared_error', max_iter=100, learning_rate=0.1)
                    # gbp = HistGradientBoostingRegressor(loss='poisson', max_iter=100, learning_rate=0.1)
                    rfr = RandomForestRegressor(n_estimators = 200, n_jobs=4)
                    # rfc = RandomForestClassifier(n_estimators = 200, n_jobs=4)

                    # model_dict = {'gbsq': gbsq, 'gbp': gbp, 'rfr': rfr}
                    # model_dict = {'rfr': rfr, 'rfc': rfc}

                    rfr.fit(train_features, train_labels)

                    ## Make the predictions and combine with the actuals
                    predictions1 = rfr.predict(test_features)

                    predict1 = pd.Series(predictions1, index=test_features_df.index, name='predicted')
                    predict1.loc[predict1 <= 0.5] = 0

                    combo1 = pd.merge(test_labels_df.reset_index(), predict1.reset_index(), on='time', how='left').set_index('time')

                    combo1['error'] = combo1['predicted'] - combo1['precipitation']
                    combo1['AE'] = combo1['error'].abs()
                    mean_actual = combo1['precipitation'].mean()
                    mean_ae = combo1['AE'].mean()
                    nae = round(mean_ae/mean_actual, 3)
                    mean_error = combo1['error'].mean()
                    bias = round(mean_error/mean_actual, 3)

                    ## Prepare the ts data
                    combo2 = pd.merge(train_labels_df.reset_index(), predict1.reset_index(), on='time', how='right')
                    combo2['residuals'] = combo2['predicted'] - combo2['precipitation']
                    combo2['geometry'] = g1
                    combo2['height'] = p_data1.height.values

                    combo3 = combo2.drop('precipitation', axis=1).rename(columns={'predicted': 'precipitation'}).set_index(['geometry', 'time', 'height']).to_xarray()

                    ## Prepare stn data
                    # stn2 = copy.deepcopy(stn)
                    stn2 = {k: v for k, v in stn.items() if k in ['station_id', 'ref', 'name', 'altitude']}
                    stn2['NAE'] = nae
                    stn2['bias'] = bias
                    stn2['geometry'] = g1

                    stn3 = pd.DataFrame([stn2]).set_index('geometry').to_xarray()

                    ## Combo
                    combo4 = xr.combine_by_coords([combo3, stn3], data_vars='minimal')

                    titan.load_results(combo4, datasets['precipitation'][0]['dataset_id'], other_attrs=stn_attrs, discrete=False)

    ########################################
    ### Save results and stations
    titan.update_results(30)

except Exception as err:
    # print(err)
    print(traceback.format_exc())
    tu.misc.email_msg(remote['remote']['email']['sender_address'], remote['remote']['email']['sender_password'], remote['remote']['email']['receiver_address'], 'Failure on tethys-extraction-flownat', traceback.format_exc(), remote['remote']['email']['smtp_server'])

try:

    ### Aggregate all stations for the dataset
    print('Aggregate all stations for the dataset and all datasets in the bucket')

    titan.update_aggregates()

    ### Timings
    end_run_date = pd.Timestamp.today(tz='utc').round('s')

    print('-- Finished!')
    print(end_run_date)

except Exception as err:
    # print(err)
    print(traceback.format_exc())
    tu.misc.email_msg(remote['remote']['email']['sender_address'], remote['remote']['email']['sender_password'], remote['remote']['email']['receiver_address'], 'Failure on tethys-extraction-flownat agg processes', traceback.format_exc(), remote['remote']['email']['smtp_server'])
