# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn import cross_validation, metrics  # Additional     scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search
import datetime
from math import *
from datetime import date
from sklearn import preprocessing
import geohash
from sklearn.model_selection import train_test_split
import datetime
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings("ignore")


# geohash 3-11
# round 7-9

################geohash######################
def get_geohash(x, precision_i):
    return geohash.encode(longitude=float(x.longitude), latitude=float(x.latitude), precision=precision_i)


def get_geohash_round1(geohash_u):
    return geohash.neighbors(geohash_u)[0]


def get_geohash_round2(geohash_u):
    return geohash.neighbors(geohash_u)[1]


def get_geohash_round3(geohash_u):
    return geohash.neighbors(geohash_u)[2]


def get_geohash_round4(geohash_u):
    return geohash.neighbors(geohash_u)[3]


def get_geohash_round5(geohash_u):
    return geohash.neighbors(geohash_u)[4]


def get_geohash_round6(geohash_u):
    return geohash.neighbors(geohash_u)[5]


def get_geohash_round7(geohash_u):
    return geohash.neighbors(geohash_u)[6]


def get_geohash_round8(geohash_u):
    return geohash.neighbors(geohash_u)[7]


shop_info = pd.read_csv("../ccf_first_round_shop_info.csv")
trade_info = pd.read_csv("../ccf_first_round_user_shop_behavior.csv")
all_info = pd.merge(trade_info, shop_info[['shop_id', 'mall_id']], on='shop_id', how='left')
test_info = pd.read_csv("../evaluation_public.csv")
train_wifi = pd.read_csv("../dealFile/train_index_wifi.csv")
test_wifi = pd.read_csv("../dealFile/test_index_wifi.csv")
train_con = pd.concat([all_info, train_wifi], axis=1)
test_con = pd.concat([test_info, test_wifi], axis=1)
for i in range(0, 10):
    train_con[['bss' + str(i)]] = train_con[['bss' + str(i)]].fillna('0')
    train_con[['sig' + str(i)]] = train_con[['sig' + str(i)]].fillna('-999')
    test_con[['bss' + str(i)]] = test_con[['bss' + str(i)]].fillna('0')
    test_con[['sig' + str(i)]] = test_con[['sig' + str(i)]].fillna('-999')

mall_list = list(set(list(shop_info.mall_id)))

mall_data = {}
for value in mall_list:
    mall_data[value] = train_con[train_con.mall_id == value]

del trade_info, all_info, test_info


def processTime(train, test):
    ##########time_stamp##########
    da = datetime.datetime.now()
    l = da
    train['day_of_week'] = train.time_stamp.astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[5:7]), int(x[8:10])).weekday() + 1)
    train['day_of_month'] = train.time_stamp.astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[5:7]), int(x[8:10])).day)
    train['hour'] = train.time_stamp.astype('str').apply(lambda x: int(x[11:13]))
    train['minute'] = train.time_stamp.astype('str').apply(lambda x: int(x[14:16]))

    # test
    test['day_of_week'] = test.time_stamp.astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[5:7]), int(x[8:10])).weekday() + 1)
    test['day_of_month'] = train.time_stamp.astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[5:7]), int(x[8:10])).day)
    test['hour'] = test.time_stamp.astype('str').apply(lambda x: int(x[11:13]))
    test['minute'] = test.time_stamp.astype('str').apply(lambda x: int(x[14:16]))

    print "process Time"
    return train, test


def processLoc(train, test):
    for i in range(8, 13):
        train['geohash_' + str(i)] = train.apply(get_geohash, axis=1, args=(i,))
        test['geohash_' + str(i)] = test.apply(get_geohash, axis=1, args=(i,))
    train['geohash_91'] = train.geohash_9.apply(get_geohash_round1)
    train['geohash_92'] = train.geohash_9.apply(get_geohash_round2)
    train['geohash_93'] = train.geohash_9.apply(get_geohash_round3)
    train['geohash_94'] = train.geohash_9.apply(get_geohash_round4)
    train['geohash_95'] = train.geohash_9.apply(get_geohash_round5)
    train['geohash_96'] = train.geohash_9.apply(get_geohash_round6)
    train['geohash_97'] = train.geohash_9.apply(get_geohash_round7)
    train['geohash_98'] = train.geohash_9.apply(get_geohash_round8)

    test['geohash_91'] = test.geohash_9.apply(get_geohash_round1)
    test['geohash_92'] = test.geohash_9.apply(get_geohash_round2)
    test['geohash_93'] = test.geohash_9.apply(get_geohash_round3)
    test['geohash_94'] = test.geohash_9.apply(get_geohash_round4)
    test['geohash_95'] = test.geohash_9.apply(get_geohash_round5)
    test['geohash_96'] = test.geohash_9.apply(get_geohash_round6)
    test['geohash_97'] = test.geohash_9.apply(get_geohash_round7)
    test['geohash_98'] = test.geohash_9.apply(get_geohash_round8)

    train['geohash_101'] = train.geohash_10.apply(get_geohash_round1)
    train['geohash_102'] = train.geohash_10.apply(get_geohash_round2)
    train['geohash_103'] = train.geohash_10.apply(get_geohash_round3)
    train['geohash_104'] = train.geohash_10.apply(get_geohash_round4)
    train['geohash_105'] = train.geohash_10.apply(get_geohash_round5)
    train['geohash_106'] = train.geohash_10.apply(get_geohash_round6)
    train['geohash_107'] = train.geohash_10.apply(get_geohash_round7)
    train['geohash_108'] = train.geohash_10.apply(get_geohash_round8)

    test['geohash_101'] = test.geohash_10.apply(get_geohash_round1)
    test['geohash_102'] = test.geohash_10.apply(get_geohash_round2)
    test['geohash_103'] = test.geohash_10.apply(get_geohash_round3)
    test['geohash_104'] = test.geohash_10.apply(get_geohash_round4)
    test['geohash_105'] = test.geohash_10.apply(get_geohash_round5)
    test['geohash_106'] = test.geohash_10.apply(get_geohash_round6)
    test['geohash_107'] = test.geohash_10.apply(get_geohash_round7)
    test['geohash_108'] = test.geohash_10.apply(get_geohash_round8)

    for i in [8, 11, 12]:
        geo_lbl_1 = preprocessing.LabelEncoder()
        geo_lbl_1.fit(list(set(train['geohash_' + str(i)].values) | set(test['geohash_' + str(i)].values)))
        train['geohash_' + str(i)] = geo_lbl_1.transform(train['geohash_' + str(i)].values)
        test['geohash_' + str(i)] = geo_lbl_1.transform(test['geohash_' + str(i)].values)

    geo_lbl = preprocessing.LabelEncoder()
    geo_lbl.fit(list(set(train['geohash_9'].values) | set(test['geohash_9'].values) | \
                     set(train['geohash_91']) | set(test['geohash_91']) | \
                     set(train['geohash_92']) | set(test['geohash_92']) | \
                     set(train['geohash_93']) | set(test['geohash_93']) | \
                     set(train['geohash_94']) | set(test['geohash_94']) | \
                     set(train['geohash_95']) | set(test['geohash_95']) | \
                     set(train['geohash_96']) | set(test['geohash_96']) | \
                     set(train['geohash_97']) | set(test['geohash_97']) | \
                     set(train['geohash_98']) | set(test['geohash_98'])))

    train['geohash_9'] = geo_lbl.transform(train['geohash_9'].values)
    train['geohash_91'] = geo_lbl.transform(train['geohash_91'].values)
    train['geohash_92'] = geo_lbl.transform(train['geohash_92'].values)
    train['geohash_93'] = geo_lbl.transform(train['geohash_93'].values)
    train['geohash_94'] = geo_lbl.transform(train['geohash_94'].values)
    train['geohash_95'] = geo_lbl.transform(train['geohash_95'].values)
    train['geohash_96'] = geo_lbl.transform(train['geohash_96'].values)
    train['geohash_97'] = geo_lbl.transform(train['geohash_97'].values)
    train['geohash_98'] = geo_lbl.transform(train['geohash_98'].values)

    test['geohash_9'] = geo_lbl.transform(test['geohash_9'].values)
    test['geohash_91'] = geo_lbl.transform(test['geohash_91'].values)
    test['geohash_92'] = geo_lbl.transform(test['geohash_92'].values)
    test['geohash_93'] = geo_lbl.transform(test['geohash_93'].values)
    test['geohash_94'] = geo_lbl.transform(test['geohash_94'].values)
    test['geohash_95'] = geo_lbl.transform(test['geohash_95'].values)
    test['geohash_96'] = geo_lbl.transform(test['geohash_96'].values)
    test['geohash_97'] = geo_lbl.transform(test['geohash_97'].values)
    test['geohash_98'] = geo_lbl.transform(test['geohash_98'].values)

    geo_lbl = preprocessing.LabelEncoder()
    geo_lbl.fit(list(set(train['geohash_10'].values) | set(test['geohash_10'].values) | \
                     set(train['geohash_101']) | set(test['geohash_101']) | \
                     set(train['geohash_102']) | set(test['geohash_102']) | \
                     set(train['geohash_103']) | set(test['geohash_103']) | \
                     set(train['geohash_104']) | set(test['geohash_104']) | \
                     set(train['geohash_105']) | set(test['geohash_105']) | \
                     set(train['geohash_106']) | set(test['geohash_106']) | \
                     set(train['geohash_107']) | set(test['geohash_107']) | \
                     set(train['geohash_108']) | set(test['geohash_108'])))

    train['geohash_10'] = geo_lbl.transform(train['geohash_10'].values)
    train['geohash_101'] = geo_lbl.transform(train['geohash_101'].values)
    train['geohash_102'] = geo_lbl.transform(train['geohash_102'].values)
    train['geohash_103'] = geo_lbl.transform(train['geohash_103'].values)
    train['geohash_104'] = geo_lbl.transform(train['geohash_104'].values)
    train['geohash_105'] = geo_lbl.transform(train['geohash_105'].values)
    train['geohash_106'] = geo_lbl.transform(train['geohash_106'].values)
    train['geohash_107'] = geo_lbl.transform(train['geohash_107'].values)
    train['geohash_108'] = geo_lbl.transform(train['geohash_108'].values)

    test['geohash_10'] = geo_lbl.transform(test['geohash_10'].values)
    test['geohash_101'] = geo_lbl.transform(test['geohash_101'].values)
    test['geohash_102'] = geo_lbl.transform(test['geohash_102'].values)
    test['geohash_103'] = geo_lbl.transform(test['geohash_103'].values)
    test['geohash_104'] = geo_lbl.transform(test['geohash_104'].values)
    test['geohash_105'] = geo_lbl.transform(test['geohash_105'].values)
    test['geohash_106'] = geo_lbl.transform(test['geohash_106'].values)
    test['geohash_107'] = geo_lbl.transform(test['geohash_107'].values)
    test['geohash_108'] = geo_lbl.transform(test['geohash_108'].values)
    print "process geo"
    return train, test


def processUser(train, test):
    a = train.user_id.values
    b = test.user_id.values
    user_share = list(set(a).intersection(set(b)))
    train_n_in = ~train.user_id.isin(user_share).values
    test_n_in = ~test.user_id.isin(user_share).values
    train.loc[train_n_in, 'user_id'] = '0'
    test.loc[test_n_in, 'user_id'] = '0'

    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(set(train['user_id'].values) | set(test['user_id'].values)))
    train['user_id'] = lbl.transform(train['user_id'].values)
    test['user_id'] = lbl.transform(test['user_id'].values)
    return train, test


def proceeWifi(train, test, num=4):
    # test
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(set(train['connect'].values) | set(train['bss0'].values) | set(train['bss1'].values) | \
                 set(train['bss2'].values) | set(train['bss3'].values) | set(train['bss4'].values) | \
                 set(train['bss5'].values) | set(train['bss6'].values) | \
                 set(train['bss7'].values) | set(train['bss8'].values) | \
                 set(train['bss9'].values) | set(test['connect'].values) | \
                 set(test['bss0'].values) | set(test['bss1'].values) | \
                 set(test['bss2'].values) | set(test['bss3'].values) | set(test['bss4'].values) | \
                 set(test['bss5'].values) | set(test['bss6'].values) | \
                 set(test['bss7'].values) | set(test['bss8'].values) | set(test['bss9'].values)))
    train['connect'] = lbl.transform(train['connect'].values)
    test['connect'] = lbl.transform(test['connect'].values)
    train['bss0'] = lbl.transform(train['bss0'].values)
    test['bss0'] = lbl.transform(test['bss0'].values)
    train['bss1'] = lbl.transform(train['bss1'].values)
    test['bss1'] = lbl.transform(test['bss1'].values)
    train['bss2'] = lbl.transform(train['bss2'].values)
    test['bss2'] = lbl.transform(test['bss2'].values)
    train['bss3'] = lbl.transform(train['bss3'].values)
    test['bss3'] = lbl.transform(test['bss3'].values)
    train['bss4'] = lbl.transform(train['bss4'].values)
    test['bss4'] = lbl.transform(test['bss4'].values)
    train['bss5'] = lbl.transform(train['bss5'].values)
    test['bss5'] = lbl.transform(test['bss5'].values)
    train['bss6'] = lbl.transform(train['bss6'].values)
    test['bss6'] = lbl.transform(test['bss6'].values)
    train['bss7'] = lbl.transform(train['bss7'].values)
    test['bss7'] = lbl.transform(test['bss7'].values)
    train['bss8'] = lbl.transform(train['bss8'].values)
    test['bss8'] = lbl.transform(test['bss8'].values)
    train['bss9'] = lbl.transform(train['bss9'].values)
    test['bss9'] = lbl.transform(test['bss9'].values)

    train1 = pd.concat([train, test])
    l = []
    wifi_dict = {}
    for index, row in train1.iterrows():
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:
            row[i[0]] = int(i[1])
            if i[0] not in wifi_dict:
                wifi_dict[i[0]] = 1
            else:
                wifi_dict[i[0]] += 1
        l.append(row)
    delate_wifi = []
    for i in wifi_dict:
        if wifi_dict[i] < 20:
            delate_wifi.append(i)
    m = []
    for row in l:
        new = {}
        for n in row.keys():
            if n not in delate_wifi:
                new[n] = row[n]
        m.append(new)
    train1 = pd.DataFrame(m)
    df_train = train1[train1.shop_id.notnull()]
    df_test = train1[train1.shop_id.isnull()]

    return df_train, df_test


s_times = 1


def generateData11(mall_data, mall_id):
    print str(s_times) + "  " + str(mall_id)
    print datetime.datetime.now()
    train = mall_data
    test = test_con[test_con.mall_id == mall_id]

    shop_dict = {}
    exam_shop = shop_info[shop_info.mall_id == mall_id]
    i1 = 0
    for index, row in exam_shop.iterrows():
        shop_dict[row.shop_id] = i1
        i1 = i1 + 1
    shop_ito_shop = dict((v, k) for k, v in shop_dict.iteritems())

    train, test = processUser(train, test)
    train, test = processTime(train, test)
    train, test = processLoc(train, test)
    train, test = proceeWifi(train, test)
    train = shuffle(train)

    label_train = train[['shop_id']]
    label_num = len(shop_info[shop_info.mall_id == mall_id].shop_id.values)
    label_to_train = []
    label_train = label_train.shop_id.values
    for l in label_train:
        label_to_train.append(shop_dict[l])

    train.drop(['shop_id', 'time_stamp', 'count', 'mall_id', 'wifi_infos', 'row_id'], axis=1, inplace=True)
    row_id_list = test.row_id.values
    test.drop(['row_id', 'time_stamp', 'mall_id', 'count', 'wifi_infos', 'shop_id'], axis=1, inplace=True)

    xg_train = train.as_matrix()
    xg_test = test.as_matrix()

    params = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 9,
        'eval_metric': 'merror',
        'seed': 0,
        'missing': -999,
        'num_class': label_num,
        'silent': 1
    }

    xgbtrain = xgb.DMatrix(train, label_to_train)
    xgbtest = xgb.DMatrix(test)

    watchlist = [(xgbtrain, 'train'), (xgbtrain, 'test')]
    num_rounds = 60
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)

    test_res = model.predict(xgbtest)

    with open("res_1103.csv", 'a+') as f:
        for i in range(len(test_res)):
            f.write(str(int(row_id_list[i])) + ',' + str(shop_ito_shop[test_res[i]] + '\n'))
    print datetime.datetime.now()


with open("res_1103.csv", 'a+') as f:
    f.write('row_id,shop_id\n')
for i in mall_data.keys():
    generateData11(mall_data[i], i)
    s_times = s_times + 1



