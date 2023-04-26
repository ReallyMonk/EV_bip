from os.path import isfile
import os
import sys
import pandas as pd
import numpy as np
import json
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import time
from statsmodels.tsa.vector_ar.var_model import VAR
import torch
from pyts.image import GramianAngularField
import re
import random

from tools import check_para
from map_req import get_info_matrix

# load the data set
data_path = './data/'
datafile_name = 'UK_data.json'


# mentain info: coordinate, location, id, name, connector_num, connector_info
def generate_ChargeDevice(item):
    # coordinate
    coor = (float(item['ChargeDeviceLocation']['Latitude']), float(item['ChargeDeviceLocation']['Longitude']))
    # location
    loc = item['ChargeDeviceLocation']['Address']['County']
    # id
    id = item['ChargeDeviceId']
    # name
    name = item['ChargeDeviceName']
    # connector_num
    con_num = len(item['Connector'])
    # connector_info
    con_info = [float(cntr['RatedOutputkW']) for cntr in item['Connector']]

    loctype = item['LocationType']

    country = item['ChargeDeviceLocation']['Address']['Country']

    posttwon = item['ChargeDeviceLocation']['Address']['PostTown']

    postCode = item['ChargeDeviceLocation']['Address']['PostCode']

    charger = {
        'id': id,
        'name': name,
        'coor': coor,
        'loc': loc,
        'con_num': con_num,
        'con_info': con_info,
        'loctype': loctype,
        'country': country,
        'posttown': posttwon,
        'postcode': postCode,
    }

    return charger


def get_charger_loc(county='Greater London'):
    with open(data_path + datafile_name, 'r') as f:
        json_obj = json.load(f)
        data = json_obj['ChargeDevice']

    # pick up chargers located in certain area
    #county = []
    GL = []
    conties = []
    for item in data:
        loc = item['ChargeDeviceLocation']['Address']['County']
        conties.append(loc)
        if loc == county and item['ChargeDeviceStatus'] == 'In service':
            #print('here')
            charger = generate_ChargeDevice(item)
            if charger['coor'][1] > 0.3 or charger['coor'][1] < -0.55:
                pass
            elif charger['coor'][0] == 51.592867 and charger['coor'][1] == 0.245586:
                pass
            elif charger['coor'][0] == 51.420081 and charger['coor'][1] == 0.162433:
                pass
            elif charger['coor'][0] == 51.491095 and charger['coor'][1] == -0.2847:
                pass
            elif charger['coor'][0] == 51.409737 and charger['coor'][1] == -0.095576:
                pass
            else:
                GL.append(charger)

    # print(GL[0])
    return GL


#plt.show()
# normalization, standardization
def standardization(data, method='Norm'):
    data = np.array(data)
    if method == 'Norm':
        _avg = np.average(data)
        _var = np.var(data)

        data = (data - _avg) / _var

    elif method == 'Std':
        _min = data.min()
        _max = data.max()

        data = (data - _min) / (_max - _min)

    return data


# print(lon_data)
def ncr_data(node_num=800):

    def is_in_London(addr):
        london_postcode = ['W', 'E', 'N', 'NW', 'SW', 'SE', 'WC', 'EC']
        for pc in london_postcode:
            if addr.startswith(pc) or addr.startswith(' ' + pc):
                if addr.startswith('EN') or addr.startswith('WD'):
                    return False
                return True
        return False

    lon_data = get_charger_loc()

    labels = []
    x = []
    y = []
    postCodes = []

    for item in lon_data:
        # lb = item['loctype']
        if is_in_London(item['postcode']):
            # print(item['postcode'])
            postCodes.append(item['postcode'])
            labels.append(0)
        else:
            labels.append(1)

        x.append(item['coor'][1])
        y.append(item['coor'][0])
        postCodes.append(item['postcode'])
        is_in_London(item['postcode'])

    # normalize
    x = standardization(x, 'Std')
    y = standardization(y, 'Std')
    # x, y = np.array(x), np.array(y)

    labels = np.array(labels)
    out_data = np.array([labels, x, y]).T

    out_data = np.unique(out_data, axis=0)  #[2:802]
    labels = out_data.T[0]

    postCodes = np.array(postCodes)
    checkCodes = np.unique(postCodes, return_counts=True)

    # random pick
    rand_idx = np.sort(random.sample([i for i in range(len(labels))], min(node_num, len(labels))))

    return out_data[rand_idx], labels[rand_idx]


# build graph with dgl
def build_graph(data, threshold=1400, image=True):
    # get dist matrix
    dist_matrix_path = './data/distance_matrix.csv'
    dura_matrix_path = './data/duration_matrix.csv'

    # get location index
    index = [item['coor'] for item in data]

    if isfile(dura_matrix_path):
        dist_matrix = pd.read_csv(dist_matrix_path)
        dura_matrix = pd.read_csv(dura_matrix_path, index_col=0)
        #print(dura_matrix)
        #print(dura_matrix[3][2])
    else:
        #print(index)
        dist_matrix, dura_matrix = get_info_matrix(index)

    # initial graph
    g = dgl.DGLGraph()

    # add nodes
    node_num = len(data)
    g.add_nodes(node_num)

    # initial edge list
    for i in range(node_num):
        for j in range(i + 1, node_num, 1):
            # print(i, j)
            # add edges below threshold in
            if dura_matrix[str(i)][j] <= threshold and i != j:
                g.add_edges(i, j)

    # g = dgl.graph((head, end))
    #print(g.nodes())
    #print(g.edges())
    #print(g.edges()[0].size())

    # return
    # save graph
    dgl.save_graphs('./data/charger_graph', g)
    #print(data[41])
    #print(data[8])

    if image:
        plt.figure(figsize=(16, 12))
        G = g.to_networkx()
        pos = nx.kamada_kawai_layout(G)
        #pos = nx.spring_layout(G)
        #pos = nx.nx_agraph.graphviz_layout(G)
        nx.draw(G, with_labels=True, node_size=100, pos=pos, arrows=False, width=0.5, font_size=8)
        plt.show()

    return int(g.nodes().size()[0]), int(g.edges()[0].size()[0])


# find the boudry of the dataset, longitude and latitude, loc:(lat,long)
def range_analysis(data):
    # print(data[0])
    # initialize range
    longitude_range = [data[0]['coor'][1], data[0]['coor'][1]]
    latitude_range = [data[0]['coor'][0], data[0]['coor'][0]]

    for item in data:
        #print(item)
        item = item['coor']
        # latitude
        if item[1] > longitude_range[1]:
            longitude_range[1] = item[1]
        if item[1] < longitude_range[0]:
            longitude_range[0] = item[1]

        # longitude
        if item[0] > latitude_range[1]:
            latitude_range[1] = item[0]
        if item[0] < latitude_range[0]:
            latitude_range[0] = item[0]

    check_para(latitude_range)
    check_para(longitude_range)

    return longitude_range, latitude_range


def edge_weighted_analysis(data, image=True):
    # initial timeline
    timeline = [300 + i * 60 for i in range(0, 26, 2)]
    # print(timeline)

    edge_list = []
    # get graph info
    for thresh in timeline:
        nodes_num, edges_num = build_graph(data, threshold=thresh, image=False)
        edge_list.append(edges_num)

    #print(timeline)
    #print(nodes_num)

    if image:
        plt.plot(timeline, edge_list)
        for x, y in zip(timeline, edge_list):
            plt.text(x, y, y)
        plt.title('graph edges according to time threshold')
        plt.show()

    return


def polt_on_real_map(data):
    # extract range info
    longitude_range, latitude_range = range_analysis(data)
    # extract coordinate inforamtion
    longitude = []
    latitude = []
    for item in data:
        coor = item['coor']
        longitude.append(coor[0])
        latitude.append(coor[1])

    res = pd.DataFrame(longitude, latitude)
    res.to_csv('./data/coor.csv')

    return res


#data = [(51.431454, 0.031175), (51.618522, -0.11167), (51.537303, -0.057225), "Aldermans Hill"]
'''data = get_charger_loc()
range_analysis(data)
#build_graph(data)
#edge_weighted_analysis(data)
polt_on_real_map(data)
'''
strdate1 = 'Thu, 09 Sep 2021 21:11:44 GMT'
strdate2 = 'Fri, 10 Sep 2021 23:53:29 GMT'


def str2date(timestr):
    date = datetime.strptime(timestr, "%a, %d %b %Y %H:%M:%S %Z")
    return date


#res = str2date(strdate2) - str2date(strdate1)
#print(res.seconds)


def pick_up_ACNdata(path):
    whole_data = []
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
        data = data['_items']

    # obtain key list
    key_list = data[0].keys()

    item_count = {}
    item_coll = {}
    key_count = ['clusterID', 'siteID', 'spaceID', 'stationID', 'timezone', 'userID']

    # inital
    for key_item in key_count:
        item_count[key_item] = 0
        item_coll[key_item] = []

    for item in data:
        # check doneChargingtime
        if item['doneChargingTime'] is None:
            item['doneChargingTime'] = item['disconnectTime']

        # insert userInput ----
        uIn = item['userInputs']
        if uIn is not None:
            uIn = uIn[-1]
            uInKeys = uIn.keys()
            uIn.pop('userID')

            for uInKey in uInKeys:
                item['userIN' + uInKey] = uIn[uInKey]
        item.pop('userInputs')

        # rebuild time
        cnnctTime = str2date(item['connectionTime'])
        discTime = str2date(item['disconnectTime'])
        if item['doneChargingTime'] is None:
            doneCharTime = str2date(item['disconnectTime'])
        else:
            doneCharTime = str2date(item['doneChargingTime'])
        item['connectDay'] = cnnctTime.day
        item['connectMonth'] = cnnctTime.month
        item['connectWeekday'] = cnnctTime.weekday()
        item['chargingTime'] = (doneCharTime - cnnctTime).seconds
        item['connectTime'] = (discTime - cnnctTime).seconds
        if cnnctTime.year == 2020:
            item['weekNo'] = cnnctTime.isocalendar()[1]
        if cnnctTime.year == 2021:
            item['weekNo'] = cnnctTime.isocalendar()[1] + 31
        item['date'] = cnnctTime.strftime("%Y-%m-%d")

        #print(item)
        whole_data.append(item)

        # check count
        for c_key in key_count:
            if item[c_key] not in item_coll[c_key]:
                item_coll[c_key].append(item[c_key])
                item_count[c_key] += 1

    #print(item_coll)
    #print(item_count)
    return whole_data


def concat_acndata(in_path, out_path):
    json_list = os.listdir(in_path)

    data = []

    for json_file in json_list:
        if '.json' in json_file:
            data.extend(pick_up_ACNdata(in_path + json_file))

    data = pd.DataFrame(data)
    data.to_csv(out_path)
    return data


def check_data_info(csv_data):
    user_data = csv_data[csv_data.spaceID == 'CA-315']
    print(user_data)
    for idx, item in user_data.iterrows():
        print(idx, item['date'])
    return


# dataform
'''

stationID
date
sessionCount
userCount
kWhDelivered
occupiedTime
chargingTime
weekNo
'''


def reform_data(csv_data):
    # sessionCOunt and kWhDelivered
    kwCssC = csv_data.groupby(by=['stationID', 'weekNo'], as_index=False).agg({'siteID': 'max', 'sessionID': 'count', 'kWhDelivered': 'sum', 'connectTime': 'sum', 'chargingTime': 'sum'})

    # seperate each data by spaceID and weekNo
    sepData = csv_data.groupby(by=['stationID', 'weekNo'], as_index=False)

    # count userID numbers
    userCount = pd.DataFrame({'userCount': [item[1]['userID'].value_counts().count() for item in sepData]})

    res = pd.concat([kwCssC, userCount], axis=1)
    stationID = list(res['stationID'].drop_duplicates())

    for idx, ID in enumerate(stationID):
        res.replace(ID, idx, inplace=True)

    return res


def cut_weeks(data, week_num=7):
    whole_data = []
    labels = []
    tri_labels = []
    for item in data.groupby('stationID'):
        labels.append(item[0])
        whole_data.append(np.array(item[1]))
        site_lb = item[1]['siteID'].iloc[0]
        site_lb = site_lb if site_lb != 19 else 0
        #site_lb = item[1]['siteID'][0]
        tri_labels.append(site_lb)

    MTS = []
    MTS_labels = []
    MTS_tri_labels = []
    for lb, item in zip(tri_labels, whole_data):
        # pad zero
        zero_rows = week_num - int(len(item) % week_num)
        zeros = np.zeros((zero_rows, 5))
        mts = item[:, 3:]
        mts = np.concatenate([mts, zeros], axis=0)

        for step in range(0, len(mts), week_num):
            MTS.append(mts[step:step + week_num])
            MTS_labels.append(lb)

    #print(MTS_tri_labels)
    #print(len(MTS_labels))
    return MTS, MTS_labels


def window_slide(data, node_num, window_size=7, step_len=3):
    whole_data = []
    labels = []
    tri_labels = []
    for item in data.groupby('stationID'):
        labels.append(item[0])
        whole_data.append(np.array(item[1]))
        site_lb = item[1]['siteID'].iloc[0]
        site_lb = site_lb if site_lb != 19 else 0
        #site_lb = item[1]['siteID'][0]
        tri_labels.append(site_lb)

    MTS = []
    MTS_labels = []
    MTS_tri_labels = []
    for lb, item in zip(tri_labels, whole_data):
        zero_rows = window_size - int(len(item) % window_size)
        zeros = np.zeros((zero_rows, 5))
        mts = item[:, 3:]
        mts = np.concatenate([mts, zeros], axis=0)

        for step in range(0, len(mts) - window_size - 1, step_len):
            MTS.append(mts[step:step + window_size])
            MTS_labels.append(lb)

    MTS, MTS_labels = np.array(MTS), np.array(MTS_labels)
    rand_idx = np.sort(random.sample([i for i in range(len(MTS_labels))], min(node_num, len(MTS_labels))))

    return MTS[rand_idx], MTS_labels[rand_idx]


def GAF(data, sum=True):
    X = np.array([i for i in range(100)]).reshape(1, -1)

    if sum:
        gaf = GramianAngularField(image_size=5, method='summation')
    else:
        gaf = GramianAngularField(image_size=5, method='diffrence')

    X_gaf = gaf.fit_transform(data)

    return X_gaf


def convert_MTS_imge(data, labels, out_path):
    dataset = []
    for station_item in data:
        #print(station_item)
        station_item = station_item.T
        # station_img = []

        fea_img = GAF(station_item)
        fea_img = fea_img.reshape(1, -1)

        dataset.append(fea_img[0])
        '''for idx, feature in enumerate(station_item):
            print(idx)
            fea_img = GAF(feature.reshape(1, -1))
            if idx == 0:
                station_img = fea_img
            else:
                station_img = np.concatenate((station_img, fea_img), axis=0)'''

        #check_para(station_img)

    labels = np.array(labels).reshape(-1, 1)
    # print(labels.shape)
    dataset = np.concatenate((labels, np.array(dataset)), axis=1)
    # print(dataset.shape)
    # pd.DataFrame(dataset).to_csv(out_path)
    np.savetxt(out_path, dataset)

    return dataset


def VARMA(data, d):
    n, t = data.size()
    Z = data[:, d:].T

    lags = np.array([i for i in range(1, d + 1)])
    for i in range(d + 1, t):
        q = data[:, i + d - lags].view(-1)
        if i == d + 1:
            Q = q
        else:
            Q = torch.cat((Q, q))

    Q.resize(t - d, n * d)
    A = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(Q.T, Q)), Q.T), Z)

    return A
