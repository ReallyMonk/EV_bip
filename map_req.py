import googlemaps
from datetime import datetime
import pandas as pd

from tools import check_para

# Personal_API_key = 'AIzaSyBko56SEQZY-nd7rcG_6tcSRNsERRUiKVM'
#Personal_API_key = 'key'

#gmaps = googlemaps.Client(key=Personal_API_key)

# Geocoding an address
#geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')

# Look up an address with reverse geocoding
#reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))

# Request directions via public transit
#now = datetime.now()
#directions_result = gmaps.directions("Sydney Town Hall", "Parramatta, NSW", mode="transit", departure_time=now)


# build origin and destination list
#origins = [(51.431454, 0.031175), (51.618522, -0.11167), (51.537303, -0.057225), "Aldermans Hill"]
#destinations = [(51.431454, 0.031175), (51.530281, 0.142523)]
def request_info_matrix(origins, destinations):
    # request gmap data
    # here we need know the gmap api only takes 10 items one time
    responce_data = gmaps.distance_matrix(origins, destinations, mode='driving')
    elements = responce_data['rows']
    # index = data
    dist_data = []
    dura_data = []

    # pick up distance data
    #print(len(elements))
    for elemt in elements:
        #print(elemt)
        elemt = elemt['elements']
        dist_data.append([item['distance']['value'] for item in elemt])
        dura_data.append([item['duration']['value'] for item in elemt])
        #print(dist_data)
        #print(dura_data)

    distance_matrix = pd.DataFrame(dist_data)
    duration_matrix = pd.DataFrame(dura_data)

    # calculate avrage
    #distance_matrix = (distance_matrix + distance_matrix.T) / 2
    #print(distance_matrix)
    #distance_matrix = distance_matrix.astype('int')

    #duration_matrix = (duration_matrix + duration_matrix.T) / 2
    #duration_matrix = duration_matrix.astype('int')

    #print(distance_matrix)
    #print(duration_matrix)
    #print(distance_matrix[0][3])
    #distance_matrix.to_csv('./data/distance_matrix.csv')
    #duration_matrix.to_csv('./data/duration_matrix.csv')
    return distance_matrix, duration_matrix


def get_info_matrix(data):

    # get matrix data
    # here we need to know that gmap api only takes 10 items in
    for i in range(0, len(data), 10):
        origins = data[i:i + 10]
        for j in range(0, len(data), 10):
            destinations = data[j:j + 10]
            #print('matrix shape: ', len(origins), len(destinations))

            tmp_dist_matrix, tmp_dura_matrix = request_info_matrix(origins, destinations)

            # column joint
            if j == 0:
                row_dist_matrix = tmp_dist_matrix
                row_dura_matrix = tmp_dura_matrix
            else:
                row_dist_matrix = pd.concat([row_dist_matrix, tmp_dist_matrix], axis=1)
                row_dura_matrix = pd.concat([row_dura_matrix, tmp_dura_matrix], axis=1)

        # row joint
        if i == 0:
            dist_matrix = row_dist_matrix
            dura_matrix = row_dura_matrix
        else:
            dist_matrix = pd.concat([dist_matrix, row_dist_matrix])
            dura_matrix = pd.concat([dura_matrix, row_dura_matrix])

    def matrix_regularizatoin(matrix):
        # data index regular
        node_index = [node for node in range(len(data))]

        matrix.index = pd.Series(node_index)
        matrix.columns = pd.Series(node_index)

        # take average distance for same nodes pairs
        matrix = (matrix + matrix.T) / 2
        matrix = matrix.astype('int')

        return matrix

    dist_matrix = matrix_regularizatoin(dist_matrix)
    dura_matrix = matrix_regularizatoin(dura_matrix)

    # check_para(dist_matrix)
    dist_matrix.to_csv('./data/distance_matrix.csv')
    dura_matrix.to_csv('./data/duration_matrix.csv')

    return dist_matrix, dura_matrix


#data = [(51.431454, 0.031175), (51.618522, -0.11167), (51.537303, -0.057225), "Aldermans Hill"]
#get_info_matrix(data)
#data = [i for i in range(23)]
#get_info_matrix(data)