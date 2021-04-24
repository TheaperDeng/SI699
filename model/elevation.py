# Top level function: get_elevation

bing_maps_api = "AuM-1joRIYWNcSlskBPLyxhsmjETsE5MAT-hHs8LyggV92IhUA0K1zRdQYRakVsg"

REQUEST_CACHE_FILE = "../cache/elevation_request_result.json"
CACHE_FILE = "../cache/elevation.json"

import json
import requests


PROXY = {
}
# PROXY = None


# regularize the number to WGS84, may not be used
def lat_lng_regularization(data:list)->list:
    return


# generate the cache key, data = [(lat, lng), (lat2, lng2)...]
def elevation_cache_key(data:list)->str:
    return "||".join([str(i[0]) + "+" + str(i[1]) for i in data])


# return json if successful, else return None
# accept a list of [(lat, lng), (lat2, lng2)...]
# need WGS84 decimal degrees. (Example: 34.2412,-119.3829)
def api_request(data:list)->json:
    try:
        with open(REQUEST_CACHE_FILE, 'r') as cache_file:
            cache_contents = cache_file.read()
            cache_dict = json.loads(cache_contents)
    except:
        cache_dict = {}

    key = elevation_cache_key(data)

    if key in cache_dict.keys():
        return cache_dict[key]
    else:
        base_url = "http://dev.virtualearth.net/REST/v1/Elevation/List?"
        lat_long_pair = "points=" + ",".join([str(i[0])+","+str(i[1]) for i in data])
        request_url = base_url + lat_long_pair + "&key=" + bing_maps_api

        resp = requests.get(url=request_url, timeout=10, proxies=PROXY).text
        cache_dict[key] = resp
        dumped_str = json.dumps(cache_dict)
        with open(REQUEST_CACHE_FILE, "w") as fw:
            fw.write(dumped_str)
        return resp


# the function to call
def get_elevation(data:list)->list:
    '''
    data = [(lat1, lng1), (lat2, lng2), ...]
    return: list(str) = [elevation1|None, ...]
    '''
    try:
        with open(CACHE_FILE, 'r') as cache_file:
            cache_contents = cache_file.read()
            cache_dict = json.loads(cache_contents)
    except:
        cache_dict = {}

    res = []
    for i in data:
        key = elevation_cache_key([i])
        if key in cache_dict.keys():
            res.append(cache_dict[key])
        else:
            ret_json = json.loads(api_request(data=[i]))
            try:
                elevation = ret_json["resourceSets"][0]["resources"][0]["elevations"][0]
            except:
                elevation = None
            
            res.append(elevation)
            cache_dict[key] = elevation

    dumped_str = json.dumps(cache_dict)
    with open(CACHE_FILE, "w") as fw:
        fw.write(dumped_str)
    return res


if __name__ == "__main__":
    # test_data = [(35.89431,-110.72522), (35.89393,-110.72578), (35.89374,-110.72606), (35.89337,-110.72662)]
    # print(get_elevation(test_data))
    country_ele = {}
    with open('country_loc_dict.json', 'r') as f:
        country_dict = json.load(f)
    for name, loc in country_dict.items():
        country_ele[name] = get_elevation([(loc['lat'], loc['lon'])])
        print(name, ":", country_ele[name])
    with open('country_ele_dict.json', 'w') as f:
        f.write(json.dumps(country_ele))