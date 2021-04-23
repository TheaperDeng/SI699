import numpy as np
import netCDF4 as nc
import os
import calendar

import random
import pickle

# global var, need initialzation
DATASET = dict()

# The org provides multiple datasets.
# Change these global variables to accomodate new datasets.
DATASET_PREFIX = "Complete_TAVG_Daily_LatLong1_"
DATASET_BEGIN = 1880
DATASET_END = 2020
DATASET_INTERVAL = 10
MonthDays = dict()
DIR="./res/"
TemperatureRecord=[]

for i in range(DATASET_BEGIN, DATASET_END + 10, 1):
    for j in range(12):
        MonthDays[str(i) + "-" + str(j + 1)] = calendar.monthrange(i, j + 1)[1]

def loadDatasets(decade=None):
    global DATASET

    if not decade:
        for i in range(DATASET_BEGIN, DATASET_END, DATASET_INTERVAL):
            fileName = "{prefix}{decd}.nc".format(prefix=DATASET_PREFIX, decd=i)
            DATASET[str(i)] = nc.Dataset(fileName, encoding="utf-8")
    else:
        try:
            fileName = "{prefix}{decd}.nc".format(prefix=DATASET_PREFIX, decd=decade)
            DATASET[str(decade)] = nc.Dataset(fileName, encoding="utf-8")
        except:
            raise NameError("Wrong input parameters for func loadDatasets()")


def getCoord(lat, lng):
    # eg. Coord for New York: 40.7128° N, 74.0060° W
    # input format: 41N, 74W
    # output format: target indices for the fields that require lat/long, such as teperature/climatology
    latIdx = lngIdx = 0

    if lat.endswith("N") or lat.endswith("n"):
        latIdx = 90 + int(lat[:-1])
        if latIdx == 180:
            latIdx = 179
    elif lat.endswith("S") or lat.endswith("s"):
        latIdx = 90 - int(lat[:-1])
    else:
        raise NameError("Wrong input parameters for func getCoord() in lat")

    if lng.endswith("E") or lng.endswith("e"):
        lngIdx = 180 + int(lng[:-1])
        if lngIdx == 360:
            lngIdx = 359
    elif lng.endswith("W") or lng.endswith("w"):
        lngIdx = 180 - int(lng[:-1])
    else:
        raise NameError("Wrong input parameters for func getCoord() in lng")

    return latIdx, lngIdx


def getHistoricalForCoord(lat, lng, freq="day", interval=None):
    # freq: enum: "day"/"week"/"month"
    # returns the mean of the given freq

    # interval: time interval. eg. 1990 / 1950-2000
    # if None, return all records

    # lat, lng: eg. 41N, 74W

    # return: A list of float numbers

    monthInterval = False
    numDays = 0
    if freq.lower() == "day":
        numDays = 1
    elif freq.lower() == "week":
        numDays = 7
    elif freq.lower() == "month":
        monthInterval = True
    elif freq.lower() == "year":
        numDays = 366
    else:
        print("Unrecognized freq in func getHistoricalForCoord. Proceeding with freq=day.")
        numDays = 1

    latIdx, lngIdx = getCoord(lat, lng)

    # record = []
    timeStart, timeEnd = 0, 0
    if not interval:
        timeStart, timeEnd = DATASET_BEGIN, DATASET_END
    else:
        timeInterval = interval.split("-")
        if len(timeInterval) == 1:
            timeStart = int(timeInterval[0])
            timeEnd = int(timeInterval[0]) + 10
        else:
            timeStart, timeEnd = int(timeInterval[0]), int(timeInterval[1]) + 10

    # Store the number of days in each of the months
    if monthInterval:
        monthDays = MonthDays

    # print(monthDays)

    ret = []
    for i in TemperatureRecord:
        if numDays == 1:
            ret.append(i[:, latIdx, lngIdx].filled(fill_value=0))
        elif monthInterval:
            prev = 0
            currYear = timeStart
            currMonth = 1
            dataArray = i[:, latIdx, lngIdx].filled(fill_value=0)
            dataLength = dataArray.shape[0]
            while True:
                duration = monthDays[str(currYear) + "-" + str(currMonth)]
                ret.append(np.mean(dataArray[prev:prev + duration]))
                prev += duration

                if prev >= dataLength:
                    break

                currMonth += 1
                if currMonth == 13:
                    currMonth = 1
                    currYear += 1
        else:
            idx = 0
            dataArray = i[:, latIdx, lngIdx].filled(fill_value=0)
            dataLength = i.shape[0]
            while idx * numDays < dataLength:
                # min used here to avoid index problems
                ret.append(np.mean(dataArray[idx * numDays:min(dataLength, (idx + 1) * numDays)]))
                idx += 1

    if numDays == 1:
        ret = list(np.concatenate(ret))

    return ret


def getHistoricalForAllCoord(freq="month", timeInterval=None, CoordInterval=45):
    # Must be int
    if type(CoordInterval) != type(2):
        raise ValueError("Wrong parameter CoordInterval in func getHistoricalForAllCoord")

    # only output the files that are not in the dir
    # currFiles = set(os.listdir("./"))

    currFiles = set(os.listdir(DIR))
    for lat in range(0, 90, CoordInterval):
        x = str(lat) + "N"

        for lng in range(0, 180, CoordInterval):
            y = str(lng) + "E"
            key = x + "-" + y

            fileName = "{}.pkl".format(key)
            if fileName not in currFiles:
                print("Current Coord: ", key)
                res = getHistoricalForCoord(x, y, freq, timeInterval)

                fileName = DIR + fileName
                f = open(fileName, 'wb')
                pickle.dump(res, f, 0)
                f.close()

        for lng in range(0, 180, CoordInterval):
            y = str(lng) + "W"
            key = x + "-" + y

            fileName = "{}.pkl".format(key)
            if fileName.strip() not in currFiles:
                print("Current Coord: ", key)
                res = getHistoricalForCoord(x, y, freq, timeInterval)

                fileName = DIR + fileName
                f = open(fileName, 'wb')
                pickle.dump(res, f, 0)
                f.close()

        x = str(lat) + "S"

        for lng in range(0, 180, CoordInterval):
            y = str(lng) + "E"
            key = x + "-" + y

            fileName = "{}.pkl".format(key)
            if fileName not in currFiles:
                print("Current Coord: ", key)
                res = getHistoricalForCoord(x, y, freq, timeInterval)

                fileName = DIR + fileName
                f = open(fileName, 'wb')
                pickle.dump(res, f, 0)
                f.close()

        for lng in range(0, 180, CoordInterval):
            y = str(lng) + "W"
            key = x + "-" + y

            fileName = "{}.pkl".format(key)
            if fileName not in currFiles:
                print("Current Coord: ", key)
                res = getHistoricalForCoord(x, y, freq, timeInterval)

                fileName = DIR + fileName
                f = open(fileName, 'wb')
                pickle.dump(res, f, 0)
                f.close()


if __name__ == '__main__':
    loadDatasets()

    for i in range(DATASET_BEGIN, DATASET_END + 10, 1):
        for j in range(12):
            MonthDays[str(i) + "-" + str(j + 1)] = calendar.monthrange(i, j + 1)[1]

    for i in range(DATASET_BEGIN, DATASET_END, DATASET_INTERVAL):
        TemperatureRecord.append(DATASET[str(i)].variables['temperature'])

    getHistoricalForAllCoord(freq="day",CoordInterval=1)
    print("Complete.")