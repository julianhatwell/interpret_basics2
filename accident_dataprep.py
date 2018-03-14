import zipfile as zf

import pandas as pd
import io, os, csv, fnmatch

accident_dir = 'c:\\Dev\\Study\\Python\\accident\\'
years = ['2016', '2015'] # , '0514'
pattern = '*.zip'
dataframes = {}

for year in years:
    acc_yr_dir = accident_dir + year + '\\'
    dir_contents = os.listdir(acc_yr_dir)

    filenames = []

    for filename in dir_contents:
        if fnmatch.fnmatch(filename, pattern):
            filenames.append(acc_yr_dir + filename)
    print(filenames)

    dataframes[year] = {}
    for filename in filenames:
        zfile = zf.ZipFile(filename, 'r')
        for fname in zfile.namelist():
            if 'Make' in fname:
                shortname = 'Make'
            elif 'Accidents' in fname:
                shortname = 'Acc'
            elif 'Veh' in fname:
                shortname = 'Veh'
            elif 'Cas' in fname:
                shortname = 'Cas'
            fcontent = zfile.open(fname)
            dataframes[year][shortname] = pd.read_csv(fcontent, header=0, delimiter=",", na_values = -1, dtype = {'make': 'str', 'model': 'str'})
            print("imported " + filename + " as " + shortname)
            zfile.close()

frames = [dataframes[yy]['Acc'] for yy in years]
Acc = pd.concat(frames)
frames = [dataframes[yy]['Cas'] for yy in years]
Cas = pd.concat(frames)
frames = [dataframes[yy]['Veh'] for yy in years]
Veh = pd.concat(frames)
frames = [dataframes[yy]['Make'] for yy in years]
Make = pd.concat(frames)
