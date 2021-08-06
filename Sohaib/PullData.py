import requests
import os
import pickle
import csv
from collections import OrderedDict
import re

def pullData():
    def strip(text):
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        str = ''.join(filter(whitelist.__contains__, text))
        str = str.strip()
        str = str.split(' ')[0]
        return str
    #
    root='./fruits-360/Training/'
    dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
    dirlist = [ strip(dir) for dir in dirlist]
    dirlist = sorted(list(set(dirlist)))


    nutritionalInfo = {}

    search_url = 'https://api.nal.usda.gov/fdc/v1/foods/search?query='
    apiheader = {
        'X-API-Key': '9LhAwEMVIhMjq0pk94i8yo8jr13FEa8JhmqszxW0'
    }

    i=0

    alt_map = {
    'Beetroot': 'Beets',
    'Carambula': 'Carambola',
    'Cocos': 'Coconut',
    'Dates': 'Date',
    'Kaki': 'Persimmon',
    'Limes': 'Lime',
    'Mangostan': 'Mangosteen',
    'Maracuja': 'Passion Fruit',
    'Pepino': 'Pepino Melon',
    'Pepper': 'Bell Pepper',
    'Physalis': 'Physalis peruviana',
    'Pomelo': 'Grapefruit',
    'Salak': 'Kiwi',
    'Tamarillo': 'Tomatillo',
    'Tomato': 'Tomato',
    }

    # Name:Beetroot Search: Clementine, raw
    # Name:Carambula Search: Clementine, raw
    # Name:Cocos Search: Cereal (Malt-O-Meal Coco-Roos)
    # Name:Dates Search: Date
    # Name:Kaki Search: Persimmons, japanese, raw
    # Name:Limes Search: Lime, raw
    # Name:Mangostan Search: Clementine, raw
    # Name:Maracuja Search: Clementine, raw
    # Name:Peach Flat Search: Emu, flat fillet, raw
    # Name:Pepino Search: Clementine, raw
    # Name:Pepper Orange Search: Orange, raw
    # Name:Physalis Search: Tomatillos, raw
    # Name:Physalis with Husk Search: Tomatillos, raw
    # Name:Pitahaya Red Search: Cabbage, red, raw
    # Name:Pomelo Sweetie Search: Clementine, raw
    # Name:Salak Search: Clementine, raw
    # Name:Tamarillo Search: Clementine, raw
    # Name:Tomato Cherry Red Search: Cherries, sour, red, raw
    # Name:Tomato Heart Search: Hearts of palm, raw

    for dir in dirlist:
        fruit_name  = dir if dir not in alt_map else alt_map[dir]
        response = requests.get(search_url+fruit_name+' raw', headers=apiheader)
        nutritionalInfo[dir] = response.json()
        try:
            print(f"{i}/{len(dirlist)}: Name:{dir} Search: {response.json()['foods'][0]['description']}")
        except:
            print(f"{i}/{len(dirlist)}: Error")
        i+=1


    with open('nutritional_info_2.pkl', 'wb') as output:
        pickle.dump(nutritionalInfo, output, pickle.HIGHEST_PROTOCOL)