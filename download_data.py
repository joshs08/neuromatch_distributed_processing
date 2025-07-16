from pathlib import Path
import json
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
from rastermap import Rastermap
from tqdm import tqdm

root = "C://Users//Josh Selfe//OneDrive - Nexus365//Other Documents//Neuromatch"


# download files from Figshare
#file_ID = [54866333, 54183860] # IDs of files
file_ID = [54184673]
BASE_URL = 'https://api.figshare.com/v2'
r = requests.get(BASE_URL + '/articles/' + str(28811129)) # 28811129 is the ID of the whole dataset
file_metadata = json.loads(r.text)
for file in file_metadata['files']:
  if file['id'] in file_ID: # only download files included in file_ID
    fn = os.path.join(root, file['name'])
    if not os.path.isfile(fn):
      response = requests.get(f"{BASE_URL}/file/download/{file['id']}", stream=True, timeout=10)
      total_size = int(response.headers.get('content-length', 0))
      with open(fn, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=file['name']) as pbar:
          for chunk in response.iter_content(chunk_size=8192):
              if chunk:
                  f.write(chunk)
                  pbar.update(len(chunk))
