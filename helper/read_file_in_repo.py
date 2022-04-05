import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from obspy import read
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import dask as dd
import random

def identify_files_in_directory(path):
  """
  Identifing all files in the given directory

  :param path: String, the path
  :return: list, list of all objects in the given path
  """
  return [f for f in listdir(path) if isfile(join(path, f))]
  
def read_data_from_disk(path, files):
  """
  This functions aims to read all the given data from the files list into memory

  :param files: string, the path towards the data
  :param files: list, list of objects to read in
  :return: Array containing the data
  """

  # set variables
  random.seed(0)
  p_win = (.5, 3) # setting limits for P-wave 
  sampling_rate = 31.25
  min_snr = 1.5

  all_raw_data = []
  all_correct_answer = []
  pwave_index = []
  for i in tqdm(files):
      st = read(f'{path}/{i}')
      # cut = int((random.random()*1000)) # change to [0,1000]
      cut = int((random.random()*p_win[1]+p_win[0])*sampling_rate)

      for tr in st:
        snr = sum(abs(tr.data[1000:1032]))/sum(abs(tr.data[1000-32:1000]))
        if "_P" in i and snr<=min_snr:
          print("  Channel: {}, passing, not high-enough P wave S/N ratio ({:4.2f})".format(tr.stats.channel, snr))      
        
        # otherwise do everything
        else:
            tr.data = tr.data[cut:1000+cut]
            all_raw_data.append(tr)
            # set correct answers
            if "_N" in i:
                # if it is a noise segment, the correct answer is None
                correct_answer = None
                pwave_ix = None
            else:
                # if it is a p-wave segment, the correct answer is the p-wave arrival time (UTCDateTime format)
                correct_answer = tr.stats.starttime + (1000-cut)/tr.stats.sampling_rate
                pwave_ix = int(1000-cut)
            all_correct_answer.append(correct_answer)
            pwave_index.append(pwave_ix)

      del st
  all_correct_answer = np.array(all_correct_answer)
  return all_raw_data, all_correct_answer, pwave_index