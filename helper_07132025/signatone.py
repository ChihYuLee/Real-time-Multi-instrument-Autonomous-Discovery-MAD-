import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import scipy 
import math 
import pathlib
from datetime import datetime
import os
import glob
from probe_station import visualization, measurement
import pyvisa
from pymeasure.instruments.keithley import Keithley2400
import matplotlib.cm as cm
from helper_07132025.config import *


class Signatone:

    def __init__(self,x, y, sample_id, stage, **kargs):
        self.x= x
        self.y= y
        self.sample_id= sample_id
        self.stage= stage
        self.probe= PROBE

    def execute(self):
            #check connection 
        try:
            print(self.stage.query("*IDN?"))
        except (pyvisa.errors.VisaIOError, OSError):
            print("Connection lost! Reconnecting...")
            self.stage = pyvisa.ResourceManager().open_resource(STAGE)
        #execute command 
        stat=[]
        print(f'#### Measurement of {self.sample_id} starts #####')
        for i in range(3):
            data= measurement.measure_r(PROBE, self.stage, [self.x, self.y ], self.sample_id)
            stat.append(data)
        data_dict= {'R': np.array(stat)}
        pd.DataFrame(data_dict).to_csv(PROBE_LOCAL_FOLDER+f"{self.sample_id}.csv", index=False)
        print(f"{self.sample_id}.csv is saved")
        return True 

    def get_return_file(self):
        return PROBE_LOCAL_FOLDER+f"{self.sample_id}.csv"



