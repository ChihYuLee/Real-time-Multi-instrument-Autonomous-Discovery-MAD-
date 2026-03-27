#live run operation
LIVE= 'SEMI'

#communication 
HEADER_SIZE = 16
CLIENT_ADDRESS = '10.175.219.108'
PORT_PROBE = 1027
PORT_XRD= 1026 #11452
SERVER_PROBE= '10.229.46.106' #'127.0.0.1'   
SERVER_XRD= '10.206.38.223' #'127.0.0.1'   

#mesage 
ENCODE_FORMAT = 'utf-8'
DISCONNECT_MSG = "@@DISCONNECT"
GADDS_MSG = "@@GADDS"
COMMAND_MSG = "@@COMMAND"
SLM_MSG='@@SLM'
PROBE_MSG='@@PROBE'


#Signatone parameters 
STAGE= 'GPIB1::5::INSTR'
PROBE= 'GPIB0::2::INSTR'

# target xy file: modify when populated 
XRD_TARGET= '/Users/chihyulee/work/multi-instr/HetGP/notebooks/live_run/xrd_target.txt'
PROBE_TARGET= '/Users/chihyulee/work/multi-instr/HetGP/notebooks/live_run/probe_target.txt'

# semi-live folder
XRD_FOLDER_SIM= '/Users/chihyulee/work/multi-instr/HetGP/notebooks/live_run/XRD_sim/'
PROBE_FOLDER_SIM= '/Users/chihyulee/work/multi-instr/HetGP/notebooks/live_run/Probe_sim/'

# full-live folder 
XRD_FOLDER= '/Users/chihyulee/work/multi-instr/HetGP/notebooks/live_run/xrd_live/' 
PROBE_FOLDER=  '/Users/chihyulee/work/multi-instr/HetGP/notebooks/live_run/probe_live/' 

# local computer to save data 
XRD_LOCAL_FOLDER= '/Users/chihyulee/work/multi-instr/HetGP/notebooks/live_run/fake_xrd_local/'
PROBE_LOCAL_FOLDER= '/Users/chihyulee/work/multi-instr/HetGP/notebooks/live_run/fake_probe_local/'

#C2 parameters
GADDS_DIR = "C:\\SAXI\\GADDSnew\\"
FRAME=3
DUMMY_DIR=  XRD_LOCAL_FOLDER
SCAN_TIME= 300*FRAME #15 mins for one scan
PROBE_TIME= 100 #1 mins for one scan 
BUFFER= 100 #time for waiting the files 
CHECK_INTERVAL= 5 #seconds 
SLM_REF= '/Users/chihyulee/work/multi-instr/HetGP/notebooks/live_run/template.slm'
TMP = "tmp"
DEBUG = True
BACKLOG = 0
XRD_Z_LIMIT=30
