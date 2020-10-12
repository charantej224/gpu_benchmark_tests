import sys, os, time, datetime, imp, json
#ibm_db, ibm_db_dbi
import pandas as pd
import numpy as np
import requests, re
#jaydebeapi
#dsx_core_utils
#from sqlalchemy import *
#from sqlalchemy.types import String, Boolean
#import cx_Oracle
#import pymssql

#transform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder,Normalizer,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, FeatureHasher
from sklearn.compose import ColumnTransformer

#v3.0
#from keras.models import Sequential
#from keras.optimizers import Adam
#from keras.layers import Dense,Dropout
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.utils import np_utils

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Activation, Input, Bidirectional, concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# pipeline
from sklearn.pipeline import make_pipeline, Pipeline
#from sklearn.externals import joblib

#v3.4.3
from smtplib import SMTP
from email.mime.text import MIMEText
import logging 

#v3.4.4.6
#os.system('pip install keras_metrics')
#import keras_metrics

#user_id = os.environ.get("DSX_USER_ID", "990")
#dataSource = dsx_core_utils.get_data_source_info(dataSet['datasource'])

####Daqi data cleansing section#######
verbose=False

# define baseline model
#v3.0 changed drop out

ncomp = 3072
#ncomp = 1024

logger_name = f"{os.getenv('APP_LOG')}.{__name__}" if os.getenv("APP_LOG") else __name__
my_logger = logging.getLogger(logger_name)

def load_model(project_path, model_name):
  
  #custom class to return array
  class mtxtoarray():
      
      def fit(self, x, y=None):
          return self
  
      def transform(self, x):
          return x.toarray().astype(np.float32)
  
  #model_name = "FaultAlarmPartsPredictor"
  model_parent_path = project_path + "/models/" + model_name + "/"
  metadata_path = model_parent_path + "metadata.json"
  
  # fetch info from metadata.json
  with open(metadata_path) as data_file:
      meta_data = json.load(data_file)
  
  version = 'latest'
  
  # if latest version, find latest version from  metadata.json
  if (version == "latest"):
      version = meta_data.get("latestModelVersion")
  
  # prepare model path using model name and version
  model_path = model_parent_path + str(version) + "/model"
  
  serialization_method = "joblib"
  
  # load model
  if serialization_method == "joblib":
      model = joblib.load(open(model_path, 'rb'))
  elif serialization_method == "pickle":
      model = pickle.load(open(model_path, 'rb'))
      
  return model



  
def load_trams(query,datasetname):
  return load_dataset(datasetname, query)
  

def load_data(query, datasetname):
    return load_dataset(datasetname, query)


#date subtrace for fileoutputs
def subdate(yearmonthday, days):
    yearmonthday = datetime.datetime.strptime(yearmonthday, '%Y%m%d') - datetime.timedelta(1)
    yearmonthday = str(yearmonthday.date())
    yr,mth,day = yearmonthday[:-6], yearmonthday[5:-3], yearmonthday[8:] 
    #+ yearmonthday[5:-3]
    return yr,mth,day

#load files
def groot_load(targettable,filename,datasetname,ydate2,pjpath):

  db_dict = retrieve_db_info("groot")
  
  #Enter the values for you database connection
  dsn_database = db_dict.get("db_name")           # e.g. "BLUDB"
  dsn_hostname = db_dict.get("db_host") # e.g.: 
  dsn_port = "50000"                  # e.g. "50000" 
  dsn_schema = "ICPD"
  #dsn_protocol = "TCPIP"              # i.e. "TCPIP"
  dsn_uid = db_dict.get("db_user")        # e.g. ""
  dsn_pwd = db_dict.get("db_password")    # e.g. ""

  dsn = (
      "DATABASE={0};"
      "HOSTNAME={1};"
      "PORT={2};"
      "PROTOCOL=TCPIP;"
      "UID={3};"
      "PWD={4};").format(dsn_database, dsn_hostname, dsn_port, dsn_uid, dsn_pwd)
  conn = ibm_db.connect(dsn, "", "")

  #####delete records for day if exist#######
  # query
  query = 'delete from ' + dsn_schema + '.' + targettable + ' where date_value = ' + '\'' + ydate2 + '\''
  #print(query)

  # run direct SQL
  now = time.time()
  stmt = ibm_db.exec_immediate(conn, query)
  #print("Number of affected rows: ", ibm_db.num_rows(stmt))
  #conn.close()

  later = time.time()
  difference = int(later - now)
  #print('sql run time: ' + str(difference))
  #########

  ####load records########
  #dbload connection details
  #v3.4.3.5 set maxerrors=0 and > /dev/null...unwanted stdout
  dbstr = (
      "/opt/dbload/dbload "
      "-host {1} "
      "-db {0} "
      "-schema ICPD "
      "-u {3} "
      "-pw {4} "
      "-t {6} "
      "-delim ',' "
      "-skipRows 1 "
      "-maxErrors 0 "
      "-outputDir {5} "
      "-df ").format(dsn_database, dsn_hostname, dsn_port, dsn_uid, dsn_pwd, pjpath, targettable)

  #load string
  dbcmd = dbstr + filename + ' 2>&1 > /dev/null'

  now = time.time()
  result_code = os.system(dbcmd)
  #if result_code!=0:
      #print('load error: '+ str(result_code))
  #else:
      #print('load success!!!')

  later = time.time()
  difference = int(later - now)
  #print('load run time: ' + str(difference))

#v3.3.1
#note: need to consolidate
def groot_realtime_load(targettable,filename,datasetname,data_path):

  db_dict = retrieve_db_info("groot")
  
  #Enter the values for you database connection
  dsn_database = db_dict.get("db_name")           # e.g. "BLUDB"
  dsn_hostname = db_dict.get("db_host") # e.g.: 
  dsn_port = "50000"                  # e.g. "50000" 
  dsn_schema = "ICPD"
  #dsn_protocol = "TCPIP"              # i.e. "TCPIP"
  dsn_uid = db_dict.get("db_user")        # e.g. ""
  dsn_pwd = db_dict.get("db_password")    # e.g. ""
    
  dsn = (
      "DATABASE={0};"
      "HOSTNAME={1};"
      "PORT={2};"
      "PROTOCOL=TCPIP;"
      "UID={3};"
      "PWD={4};").format(dsn_database, dsn_hostname, dsn_port, dsn_uid, dsn_pwd)
  conn = ibm_db.connect(dsn, "", "")

  ####load records########
  #dbload connection details
  #v3.4.3.5 set maxerrors=0 and > /dev/null...unwanted stdout
  dbstr = (
      "/opt/dbload/dbload "
      "-host {1} "
      "-db {0} "
      "-schema ICPD "
      "-u {3} "
      "-pw {4} "
      "-t {6} "
      "-delim ';' "
      "-skipRows 1 "
      "-maxErrors 0 "
      "-quotedValue DOUBLE "
      "-outputDir {5} "
      "-df ").format(dsn_database, dsn_hostname, dsn_port, dsn_uid, dsn_pwd, data_path, targettable)

  #load string
  dbcmd = dbstr + filename + ' 2>&1 > /dev/null'

  now = time.time()
  result_code = os.system(dbcmd)
  
  records = ''
  #if result_code!=0:
    #print('load error: '+ str(result_code))
  #else:
  #  print('load/update success!!!')

  later = time.time()
  difference = int(later - now)
  #print('sql run time: ' + str(difference))
  #########
  
  return result_code

#6+ months netcool alarm history for training/test
def load_alarm_history(datasetname):
    return load_dataset(datasetname)

def load_chat_history(query, dbsource, datasetname):
  
    # dfp = load_chat_history(chat_query,'chat','chat_sample')
    if dbsource == "chat":
      #load chat history from CHAT MSSQL Server!
      dataset = datasetname if datasetname else "part_history"
    else:
      raise NameError('dbsource')
        
    #daqi code below, incorrect source...chat is always MSSQL, we don't have any chat data on groot
    '''
    if dbsource == "groot":
        dataset = datasetname if datasetname else "chat_sample"
    elif dbsource == "groot_dev":
        dataset = datasetname if datasetname else "chat_sample_dev"
    elif dbsource == "chat":
        #load chat history from CHAT MSSQL Server!
        dataset = datasetname if datasetname else "part_history"
    else:
        raise NameError('dbsource')
    '''
    
    return load_dataset(dataset, query)
    
#v2.0 new add to batch score
#v3.0 update
def get_nparts(dfp1,dfa1,nparts,partslist):
    
    #merge part df with alarms to get frequency of ea. part
    dfpa = dfp1.merge(pd.DataFrame({"TICK_NBR": dfa1['TICK_NBR'].unique()}), on='TICK_NBR', how='inner')[['TICK_NBR', 'IMN_NO']]
    
    if partslist:
        #print('existing static parts list')
        dfpa = dfpa[dfpa['IMN_NO'].isin(partslist)].copy()
        dfpf = pd.DataFrame({'IMN_NO' : dfpa.drop_duplicates().groupby(['IMN_NO']).count()['TICK_NBR'].nlargest(nparts).index})
    else:
        #print('auto generating top ' + str(nparts) + ' list')
        dfpf = pd.DataFrame({'IMN_NO' : dfpa.drop_duplicates().groupby(['IMN_NO']).count()['TICK_NBR'].nlargest(nparts).index})

    #add frequency to parts for future reference
    dfpf = dfpf.merge(dfpa.groupby('IMN_NO').count(), on='IMN_NO', how='inner').rename(index=str, columns={'TICK_NBR': 'FREQ'})

    # select alarms with parts filterd by nparts above setting
    dfpre = dfpa.merge(dfpf, on='IMN_NO', how='inner')

    #distinct tick_nbr, imn_no. identifey # of times parts was used for ea. ticket
    dfpre = dfpre.groupby(['TICK_NBR','IMN_NO'])\
                   .agg({'IMN_NO':'count'})\
                   .rename(columns={'IMN_NO':'COUNT'})\
                   .reset_index(drop=False)

    # setting values to 1
    # NEED TO CHECK IF > 1 is CORRECT HERE!!!!!
    dfpre.loc[dfpre['COUNT'] > 1, 'COUNT'] = 1
    
        # Pivot to turn tick_nbr into index and part numbers to columns, counts as values
    dfl = dfpre.pivot(index='TICK_NBR', columns='IMN_NO', values='COUNT')\
                    .fillna(0)\
                    .astype(int)\
                    .reset_index(drop=False)
    
    #v2.0 batch score only!
    #v3.0 bug fix, added partlist loop only if partlist exists
    #add part to df if doesn't exist and set to 0
    if partslist:
        for i in partslist:
            if i not in dfl.columns.tolist():
                dfl[i] = 0
                dfl[i] = dfl[i].astype('int')
                
        #fix sort order here
        dfl = dfl[['TICK_NBR'] + partslist]
    
    # Name columns with a P in front 
    dfl = dfl.rename(columns=dict(zip(dfl.columns[1:], ["P{}".format(part) for part in dfl.columns[1:] ])))
    
    '''
    if not partslist:
        #NOTE: only save these files for pyscript, model create/export should be done from pyscript job only!
        #v2.0
        #print('Warning: creating new part list files...')
        dfl.to_csv(os.getenv("DSX_PROJECT_DIR") + '/datasets/' + 'parts_per_ticket.csv', index=False, header=True)
        dfpf.to_csv(os.getenv("DSX_PROJECT_DIR") + '/datasets/' + 'topn_parts_freq.csv', index=False, header=True)
    '''
    
    return dfl, dfpf

def load_atlas(dbsource,datasetname):
  return load_dataset(datasetname)


def load_netcool_history(query,dbsource,datasetname):
    if dbsource.lower() == "groot":
        dataset = datasetname if datasetname else "netcool_history_sample"
    elif dbsource.lower() == "groot_dev":
        #dataset = datasetname
        dataset = datasetname if datasetname else "netcool_history_sample_dev"
    elif dbsource.lower() == "fms":
        dataset = datasetname if datasetname else "fms_alarm_history"
    else:
        raise NameError('dbsource')
        
    return load_dataset(dataset, query)   
        

# define baseline model
#v3.4.4.6
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(1024, input_dim=ncomp, activation='relu'))
    model.add(Dropout(rate = 0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate = 0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate = 0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate = 0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(rate = 0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(rate = 0.3))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    #adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy',keras_metrics.precision(), keras_metrics.recall(),keras_metrics.f1_score()])
    return model

#v3.1
#new add
def replace_multiples(string, replacements):
    """
    Given a string and a replacement map, it returns the replaced string.

    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :param dict replacements: replacement list [value to be replaced with space]
    :re_flags can be re.M, re.I and their combinations re.I | re.M 
    :rtype: str
     remove the flags, got errors sometime. 
    """
    if isinstance(replacements, list):
        replacements = {key : "" for key in replacements}
    # Place longer ones first to keep shorter substrings from matching
    # where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against 
    # the string 'hey abc', it should produce 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0)], string)

#v3.1
#new add
def extract_keepitems(str, keepitems=[]):
    regexp = re.compile('|'.join(map(re.escape, keepitems)))
    keeplist = regexp.findall(str)
    keeplist.insert(0, ' ')
    tmpstr = " ".join(keeplist) 
    return tmpstr

#v3.1
#new add
def remove_substring(row):
    s1 = row["s1"]
    sub_list = row["s2"].split(" ")
    return replace_multiples(s1, sub_list)

#v3.1
#added keeps (daqi)
#removed ALL major cleansing steps that impacted performance
#new add
def remove_keepitems(s1, s2):
    df = pd.DataFrame({"s1": s1.values, "s2":s2.values})
    df.index = s1.index
    df["s3"] = df.apply(remove_substring, axis=1)
    return df["s3"] 

#v3.1
#modify by adding/processing extra keeps list.
def clean_tokens(sr, keeps=[], verbose=False):

    if keeps:
        tmpsr  = sr.str.lower().apply(extract_keepitems, keepitems=keeps)  
        sr = remove_keepitems(sr.str.lower(), tmpsr)
    
    #remove leading, trailing white space
    sr = sr.str.lower()
    sr = sr.str.strip()
    
    if keeps: 
        return tmpsr.str.replace(' ', '')
    else:
        return sr

def separate_tokens(sr, tokens, verbose=False):
    for index, token in enumerate(tokens):
        if verbose: print('SEPARATING TOKEN {0:>3d} of {1:>3d}. \t TOKEN: {2:<20s}'.format(index+1, len(tokens), token), end='\r', flush=True)
        pat = r'(?<=[a-z]){}(.*)'.format(token)
        rep = r' {}\1'.format(token)
        sr = sr.str.replace(pat=pat, repl=rep)
    return sr

def replace_tokens(sr, fixes, verbose=False):
    for index, (pat, rep) in enumerate(fixes):
        if verbose: print('REPLACING TOKEN {0:>3d} of {1:>3d}. \t TOKEN: {2:<20s}'.format(index+1, len(fixes), rep), end='\r', flush=True)
        sr = sr.str.replace(pat=pat, repl=rep)
        #sr = sr.replace(to_replace = pat,value = rep)
    return sr 
    
#v3.1
#v3.2
#modify by adding extra keeps list and invoking clean_tokens
# This is only used for ['HOSTNAME', 'NODEALIAS', 'ALERTKEY', 'ALERTNAME', 'ALERTGROUP']
def process_text_column(df, col, tokens=[], fixes=[], keeps=[], verbose=False, padding=None):
    
    sr = df[col].copy()
    sr = clean_tokens(sr, keeps, verbose)  #completing lower and stripping and shrinking spaces
    if fixes:
        sr = replace_tokens(sr, fixes, verbose)
    if tokens:
        sr = separate_tokens(sr, tokens, verbose)

    if padding:
        sr = sr.str.pad(padding, side='right', fillchar=" ")      

    return sr 

#v3.4.4.5 cpri port logic add
def fix_freqband(row, verbose=False):
    
    #rrh to samsung band lookups
    samsung_band_cdma = {'rrh040':'800','rrh100':'1900'}
    samsung_band_cdu = {'rrh0100':'800','rrh060':'800','rrh080':'800'}
    
    #v3.4.4.5 cpri keys
    samsung_band = {'rrh000': '1900',
    'rrh010': '1900',
    'rrh0110': '800',
    'rrh020': '1900',
    'rrh030': '800',
    'rrh050': '800',
    'rrh070': '800',
    'rrh090': '800',
    'rrh110': '1900',
    'rrh120': '1900',
    'rrh140': '2.5G',
    'rrh180': '2.5G',
    'rrh200': '800',
    'rrh210': '800',
    'rrh220': '800',
    'rrh230': '800',
    'rrh240': '800',
    'rrh250': '800',
    'rrh040': '1900',
    'rrh100': '2.5G',
    'rrh0100': '2.5G',
    'rrh060': '2.5G',
    'rrh080': '2.5G',
    'ecp0cpriport0': '1900',
    'ecp0cpriport1': '1900',
    'ecp0cpriport2': '1900',
    'ecp0cpriport3': '1900',
    'ecp0cpriport4': '1900',
    'ecp0cpriport5': '1900',
    'ecp0cpriport7': '1900',
    'ecp0cpriport9': '1900',
    'ecp0cpriport11': '1900',
    'ecp0cpriport6': '800',
    'ecp0cpriport8': '800',
    'ecp0cpriport10': '800',
    'ecp1cpriport0': '2.5G',
    'ecp1cpriport1': '2.5G',
    'ecp2cpriport0': '2.5G',
    'ecp1cpriport4': '2.5G',
    'ecp1cpriport5': '2.5G',
    'ecp2cpriport4': '2.5G',
    'ecp1cpriport8': '2.5G',
    'ecp1cpriport9': '2.5G',
    'ecp2cpriport8': '2.5G'}

    s1 = row['IDENTIFIER'] 
    s1 = s1.lower().replace('_','')
    #v3.4.4.5 cpri port logic add, use rrh as primary
    s1 = s1.lower().replace('/','')
    s1 = s1.lower().replace('[','')
    s1 = s1.lower().replace(']','')
    s2 = row['FREQBAND']

    #samsung logic, update alertkey rrh*
    
    #v3.4.4.5 cpri port logic add, use rrh as primary
    if 'samsung' in s1 and re.search(r'ecp\d+cpriport\d+',s1):
        
        rrhstr = re.findall(r'ecp\d+cpriport\d+',s1)[0]
        
        if rrhstr in samsung_band.keys():
            s2 = samsung_band[rrhstr]

    #v3.4.3 bug fix!
    if 'samsung' in s1 and re.search(r'rrh\d+',s1):
        
        rrhstr = re.findall(r'rrh\d+',s1)[0]

        if rrhstr in samsung_band.keys():
            s2 = samsung_band[rrhstr]

        if 'cdma' in s1 and rrhstr in samsung_band_cdma.keys():
            s2 = samsung_band_cdma[rrhstr]

        if 'cdu30' in s1 and rrhstr in samsung_band_cdu.keys():
            s2 = samsung_band_cdu[rrhstr]
    

    s3 = row['SUMMARY']
    s3 = s3.lower()
            
    #alu logic
    #faulty_cells 25,26,27 800
    #v3.3
    #v3.4.4.5 - added faultycells OR condition
    #if ('nsn' in s3) and ('degraded_cells' in s3):
    if ('nsn' in s3) and (('degraded_cells' in s3) or ('faulty_cells' in s3)):
      
        if re.search(r'lte:25',s3) or re.search(r'lte:26',s3) or re.search(r'lte:27',s3):
            s2 = '800'
        else:
            s2 = '1900'
            
    return s2

#3.3
#v3.4 updated logic!
def fix_oem(row, verbose=False):
    
    s1 = row['ALERTNAME']
    s1 = s1.lower()
    s2 = row['OEMMARKETVENDOR']
    
    if 'nsn-snmp-nbi' in s1 or 'ecp_luc' in s1:
        s2 = 'Alcatel Lucent'
        
    if 'alu_enb' in s1 or 'alu_csr' in s1:
        s2 = 'Alcatel Lucent'
    
    if 'samsung' in s1:
        s2 = 'Samsung'

    if 'ericsson' in s1 or '_cems_' in s1:
        s2 = 'Ericsson'
        
    return s2


# FOR CLEANING MGR Notes
#v3.2
mgrnotes_tokens = ['intermittent', 'unreachable', 'operational', 'performance', 'breakerpart', 'maintenance', 'necessary', 'permanent', 'microwave', 'generator', 
                   'connector', 'portable', 'multiple', 'previous', 'backhaul', 'standing', 'proofing', 'unknown', 'chronic', 'offline', 'digital', 'correct', 
                   'antenna', 'problem', 'traffic', 'process', 'weather', 'passing', 'replace', 'persist', 'outage', 'active', 'unable', 'online', 'stable', 
                   'relate', 'supply', 'triage', 'window', 'sector', 'module', 'output', 'adjust', 'return', 'reset', 'clean', 'prior', 'optic', 'power', 
                   'fault', 'clear', 'macro', 'spare', 'close', 'light', 'relay', 'cable', 'fiber', 'issue', 'event', 'cause', 'storm', 'radio', 'order', 
                   'metro', 'input', 'clock', 'check', 'open', 'gain', 'cdma', 'evdo', 'dead', 'high', 'temp', 'mini', 'good', 'next', 'vswr', 'fail', 'time', 
                   'csms', 'step', 'root', 'port', 'cell', 'site', 'unit', 'test', 'miss', 'down', 'loss', 'all', 'oos', 'low', 'lte', 'rrh', 'bad', 'far', 'cmc', 
                   'bcp', 'das', 'ac', '3g', '4g', '5g', 'tx', 'hw']


mgrnotes_fixes = [('faulty', 'fault'), ('vender', 'vendor'),('connector','connecter'), ('weather proofed','weatherproofing'),('mini-macro','mini')]
mgrnotes_tokens.sort(key=lambda x: len(x), reverse=True)
mgrnotes_keepitems = []

#v3.2
#v3.3
#v3.4 added mtype
#modify the globals, and invoking process_text_column with extra keeps list
def transform_alarms_dataframe(df, mtype, verbose=False):
    
    global alertkey_keepitems, alertname_keepitems, alertgroup_keepitems
               
    cleaned_df = df.copy()
    
    total_nans = cleaned_df.isnull().sum().sum()
    
    if total_nans > 0:
        cleaned_df.fillna(0, downcast = False, inplace=True)
    
    #v3.3 - section only for neural net binary model.   oemvendor cleanse logic doesn't help RF part models!
    #v3.4 added mtype to cleanse parts model vs. binary

    if mtype == 'binary':
        cleaned_df["MGR_SUGG_TRIAGE_TXT"] = process_text_column(cleaned_df, 'MGR_SUGG_TRIAGE_TXT', mgrnotes_tokens, mgrnotes_fixes, mgrnotes_keepitems, verbose=verbose, padding=None)
        cleaned_df['OEMMARKETVENDOR'] = cleaned_df.apply(fix_oem,axis=1,verbose=verbose)

    #v3.3...added NSN/ALU RRH band logic
    cleaned_df['FREQBAND'] = cleaned_df.apply(fix_freqband,axis=1,verbose=verbose)
    
    #py script only!
    cleaned_df = cleaned_df.drop(['IDENTIFIER','SUMMARY'],axis=1)
    
    return cleaned_df

#v2.0
#custom class to return array
class mtxtoarray():
    
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.toarray().astype(np.float32)
        
###########################
# Daqi cleaning section end
###########################

#added custom clean transformer for data cleansing (wraps Daqi cleansing and transformation code!)
#purpose to pipeline, into production deployment process
#v3.4 mtype
class CleanTransformerP(TransformerMixin):
    
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            return transform_alarms_dataframe(x, 'parts')
        return x

class CleanTransformerB(TransformerMixin):
    
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            return transform_alarms_dataframe(x, 'binary')
        return x

class CleanTransformer(TransformerMixin):

    def __init__( self, mytype ):
        #mytype can only be "binary" or "parts", though "parts", especially for "binary" per implementation of the below invokation.
        self._type = mytype
    
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            return transform_alarms_dataframe(x, self._type)
        return x


def retrieve_dset_info(ds_name):
    try:
        dataset = dsx_core_utils.get_remote_data_set_info(ds_name)      
    except (Exception,) as e:
        #print ("ds_name can't be retrieved !")
        raise(e)
    else: 
        return (dataset["description"], dataset["datasource"], dataset["query"])    
        

def load_dataset(ds_name, query=None, showquery = False):
    now = time.time()
    (ds_description, ds_datasource, ds_query) = retrieve_dset_info(ds_name)
    ds_query = query if query else ds_query
    
    #(db_description, db_type, db_host, db_port, db_name, db_user, db_password) = retrieve_database_info(ds_datasource)
    db_details = retrieve_db_info(ds_datasource)
    #print ("data source is: {0}, data set is: {1}, dataset description is: {2} \n".format(ds_datasource, ds_name, ds_description))
    #print (db_details, "\n")
    #query_source = "The Given" if query else "The Retrieved"
    #if showquery:
    #    print ("{0} ds_query is:\n {1} \n".format(query_source, ds_query))

    if (db_details["db_type"] == "MSSQL"):
        df = load_mssql(db_details, ds_query)
        #df=None
    elif(db_details["db_type"] == "Oracle"):
        df = load_oracle(db_details, ds_query)
        #df=None
    elif(db_details["db_type"] == "DB2"):
        df = load_db2(db_details, ds_query)
    else:
        #print ("getting unknow db within load_remote_dataset function")
        raise Exception("unknow db_type: {}".format(db_type))   
        
    #print ("Time to load data:", time.time() - now)
    return df
        
def load_query(datasource, query):
    now = time.time()
    try:
        db_details = retrieve_remote_db_info(datasource)     
    except (Exception,) as e:
        #print ("datasource can't be retrived !")
        raise(e)
    else: 
        #print ("ds_query is:\n {0} \n".format(query))
        if (db_details["db_type"] == "MSSQL"):
            df = load_mssql(db_details, query)
        elif(db_details["db_type"] == "Oracle"):
            #print ("+++++++++++++++++++")
            df = load_oracle(db_details, query)
        elif(db_details["db_type"] == "DB2"):
            df = load_db2(db_details, query)
        else:
            #print ("getting unknow db within load_remote_dataset function")
            raise Exception("unknow db_type: {}".format(db_details["db_type"]))   
            
        #print ("Time to load data:", time.time() - now)
        return df
        
#cp4d
def retrieve_db_info(ds_datasource):
    
    project = Project.access()
    dataSource = project.get_connected_data(name=ds_datasource)
    db_description = "unknown"
    db_type = 'DB2'
    db_user = dataSource["username"]
    db_password = dataSource["password"]
    
    try: 
        dataSource = project.get_connected_data(name=ds_datasource)
        db_type = 'DB2'
        db_user = dataSource["username"]
        db_password = dataSource["password"]
        db_host = dataSource["host"]
        db_port = dataSource["port"]        
        db_name = dataSource["database"] 
    except Exception as e:
        #print("can't parse/retrieve database details")
        raise(e)
    else:
        return {"db_description": db_description, "db_type": db_type, "db_host": db_host, "db_port": db_port, "db_name": db_name, "db_user": db_user, "db_password": db_password}

#cp4d
def load_db2(db_details,ds_query):
    #cp4d
    #ds_datasource = 'groot'
    #db_details = retrieve_db_info(ds_datasource)
    #hard code db_part to be 50000 instead of 50001
    dsn = (
        "DATABASE={0};"
        "HOSTNAME={1};"
        "PORT={2};"
        "PROTOCOL=TCPIP;"
        "UID={3};"
        "PWD={4};").format(db_details["db_name"], db_details["db_host"], 50000, db_details["db_user"], db_details["db_password"])
    
    conn = ibm_db.connect(dsn, "", "")

    # run direct SQL
    stmt = ibm_db.exec_immediate(conn, ds_query)
    ibm_db.fetch_both(stmt)
    pconn = ibm_db_dbi.Connection(conn)

    now = time.time()
    df = pd.read_sql(ds_query, pconn)
    pconn.close()
    return df
    
    
def load_oracle(db_details, ds_query):
    # makedsn - Returns a string suitable for use as the dsn parameter for connect()
    dsn = cx_Oracle.makedsn(db_details["db_host"], db_details["db_port"], db_details["db_name"])    
    conn = cx_Oracle.connect(db_details['db_user'], db_details['db_password'], dsn)

    cursor = conn.cursor()
    cursor.execute(ds_query)
    queryData = cursor.fetchall()

    colNames = []
    for i in range(0, len(cursor.description)):
        colNames.append(cursor.description[i][0])

    cursor.close()
    conn.close()

    df = pd.DataFrame(data=queryData, columns=colNames)
    return df
    
def load_mssql(db_details, ds_query):
    # Establish a connection to MS Sql Database
    conn = pymssql.connect(server=db_details["db_host"], user=db_details['db_user'], password=db_details['db_password'], database=db_details["db_name"])

    # Execute the query and fetch data
    curs = conn.cursor()
    curs.execute(ds_query)
    queryData = curs.fetchall()
    # Attach column names to pandas dataFrame
    colNames = []
    for i in range(0, len(curs.description)):
        colNames.append(curs.description[i][0])

    df = pd.DataFrame(data=queryData, columns=colNames)
    return df
    
#v3.4.3
def sendmail(destination,subject,content):

    #! /usr/local/bin/python
    SMTPserver = 'mailhost.corp.sprint.com'
    sender = 'networkai@sprint.com'

    # typical values for text_subtype are plain, html, xml
    text_subtype = 'plain'

    try:
        msg = MIMEText(content, text_subtype)
        msg['Subject']= subject
        msg['From'] = sender # some SMTP servers will do this automatically, not all
        msg['To'] = ', '.join(destination)
        conn = SMTP(SMTPserver)
        conn.set_debuglevel(False)
        #conn.login(USERNAME, PASSWORD)
        try:
            conn.sendmail(sender, destination, msg.as_string())
        finally:
            conn.quit()
        exit_code = 0
    except:
        sys.exit( "mail failed; %s" % "CUSTOM_ERROR" ) # give an error message
        exit_code = 255
        
    return exit_code
