from lstm_model.faultalarmclasses_cp4d import *
import logging, platform
from logging.handlers import RotatingFileHandler

#cp4d
#from project_lib import Project

logger_name = f"{os.getenv('APP_LOG')}.{__name__}" if os.getenv("APP_LOG") else __name__
my_logger = logging.getLogger(logger_name)

#cp4d
def retrieve_db_info_cp4d(db_type,ds_datasource):
    
    project = Project.access()
    dataSource = project.get_connected_data(name=ds_datasource)
    db_description = "unknown"
    db_user = dataSource["username"]
    db_password = dataSource["password"]
    
    try: 
        dataSource = project.get_connected_data(name=ds_datasource)
        db_user = dataSource["username"]
        db_password = dataSource["password"]
        db_host = dataSource["host"]
        db_port = dataSource["port"]
        if db_type == 'Oracle':
            db_name = dataSource["sid"] 
        else:
            db_name = dataSource["database"] 
    except Exception as e:
        #print("can't parse/retrieve database details")
        raise(e)
    else:
        return {"db_description": db_description, "db_type": db_type, "db_host": db_host, "db_port": db_port, "db_name": db_name, "db_user": db_user, "db_password": db_password}

#Daqi modify the parameter to allows date?
#cp4d
def load_part_history(phase,select_type,query):
    #phase can be TRRAIN, SCORING, BATCH_DAILY
    if phase in ["TRAIN"]:
        assert (query is not None)
        assert(select_type in ["TICKET"])     
    elif phase in ["SCORING", "BATCH_DAILY"]:         
        assert (query is not None)
        assert(select_type in ["CASCADEID", "TICKET"])
    else:
        raise Exception("Unknown phase value of: {}".format(phase))
    #Daqi, this is confusing, chat ~ part_history, groot ~ chat_sample. 
    #dfp = load_chat_history(query,'chat','chat_sample')  
    #dfp = load_chat_history(query,'chat','part_history')  
    db_details = retrieve_db_info_cp4d('MSSQL','chat')
    dfp = load_mssql(db_details,query)
    dfp = convert_imn_toPrimeIMN(dfp, select_type)

    return dfp    


#Daqi to aggregate after the join and replacement.
#cp4d
def convert_imn_toPrimeIMN(dfp1, select_type = "TICKET"):
    #dfp_type can be TICK_NBR or CASCADEID, i.e. the column name within dfp1
    
    dfp = dfp1.copy()

    imno_query = '''select right(concat('000',cast(prime_imn as varchar(20))),6) prime_imn, right(concat('000',cast(imn as varchar(20))),6) imn_no
    from ICPD.IMN_PRIME_CATALOG
    group by right(concat('000',cast(prime_imn as varchar(20))),6), right(concat('000',cast(imn as varchar(20))),6)
    '''
     
    db_details = retrieve_db_info_cp4d('DB2','groot')
    dfimno = load_db2(db_details,imno_query)
    #dfimno = load_netcool_history(imno_query,'groot','netcool_history_sample')
    #dfimno = dfimno.set_index('TICK_NBR')

    #v3.0 update chat IMNO's with correct primes
    dfp_labels = dfp.columns.tolist()
    dfp = dfp.merge(dfimno, on='IMN_NO', how='inner')
    dfp['IMN_NO'] = dfp['PRIME_IMN']
    dfp = dfp[dfp_labels]
    
    #Daqi need re-aggregated due to duplicates after merging and renaming
    if select_type == "TICKET":
        dfp = dfp.groupby(["TICK_NBR", "IMN_NO"]).agg({"TOTAL_PART_CNT" : sum}).reset_index(drop=False) 
    else:
        dfp = dfp.groupby(["CASCADEID", "IMN_NO"]).agg({"TOTAL_PART_CNT" : sum}).reset_index(drop=False)
        
    return dfp
    
#Daqi this is to see if/how difference between site_id from dfp table itself, and that from dfa tickets
def convert_part_history(dfp_ticket):
    #since normal dpf_labels only carry ticketnbr, which won't be able to join with the ticket submitted for real-time scoring
    #so we will coverrt dfp with tick_nbr, imn_no and total_part_cnt to cascadeid, imn_no and total_part_cnt
    
    dfp = dfp_ticket.copy()
    #print (dfp.dtypes)
    dfa_history = load_alarms("TRAIN") #def load_alarms(phase, datasetname, query = None):
    #dfa_history = load_alarm_history('groot_partalarm_model_create')
    #print(dfa_history.dtypes)
    #print (dfa_history.info())
    
    dfa = dfa_history[["TICK_NBR", "CASCADEID"]].drop_duplicates()
    dfa = dfa.merge(dfp, on="TICK_NBR", how = "inner").drop(columns = "TICK_NBR")
    dfp_cascadeid = dfa.groupby(["CASCADEID", "IMN_NO"]).sum().reset_index(drop = False)
    return dfp_cascadeid


#Daqi same as add_hist_atlas_features, dfp1 here could have different columns, depending on phase of TRAIN vs. for SCORING and BATCH_DAILY 
#Daqi, pay attention on different asserts used for each phase. especially SCORING here, it is try to resolve a logical bug.
#Daqi, for BATCH_DAILY, here it is forced to use same logic for TRAIN, really intended? 
def add_part_history(phase, dfa1, dfp1, nparts, partslist, cascadeid = None):
    dfa = dfa1.copy()
    dfp = dfp1.copy()
    #print (dfa["CASCADEID"][0])
    #print (dfp[dfp["CASCADEID"] == "RA03XC095"])
    
    if partslist:
        dftc_labels = ["HIST_P{}".format(part) for part in partslist]
    
    if phase == "SCORING":  
        assert(cascadeid is not None)
        assert(partslist is not None)
        assert(set(['CASCADEID', 'IMN_NO', 'TOTAL_PART_CNT']) == set(dfp.columns))
        assert (dfp.duplicated(subset =["CASCADEID", "IMN_NO"],  keep=False).sum() == 0)
        dfp = dfp[dfp["CASCADEID"] == cascadeid]
        #print (dfp[dfp["CASCADEID"] == "RA03XC095"])
        #dfp_labels = dfp.pivot(index = "CASCADEID", columns = "IMN_NO").fillna(0).astype(int).clip_upper(1)
        #dfp_labels.columns = dfp_labels.columns.droplevel()
        dfp_labels = dfp.pivot(index = "CASCADEID", columns = "IMN_NO",  values= "TOTAL_PART_CNT").fillna(0).astype(int).clip_upper(1)
        dfa[partslist] = pd.DataFrame([[0]*len(partslist)], index = dfa.index)
        dfa.set_index('CASCADEID', inplace=True)
        dfa.update(dfp_labels) # this set dtype to be float even both dfa and dfp_labels is type of int
        dfa[partslist] = dfa[partslist].astype(int)
        dfa = dfa.reset_index()
        dfa.rename(columns=dict(zip(partslist, dftc_labels)), inplace=True)   
    
    elif phase == "UNUSED_BATCH_DAILY":
        assert(partslist is not None)
        assert(set(['CASCADEID', 'IMN_NO', 'TOTAL_PART_CNT']) == set(dfp.columns))
        assert (dfp.duplicated(subset =["CASCADEID", "IMN_NO"],  keep=False).sum() == 0)
        # raise Exception("To be developed with BATCH_DAILY as template, see concerns")
        # In TRAIN phase, the dfp is filtered in get_nparts to be only entries having matching ticekt_nbr available from the input dfa
        # In SCORING phase, the dfp is selected only history tickets, it doesn't do filter as done in TRAIN phase.
        # in BATCH_DAILY, which is daily for yesterday tickets, theratically, it should be done similarly as in SCORING phase
        dfp = dfp[dfp["IMN_NO"].isin(partslist)]
        #dfp = dfp.pivot(index = "CASCADEID", columns = "IMN_NO", values = "TOTAL_PART_CNT").fillna(0).astype(int).clip_upper(1)
        dfp_labels = pd.crosstab(index = dfp.CASCADEID, columns = dfp.IMN_NO, dropna=False)
        dfa[partslist] = pd.DataFrame([[0]*len(partslist)], index = dfa.index)
        dfa.set_index('CASCADEID', inplace=True)
        dfa.update(dfp_labels)
        dfa = dfa.reset_index()
        dfa.rename(columns=dict(zip(partslist, dftc_labels)), inplace=True)   
        dfa[dftc_labels] = dfa[dftc_labels].astype(int)        
        
    elif phase in ["TRAIN", "BATCH_DAILY"]:     
        assert(nparts > 10) 
        assert(partslist is not None)
        assert(set(partslist + ["TICK_NBR"]) == set(dfp.columns)) #in addition to TICK_NBR, also columns of partslist 
        assert (dfp.duplicated(subset =["TICK_NBR"],  keep=False).sum() == 0)
        # raise Exception("To be developed")      
        # double check dfp doesn't have None, null value
        dftc = dfa[['TICK_NBR','CASCADEID']]
        dftc = dftc.drop_duplicates()
        dftc = dftc.merge(dfp, on='TICK_NBR', how='inner').reset_index(drop=True)
        dftc = dftc.set_index('TICK_NBR').sort_index(ascending=True).fillna(0)  # double checking, fillna(0) doesn't seem needed. 
        
        if len(dftc) > 0:  #transform fails if dftcp is empty
            dftc_tmp = dftc.groupby('CASCADEID').transform(pd.Series.cumsum)
            dftc[dftc.columns[1:]] = dftc_tmp - dftc[dftc.columns[1:]] 
        
        dfa = dfa.merge(dftc,on=['TICK_NBR','CASCADEID'],how='left') # double checking, on can be index for merge??
        dfa[dftc.columns[1:]] = dfa[dftc.columns[1:]].fillna(0).astype(int)  # double checking, doesn't needed
        dfa.rename(columns=dict(zip(partslist, dftc_labels)), inplace=True)   
       
    else:
        raise Exception("Unknown phase value of: {0}".format(phase))
        
    return dfa

#Daqi, this filter out those entries in dfp, but not in dfa, wondering why these tickets in dfp, but not in dfa 
#Also, the return dfp, still have raw IMN_NO, without any prefix P, this is different from Lance's get_nparts
def get_nparts_new(dfp,dfa,nparts,partslist):
    dfp1 = dfp.copy()
    dfa1 = dfa.copy()
    #daqi what are those being filtered out, if they are not associated with tickets within dfa
    #this is the difference between lance' and mine
    dfp1 = dfp.merge(pd.DataFrame({"TICK_NBR": dfa1['TICK_NBR'].unique()}), on='TICK_NBR', how='inner')

    if partslist:
        #print('existing static parts list')
        dfp1 = dfp1[dfp1["IMN_NO"].isin(partslist)]
        #dfa = dfa1[dfa1['IMN_NO'].isin(partslist)].copy()
    else:
        #print('auto generating top ' + str(nparts) + ' list')
        #dfp = dfp1        
        my_partslist = dfp1.groupby(["IMN_NO"])['TICK_NBR'].count().nlargest(nparts).index
        dfp1 = dfp1[dfp1["IMN_NO"].isin(my_partslist)]        

        
    dfpf = pd.DataFrame(dfp1.groupby(["IMN_NO"])['TICK_NBR'].count().nlargest(nparts))
    dfpf = dfpf.reset_index(drop=False).rename(columns = {"TICK_NBR" : "FREQ"})
    
    dfa = dfa1[["TICK_NBR"]].drop_duplicates()
    my_logger.debug(f"inside get_nparts_new before merge, dfp1 shape is {dfp1.shape}")
    dfp1 = dfp1.merge(dfa, on = ["TICK_NBR"], how="inner")
    my_logger.debug(f"inside get_nparts_new after merge, dfp1 shape is {dfp1.shape}")
    
    #print (dfp1.IMN_NO.nunique())   
    
    #pd.crosstab doesn't carry index, columns if it is empty, while df.pivot would still carry.
    #dfp1 = pd.crosstab(index = dfp1.TICK_NBR, columns = dfp1.IMN_NO, dropna=False)
    dfp1 = dfp1.pivot(index = "TICK_NBR", columns = "IMN_NO",  values= "TOTAL_PART_CNT").fillna(0).astype(int).clip_upper(1)

    if partslist:
        set1 = set(partslist)
        set2 = set(dfp1.columns)
        missing_set = set1 -set2
        
        for part in missing_set:
            dfp1[part] = 0
            
        dfp1 = dfp1[partslist]
    
    #dfp1 = dfp1.rename(columns=dict(zip(dfp1.columns, ["P{}".format(part) for part in dfp1.columns])))
    dfp1.reset_index(drop = False, inplace=True)       

    if not partslist:
        #NOTE: only save these files for pyscript, model create/export should be done from pyscript job only!
        my_logger.info('Warning: creating new part list files...')
        dfp1.to_csv(os.getenv("DSX_PROJECT_DIR") + '/datasets/' + 'parts_per_ticket.csv', index=False, header=True)
        dfpf.to_csv(os.getenv("DSX_PROJECT_DIR") + '/datasets/' + 'topn_parts_freq.csv', index=False, header=True)
    
    return dfp1, dfpf
        
#Daqi modify the parameter to allow date? 
#Daqi, the source parameter is confusing, it is due to BATCH_DAILY from Lance's coding, for flexibility to point to different databases
def load_alarms(phase, datasetname, query = None):
    if phase == "SCORING":
        assert(query is not None)
        dfa = load_dataset(datasetname, query)
    elif phase == "TRAIN":
        dataset = datasetname if datasetname else "groot_partalarm_model_create"
        dfa = load_alarm_history(dataset)
    elif phase == "BATCH_DAILY":
        dataset = datasetname if datasetname else "netcool_batch_score"
        dfa = load_dataset(dataset, query)      
    else:
        raise Exception("Unknown phase value of: {}".format(phase))
        
    return dfa

    
#Daqi still have raw IMN_NO, no EQP_P prefix, which is set in add_part_atlas
#cp4d
def load_part_atlas(phase, query):
    if phase in ["SCORING", "BATCH_DAILY", "TRAIN"]:
        db_details = retrieve_db_info_cp4d('DB2','groot')
        dfeqp = load_db2(db_details,query)
        #dfeqp = load_atlas('groot',datasetname)
        dfeqp = dfeqp.set_index('CASCADEID')
        dfeqp = dfeqp.pivot(columns='IMN_NO').fillna(0).astype(int)
        dfeqp.columns = dfeqp.columns.droplevel()
        #dfeqp = dfeqp.reset_index()
        #rename col
        #dfeqp= dfeqp.rename(columns=dict(zip(dfeqp.columns[1:], ["EQP_P{}".format(part) for part in dfeqp.columns[1:]])))
    else:
        raise Exception("Unknown phase value of: {0}".format(phase))
        
    return dfeqp
    
    
#Daqi, below is copied from scoring script, enhanced with different phases
#v3.4 score function only!
def alarm_catch(phase, df_trms_d1,dfa1,timedelta_thresh):
  
    #v3.4 alarm association by cascade, prior was only by ticket #
    #cp4d - added TRAIN as moved to TRMS
    df_trms_d1['ORGNT_DT'] = df_trms_d1['ORGNT_DT'].astype('datetime64[ns]')
    dfa1 = dfa1.merge(df_trms_d1[['TICK_NBR','CASCADEID','ORGNT_DT','MGR_SUGG_TRIAGE_TXT','TRAIN']],on='CASCADEID',how='inner')
 
    if phase =="SCORING": 
        #v3.4.2.1 remove (dfa['FIRSTOCCURRENCE'] < dfa['ORGNT_DT']) for realtime score
        dfa1 = dfa1.loc[(dfa1['FIRSTOCCURRENCE'] > (dfa1['ORGNT_DT'] - datetime.timedelta(hours=timedelta_thresh))),]
    
        #try join by tick_nbr if no alarms
        if len(dfa1) == 0: #Daqi, what this try to do to merge with an empty dataframe? why introducing new NTICK_NBR, instead of TICK_NBR, in alarm query??
          dfa1 = dfa1.merge(df_trms_d1[['TICK_NBR','ORGNT_DT','MGR_SUGG_TRIAGE_TXT']],left_on='NTICK_NBR',right_on='TICK_NBR',how='inner')
        #remove alarm table ntick_nbr
        dfa1 = dfa1.drop('NTICK_NBR',axis=1)
        
    elif phase in ["TRAIN", "BATCH_DAILY"]:
        #use following filter when training
        dfa1 = dfa1.loc[(dfa1['FIRSTOCCURRENCE'] > (dfa1['ORGNT_DT'] - datetime.timedelta(hours=timedelta_thresh))) & (dfa1['FIRSTOCCURRENCE'] < dfa1['ORGNT_DT']),]
    
    return dfa1 
    

#Daqi, this one can pass in dfp_cascadeid, or dfp_labels for dfp1, depending on both phase and orig_flag, which is a little bit confusing
#Daqi, but the justification is that it integrates all similarities same place, so easier for maintenance and change.
def add_hist_atlas_features (phase, dfa1, dfp1, dfeqp1, nparts, partslist, ticketnbr =None, cascadeid =None, orig_flag = False): 
    # phase could be TRAIN, SCORING, BATCH_DAILY
    # parameter ticketnbr, cascadeid and orig_flag are all for phase of SCORING
    # orig_flag TRUE is to force using dfp_ticket, then get_nparts_new to get dfp_labels for dfp1, same as TRAIN, which ends with 0 for HIST_P### 
    # local variable return_type is also for "SCORING", but no hurts to return the same for DAILY_BATCH

    result_type = ''  
    part_labels = ["P{}".format(part) for part in partslist]
    dftc_labels = ["HIST_P{}".format(part) for part in partslist]
    dfeqp_labels = ["EQP_P{}".format(part) for part in partslist]
    
    if phase == "SCORING":  #Daqi, using orig_flag to expect different formats of dfp1 input, thus different implementation.
        assert (ticketnbr is not None)
        assert (cascadeid is not None)

        if orig_flag: #dfp1 is dfp_labels gotten from get_nparts_new
            dfa = add_part_history("TRAIN", dfa1, dfp1, nparts, partslist, cascadeid)
        else:  #dfp1 is dfp_cascadeid, which is to fix
            dfa = add_part_history(phase, dfa1, dfp1, nparts, partslist, cascadeid)
        my_logger.debug ("adding part_history ", dfa.shape)
        #print (dfa[dftc_labels])
        dfa.head()
        
    elif phase in ["TRAIN", "BATCH_DAILY"]: #Daqi, theatrically speaking, it should have same concern as for phase SCORING, but to be simplified. 
        dfa = add_part_history(phase, dfa1, dfp1, nparts, partslist)
        my_logger.debug ("adding part_history ", dfa.shape)
        #print (dfa[dftc_labels])
        dfa.head()
        
    else:
        raise Exception("Unknown phase")

    dfa = add_part_atlas(phase, dfa, dfeqp1, partslist)
    my_logger.debug ("adding part_atlas ", dfa.shape)

    #NOTE: removing ALL alarms that we don't have Top 25 parts on pulled from Atlas/current install base
    dfa = dfa.loc[dfa[dfeqp_labels].sum(axis=1) > 0]
    
    #v3.2.3 log
    if len(dfa) == 0:
        result_type = 'no_eqp'
        
    #remove unwanted fields (column transformer passthrough disabled, therefore optional drop here!)
    dfa = dfa.drop(['CASCADEID'],axis=1)
    #dfa = dfa.drop(['CASCADEID','ALARM_RANK_ID'],axis=1) #this is invoked in TRAIN module, so not sure about the ALARM consistency for each phase

    #set all part fields to int
    dfa[list(dftc_labels)+list(dfeqp_labels)] = dfa[list(dftc_labels)+list(dfeqp_labels)].fillna(0).astype(int)
    
#         #v3.0 set > 0 values to 1 for part equip AND history
#         for i in list(dftc_labels)+list(dfeqp_labels):
#           dfa.loc[dfa[i] > 0, i] = 1
    dfa[list(dftc_labels)+list(dfeqp_labels)] = dfa[list(dftc_labels)+list(dfeqp_labels)].clip_upper(1)

    dfa[list(dftc_labels)+list(dfeqp_labels)] = dfa[list(dftc_labels)+list(dfeqp_labels)].astype(str)

    #v3.1 add alarm association(using counts)
    alarmcnt = pd.DataFrame(dfa.groupby(by='TICK_NBR').count()['REGION'],).rename(columns={'REGION': 'ALARMGRP'})
    dfa = dfa.merge(alarmcnt,on='TICK_NBR',how='left')
    my_logger.debug ("adding alarmcnt ", dfa.shape)

    #set ALL to string (Part HIST and EQUIP for OHE!)
    dfa = dfa.astype('str')
    dfa[['TICK_NBR','ALARMGRP']] = dfa[['TICK_NBR','ALARMGRP']].astype('int')
   
    return dfa, result_type


    
#Daqi be sure the input dfeqp1 still with raw IMN_NO, no prefix EQP_P yet.
def add_part_atlas(phase, dfa1, dfeqp1, partslist):
    dfa = dfa1.copy()
    dfeqp_labels = ["EQP_P{}".format(part) for part in partslist]
    
    if phase in ["SCORING", "BATCH_DAILY", "TRAIN"]:
    
        dfa[partslist] = pd.DataFrame([[0]*len(partslist)], index = dfa.index)
 
        #v3.0 update with atlas parts
        #merge with atlas data with alarms
        dfa.set_index('CASCADEID', inplace=True)
        #dfa.update(dfeqp1.set_index('CASCADEID'))
        dfa.update(dfeqp1)
        dfa = dfa.reset_index()
        #dfa.columns.tolist()
        #print (dfa.shape)
        
        dfa.rename(columns=dict(zip(partslist, dfeqp_labels)), inplace=True)  
        dfa[dfeqp_labels] = dfa[dfeqp_labels].astype(int) #even both are int type, but they bc float type after update
        #NOTE: removing ALL alarms that we don't have Top 25 parts on pulled from Atlas/current install base
        #dfa = dfa.loc[dfa[dfeqp_labels].sum(axis=1) > 0]
        #dfa.rename(columns=dict(zip(partslist, dfeqp_labels)), inplace=True)  
    else:
        raise Exception("Unknown phase value of: {0}".format(phase))
        
    return dfa
    

# to be added into dql_forLance.py, similarly as get_npart_new, while get_parts is in faultalarmclasses.py
import pickle
#from sklearn.externals import joblib
def load_model_new(project_path, model_name, version ="latest", serialization_method = "joblib"):
  #lib can be either joblib or pickle
  
  #Daqi, do we till need this here?, there is one in faultalarmclasses.py
  #custom class to return array
#   class mtxtoarray():
      
#       def fit(self, x, y=None):
#           return self
  
#       def transform(self, x):
#           return x.toarray().astype(np.float32)
  
  #model_name = "FaultAlarmPartsPredictor"
  model_parent_path = project_path + "/models/" + model_name + "/"
  metadata_path = model_parent_path + "metadata.json"
  
  # fetch info from metadata.json
  with open(metadata_path) as data_file:
      meta_data = json.load(data_file)
 
  # if latest version, find latest version from  metadata.json
  if (version == "latest"):
      version = meta_data.get("latestModelVersion")
  
  # prepare model path using model name and version
  model_path = model_parent_path + str(version) + "/model"
  my_logger.debug (f"model_path is: {model_path}")
  
  # load model
  if serialization_method == "joblib":
      model = joblib.load(open(model_path, 'rb'))
  elif serialization_method == "pickle":
      model = pickle.load(open(model_path, 'rb'))
      
  return model

def model_predict(model, mtype, df, thresh, x_labels1):
  #v3.2.4
  if mtype == 'binary':
    #keep MGR TXT only for binary model
    x_labels = x_labels1 + ['MGR_SUGG_TRIAGE_TXT']
  else:
    x_labels = x_labels1
  
  #print(mtype,x_labels)

  #y_predp = model.predict_proba(df).toarray()
  y_predp = model.predict_proba(df[x_labels])

  if not isinstance(y_predp, np.ndarray):
    y_predp = y_predp.toarray()
  
  #v3.3.1  
  if mtype == 'binary':
    y_predp = y_predp[:,1]

  y_pred = y_predp.copy()
  y_pred[y_pred >= thresh] = 1
  y_pred[y_pred < thresh] = 0
  y_pred=y_pred.astype(int)
    
  return y_pred,y_predp
  

#this is initial development, after seeing other places, it seems not good to mix both predictions, use merge_ticket_prediction instead     
def integrate_ticket_prediction(dfa, binary_predp, part_predp, plabels, b_thresh, p_thresh):   
    dfeqp_labels = ["EQP_{}".format(part) for part in plabels]
    
    # create df's on part prediction model results
    #y_predp = pd.DataFrame(y_predp, columns=dfp_labels.columns[1:])
    y_predp = pd.DataFrame(part_predp, columns=plabels)
    b_predp = (binary_predp >= b_thresh).astype(int)
    df_b = pd.DataFrame(b_predp, columns=['binary_flag'])

    # merge binary with parts model results
    y_predp = y_predp.merge(df_b.reset_index(drop=True).astype(int),left_index=True, right_index=True)

    #add tick_nbr to result
    y_predp['TICK_NBR'] = np.array(dfa['TICK_NBR'])
    
    # add eqp parts
    y_predp = y_predp.merge(dfa[dfeqp_labels].reset_index(drop=True).astype(int),left_index=True, right_index=True).copy()
    #print('resulta:',y_predp.values.tolist())
    
    #take max of all alarm predictions
    y_predp = y_predp.groupby('TICK_NBR').max(axis=0)
    #print('result0:', y_predp.values.tolist())
    
  
    #filter only results where binary predicts = 1, remove tick_nbr
    y_predp = y_predp.loc[y_predp['binary_flag']==1,]
    #print('result1:', y_predp.values.tolist())
    
    #skip part compatible check if binary doesn't predict > threshold
    if len(y_predp) > 0:     
      #make sure compatible parts are predicted
      #v3.0 release predictions on compatible parts only!
      #don't send parts to sites that are not compatible
      #model does good but not perfect, why rule logic is used here!
    
      #filter predictions that don't have atlas install info, multiply by -1 for tracking purposes, these predictions when > prediction threshold is applied
      #for i in dfp_labels.columns[1:]:
      for i in plabels:
          y_predp.loc[(y_predp['EQP_' + i] == 0) & (y_predp[i] > 0), i] = -1 * y_predp.loc[(y_predp['EQP_' + i] == 0) & (y_predp[i] > 0), i]

        
      y_predp = y_predp[plabels]
      
      y_pred = y_predp.copy()
      y_pred= (y_pred >= p_thresh).astype(int)
      #remove negative values for proba scores
      #keep negative values
      #y_predp[y_predp <= -0.0] = 0.0
      
      y_pred = y_pred.values.tolist()
      y_predp = y_predp.values.tolist()
    else:
      y_pred = None
      y_predp = None
    return y_pred, y_predp


def capture_result_summary(o_result, n_result):
    result = ""
    if o_result:
        if n_result:
            result = o_result + "-" + n_result
        else:
            result = o_result
    else:
        result = n_result
    
    return result
    
    
def integrate_ticket_binary_prediction(dfa, predp, thresh):   

    df_predp = pd.DataFrame(predp, columns=['PARTS'])

    #add tick_nbr to result
    df_predp['TICK_NBR'] = np.array(dfa['TICK_NBR'])

    #take max of all alarm predictions
    df_predp = df_predp.groupby('TICK_NBR').max(axis=0)

    df_pred = df_predp.copy()
    
    df_pred["PARTS"] = (df_pred["PARTS"] >= thresh).astype(int)
    
    return df_pred, df_predp
    

def integrate_ticket_parts_prediction(dfa, part_predp, plabels, p_thresh):   
    dfeqp_labels = ["EQP_{}".format(part) for part in plabels]
    
    # create df's on part prediction model results
    df_predp = pd.DataFrame(part_predp, columns=plabels)

    #add tick_nbr to result
    df_predp['TICK_NBR'] = np.array(dfa['TICK_NBR'])
    
    # add eqp parts
    df_predp = df_predp.merge(dfa[dfeqp_labels].reset_index(drop=True).astype(int),left_index=True, right_index=True).copy()
    #print('resulta:',y_predp.values.tolist())
    
    #take max of all alarm predictions
    df_predp = df_predp.groupby('TICK_NBR').max(axis=0)

    #make sure compatible parts are predicted
    #v3.0 release predictions on compatible parts only!
    #don't send parts to sites that are not compatible
    #model does good but not perfect, why rule logic is used here!

    #filter predictions that don't have atlas install info, multiply by -1 for tracking purposes, these predictions when > prediction threshold is applied
    #for i in dfp_labels.columns[1:]:
    for i in plabels:
      df_predp.loc[(df_predp['EQP_' + i] == 0) & (df_predp[i] > 0), i] = -1 * df_predp.loc[(df_predp['EQP_' + i] == 0) & (df_predp[i] > 0), i]
    
    df_predp = df_predp[plabels]

    df_pred = df_predp.copy()
    df_pred= (df_pred >= p_thresh).astype(int)
    
    #remove negative values for proba scores
    #y_predp[y_predp <= -0.0] = 0.0

    return df_pred, df_predp  
    

def save_ticket_predictions(mtype, batchtype, df_pred, df_predp, data_path):
    #set_trace()
    assert (mtype in ["binary", "parts"])
    assert (batchtype in ["daily", "rebaseline"])
    assert (isinstance(df_pred, pd.DataFrame))
    assert (isinstance(df_predp, pd.DataFrame))
    assert (df_pred.index.name == "TICK_NBR")
    assert (df_predp.index.name == "TICK_NBR")

    today = str(datetime.datetime.now().strftime('%Y%m%d'))

    #run date - 1
    yr,mth,day = subdate(today,1)

    ydate1 = yr + mth + day
    ydate2 = yr + '-' + mth + '-' + day

    #add datevalue to tables
    df_pred['DATE_VALUE'] = ydate2
    df_predp['DATE_VALUE'] = ydate2
    
    if (mtype == "binary"):
        name_prefix = 'faultalarmpredictbinary_'        
    else:
        name_prefix = 'faultalarmpredict_' 
        
    #export to files
    filename_score = data_path + name_prefix + batchtype + '_batch_score_' + ydate1 + '.csv'
    filename_score_p = data_path + name_prefix + batchtype + '_batch_score_proba_' + ydate1 + '.csv'

    df_pred.to_csv(filename_score, encoding='utf-8')
    df_predp.to_csv(filename_score_p, encoding='utf-8')
    
    time.sleep(5)
    groot_load(name_prefix + batchtype + '_batch_score', filename_score,'batch_score_table',ydate2,data_path)
    groot_load(name_prefix + batchtype + '_batch_score_proba', filename_score_p,'batch_score_table',ydate2,data_path)


# to capture the pod_name in the log: ww3-pyscript-faultalarm-7947f775b4-rzdx8
# import platform
# "-".join(platform.node().split("-")[-2:])
# platform.node().split("-")[-1]
# '7947f775b4-rzdx8'
# "-".join(os.uname()[1].split("-")[-2:])
# pod_name = platform.node().split("-")[-1]
# app_logger = setup_app_logger(logpath=logpath, filename="web_score", pod_name = pod_name)
# instead of modify the calling, making the modification inside of its implementation
def setup_app_logger(logpath, filename, ephemeral=False, level=logging.DEBUG): 
    '''This function should only be invoked within main'''

    logger_name = os.getenv("APP_LOG") or "__main__"
    file_name =  os.getenv("APP_LOG") or filename
    file_name = f"{logpath}{file_name}"
    
    
    #pod_name = platform.node().split("-")[-1]
    pod_name = "-".join(platform.node().split("-")[-2:])
    mine_extra = {'pod_name' : pod_name}
    
    if ephemeral:
      timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
      file_name = f"{file_name}_{timestamp}.log"
    else:
      file_name = f"{file_name}.log"
      
    #log_format = "%(asctime)s - %(levelname)s - %(name)s - {%(filename)s:%(lineno)d:%(funcName)s} - %(message)s"
    #log_format = "%(asctime)s - %(levelname)s - {%(filename)s:%(lineno)d:%(funcName)s} - %(message)s"
    #log_format = "%(asctime)s - %(levelname)s - %(process)d - {%(filename)s:%(lineno)d:%(funcName)s} - %(message)s"
    log_format = "%(asctime)s - %(levelname)s - %(pod_name)s - %(process)d - {%(filename)s:%(lineno)d:%(funcName)s} - %(message)s"
      
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(log_format)
    logger.setLevel(level)
    
    # add a handler to send DEBUG level messages to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    #logger.addHandler(console_handler)    
    
    # add a handler to send INFO level messages to file 
    #file_handler = logging.FileHandler(file_name, 'a') 
    file_handler = RotatingFileHandler(file_name, maxBytes=1000000000, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger = logging.LoggerAdapter(logger, mine_extra)
    
    # return the logger object
    return logger 
      
def dummy_testing_logging():
    print("this is from print statement in module")
    my_logger.debug("i am debugging")
    my_logger.info("i am info")
    my_logger.error("i am error")
    my_logger.exception("i am exception")


#Daqi2
def model_predict_dynamic(model, mtype, df, thresh, x_labels, partslist = None):
  #x_labels = x_labels1
  assert(mtype == "parts")
  assert(isinstance(thresh, dict)) 
  assert(partslist is not None)
  assert(set(thresh.keys()) == set(partslist))
  
  dfeqp_labels = ["EQP_P{}".format(part) for part in partslist]
  
  thresh_df = pd.DataFrame([thresh])
  thresh_df = thresh_df[partslist]
  thresh_ar = thresh_df.values
#   print(f"partslist is {partslist}")
#   print (f"input thresh is {thresh}")
#   print(f"thresh_ar is {thresh_ar}")
    
  #y_predp = model.predict_proba(df).toarray()
  y_predp = model.predict_proba(df[x_labels])

  if not isinstance(y_predp, np.ndarray):
    y_predp = y_predp.toarray()
  
  y_pred = y_predp.copy()  
  
  #Daqi: Below seems duplicate steps as that in integrate_ticket_parts_prediction_dynamic
  #as y_predp is return unchanged, thus its maximum can be returned first in group inside integrate_ticket_parts_prediction_dynamic, which further set to be negative.
  df_pred = pd.DataFrame(y_pred, columns = partslist)
  assert(df_pred.index.equals(df.index))
  df_pred = df_pred.join(df[dfeqp_labels].astype(int))  #do I need reset index here, adding assert above to double check???
  for i in partslist:
    df_pred.loc[(df_pred['EQP_P' + i] == 0) & (df_pred[i] > 0), i] = -1 * df_pred.loc[(df_pred['EQP_P' + i] == 0) & (df_pred[i] > 0), i]
  df_pred = df_pred[partslist]
  p_pred = df_pred.values
    
#   y_pred[y_pred >= thresh] = 1
#   y_pred[y_pred < thresh] = 0
#   y_pred=y_pred.astype(int)
  b_pred = (p_pred >= thresh_ar).astype(int)
#   print (y_pred)
#   print (y_predp)
  #returned unchanged y_predp, with positive probability for integrate_ticket_parts_prediction_dynamic  
  return b_pred,p_pred, y_predp
  

def integrate_ticket_parts_prediction_dynamic(dfa, part_predp, plabels, p_thresh):   
    assert(isinstance(p_thresh, dict)) 
    dfeqp_labels = ["EQP_{}".format(part) for part in plabels]
    
    df_thresh = pd.DataFrame([p_thresh])
    df_thresh = df_thresh.add_prefix("P")
    df_thresh = df_thresh[plabels]
    arr_thresh = df_thresh.values

    
    # create df's on part prediction model results
    df_predp = pd.DataFrame(part_predp, columns=plabels)

    #add tick_nbr to result
    df_predp['TICK_NBR'] = np.array(dfa['TICK_NBR'])
    
    # add eqp parts
    df_predp = df_predp.merge(dfa[dfeqp_labels].reset_index(drop=True).astype(int),left_index=True, right_index=True).copy()
    #print('resulta:',y_predp.values.tolist())
    
    #take max of all alarm predictions
    df_predp = df_predp.groupby('TICK_NBR').max(axis=0)

    #make sure compatible parts are predicted
    #v3.0 release predictions on compatible parts only!
    #don't send parts to sites that are not compatible
    #model does good but not perfect, why rule logic is used here!

    #filter predictions that don't have atlas install info, multiply by -1 for tracking purposes, these predictions when > prediction threshold is applied
    #for i in dfp_labels.columns[1:]:
    for i in plabels:
      df_predp.loc[(df_predp['EQP_' + i] == 0) & (df_predp[i] > 0), i] = -1 * df_predp.loc[(df_predp['EQP_' + i] == 0) & (df_predp[i] > 0), i]
    
    df_predp = df_predp[plabels]

    df_pred = df_predp.copy()
    
    index_values = df_pred.index
    pred_values = (df_pred.values >= arr_thresh).astype(int)

    df_pred = pd.DataFrame(pred_values, columns=plabels, index=index_values)
    
    #remove negative values for proba scores
    #y_predp[y_predp <= -0.0] = 0.0
    
    #app_logger.debug(part_predp)
    # my_logger.debug(arr_thresh)
    # my_logger.debug(df_pred.head(3))  
    # my_logger.debug(df_predp.head(3))  

    return df_pred, df_predp  
    
    
def merge_ticket_prediction(dfa, binary_predp, part_predp, plabels, b_thresh, p_thresh, dynamic=False):      

    df_b_pred, df_b_predp = integrate_ticket_binary_prediction(dfa, binary_predp, b_thresh)
    if dynamic:
        df_p_pred, df_p_predp = integrate_ticket_parts_prediction_dynamic(dfa, part_predp, plabels, p_thresh)      
    else:  
        df_p_pred, df_p_predp = integrate_ticket_parts_prediction(dfa, part_predp, plabels, p_thresh)
    
    assert(df_b_pred.index == df_p_pred.index)
    
    df_merged = df_b_pred.join(df_p_pred)
    dfp_merged = df_b_pred.join(df_p_predp)
    
    df_merged = df_merged.loc[df_merged['PARTS']==1,]
    dfp_merged = dfp_merged.loc[dfp_merged['PARTS']==1,]
    
    if len(df_merged) > 0:
        pred_merged = df_merged[plabels].values.tolist()
        predp_merged = dfp_merged[plabels].values.tolist()
    
    else:
        pred_merged = None
        predp_merged = None
        
    return pred_merged, predp_merged, df_p_predp.values.tolist()
