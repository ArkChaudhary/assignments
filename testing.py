import pandas_ta as pta,ast
from catboost import CatBoostRegressor,Pool,FeaturesData
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures,contextlib
from datetime import datetime,timedelta
from functools import partial
import gc
from hyperopt import hp,fmin,tpe,Trials,STATUS_OK
import io
from IPython.display import clear_output
from itertools import product
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.layers import Dense,Dropout,BatchNormalization
import lightgbm as lgb,keras_tuner as kt
from keras_tuner.tuners import Hyperband
import pandas as pd,math,matplotlib.pyplot as plt,numpy as np,os
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objs as go,pickle,random,re,seaborn as sns
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression,RFE
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split,TimeSeriesSplit
from sklearn.preprocessing import PowerTransformer,RobustScaler,LabelEncoder
from sklearn.utils import resample
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import sys,ta,tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import Input,Dense,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import threading,time
from tqdm.auto import tqdm
import traceback
from tslearn.metrics import dtw
from tvDatafeed import TvDatafeed,Interval
from typing import Dict,List,Union,Tuple
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping as XGBEarlyStopping
tv=TvDatafeed()
df1=pd.DataFrame(['high-volatility-cluster','moderate-volatility-cluster','isolated-high-volatility','neutral','low-volatility-cluster','volatility-reduction'])
le=LabelEncoder().fit(df1[0])
interval=Interval.in_30_minute
n=48
symbol_list=['PASTERNAKUSDT','FXYUSDT','MOODENGUSDT','JSMUSDT','SXTUSDT','GOATUSDT','PNUTUSDT','MRSUSDT','PIUSDT','NEIROUSDT','ETHFIUSDT','PEOPLEUSDT','NXQUSDT','BZRUSDT','TOSHIUSDT','INITUSDT','EIGENUSDT','BOMEUSDT','WHITEUSDT','CSPRUSDT','GIGAUSDT','DSYNCUSDT','WIFUSDT','KAITOUSDT','BRETTUSDT','SYRUPUSDT','PEPEUSDT','IOUSDT','MOGUSDT','ENAUSDT','SEEDUSDT','SOSUSDT','BABYDOGEUSDT','VIRTUALUSDT','SPXUSDT','AIXBTUSDT','PIXELUSDT','MSQUSDT','AI16ZUSDT','ORDIUSDT','PRIMEUSDT','SUMMITUSDT','PENGUUSDT','BERAUSDT','FARTCOINUSDT','ARBUSDT','BONKUSDT','ZKUSDT','OPUSDT','ATHUSDT','RAYUSDT','FLUIDUSDT','MELANIAUSDT','PYTHUSDT','POPCATUSDT','STRKUSDT','MEWUSDT','ARKMUSDT','BITUSDT','OCEANUSDT','UNIUSDT','GRASSUSDT','CHEXUSDT','FLOKIUSDT','TRUMPUSDT','ETHWUSDT','NOTUSDT','DOGEUSDT','COWUSDT','LDOUSDT','LEMXUSDT','OMUSDT','CHEEMSUSDT','IPUSDT','DEEPUSDT','MOVEUSDT','DOGUSDT','BABYUSDT','AIOZUSDT','AAIUSDT','ROSEUSDT','LAYERUSDT','TIAUSDT','SUSHIUSDT','MORPHOUSDT','ENSUSDT','SAFEUSDT','IMXUSDT','RONUSDT','KSMUSDT','TURBOUSDT','GALAUSDT','RUNEUSDT','FETUSDT','CFXUSDT','KASUSDT','WETHUSDT','SHIBUSDT','BEAMUSDT','LPTUSDT','ZENUSDT','HOTUSDT','CMETHUSDT','WOOUSDT','BETHUSDT','NEARUSDT','FXSUSDT','THETAUSDT','JUPUSDT','AKTUSDT','ETHUSDT','METHUSDT','WLDUSDT','PENDLEUSDT','AEROUSDT','SNEKUSDT','KDAUSDT','STETHUSDT','WBETHUSDT','CBETHUSDT','GRTUSDT','SUPERUSDT','WEETHUSDT','EGLDUSDT','LSETHUSDT','CVXUSDT','SNXUSDT','VANAUSDT','AXSUSDT','SMARTUSDT','WSTETHUSDT','ARUSDT','ZROUSDT','RSRUSDT','RETHUSDT','APEUSDT','RSETHUSDT','TELUSDT','APEPEUSDT','EZETHUSDT','ENJUSDT','QUBICUSDT','INJUSDT','PUFETHUSDT','WUSDT','WALUSDT','NEOUSDT','IOTAUSDT','RENDERUSDT','ONDOUSDT','POLYXUSDT','MINAUSDT','MEUSDT','AAVEUSDT','BORGUSDT','SANDUSDT','DYDXUSDT','DGBUSDT','SUIUSDT','IOTXUSDT','ONEUSDT','DRIFTUSDT','CELOUSDT','JTOUSDT','AVAXUSDT','FORMUSDT','EOSUSDT','PLUMEUSDT','HYPEUSDT','FLZUSDT','SEIUSDT','AXLUSDT','JASMYUSDT','CRVUSDT','MOCAUSDT','BLURUSDT','SFRXETHUSDT','VETUSDT','APTUSDT','SUSDT','BIGTIMEUSDT','GMXUSDT','HASUIUSDT','ALGOUSDT','IDUSDT','CKBUSDT','1INCHUSDT','EULUSDT','COTIUSDT','SKLUSDT','MANAUSDT','LINKUSDT','GNOUSDT','L1XUSDT','POLUSDT','DOTUSDT','WAVESUSDT','TAOUSDT','ADAUSDT','HUGEUSDT','MKRUSDT','SKYUSDT','YFIUSDT','ULTIMAUSDT','ETCUSDT','ACHUSDT','GASUSDT','LRCUSDT','HBARUSDT','XCHUSDT','FILUSDT','RYOUSDT','ZETAUSDT','GMTUSDT','QTUMUSDT','COMPUSDT','STXUSDT','RVNUSDT','LIONUSDT','CAKEUSDT','FTTUSDT','TRACUSDT','CHZUSDT','OSMOUSDT','ICPUSDT','BSVUSDT','ORCAUSDT','XTZUSDT','ZRXUSDT','VTHOUSDT','ASTRUSDT','ZILUSDT','ATOMUSDT','HNTUSDT','SOLUSDT','XCNUSDT','XLMUSDT','BARAUSDT','BNSOLUSDT','UXLINKUSDT','COREUSDT','DEXEUSDT','VCITYUSDT','FLOWUSDT','TONUSDT','BATUSDT','XRPUSDT','KAVAUSDT','JITOSOLUSDT','MSOLUSDT','ANKRUSDT','ZECUSDT','QNTUSDT','XECUSDT','HMSTRUSDT','LUNCUSDT','SCUSDT','AMPUSDT','CROUSDT','XEMUSDT','BCHUSDT','CTCUSDT','DASHUSDT','SUTUSDT','TFUELUSDT','BSOLUSDT','TUSDT','ABUSDT','KLAYUSDT','TWTUSDT','XYOUSDT','GLMUSDT','LTCUSDT','ZBUUSDT','KAIAUSDT','XMRUSDT','DCRUSDT','BGBUSDT','JSTUSDT','MNTUSDT','BTTUSDT','SFPUSDT','FLRUSDT','XDCUSDT','SAROSUSDT','TRIBEUSDT','BNBUSDT','SLISBNBUSDT']
high_volatility_crypto=['KTAUSDT','MRSUSDT','MOODENGUSDT','NEIROUSDT','PNUTUSDT','WLDUSDT','WIFUSDT','EIGENUSDT','FXSUSDT','TURBOUSDT','MASKUSDT','VIRTUALUSDT','ORDIUSDT','PEPEUSDT','ENAUSDT','SFPUSDT','KAITOUSDT','BABYUSDT','STRKUSDT','BONKUSDT','ETHFIUSDT','RUNEUSDT','FLOKIUSDT','SUPERUSDT','NOTUSDT','TRUMPUSDT','INJUSDT','PENGUUSDT','FETUSDT','PYTHUSDT','WUSDT','LDOUSDT','OMUSDT','SUSHIUSDT','AAVEUSDT','JUPUSDT','TIAUSDT','ROSEUSDT','OPUSDT','RAYUSDT','GALAUSDT','CRVUSDT','AEROUSDT','LPTUSDT','ZROUSDT','ONEUSDT','CVXUSDT','DYDXUSDT','THETAUSDT','UNIUSDT','RENDERUSDT','STXUSDT','TAOUSDT','ZKUSDT','RSRUSDT','GRTUSDT','JASMYUSDT','ARUSDT','IMXUSDT','SNXUSDT','IDUSDT','NEARUSDT','ARBUSDT','BERAUSDT','AMPUSDT','PENDLEUSDT','VANAUSDT','SUIUSDT','MANAUSDT','AVAXUSDT','KSMUSDT','CFXUSDT','DOGEUSDT','HOTUSDT','CELOUSDT','APEUSDT','ONDOUSDT','AXSUSDT','BLURUSDT','APTUSDT','ZECUSDT','ENSUSDT','LAYERUSDT','SANDUSDT','CKBUSDT','SUSDT','RVNUSDT','EOSUSDT','IOTAUSDT','MINAUSDT','SEIUSDT','EGLDUSDT','1INCHUSDT','JTOUSDT','ACHUSDT','LINKUSDT','AXLUSDT','NEOUSDT','MOVEUSDT','VETUSDT','MKRUSDT','SHIBUSDT','ATOMUSDT','ICPUSDT','VTHOUSDT','ETHUSDT','WBETHUSDT','DOTUSDT','ALGOUSDT','ADAUSDT','ZILUSDT','FILUSDT','CAKEUSDT','FLOWUSDT','FORMUSDT','CHZUSDT','ZRXUSDT','WSTETHUSDT','COMPUSDT','IOTXUSDT','FTTUSDT','BATUSDT','ETCUSDT','POLUSDT','SOLUSDT','QTUMUSDT','BNSOLUSDT','XTZUSDT','QNTUSDT','LTCUSDT','BCHUSDT','LUNCUSDT','DCRUSDT','XECUSDT','ASTRUSDT','HBARUSDT','GNOUSDT','YFIUSDT','GLMUSDT','DASHUSDT','KAVAUSDT','TFUELUSDT','TONUSDT','XLMUSDT','SCUSDT','GASUSDT','XRPUSDT','KAIAUSDT','JSTUSDT','TWTUSDT']
high_volatility_crypto=high_volatility_crypto+['BTCUSDT','ETHUSDT']
stock_symbol_list=['ABB','ACC','APLAPOLLO','AUBANK','AARTIIND','ADANIENSOL','ADANIENT','ADANIGREEN','ADANIPORTS','ATGL','ABCAPITAL','ABFRL','ALKEM','AMBUJACEM','ANGELONE','APOLLOHOSP','APOLLOTYRE','ASHOKLEY','ASIANPAINT','ASTRAL','AUROPHARMA','DMART','AXISBANK','BSOFT','BSE','AUTO','BAJFINANCE','BAJAJFINSV','BALKRISIND','BANDHANBNK','BANKBARODA','BANKINDIA','BERGEPAINT','BEL','BHARATFORG','BHEL','BPCL','BHARTIARTL','BIOCON','BOSCHLTD','BRITANNIA','CESC','CGPOWER','CANBK','CDSL','CHAMBLFERT','CHOLAFIN','CIPLA','COALINDIA','COFORGE','COLPAL','CAMS','CONCOR','CROMPTON','CUMMINSIND','CYIENT','DLF','DABUR','DALBHARAT','DEEPAKNTR','DELHIVERY','DIVISLAB','DIXON','DRREDDY','EICHERMOT','ESCORTS','EXIDEIND','NYKAA','GAIL','GMRAIRPORT','GLENMARK','GODREJCP','GODREJPROP','GRANULES','GRASIM','HCLTECH','HDFCAMC','HDFCBANK','HDFCLIFE','HFCL','HAVELLS','HEROMOTOCO','HINDALCO','HAL','HINDCOPPER','HINDPETRO','HINDUNILVR','HUDCO','ICICIBANK','ICICIGI','ICICIPRULI','IDFCFIRSTB','IIFL','IRB','ITC','INDIANB','IEX','IOC','IRCTC','IRFC','IREDA','IGL','INDUSTOWER','INDUSINDBK','NAUKRI','INFY','INDIGO','JKCEMENT','JSWENERGY','JSWSTEEL','JSL','JINDALSTEL','JIOFIN','JUBLFOOD','KEI','KPITTECH','KALYANKJIL','KOTAKBANK','LTF','LTTS','LICHSGFIN','LTIM','LT','LAURUSLABS','LICI','LUPIN','MRF','LODHA','MGL','MFIN','M','MANAPPURAM','MARICO','MARUTI','MFSL','MAXHEALTH','MPHASIS','MCX','MUTHOOTFIN','NBCC','NCC','NHPC','NMDC','NTPC','NATIONALUM','NESTLEIND','OBEROIRLTY','ONGC','OIL','PAYTM','OFSS','POLICYBZR','PIIND','PAGEIND','PATANJALI','PERSISTENT','PETRONET','PIDILITIND','PEL','POLYCAB','POONAWALLA','PFC','POWERGRID','PRESTIGE','PNB','RBLBANK','RECLTD','RELIANCE','SBICARD','SBILIFE','SHREECEM','SJVN','SRF','MOTHERSON','SHRIRAMFIN','SIEMENS','SOLARINDS','SONACOMS','SBIN','SAIL','SUNPHARMA','SUPREMEIND','SYNGENE','TATACONSUM','TITAGARH','TVSMOTOR','TATACHEM','TATACOMM','TCS','TATAELXSI','TATAMOTORS','TATAPOWER','TATASTEEL','TATATECH','TECHM','FEDERALBNK','INDHOTEL','PHOENIXLTD','RAMCOCEM','TITAN','TORNTPHARM','TORNTPOWER','TRENT','TIINDIA','UPL','ULTRACEMCO','UNIONBANK','UNITDSPR','VBL','VEDL','IDEA','VOLTAS','WIPRO','YESBANK','ZOMATO','ZYDUSLIFE']
high_volatility_stock=['WBD','TSLA','INTC','ENPH','FSLR','PCG','EIX','ALB','SJM','UHS','APA','ORCL','PLTR','ON','UAL','MRNA','VST','CZR','SMCI','MPWR','DAL','BG','BLDR','CPAY','COIN','ANET','CEG','LEN','TGT','AXON','AES','DHI','GEV','DOW','NCLH','PHM','MCHP','BF.B','EQT','CF','ISRG','NUE','FICO','HAL','NRG','WSM','NKE','BRO','TECH','HCA','EL','AMD','MOS','EXE','CRL','LVS','LULU','ODFL','AVGO','FANG','LUV','HII','APTV','DECK','TPL','CRWD','MU','REGN','CTRA','SWK','BBY','ALGN','PWR','MHK','RVTY','AJG','NEM','LYB','IQV','PODD','GE','DASH','HWM','BA','NXPI','COP','DVN','MGM','PAYC','SWKS','TDG','LRCX','WDC','VTRS','TMO','ALL','ADM','NVR','DELL','FCX','LMT','JBHT','LW','BAX','DAY','RTX','FTNT','ACGL','HSY','STX','MAS','SLB','STZ','TER','HUM','PYPL','ORLY','IPG','IVZ','ROST','CCL','ZTS','CAH','EOG','TXT','CAG','SYF','OMC','HPQ','POOL','TRGP','WST','PSX','ETN','MTD','KLAC','GPN','EPAM','RL','MTCH','EMN','UBER','WAT','MMC','APO','DXCM','SBUX','AIG','HSIC','MOH','FOX','AMAT','WTW','PGR','A','UNH','COO','QCOM','FOXA','DPZ','ERIE','CMG','V','TXN','WMB','RCL','AON','AAPL','GNRC','STLD','BX','NOC','ADI','TRV','NTAP','MRK','TPR','LDOS','IP','DVA','OXY','DLTR','LLY','SRE','NFLX','BIIB','KMX','LHX','INCY','AIZ','ULTA','FTV','APH','CRM','TMUS','WRB','LOW','SNPS','DHR','GPC','DG','VLO','GM','IT','GOOGL','GOOG','NDSN','CSGP','KVUE','ZBH']
crypto_path='/content/drive/MyDrive/Crypto'
stock_path='/content/drive/MyDrive/Stock'
used_symbols1=['LODHA','AUTO','NHPC','NATIONALUM','CYIENT','GAIL','HFCL','CANBK','INDUSTOWER','AUBANK','TATASTEEL','ESCORTS','HAL','NCC','TCS','SBIN','JSL','SONACOMS','AARTIIND','MANAPPURAM','BAJFINANCE','TVSMOTOR','POONAWALLA','BOSCHLTD','MCX','COALINDIA','NAUKRI','AXISBANK','GRASIM','KPITTECH','GRANULES','JSWSTEEL','AUROPHARMA','PATANJALI','PHOENIXLTD','SBICARD','APOLLOHOSP','PFC','KALYANKJIL','ADANIPORTS','ONGC','HDFCAMC','ABCAPITAL','PAGEIND','MUTHOOTFIN','HUDCO','ICICIGI','PEL','DLF','ASTRAL','INDIANB','KOTAKBANK','BANKBARODA','IRCTC','BAJAJFINSV','NESTLEIND','BHARTIARTL','SUPREMEIND','ZOMATO','NTPC']
used_symbols2=['LTF','LUPIN','DALBHARAT','BHARATFORG','CROMPTON','PNB','YESBANK','LTIM','PETRONET','SHRIRAMFIN','TITAN','DELHIVERY','RECLTD','FEDERALBNK','IOC','ITC','HEROMOTOCO','JUBLFOOD','GODREJPROP','SOLARINDS','TECHM','NBCC','VOLTAS','VBL','OIL','UPL','CHAMBLFERT','JKCEMENT','COLPAL','LICI','BEL','HINDALCO','TATATECH','INDHOTEL','APOLLOTYRE','RBLBANK','BRITANNIA','BERGEPAINT','CESC','TATACONSUM','MOTHERSON','SRF','ABB','GODREJCP','HINDUNILVR','PIDILITIND','TATACHEM','OFSS','IGL','TRENT','INFY','HINDPETRO','PRESTIGE','TIINDIA','HDFCBANK','RAMCOCEM','CIPLA','LICHSGFIN','BSE','IEX']
syms_in_pred=['CONCOR','ABFRL','BHEL','PERSISTENT','MFIN','TORNTPHARM','TATACOMM','PIIND','ADANIGREEN','PAYTM','MARICO','UNIONBANK','APLAPOLLO','ICICIBANK','LTTS','MRF','IDEA','MFSL','IRB','ANGELONE','SHREECEM','ADANIENT','LT','INDUSINDBK','IRFC','BALKRISIND','ALKEM','LAURUSLABS','ZYDUSLIFE','DEEPAKNTR','TORNTPOWER','ICICIPRULI','UNITDSPR','ADANIENSOL','BANKINDIA','SYNGENE','SBILIFE','ATGL','RELIANCE','HCLTECH','HDFCLIFE','MPHASIS','CAMS','ULTRACEMCO','TATAPOWER','TITAGARH','EICHERMOT','INDIGO','OBEROIRLTY','EXIDEIND','DABUR','JSWENERGY','CGPOWER','MGL','KEI','ASHOKLEY','CUMMINSIND','JINDALSTEL','AMBUJACEM','GLENMARK','DIVISLAB','MARUTI','SAIL','MAXHEALTH','BANDHANBNK','SJVN','TATAELXSI','COFORGE','GMRAIRPORT','WIPRO','POLYCAB','IREDA','ACC','SUNPHARMA','HINDCOPPER','CHOLAFIN','DRREDDY','JIOFIN','M','POWERGRID','VEDL','CDSL','BPCL','IDFCFIRSTB','NMDC','POLICYBZR','BSOFT','ASIANPAINT','TATAMOTORS','SIEMENS','DMART','HAVELLS','NYKAA','IIFL','BIOCON','DIXON']
used_syms=used_symbols1+used_symbols2
def get_unique_random_elements(available_items,excluded_items,num_elements):
	import random;valid_elements=[item for item in available_items if item not in excluded_items]
	if len(valid_elements)<num_elements:raise ValueError(f"Can only select {len(valid_elements)} unique elements, but {num_elements} were requested")
	return random.sample(valid_elements,num_elements)
_start_time=time.time()
def tic():global _start_time;_start_time=time.time()
def tok():t_sec=round(time.time()-_start_time);t_min,t_sec=divmod(t_sec,60);t_hour,t_min=divmod(t_min,60);print(f"----< TIME TAKEN: {t_hour} HOURS : {t_min} MINS : {t_sec} SECS >----")
def get_normalizer(folder_path):
	folder=f"{folder_path}/dataset/normalizers"
	if not os.path.exists(folder):os.makedirs(folder)
	try:
		with open(f"{folder}/scaler.pkl",'rb')as scaler_file:scaler=pickle.load(scaler_file)
		with open(f"{folder}/transformer.pkl",'rb')as transformer_file:transformer=pickle.load(transformer_file)
	except(FileNotFoundError,ValueError):scaler=RobustScaler();transformer=PowerTransformer(method='yeo-johnson')
	return scaler,transformer
def save_normalizer(scaler,transformer,folder_path):
	folder=f"{folder_path}/dataset/normalizers"
	if not os.path.exists(folder):os.makedirs(folder)
	scaler_file=f"{folder}/scaler.pkl";transformer_file=f"{folder}/transformer.pkl"
	with open(scaler_file,'wb')as sf:pickle.dump(scaler,sf)
	with open(transformer_file,'wb')as tf:pickle.dump(transformer,tf)
def fetch_and_save_data(symbol,exchange,interval,n_bars,file_name,folder_path):
	df=tv.get_hist(symbol=symbol,exchange=exchange,interval=interval,n_bars=n_bars)
	if df is None or df.empty:print(f"Failed to fetch data for {symbol}.");return False
	folder=f"{folder_path}/dataset/data"
	if not os.path.exists(folder):os.makedirs(folder)
	file_path=os.path.join(folder,f"{file_name}.csv")
	if os.path.exists(file_path):
		try:
			existing_data=pd.read_csv(file_path,index_col=0,parse_dates=True);new_data=df[~df.index.isin(existing_data.index)]
			if not new_data.empty:updated_data=pd.concat([existing_data,new_data]).sort_index();updated_data.to_csv(file_path,index=True);print(f"New data appended for {file_name}.")
			else:print(f"No new data to append for {file_name}.")
		except Exception as e:print(f"Error reading {file_path}: {e}");return False
	else:df.to_csv(file_path,index=True);print(f"Data saved for {file_name}.")
	return True
def add_data(symbol_list,exchange,ref,ref_exchange,interval=None,n_bars=15000,folder_path=None,max_workers=12):
	def process_symbol(symbol,is_reference=False):
		try:
			if is_reference:result=fetch_and_save_data(symbol,ref_exchange,interval,n_bars,folder_path=folder_path,file_name='reference')
			else:result=fetch_and_save_data(symbol,exchange,interval,n_bars,file_name=symbol,folder_path=folder_path)
			return{'symbol':symbol,'is_reference':is_reference,'result':result}
		except Exception as exc:error_msg=f"Error processing {"reference"if is_reference else"symbol"} {symbol}: {exc}";print(error_msg);return{'symbol':symbol,'is_reference':is_reference,'result':False}
	all_tasks=[(symbol,False)for symbol in symbol_list]+[(ref,True)];results=[];unsuccessful=[];print(f"Starting parallel processing for {len(all_tasks)} tasks...")
	with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)as executor:
		future_to_task={executor.submit(process_symbol,symbol,is_ref):(symbol,is_ref)for(symbol,is_ref)in all_tasks}
		for future in concurrent.futures.as_completed(future_to_task):
			symbol,is_ref=future_to_task[future]
			try:result=future.result();results.append(result);print(f"Completed {"reference"if is_ref else"symbol"}: {symbol}")
			except Exception as exc:
				print(f"Unexpected error with {"reference"if is_ref else"symbol"} {symbol}: {exc}")
				if is_ref:unsuccessful.append('REFERENCE ERROR')
				else:unsuccessful.append(symbol)
	print(f"\nProcess complete. Processed {len(results)} items.");return unsuccessful
def add_prediction_data_optimized(symbol_list,folder_path=None):
	folder_path=Path(folder_path or'.').resolve();pred_data_dir=folder_path/'dataset'/'predicted_data';data_dir=folder_path/'dataset'/'data';pred_data_dir.mkdir(parents=True,exist_ok=True);results=[];logs=[];print(f"[INFO] Base folder resolved to: {folder_path}");print('[INFO] Phase 1: Loading source data and existing prediction files...');symbol_data={};loaded_count=0
	for(i,symbol)in enumerate(symbol_list):
		data_file=data_dir/f"{symbol}.csv";pred_file=pred_data_dir/f"{symbol}_preds.csv"
		if i%10==0 or i==len(symbol_list)-1:print(f"[INFO] Loading {i+1}/{len(symbol_list)} symbols")
		if not data_file.exists():results.append(f"Source data missing for {symbol}");continue
		try:
			df=pd.read_csv(data_file,index_col='datetime',parse_dates=True)
			if pred_file.exists():
				try:existing_preds=pd.read_csv(pred_file,index_col=0,parse_dates=True);start_date=existing_preds.dropna().index.max()if not existing_preds.empty else None;action='appended'
				except Exception:existing_preds=pd.DataFrame();start_date=None;action='recreated'
			else:existing_preds=pd.DataFrame();start_date=None;action='created'
			symbol_data[symbol]={'source_df':df,'existing_preds':existing_preds,'start_date':start_date,'action':action,'pred_file':pred_file};loaded_count+=1
		except Exception as e:logs.append(f"--- Error loading {symbol} ---\n{traceback.format_exc()}");results.append(f"Error loading {symbol}: {str(e)}")
	print(f"[INFO] Loaded data for {loaded_count}/{len(symbol_list)} symbols")
	if not symbol_data:print('[WARN] No valid data to process!');return results
	batch_size=min(20,len(symbol_data));all_symbols=list(symbol_data.keys());total_batches=(len(all_symbols)+batch_size-1)//batch_size;print(f"[INFO] Phase 2: Processing predictions in {total_batches} batches of {batch_size}...")
	for batch_idx in range(0,len(all_symbols),batch_size):
		batch_symbols=all_symbols[batch_idx:batch_idx+batch_size];print(f"[INFO] Processing batch {batch_idx//batch_size+1}/{total_batches}...")
		with concurrent.futures.ThreadPoolExecutor(max_workers=min(8,len(batch_symbols)))as executor:
			def process_symbol(symbol):
				try:
					data=symbol_data[symbol];df=data['source_df'];df,df_main=add_columns_for_prediction(df,add_future_vals=True,add_volatility=True,folder_path=folder_path);print(f"[DEBUG] start date for symbol {symbol}: {data["start_date"]}");df=stack_predict(df,df_main,data['start_date'],adjust_preds=True,folder_path=folder_path)
					if df.empty:print(f"[DEBUG] No predictions generated for {symbol} (empty df after stack_predict)");return symbol,f"No predictions for {symbol}",''
					updated_data=pd.concat([data['existing_preds'],df],sort=False)if not data['existing_preds'].empty else df;updated_data=updated_data[~updated_data.index.duplicated(keep='last')].sort_index();updated_data=updated_data.copy()
					try:updated_data.to_csv(data['pred_file']);file_size=data['pred_file'].stat().st_size;print(f"[INFO] Saved {len(updated_data)} rows to {data["pred_file"]} (size: {file_size} bytes)")
					except Exception as save_err:return symbol,f"Failed to save predictions for {symbol}: {save_err}",traceback.format_exc()
					return symbol,f"Prediction data {data["action"]} for {symbol}",''
				except Exception as e:return symbol,f"Error processing {symbol}: {e}",traceback.format_exc()
			futures={executor.submit(process_symbol,sym):sym for sym in batch_symbols}
			for future in concurrent.futures.as_completed(futures):
				symbol=futures[future]
				try:
					sym,result,log=future.result();results.append(result)
					if log:logs.append(f"--- Error in {sym} ---\n{log}")
				except Exception as exc:results.append(f"Unhandled error processing {symbol}: {exc}")
		for sym in batch_symbols:symbol_data[sym].clear()
		gc.collect()
	summary={'created':sum('created'in r for r in results),'appended':sum('appended'in r for r in results),'recreated':sum('recreated'in r for r in results),'errors':sum('Error'in r for r in results)};print('\n[INFO] Summary:')
	for(k,v)in summary.items():print(f"- {k.title()}: {v}")
	log_file=folder_path/'prediction_logs.txt'
	with open(log_file,'w')as f:f.write('\n'.join(logs))
	print('[INFO] Phase 3: Running post-prediction analysis...')
	try:analysis=analyse_predictions(symbol_list,'day_signal',folder_path);analysis.to_csv(folder_path/'analysis.csv');print('[INFO] Analysis saved successfully.')
	except Exception as e:print(f"[ERROR] Failed during analysis: {e}");logs.append(f"--- Analysis Error ---\n{traceback.format_exc()}")
	return results
def refresh_crypto(n_bars=1600):add_data(high_volatility_crypto,'BINANCE','BTCUSDT','BINANCE',interval=interval,n_bars=n_bars,folder_path=crypto_path,max_workers=4);gc.collect();add_prediction_data_optimized(high_volatility_crypto,folder_path=crypto_path);gc.collect()
def refresh_stocks(n_bars=1600):add_data(high_volatility_stock,'NSE','NIFTY','NSE',interval=interval,n_bars=n_bars,folder_path=stock_path,max_workers=4);clear_output();gc.collect();add_prediction_data_optimized(high_volatility_stock,folder_path=stock_path);clear_output();gc.collect()
def join_closest_datetime(df1,df2):df1_temp=df1.reset_index();df2_temp=df2.reset_index();df1_temp=df1_temp.rename(columns={'index':'datetime'});df2_temp=df2_temp.rename(columns={'index':'datetime'});df1_temp=df1_temp.sort_values('datetime');df2_temp=df2_temp.sort_values('datetime');merged_df=pd.merge_asof(df1_temp,df2_temp,on='datetime',direction='nearest');merged_df.set_index('datetime',inplace=True);return merged_df
def add_reference(df_,folder_path):
	df=df_.copy();df_ref=pd.read_csv(f"{folder_path}/dataset/data/reference.csv",index_col=0,parse_dates=True)
	if'symbol'in df_ref.columns:df_ref.drop('symbol',axis=1,inplace=True)
	df_ref.rename(columns={'open':'ref_open','high':'ref_high','low':'ref_low','close':'ref_close','volume':'ref_volume'},inplace=True)
	if'ref_volume'in df_ref.columns:df_ref.drop('ref_volume',axis=1,inplace=True)
	df=join_closest_datetime(df,df_ref);df_ref=df.loc[:,'ref_open':];return df,df_ref
def add_base_columns(df):original_index=df.index;df['prev_day_close']=df['close'].shift(n);df['daily_volume_change%']=df['volume'].pct_change(periods=n)*100;df['weekly_volume_change%']=df['volume'].pct_change(periods=n*7)*100;df['fortnightly_volume_change%']=df['volume'].pct_change(periods=n*14)*100;df['monthly_volume_change%']=df['volume'].pct_change(periods=n*30)*100;df['log_return']=np.log(df['close']/df['close'].shift(1));df['daily_volatility%']=abs((df['log_return'].rolling(window=n).std()*np.sqrt(n)*100)**2);df['weekly_volatility%']=abs((df['log_return'].rolling(window=n*7).std()*np.sqrt(7)*100)**2);df['fortnightly_volatility%']=abs((df['log_return'].rolling(window=n*14).std()*np.sqrt(14)*100)**2);df['monthly_volatility%']=abs((df['log_return'].rolling(window=n*30).std()*np.sqrt(30)*100)**2);df['daily_volatility_change%']=df['daily_volatility%']-df['daily_volatility%'].shift(n);df['daily_high_%chg']=df['high'].pct_change(periods=n*1)*100;df['daily_low_%chg']=df['low'].pct_change(periods=n*1)*100;df['daily_close_%chg']=df['close'].pct_change(periods=n*1)*100;df['weekly_high_%chg']=df['high'].pct_change(periods=n*7)*100;df['weekly_low_%chg']=df['low'].pct_change(periods=n*7)*100;df['weekly_close_%chg']=df['close'].pct_change(periods=n*7)*100;df['daily_high']=df['high'].rolling('1D').max();df[f"price_high_daily_%diff"]=(df['close']-df['daily_high'])/df['close']*100;df['daily_low']=df['low'].rolling('1D').min();df[f"price_low_daily_%diff"]=(df['close']-df['daily_low'])/df['close']*100;df['weekly_high']=df['high'].rolling('7D').max();df[f"price_high_weekly_%diff"]=(df['close']-df['weekly_high'])/df['close']*100;df['weekly_low']=df['low'].rolling('7D').min();df[f"price_low_weekly_%diff"]=(df['close']-df['weekly_low'])/df['close']*100;return df
def add_base_columns_ref(df_):df=df_.copy();df['ref_log_return']=np.log(df['ref_close']/df['ref_close'].shift(1));df['ref_daily_volatility%']=abs((df['ref_log_return'].rolling(window=n).std()*np.sqrt(n)*100)**2);df['ref_weekly_volatility%']=abs((df['ref_log_return'].rolling(window=n*7).std()*np.sqrt(7)*100)**2);df['ref_fortnightly_volatility%']=abs((df['ref_log_return'].rolling(window=n*14).std()*np.sqrt(14)*100)**2);df['ref_monthly_volatility%']=abs((df['ref_log_return'].rolling(window=n*30).std()*np.sqrt(30)*100)**2);df['close_%chg']=df['close'].pct_change(periods=n)*100;df['ref_close_%chg']=df['ref_close'].pct_change(periods=n)*100;df['correlation_daily']=df['close_%chg'].rolling(window=n).corr(df['ref_close_%chg']);df['correlation_weekly']=df['close_%chg'].rolling(window=n*7).corr(df['ref_close_%chg']);df['correlation_fortnightly']=df['close_%chg'].rolling(window=n*14).corr(df['ref_close_%chg']);df['correlation_monthly']=df['close_%chg'].rolling(window=n*30).corr(df['ref_close_%chg']);return df
def add_p_chg_columns(df_,columns,n):
	df=df_.copy()
	for col in columns:columns=df.loc[:,col:];df=df.loc[:,:col];df[f"{col}_%chg"]=df[col].pct_change(periods=n)*100;df=df.join(columns.iloc[:,1:],how='left')
	return df
def calculate_means(df_,windows=[50,75,100,200]):
	df=df_.copy();means={}
	for window in windows:means[f"close_mean_{window}"]=df['close'].rolling(window=window,min_periods=1).mean();means[f"high_mean_{window}"]=df['high'].rolling(window=window,min_periods=1).mean();means[f"low_mean_{window}"]=df['low'].rolling(window=window,min_periods=1).mean();means[f"daily_volatility_mean_{window}"]=df['daily_volatility%'].rolling(window=window,min_periods=1).mean();means[f"volume_mean_{window}"]=df['volume'].rolling(window=window,min_periods=1).mean()
	means['rsi_mean_14']=df['RSI'].rolling(window=14,min_periods=1).mean();means['adx_mean_14']=df['ADX'].rolling(window=14,min_periods=1).mean();means['cci_mean_20']=df['cci-20'].rolling(window=20,min_periods=1).mean()
	for(key,value)in means.items():df[key]=value
	return df
def calculate_zscores(df_,windows=[50,75,100,200]):
	df=df_.copy()
	def rolling_zscore(series,window):mean=series.rolling(window=window,min_periods=1).mean();std=series.rolling(window=window,min_periods=1).std();std=std.replace(0,np.nan);zscore=(series-mean)/std;zscore=zscore.bfill().ffill();return zscore
	for window in windows:df[f"close_zscore_{window}"]=rolling_zscore(df['close'],window);df[f"high_zscore_{window}"]=rolling_zscore(df['high'],window);df[f"low_zscore_{window}"]=rolling_zscore(df['low'],window);df[f"volume_zscore_{window}"]=rolling_zscore(df['volume'],window);df[f"volatility_zscore_{window}"]=rolling_zscore(df['daily_volatility%'],window)
	df['rsi_zscore']=rolling_zscore(df['RSI'],14);df['adx_zscore']=rolling_zscore(df['ADX'],14);df['cci_zscore']=rolling_zscore(df['cci-20'],20);return df
def categorize_trends_for_window(df_,window,reversion_threshold=.8):
	df=df_;mean_col=f"close_mean_{window}";categories=pd.Series(index=df.index,data='neutral')
	if mean_col not in df.columns:df[mean_col]=df['close'].rolling(window=window).mean()
	valid_indices=df.index[window:];recent_daily_vol=df['daily_volatility%'].abs().rolling(window=window).mean().shift(1);recent_weekly_vol=df['weekly_volatility%'].abs().rolling(window=window).mean().shift(1);recent_fortnightly_vol=df['fortnightly_volatility%'].abs().rolling(window=window).mean().shift(1);recent_monthly_vol=df['monthly_volatility%'].abs().rolling(window=window).mean().shift(1);current_deviation=(df['close']-df[mean_col]).abs()/df[mean_col]*100;high_daily_vol_mask=recent_daily_vol>18.5;high_deviation_mask=current_deviation>2.;complex_volatility_mask=(recent_weekly_vol>4.5)|(recent_monthly_vol>17.5)|(recent_fortnightly_vol>8.7);low_fortnight_vol_mask=recent_fortnightly_vol<reversion_threshold**2*4.;categories[valid_indices]='neutral';high_vol_indices=valid_indices[high_daily_vol_mask[valid_indices]];high_dev_indices=high_vol_indices[high_deviation_mask[high_vol_indices]];complex_vol_indices=high_dev_indices[complex_volatility_mask[high_dev_indices]];moderate_vol_indices=high_dev_indices[~complex_volatility_mask[high_dev_indices]];isolated_vol_indices=high_vol_indices[~high_deviation_mask[high_vol_indices]];categories[complex_vol_indices]='high-volatility-cluster';categories[moderate_vol_indices]='moderate-volatility-cluster';categories[isolated_vol_indices]='isolated-high-volatility';low_vol_indices=valid_indices[~high_daily_vol_mask[valid_indices]];low_fortnight_indices=low_vol_indices[low_fortnight_vol_mask[low_vol_indices]];vol_reduction_indices=low_vol_indices[~low_fortnight_vol_mask[low_vol_indices]];categories[low_fortnight_indices]='low-volatility-cluster';categories[vol_reduction_indices]='volatility-reduction';return categories
def categorize_trends(df_,windows=[int(n*.25),n*1,n*2,n*3],reversion_threshold=.8):
	df=df_.copy()
	for window in windows:df[f"close_mean_{window}"]=df['close'].rolling(window=window).mean()
	trend_categories={}
	for window in windows:trend_categories[f"trend_category_{window}"]=categorize_trends_for_window(df,window,reversion_threshold)
	for(key,value)in trend_categories.items():df[key]=value
	return df.loc[:,f"trend_category_{windows[0]}":]
def analyze_mean_reversion(df_,windows=[int(n*.25),n*1,n*2,n*3],trend_columns=True):
	df=df_.copy();df=calculate_means(df,windows);df=calculate_zscores(df,windows);trend_columns=None
	if not trend_columns:trend_columns=categorize_trends(df,windows)
	return df,trend_columns
def add_rolling_correlation(df_,col1,col2,n):
	df=df_.copy();windows=[n*1,n*3,n*7]
	for window in windows:column_name=f"rolling_corr_price-ref_{window}";df[column_name]=(1/df[col1].rolling(window=window).corr(df[col2]))**2
	return df
def cap_extreme_values(df,cap=1e6):
	df=df.copy();numeric_cols=df.select_dtypes(include=[np.number]).columns
	for col in numeric_cols:df[col]=df[col].clip(lower=-cap,upper=cap)
	return df
def add_target_variables(df_):df=df_.copy();df['next_day_close_%chg']=(df['close'].shift(-n)-df['close'])/df['close']*100;df['next_day_high_%chg']=(df['high'].rolling(window=n).max().shift(-n)-df['close'])/df['close']*100;df['next_day_low_%chg']=(df['low'].rolling(window=n).min().shift(-n)-df['close'])/df['close']*100;df['next_week_close_%chg']=(df['high'].rolling(window=n).mean().shift(-n*7)-df['close'])/df['close']*100;df['next_week_high_%chg']=(df['high'].rolling(window=n*7).max().shift(-n*7)-df['close'])/df['close']*100;df['next_week_low_%chg']=(df['low'].rolling(window=n*7).min().shift(-n*7)-df['close'])/df['close']*100;return df.loc[:,'next_day_close_%chg':]
def normalize_data(df_,scaler,transformer,fit=True):
	df=df_.copy();original_index=df.index;split_idx=len(df)//2;train_data=df.iloc[:split_idx,1:];test_data=df.iloc[split_idx:,1:]
	if fit:scaler.fit(train_data);train_scaled=scaler.transform(train_data);transformer.fit(train_scaled)
	else:train_scaled=scaler.transform(train_data)
	train_transformed=scaler.transform(train_scaled);test_scaled=scaler.transform(test_data);test_transformed=scaler.transform(test_scaled);transformed_data=pd.concat([pd.DataFrame(train_transformed,columns=df.columns[1:],index=original_index[:split_idx]),pd.DataFrame(test_transformed,columns=df.columns[1:],index=original_index[split_idx:])]);transformed_data.insert(0,df.columns[0],df[df.columns[0]]);return transformed_data
def add_ema_diff(df,col_name,multiplier,ema_period,suffix):ema_col=f"EMA_{ema_period}{suffix}_{col_name}";diff_col=f"Price_{ema_col}{suffix}_%diff";df[ema_col]=pta.ema(df[col_name],length=multiplier*ema_period);df[diff_col]=(df['close']-df[ema_col])/df[ema_col]*100*((df['close']-df[ema_col])/df[ema_col])*100;return df
def add_EMA(df):m=2;add_ema_diff(df,'high',1,20,'');add_ema_diff(df,'low',1,20,'');add_ema_diff(df,'close',1,100,'');add_ema_diff(df,'close',1,200,'');add_ema_diff(df,'high',m,20,'_daily');add_ema_diff(df,'low',m,20,'_daily');add_ema_diff(df,'close',m,100,'_daily');add_ema_diff(df,'close',m,200,'_daily');add_ema_diff(df,'high',m*7,20,'_weekly');add_ema_diff(df,'low',m*7,20,'_weekly');return df
def add_bollinger_bands(df,col_name,period,std_dev):rolling_mean=df[col_name].rolling(window=period).mean();rolling_std=df[col_name].rolling(window=period).std();Bollinger_upper=rolling_mean+std_dev*rolling_std;Bollinger_lower=rolling_mean-std_dev*rolling_std;df[f"Price_Bollinger_upper_{period}_%diff"]=(1/((df[col_name]-Bollinger_upper)/Bollinger_upper)*100)**2;df[f"Price_Bollinger_lower_{period}_%diff"]=(1/((df[col_name]-Bollinger_lower)/Bollinger_lower)*100)**2;return df
def add_bb(df):df=add_bollinger_bands(df,'close',20,2);df=add_bollinger_bands(df,'daily_high',20,2);df=add_bollinger_bands(df,'daily_low',20,2);return df
def add_sma_diff(df,col_name,sma_period):sma_col=f"SMA_{sma_period}_{col_name}";diff_col=f"Price_{sma_col}_%diff";df[sma_col]=df[col_name].rolling(window=sma_period).mean();df[diff_col]=(df['close']-df[sma_col])/df[sma_col]*100*((df['close']-df[sma_col])/df[sma_col])*100;return df
def add_sma(df):add_sma_diff(df,'close',50);add_sma_diff(df,'daily_high',50);add_sma_diff(df,'daily_low',50);return df
def resample_to_4hr(df):df=df.resample('4H').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'});return df
def dmi(df,length=14):dmi=pta.adx(df['high'],df['low'],df['close'],length=length);df['DMI_positive']=dmi[f"DMP_{length}"];df['DMI_negative']=dmi[f"DMN_{length}"];return df
def adx(df,length=14):df['ADX']=pta.adx(df['high'],df['low'],df['close'],length=length)['ADX_14'];return df
def rsi(df,length=14):df['RSI']=pta.rsi(df['close'],length=length);return df
def atr(df,length=14):df['ATR']=ta.volatility.AverageTrueRange(high=df['high'],low=df['low'],close=df['close'],window=length).average_true_range();return df
def add_advanced_indicators(df):
	for period in[5,10,15]:df[f"stochs-{period}"]=(df['close']-df['low'].rolling(period).min())/(df['high'].rolling(period).max()-df['low'].rolling(period).min())*100;df[f"slope_stochs-{period}"]=df[f"stochs-{period}"].diff()
	df['acc_dist']=(df['close']-df['low']-(df['high']-df['close']))/(df['high']-df['low']+1e-09)*df['volume'];df['slope_acc_dist']=df['acc_dist'].diff()
	for period in[5,10,20]:distance_moved=((df['high']+df['low'])/2).diff();box_ratio=df['volume']/(df['high']-df['low']+1e-09);df[f"eom-{period}"]=(distance_moved/box_ratio).rolling(period).mean();df[f"slope_eom-{period}"]=df[f"eom-{period}"].diff()
	for period in[5,10,20]:tp=(df['high']+df['low']+df['close'])/3;ma=tp.rolling(period).mean();md=tp.rolling(period).apply(lambda x:np.mean(np.abs(x-x.mean())),raw=True);df[f"cci-{period}"]=(tp-ma)/(.015*md);df[f"slope_cci-{period}"]=df[f"cci-{period}"].diff()
	return df
def add_indicators(df_):df=df_.copy();df=add_EMA(df);df=add_sma(df);df=add_bb(df);df=dmi(df);df=adx(df);df=rsi(df);df=atr(df);df=add_advanced_indicators(df);return df
percent_change_columns=['open','high','low','volume','prev_day_close','EMA_20_high','EMA_20_low','EMA_100_close','EMA_200_close','EMA_20_daily_high','EMA_20_daily_low','EMA_100_daily_close','EMA_200_daily_close','SMA_50_close','SMA_50_daily_high','SMA_50_daily_low','ref_open','ref_high','ref_low']
drop_columns=['close','ref_close','daily_high','daily_low','daily_close','weekly_high','weekly_low','weekly_close','fortnightly_high','fortnightly_low','fortnightly_close','monthly_high','monthly_low','monthly_close']+[f"EMA_20_{timeframe}_high"for timeframe in['weekly','fortnightly','monthly']]+[f"EMA_20_{timeframe}_low"for timeframe in['weekly','fortnightly','monthly']]+[f"EMA_100_{timeframe}_close"for timeframe in['weekly','fortnightly','monthly']]+[f"EMA_200_{timeframe}_close"for timeframe in['weekly','fortnightly','monthly']]+[f"high_mean_{int(n*mult)}"for mult in[.25,1,2,3]]+[f"low_mean_{int(n*mult)}"for mult in[.25,1,2,3]]+[f"volume_mean_{int(n*mult)}"for mult in[.25,1,2,3]]+[f"close_mean_{int(n*mult)}"for mult in[.25,1,2,3]]
def preprocess_symbol(symbol,folder_path):
	df_main=pd.read_csv(f"{folder_path}/dataset/data/{symbol}.csv",index_col=['datetime'],parse_dates=True);df=df_main.copy();df_processed=add_base_columns(df);df_processed=add_indicators(df_processed);df_processed,df_ref=add_reference(df_processed,folder_path=folder_path);df_processed=add_base_columns_ref(df_processed);df_processed,trend_columns=analyze_mean_reversion(df_processed,[int(n*.25),n*1,n*2,n*3]);df_processed=add_p_chg_columns(df_processed,percent_change_columns,1);target=add_target_variables(df_processed);df_processed=add_rolling_correlation(df_processed,'close_%chg','ref_close_%chg',n)
	for column in drop_columns:
		if column in df_processed.columns:df_processed=df_processed.drop(column,axis=1,inplace=False)
	df_processed=df_processed.drop(percent_change_columns,axis=1,inplace=False);df_processed=cap_extreme_values(df_processed);df_processed=join_closest_datetime(df_processed,trend_columns);df_processed=join_closest_datetime(df_processed,target);df_main.drop('symbol',axis=1,inplace=True);df_processed.replace([np.inf,-np.inf],np.nan,inplace=True);df_processed=df_processed.dropna();return df_processed,df_main,df_ref
def test_train_split_main(df_,test_perc,num_rows,target_column):
	data=df_.copy();data=data.drop(['symbol'],axis=1);columns_to_encode=[f"trend_category_{mult}"for mult in[int(n*.25),n*1,n*2,n*3]];target_columns=['next_day_close_%chg','next_day_high_%chg','next_day_low_%chg','next_week_close_%chg','next_week_high_%chg','next_week_low_%chg','next_fortnight_close_%chg','next_fortnight_high_%chg','next_fortnight_low_%chg'];data.drop(columns=[col for col in target_columns if col in data.columns and col!=target_column],inplace=True);data.replace([np.inf,-np.inf],np.nan,inplace=True);data.dropna(inplace=True)
	if len(data)<num_rows:print(f"Warning: Requested {num_rows} rows, but only {len(data)} available after balancing.");num_rows=len(data)
	data=data.sample(n=num_rows,random_state=42);data.sort_index(inplace=True)
	for column in columns_to_encode:
		if column in data.columns:data[column]=le.transform(data[column])
	Q1=data[target_column].quantile(.25);Q3=data[target_column].quantile(.75);IQR=Q3-Q1;lower_bound=Q1-2.8*IQR;upper_bound=Q3+2.8*IQR;filtered_data=data[(data[target_column]>=lower_bound)&(data[target_column]<=upper_bound)];mean_target=filtered_data[target_column].mean();std_target=filtered_data[target_column].std();data['weights']=np.exp(abs((data[target_column]-mean_target)/std_target));data['weights']=np.clip(data['weights'],None,35);random_state=48;total_samples=len(data);test_size=int(total_samples*test_perc/100);val_size=int((total_samples-test_size)*test_perc/100);test_start=total_samples-test_size;val_start=test_start-val_size;test_data=data.iloc[test_start:];val_data=data.iloc[val_start:test_start];train_data=data.iloc[:val_start];test_data=test_data.sample(frac=1,random_state=random_state);val_data=val_data.sample(frac=1,random_state=random_state);train_data=train_data.sample(frac=1,random_state=random_state);X_test=test_data.drop(columns=[target_column,'weights','datetime']);y_test=test_data[target_column];weights_test=test_data['weights'];X_val=val_data.drop(columns=[target_column,'weights','datetime']);y_val=val_data[target_column];weights_val=val_data['weights'];X_train=train_data.drop(columns=[target_column,'weights','datetime']);y_train=train_data[target_column];weights_train=train_data['weights'];return X_train,y_train,weights_train,X_val,y_val,weights_val,X_test,y_test,weights_test
def save_models(models,filename,folder_path):
	with open(f"{folder_path}/models/{filename}",'wb')as f:pickle.dump(models,f)
def load_models(filename,folder_path):
	with open(f"{folder_path}/models/{filename}",'rb')as f:return pickle.load(f)
target_column_list=['next_day_close_%chg','next_day_high_%chg','next_day_low_%chg','next_week_close_%chg','next_week_high_%chg','next_week_low_%chg']
features_week=['ref_monthly_volatility%','monthly_volatility%','ref_weekly_volatility%','fortnightly_volatility%','ATR','ref_fortnightly_volatility%','daily_volatility_change%','correlation_monthly','correlation_fortnightly','weekly_volatility%','SMA_50_daily_high_%chg','ref_daily_volatility%','EMA_200_close_%chg','SMA_50_daily_low_%chg','correlation_weekly','daily_volatility_mean_144','weekly_low_%chg','daily_volatility_mean_96','price_high_weekly_%diff','price_low_weekly_%diff','weekly_high_%chg','EMA_200_daily_close_%chg','daily_volatility_mean_48','rolling_corr_price-ref_144','weekly_close_%chg','ref_close_%chg','daily_volatility%','volatility_zscore_144','Price_EMA_20_weekly_low_weekly_%diff','daily_volatility_mean_12','EMA_100_close_%chg','high_zscore_144','EMA_100_daily_close_%chg','Price_EMA_200_daily_close_daily_%diff','low_zscore_144','Price_EMA_20_weekly_high_weekly_%diff','close_zscore_144','correlation_daily','price_low_daily_%diff','rolling_corr_price-ref_336','ref_open_%chg','ref_log_return','Price_SMA_50_daily_low_%diff','ref_low_%chg','Price_SMA_50_daily_high_%diff']
features_day=['ref_weekly_volatility%','ref_monthly_volatility%','ref_fortnightly_volatility%','ref_daily_volatility%','EMA_100_close_%chg','ref_close_%chg','Price_SMA_50_daily_high_%diff','SMA_50_daily_low_%chg','Price_EMA_20_weekly_low_weekly_%diff','SMA_50_daily_high_%chg','weekly_close_%chg','price_high_daily_%diff','monthly_volatility%','weekly_high_%chg','correlation_monthly','fortnightly_volatility%','correlation_fortnightly','ATR','volatility_zscore_144','EMA_20_daily_low_%chg','daily_volatility_mean_144','volatility_zscore_96','EMA_200_close_%chg','weekly_volatility%','Price_EMA_200_close_%diff','correlation_weekly','daily_volatility_mean_96','correlation_daily','Price_EMA_200_daily_close_daily_%diff','weekly_low_%chg','EMA_20_daily_high_%chg','Price_EMA_20_high_%diff','daily_volatility_mean_12','daily_volatility%','EMA_200_daily_close_%chg','daily_volatility_mean_48','ref_low_%chg','close_zscore_96','rolling_corr_price-ref_336','ref_open_%chg','price_high_weekly_%diff','rolling_corr_price-ref_48','price_low_daily_%diff','Price_EMA_100_close_%diff','Price_SMA_50_close_%diff']
features_meta_week=['ref_fortnightly_volatility%','monthly_volatility%','ref_monthly_volatility%','ref_weekly_volatility%','correlation_monthly','correlation_fortnightly','fortnightly_volatility%','weekly_volatility%','ATR','rolling_corr_price-ref_144','Price_EMA_200_close_%diff','ref_daily_volatility%','weekly_high_%chg','weekly_low_%chg','SMA_50_daily_low_%chg','daily_volatility_mean_144','weekly_close_%chg','daily_volatility_mean_96','price_low_weekly_%diff','ref_close_%chg','price_high_weekly_%diff','Price_EMA_20_weekly_high_weekly_%diff','daily_volatility_mean_48','daily_volatility_change%','Price_EMA_200_daily_close_daily_%diff','price_low_daily_%diff','volatility_zscore_48','low_zscore_144','Price_Bollinger_upper_20_%diff','price_high_daily_%diff','rolling_corr_price-ref_336','trend_category_48','eom-20','eom-5']+[f"Lightgbm_{target_column}_pred"for target_column in target_column_list]+[f"MLP_{target_column}_pred"for target_column in target_column_list]+[f"CatBoost_{target_column}_pred"for target_column in target_column_list]+[f"NN_{target_column}_pred"for target_column in target_column_list]
features_meta_day=['ref_fortnightly_volatility%','ref_monthly_volatility%','ref_weekly_volatility%','ref_daily_volatility%','ref_close_%chg','monthly_volatility%','weekly_volatility%','rolling_corr_price-ref_144','correlation_monthly','daily_volatility_mean_144','fortnightly_volatility%','SMA_50_daily_low_%chg','correlation_fortnightly','EMA_200_close_%chg','SMA_50_daily_high_%chg','weekly_close_%chg','EMA_200_daily_close_%chg','daily_volatility_mean_48','price_low_weekly_%diff','daily_volatility_mean_96','weekly_low_%chg','daily_volatility%','weekly_high_%chg','ATR','Price_EMA_20_weekly_low_weekly_%diff','Price_SMA_50_daily_high_%diff','trend_category_48','Price_EMA_200_close_%diff','trend_category_144','Price_SMA_50_daily_low_%diff','rolling_corr_price-ref_336','eom-20','low_zscore_96','high_zscore_48']+[f"Lightgbm_{target_column}_pred"for target_column in target_column_list]+[f"MLP_{target_column}_pred"for target_column in target_column_list]+[f"CatBoost_{target_column}_pred"for target_column in target_column_list]+[f"NN_{target_column}_pred"for target_column in target_column_list]
def objective_lightgbm(trial,X_train,y_train,weights_train,X_val,y_val):params={'num_leaves':trial.suggest_int('num_leaves',31,100),'learning_rate':trial.suggest_float('learning_rate',.01,.1,log=True),'n_estimators':trial.suggest_int('n_estimators',500,2000),'max_depth':trial.suggest_int('max_depth',-1,12),'objective':'regression','metric':'l2','verbosity':-1};model=lgb.LGBMRegressor(**params);model.fit(X_train,y_train,sample_weight=weights_train,eval_set=[(X_val,y_val)],callbacks=[lgb.early_stopping(stopping_rounds=50)]);y_val_pred=model.predict(X_val);val_score=mean_squared_error(y_val,y_val_pred);return val_score
def optimize_lightgbm(X_train,y_train,weights_train,X_val,y_val,weights_val,n_trials=20):
	def objective(trial):return objective_lightgbm(trial,X_train,y_train,weights_train,X_val,y_val)
	study=optuna.create_study(direction='minimize');study.optimize(objective,n_trials=n_trials);return study.best_params
def train_lightgbm(X_train,y_train,weights_train,X_val,y_val,best_params,target_column):model=lgb.LGBMRegressor(**best_params);model.fit(X_train,y_train,sample_weight=weights_train,eval_set=[(X_val,y_val)],callbacks=[lgb.early_stopping(stopping_rounds=50)]);y_val_pred=model.predict(X_val);val_mse=mean_squared_error(y_val,y_val_pred);val_rmse=np.sqrt(val_mse);print(f"Validation MSE: {val_mse:.4f}");print(f"Validation RMSE: {val_rmse:.4f}");return model
def build_mlp_model(hp):
	model=Sequential();initial_units=hp.Int('initial_units',64,256,step=64);model.add(Dense(initial_units,activation='relu',input_dim=len(features),kernel_initializer='he_normal',kernel_regularizer=l2(hp.Float('l2_reg',1e-06,.001,sampling='log')),bias_initializer='zeros'));model.add(BatchNormalization());model.add(Dropout(hp.Float('initial_dropout',.1,.5,step=.05)));n_layers=hp.Int('layers',min_value=2,max_value=15,step=1)
	for i in range(1,n_layers):
		max_units=max(300,initial_units//(i+1));min_units=max(32,max_units//2);min_units=min(min_units,max_units-32)
		if min_units<max_units:model.add(Dense(hp.Int(f"layer_{i}_units",min_units,max_units,step=32),activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(hp.Float(f"l2_reg_{i}",1e-06,.001,sampling='log')),bias_initializer='zeros'));model.add(BatchNormalization());model.add(Dropout(hp.Float(f"dropout_{i}",.1,.5,step=.05)))
	model.add(Dense(1,activation='linear',kernel_initializer='glorot_normal',bias_initializer='zeros'));optimizer=Adam(learning_rate=hp.Float('learning_rate',.0001,.01,sampling='log'),clipnorm=1.);model.compile(optimizer=optimizer,loss='huber',metrics=['mse']);return model
def optimize_mlp(X_train,y_train,weights_train,X_val,y_val,weights_val,epoch=20):callbacks=[EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,min_delta=.0001),ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=5,min_lr=1e-06,min_delta=.0001)];tuner=kt.Hyperband(build_mlp_model,objective='val_loss',max_epochs=epoch,factor=3,directory='mlp_tuning',project_name='stock_mlp',overwrite=True);batch_size=300;tuner.search(X_train,y_train,sample_weight=weights_train,validation_data=(X_val,y_val,weights_val),epochs=epoch,batch_size=batch_size,callbacks=callbacks,verbose=1);return tuner.get_best_hyperparameters(num_trials=1)[0]
def train_mlp(X_train,y_train,weights_train,X_test,y_test,weights_test,best_hps):callbacks=[EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True,min_delta=.0001),ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=7,min_lr=1e-06,min_delta=.0001)];model=build_mlp_model(best_hps);batch_size=300;history=model.fit(X_train,y_train,sample_weight=weights_train,validation_data=(X_test,y_test,weights_test),epochs=50,batch_size=batch_size,callbacks=callbacks,verbose=1);return model
def objective_catboost(trial):params={'iterations':trial.suggest_int('iterations',500,2000),'depth':trial.suggest_int('depth',4,10),'learning_rate':trial.suggest_loguniform('learning_rate',.01,.1),'l2_leaf_reg':trial.suggest_loguniform('l2_leaf_reg',1,10),'loss_function':'RMSE'};model=CatBoostRegressor(**params,verbose=0);model.fit(X_train,y_train,sample_weight=weights_train,eval_set=(X_val,y_val),early_stopping_rounds=50,verbose=0);return model.get_best_score()['validation']['RMSE']
def optimize_catboost(X_train,y_train,weights_train,X_val,y_val,weights_val):study=optuna.create_study(direction='minimize');study.optimize(objective_catboost,n_trials=20);return study.best_params
def train_catboost(X_train,y_train,weights_train,X_test,y_test,weights_test,best_params):model=CatBoostRegressor(**best_params,loss_function='RMSE',verbose=100);model.fit(X_train,y_train,sample_weight=weights_train);return model
def build_nn_model(hp):
	model=Sequential();initial_units=hp.Int('Input_neurons',min_value=40,max_value=300,step=20);model.add(Dense(initial_units,input_shape=(X_train[features].shape[1],),activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(hp.Float('input_l2_reg',1e-06,.001,sampling='log')),bias_initializer='zeros'));model.add(BatchNormalization());model.add(Dropout(hp.Float('Input_dropout',min_value=.02,max_value=.4,step=.02)));n_layers=hp.Int('num_hidden_layers',1,16)
	for i in range(n_layers):
		max_units=max(300,initial_units//(i+2));min_units=max(40,max_units//2);min_units=min(min_units,max_units-20)
		if min_units<max_units:model.add(Dense(hp.Int(f"Layer_{i+1}",min_units,max_units,step=20),activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(hp.Float(f"layer_{i+1}_l2_reg",1e-06,.001,sampling='log')),bias_initializer='zeros'));model.add(BatchNormalization());model.add(Dropout(hp.Float(f"Layer_{i+1}_dropout",min_value=.02,max_value=.46,step=.02)))
	model.add(Dense(1,kernel_initializer='glorot_normal',bias_initializer='zeros'));optimizer=Adam(learning_rate=hp.Float('learning_rate',.0001,.01,sampling='log'),clipnorm=1.);model.compile(optimizer=optimizer,loss='huber',metrics=['mse']);return model
def optimize_nn(X_train,y_train,weights_train,X_val,y_val,weights_val,epoch=50):callbacks=[EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,min_delta=.0001),ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=5,min_lr=1e-06,min_delta=.0001)];tuner=kt.Hyperband(build_nn_model,objective='val_loss',max_epochs=epoch,factor=3,directory='dir_scoring',project_name='nn_tuning',overwrite=True);batch_size=300;tuner.search(X_train,y_train,sample_weight=weights_train,validation_data=(X_val,y_val,weights_val),epochs=epoch,batch_size=batch_size,callbacks=callbacks,verbose=1);return tuner.get_best_hyperparameters(num_trials=1)[0]
def train_nn(X_train,y_train,weights_train,X_test,y_test,weights_test,best_hps):callbacks=[EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True,min_delta=.0001),ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=7,min_lr=1e-06,min_delta=.0001)];model=build_nn_model(best_hps);batch_size=300;history=model.fit(X_train,y_train,sample_weight=weights_train,validation_data=(X_test,y_test,weights_test),epochs=50,batch_size=batch_size,callbacks=callbacks,verbose=1);return model
def generate_meta_features(df_,folder_path):
	df=df_.copy();target_columns=['next_day_close_%chg','next_day_low_%chg','next_day_high_%chg','next_week_close_%chg','next_week_low_%chg','next_week_high_%chg']
	for target_column in target_columns:
		if target_column in['next_day_close_%chg','next_day_low_%chg','next_day_high_%chg']:features=features_day
		elif target_column in['next_week_close_%chg','next_week_low_%chg','next_week_high_%chg']:features=features_week
		else:raise ValueError(f"Unknown target column: {target_column}")
		missing=[col for col in features if col not in df.columns]
		if missing:print(f"[ERROR] Missing features for {target_column}: {missing}");continue
		if df[features].empty or len(df[features])==0:print(f"[ERROR] DataFrame empty after selecting features for {target_column}. Shape: {df[features].shape}");print(f"[DEBUG] Available columns: {df.columns.tolist()}");continue
		model1=load_models(f"lightgbm_{target_column}.pkl",folder_path);model2=load_models(f"mlp_{target_column}.pkl",folder_path);model3=load_models(f"catboost_{target_column}.pkl",folder_path);model4=load_models(f"nn_{target_column}.pkl",folder_path);base_models={'Lightgbm':model1,'MLP':model2,'CatBoost':model3,'NN':model4}
		for(name,model)in base_models.items():df[f"{name}_{target_column}_pred"]=model.predict(df[features])
	return df
def optimize_meta_learner(X_train_meta,y_train_meta,X_val_meta,y_val_meta,weights_train,weights_val):
	def objective(params):model=xgb.XGBRegressor(objective='reg:squarederror',n_estimators=int(params['n_estimators']),max_depth=int(params['max_depth']),learning_rate=params['learning_rate'],subsample=params['subsample'],colsample_bytree=params['colsample_bytree'],gamma=params['gamma'],reg_lambda=params['reg_lambda'],random_state=42,n_jobs=-1);model.fit(X_train_meta,y_train_meta,eval_set=[(X_val_meta,y_val_meta)],sample_weight=weights_train,sample_weight_eval_set=[weights_val],verbose=False);y_pred=model.predict(X_val_meta);loss=mean_absolute_error(y_val_meta,y_pred,sample_weight=weights_val);return{'loss':loss,'status':STATUS_OK}
	space={'n_estimators':hp.quniform('n_estimators',500,1000,25),'max_depth':hp.quniform('max_depth',3,18,1),'learning_rate':hp.loguniform('learning_rate',np.log(.01),np.log(.2)),'subsample':hp.uniform('subsample',.5,1.),'colsample_bytree':hp.uniform('colsample_bytree',.5,1.),'gamma':hp.uniform('gamma',0,5),'reg_lambda':hp.uniform('reg_lambda',0,10)};trials=Trials();best_params=fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=50,trials=trials);best_params['n_estimators']=int(best_params['n_estimators']);best_params['max_depth']=int(best_params['max_depth']);return best_params
def train_meta_learner(X_train_meta,y_train_meta,X_test_meta,y_test_meta,weights_train,weights_test,best_params):model=xgb.XGBRegressor(objective='reg:squarederror',**best_params,random_state=42,n_jobs=-1);model.fit(X_train_meta,y_train_meta,sample_weight=weights_train,eval_set=[(X_test_meta,y_test_meta)],sample_weight_eval_set=[weights_test],verbose=True);return model
def stack_predict(df_,df_main_,start_date,adjust_preds=False,folder_path=None):
	df=df_.loc[start_date:].sort_index();df_main=df_main_.loc[start_date:].sort_index();print(f"[DEBUG] stack_predict() called with start_date = {start_date}");print(f"[DEBUG] df_ index range: {df_.index.min()} to {df_.index.max()} | Shape: {df_.shape}");print(f"[DEBUG] df_main_ index range: {df_main_.index.min()} to {df_main_.index.max()} | Shape: {df_main_.shape}")
	if pd.isna(start_date):print(f"[WARN] Invalid start_date ({start_date}) — using full data range instead.");df=df_.sort_index();df_main=df_main_.sort_index()
	else:df=df_.loc[start_date:].sort_index();df_main=df_main_.loc[start_date:].sort_index()
	print(f"[DEBUG] After slicing: df shape = {df.shape}, df_main shape = {df_main.shape}");df=generate_meta_features(df,folder_path)
	if df.empty:print(f"[WARN] Sliced df is empty for start_date = {start_date}. Using full df as fallback.");df=df_.sort_index();df_main=df_main_.sort_index()
	day_window=int(n/4);base_close=df_main['close'].values;models={target:load_models(f"xgb_{target}.pkl",folder_path)for target in target_column_list}
	for target_column in target_column_list:
		if target_column in['next_day_close_%chg','next_day_low_%chg','next_day_high_%chg']:features_meta=features_meta_day
		elif target_column in['next_week_close_%chg','next_week_low_%chg','next_week_high_%chg']:features_meta=features_meta_week
		else:raise ValueError(f"Unknown target column: {target_column}")
		feature_data=df[features_meta];model=models[target_column];y_pred=model.predict(feature_data);window_size=n if target_column.startswith('next_week')else day_window;y_pred=moving_average_numpy(y_pred,window_size);df_main[f"{target_column}_pred"]=y_pred
	columns_to_convert=['next_day_close_%chg_pred','next_day_high_%chg_pred','next_day_low_%chg_pred','next_week_close_%chg_pred','next_week_high_%chg_pred','next_week_low_%chg_pred']
	for col in columns_to_convert:df_main[f"{col}_value"]=base_close*(1+df_main[col]/100)
	if adjust_preds:df_main=insert_future_predictions(df_main,n)
	return df_main
def add_columns_for_prediction(df_,add_future_vals=False,add_volatility=False,folder_path=None):
	df_main=df_.copy();window_day=n;window_week=n*7;quarter_n=int(n*.25);mult_values=[quarter_n,n,n*2,n*3]
	if add_future_vals:df_main['next_day_high_val']=df_main['high'].rolling(window=window_day).max().shift(-window_day);df_main['next_day_low_val']=df_main['low'].rolling(window=window_day).min().shift(-window_day);df_main['next_week_high_val']=df_main['high'].rolling(window=window_week).max().shift(-window_week);df_main['next_week_low_val']=df_main['low'].rolling(window=window_week).min().shift(-window_week)
	if add_volatility:log_return=np.log(df_main['close']/df_main['close'].shift(1));df_main['daily_volatility']=log_return.rolling(window=window_day).std()*np.sqrt(window_day)*100;df_main['weekly_volatility']=log_return.rolling(window=window_week).std()*np.sqrt(window_week)*100
	df=df_.copy();df=add_base_columns(df);df=add_indicators(df);df,_=add_reference(df,folder_path);df=add_base_columns_ref(df);df,trend_columns=analyze_mean_reversion(df,mult_values);df=add_p_chg_columns(df,percent_change_columns,1);df=add_rolling_correlation(df,'close_%chg','ref_close_%chg',n);columns_to_drop=[col for col in drop_columns if col in df.columns];columns_to_drop.extend(percent_change_columns)
	if columns_to_drop:df=df.drop(columns=columns_to_drop)
	df=cap_extreme_values(df);scaler,transformer=get_normalizer(folder_path);df=join_closest_datetime(df,trend_columns);split_point=df.columns.get_loc(f"trend_category_{quarter_n}");numerical_main=df.iloc[:,:split_point];rest_main=df.iloc[:,split_point:];numerical_main=normalize_data(numerical_main,scaler,transformer,fit=False);df=pd.concat([numerical_main,rest_main],axis=1);trend_category_columns=[f"trend_category_{mult}"for mult in mult_values if f"trend_category_{mult}"in df.columns]
	if trend_category_columns:df[trend_category_columns]=df[trend_category_columns].apply(lambda col:le.transform(col))
	nan_mask=df.isna().any(axis=1);nan_indices=df.index[nan_mask];df=df[~nan_mask];df_main_index_set=set(df_main.index);missing_indices=[idx for idx in nan_indices if idx not in df_main_index_set]
	if missing_indices:print(f"[DEBUG] Warning: {len(missing_indices)} indices not found in df_main: {missing_indices[:5]}...")
	indices_to_drop=[idx for idx in nan_indices if idx in df_main_index_set]
	if indices_to_drop:df_main=df_main.drop(index=indices_to_drop)
	if len(df)!=len(df_main):print(f"[DEBUG] Warning: Cleaned dataframes have different lengths. df: {len(df)}, df_main: {len(df_main)}")
	return df,df_main
def insert_future_predictions(df,n):
	df_main=df.copy();original_columns=df_main.columns.tolist();future_data=[];shifts={'next_day_close_%chg_pred_value':n,'next_day_high_%chg_pred_value':n,'next_day_low_%chg_pred_value':n,'next_week_close_%chg_pred_value':n*7,'next_week_high_%chg_pred_value':n*7,'next_week_low_%chg_pred_value':n*7}
	for(col,shift_val)in shifts.items():temp_df=df_main[[col]].copy();temp_df.index=temp_df.index+pd.DateOffset(days=shift_val//n);temp_df.rename(columns={col:f"{col}_adjusted"},inplace=True);future_data.append(temp_df)
	future_df=pd.concat(future_data,axis=1);df_main=df_main.combine_first(future_df);adjusted_columns=[col+'_adjusted'for col in shifts.keys()];df_main=df_main[original_columns+adjusted_columns];return df_main
def moving_average_numpy(arr,n):cumsum=np.cumsum(arr,dtype=float);cumsum[n:]=cumsum[n:]-cumsum[:-n];counts=np.arange(1,len(arr)+1);counts[n:]=n;return cumsum/counts
def generate_trade_signals(df_):df=df_.copy();day_close_higher=df['next_day_close_%chg_pred_value']>df['close'];week_close_higher=df['next_week_close_%chg_pred_value']>df['close'];buy_condition=day_close_higher&week_close_higher;day_close_lower=df['next_day_close_%chg_pred_value']<df['close'];week_close_lower=df['next_week_close_%chg_pred_value']<df['close'];sell_condition=day_close_lower&week_close_lower;df['signal']='Hold';df.loc[buy_condition&~sell_condition,'signal']='Buy';df.loc[sell_condition&~buy_condition,'signal']='Sell';day_buy_condition=(df['next_day_close_%chg_pred_value']>df['close'])&(df['next_week_close_%chg_pred_value']>df['close']);day_sell_condition=df['next_day_close_%chg_pred_value']<df['close'];df['day_signal']='Hold';df.loc[day_buy_condition&~day_sell_condition,'day_signal']='Buy';df.loc[day_sell_condition&~day_buy_condition,'day_signal']='Sell';return df
def add_volatility_columns(df_):df=df_.copy();df['daily_pred_volatility']=df['next_day_high_%chg_pred']-df['next_day_low_%chg_pred'];df['weekly_pred_volatility']=df['next_week_high_%chg_pred']-df['next_week_low_%chg_pred'];return df
def add_pressure_columns(df_):df=df_.copy();closern=df['close'].values;day_high=df['next_day_high_%chg_pred_value'].values;day_low=df['next_day_low_%chg_pred_value'].values;day_close=df['next_day_close_%chg_pred_value'].values;daily_positive_mask=day_close>closern;daily_negative_mask=~daily_positive_mask;daily_pressure=np.zeros(len(df));high_adjusted=day_high[daily_positive_mask]-day_low[daily_positive_mask];close_adjusted=day_close[daily_positive_mask]-day_low[daily_positive_mask];high_adjusted=np.where(high_adjusted==0,np.finfo(float).eps,high_adjusted);daily_pressure[daily_positive_mask]=close_adjusted/high_adjusted*100;close_adjusted=day_close[daily_negative_mask]-day_high[daily_negative_mask];low_adjusted=day_low[daily_negative_mask]-day_high[daily_negative_mask];low_adjusted=np.where(low_adjusted==0,np.finfo(float).eps,low_adjusted);daily_pressure[daily_negative_mask]=-(close_adjusted/low_adjusted*100);week_high=df['next_week_high_%chg_pred_value'].values;week_low=df['next_week_low_%chg_pred_value'].values;week_close=df['next_week_close_%chg_pred_value'].values;weekly_positive_mask=week_close>closern;weekly_negative_mask=~weekly_positive_mask;weekly_pressure=np.zeros(len(df));high_adjusted=week_high[weekly_positive_mask]-week_low[weekly_positive_mask];close_adjusted=week_close[weekly_positive_mask]-week_low[weekly_positive_mask];high_adjusted=np.where(high_adjusted==0,np.finfo(float).eps,high_adjusted);weekly_pressure[weekly_positive_mask]=close_adjusted/high_adjusted*100;close_adjusted=week_close[weekly_negative_mask]-week_high[weekly_negative_mask];low_adjusted=week_low[weekly_negative_mask]-week_high[weekly_negative_mask];low_adjusted=np.where(low_adjusted==0,np.finfo(float).eps,low_adjusted);weekly_pressure[weekly_negative_mask]=-(close_adjusted/low_adjusted*100);df['daily_pressure']=daily_pressure;df['weekly_pressure']=weekly_pressure;return df
def soften_predictions(test_pred,adjusted_columns=False):
	df=test_pred.copy();pred_cols=['close','next_day_close_%chg_pred_value','next_day_high_%chg_pred_value','next_day_low_%chg_pred_value','next_week_close_%chg_pred_value','next_week_high_%chg_pred_value','next_week_low_%chg_pred_value']
	if adjusted_columns:pred_cols=pred_cols+['next_day_close_%chg_pred_value_adjusted','next_day_high_%chg_pred_value_adjusted','next_day_low_%chg_pred_value_adjusted','next_week_close_%chg_pred_value_adjusted','next_week_high_%chg_pred_value_adjusted','next_week_low_%chg_pred_value_adjusted']
	for column in pred_cols:df[column]=pta.ema(df[column],length=12)
	return df
def add_expected_outcome(df_,folder_path):
	df=df_.copy();folder=f"{folder_path}/dataset/processed_data";distribution_files={'weekly_pred':pd.read_csv(f"{folder}/expected_result_weekly_pred.csv"),'weekly_pred_1st_exit':pd.read_csv(f"{folder}/expected_result_weekly_pred_1st_exit.csv"),'daily_pred':pd.read_csv(f"{folder}/expected_result_daily_pred.csv"),'daily_pred_2nd_exit':pd.read_csv(f"{folder}/expected_result_daily_pred_2nd_exit.csv")};bin_lookups={}
	for(name,dist_df)in distribution_files.items():
		bins=[];means=[]
		for(i,row)in dist_df.iterrows():
			bin_str=row['bin'];mean_val=row['mean']
			if'<'in bin_str:upper_bound=float(bin_str.split('<')[1].strip().replace('%',''));bins.append((-float('inf'),upper_bound))
			elif'>'in bin_str:lower_bound=float(bin_str.split('>')[1].strip().replace('%',''));bins.append((lower_bound,float('inf')))
			elif'to'in bin_str:parts=bin_str.split('to');lower_bound=float(parts[0].strip().replace('%',''));upper_bound=float(parts[1].strip().replace('%',''));bins.append((lower_bound,upper_bound))
			means.append(mean_val)
		bin_lookups[name]=bins,means
	def map_to_bin(value,lookup_key):
		bins,means=bin_lookups[lookup_key]
		if value<=bins[0][1]:return means[0]
		elif value>=bins[-1][0]:return means[-1]
		left,right=0,len(bins)-1
		while left<=right:
			mid=(left+right)//2;lower,upper=bins[mid]
			if lower<=value<=upper:return means[mid]
			elif value<lower:right=mid-1
			else:left=mid+1
		return float('nan')
	mappings={'expected_outcome_weekly_pred':('next_week_close_%chg_pred','weekly_pred'),'expected_outcome_weekly_pred_1st_exit':('next_week_close_%chg_pred','weekly_pred_1st_exit'),'expected_outcome_daily_pred':('next_day_close_%chg_pred','daily_pred'),'expected_outcome_daily_pred_2nd_exit':('next_day_close_%chg_pred','daily_pred_2nd_exit')}
	def process_column(df,output_col,input_col,lookup_key):df[output_col]=df[input_col].apply(lambda x:map_to_bin(x,lookup_key));return df
	with ThreadPoolExecutor(max_workers=4)as executor:
		futures=[]
		for(output_col,(input_col,lookup_key))in mappings.items():func=partial(process_column,df=df,output_col=output_col,input_col=input_col,lookup_key=lookup_key);futures.append(executor.submit(func))
		for future in futures:df=future.result()
	df['expected_outcome_weekly']=df[['expected_outcome_weekly_pred','expected_outcome_daily_pred_2nd_exit']].mean(axis=1);df['expected_outcome_daily']=df[['expected_outcome_daily_pred','expected_outcome_weekly_pred_1st_exit']].mean(axis=1);return df
def calculate_signal_results(df_,exit_perc):
	df=df_.copy();buy_signals=df[df['signal'].isin(['Buy','Hold'])].copy()
	if len(buy_signals)==0:return{'average_gain_pct':0,'number_of_trades':0,'max_gain_pct':0,'taxes_paid':0,'gross_profit':0,'net_profit':0}
	trading_amount=1;results=[]
	for(_,row)in buy_signals.iterrows():
		close_price=row['close'];shares_bought=trading_amount/close_price
		if row['next_week_high_%chg_pred_value']>row['next_week_high_val']:exit_perc=0
		pred_pct_change=row['next_week_high_%chg_pred']/100;exit1_price=close_price*(1+pred_pct_change);exit1_shares=shares_bought*exit_perc;exit1_value=exit1_shares*exit1_price;exit2_price=row['next_week_high_val'];exit2_shares=shares_bought*(1-exit_perc);exit2_value=exit2_shares*exit2_price;total_exit_value=exit1_value+exit2_value;pnl=total_exit_value-trading_amount;pnl_pct=pnl/trading_amount*100;taxes=trading_amount*.026;results.append({'pnl':pnl,'pnl_pct':pnl_pct,'taxes':taxes})
	results_df=pd.DataFrame(results);number_of_trades=len(results_df);average_gain_pct=results_df['pnl_pct'].mean();max_gain=results_df['pnl_pct'].max();taxes_paid=results_df['taxes'].sum();gross_profit=results_df['pnl'].sum();net_profit=gross_profit-taxes_paid;return{'average_gain_pct':average_gain_pct,'number_of_trades':number_of_trades,'max_gain_pct':max_gain,'taxes_paid':taxes_paid,'gross_profit':gross_profit,'net_profit':net_profit}
def sort_and_split_dataframe(df,train_ratio=.7,val_ratio=.15,test_ratio=.15):
	if abs(train_ratio+val_ratio+test_ratio-1.)>1e-10:raise ValueError('Train, validation, and test ratios must sum to 1')
	df_sorted=df.sort_index().copy();n=len(df_sorted);train_end=int(n*train_ratio);val_end=train_end+int(n*val_ratio);df_train=df_sorted.iloc[:train_end].copy();df_val=df_sorted.iloc[train_end:val_end].copy();df_test=df_sorted.iloc[val_end:].copy();print(f"DataFrame split into: {len(df_train)} training samples, {len(df_val)} validation samples, {len(df_test)} test samples");return df_train,df_val,df_test
def df_for_agent(symbol_list,exchange,snapshot=None,n=1550,add_future_vals=False,add_volatility=False,folder_path=None):
	df_list=[];df_main_list=[];add_data([],'BINANCE','BTCUSDT','BINANCE',interval=interval,n_bars=200,folder_path=stock_path);add_data([],'NSE','NIFTY','NSE',interval=interval,n_bars=200,folder_path=crypto_path)
	for symbol in symbol_list:
		try:x=tv.get_hist(symbol=symbol,exchange=exchange,interval=interval,n_bars=n);x,x_main=add_columns_for_prediction(x,add_future_vals=add_future_vals,add_volatility=add_volatility,folder_path=folder_path);x=x.loc[:snapshot];x_main=x_main.loc[:snapshot];df_list.append(x.tail(1));df_main_list.append(x_main.tail(1))
		except Exception as e:print(f"Error processing {symbol}: {e}")
	if df_list and df_main_list:df=pd.concat(df_list,ignore_index=True);df_main=pd.concat(df_main_list,ignore_index=True);df2=stack_predict(df,df_main,start_date='01-01-1970',adjust_preds=False,folder_path=folder_path);return df2
	else:print('No data was processed successfully.');return pd.DataFrame()
def calc_stats(df,signal_column,volatility_columns,high_close_columns,low_columns,ref):
	df=df.copy();ref['ref_high']=ref['high'];ref['ref_low']=ref['low'];df=join_closest_datetime(df,ref.loc[:,'ref_high':]);result={};ema_span=8;result['datetime']=df.index.max()
	for(cols,prefix)in[(volatility_columns,''),(high_close_columns,''),(low_columns,'')]:
		for col in cols:result[f"{col}_avg"]=float(df[col].ewm(span=ema_span).mean().iloc[-1])
	price=df['close'].iloc[-1];low=result.get('next_week_low_%chg_pred_avg',0)or 0;high=result.get('next_week_high_%chg_pred_avg',0)or 0;volatility=result.get('weekly_pred_volatility_avg',0)or 0;result['high_pred']=(price+price*low/100+price*volatility/100-price)*100/price;result['low_pred']=(price-price*high/100-price*volatility/100-price)*100/price;result['correlation_long']=df['high'].corr(df['ref_high']);result['correlation_short']=df['low'].corr(df['ref_low']);result['day_high']=(df['next_day_high_val'].max()-price)/price*100;result['day_low']=(df['next_day_low_val'].max()-price)/price*100;result['week_high']=(df['next_week_high_val'].max()-price)/price*100;result['week_low']=(df['next_week_low_val'].max()-price)/price*100;result['price']=price
	try:atr=pta.atr(high=df['high'],low=df['low'],close=df['close'],length=8).iloc[-1];result['atr']=atr/price*100
	except:result['atr']=np.nan
	try:actual_prices=df['close'].values;high_preds=[df['close'].iloc[i]*(1+df['next_day_high_%chg_pred_value'].iloc[i]/100)for i in range(len(df))];result['dtw_high_avg']=float(dtw(actual_prices,np.array(high_preds)));low_preds=[df['close'].iloc[i]*(1+df['next_day_low_%chg_pred_value'].iloc[i]/100)for i in range(len(df))];result['dtw_low_avg']=float(dtw(actual_prices,np.array(low_preds)));close_preds=[df['close'].iloc[i]*(1+df['next_day_close_%chg_pred_value'].iloc[i]/100)for i in range(len(df))];result['dtw_close_avg']=float(dtw(actual_prices,np.array(close_preds)))
	except:result['dtw_high_avg']=result['dtw_low_avg']=result['dtw_close_avg']=np.nan
	expected=result.get('high_pred',0)or 0;result['sharpe+']=expected/volatility;expected=result.get('low_pred',0)or 0;result['sharpe-']=expected/volatility;result['signal']=df[signal_column].iloc[-1];return result
def analyse_predictions(sym_list,signal_column,path,snapshot=None):
	n=48;dir=f"{path}/dataset/predicted_data";files=os.listdir(dir);results_dict={};drop_columns=['next_day_close_%chg_pred_value_adjusted','next_week_close_%chg_pred_value_adjusted','next_day_high_%chg_pred_value_adjusted','next_day_low_%chg_pred_value_adjusted','next_week_high_%chg_pred_value_adjusted','next_week_low_%chg_pred_value_adjusted'];volatility_columns=['daily_volatility','weekly_volatility','daily_pred_volatility','weekly_pred_volatility'];high_close_columns=['next_day_close_%chg_pred','next_week_close_%chg_pred','next_day_high_%chg_pred','next_week_high_%chg_pred'];low_columns=['next_day_low_%chg_pred','next_week_low_%chg_pred'];ref=pd.read_csv(f"{path}/dataset/data/reference.csv",index_col=0,parse_dates=True)
	for symbol in sym_list:
		file_name=f"{symbol}_preds.csv"
		if file_name in files:
			try:
				df=pd.read_csv(f"{path}/dataset/predicted_data/{file_name}",index_col=0,parse_dates=True);df.dropna(subset=['close'],inplace=True);df=df.loc[:snapshot];strengths=calculate_strengths(df);df=add_pressure_columns(df);df=add_volatility_columns(df);df=df.tail(n);df=generate_trade_signals(df);df.drop(drop_columns,axis=1,inplace=True);stats=calc_stats(df,signal_column,volatility_columns,high_close_columns,low_columns,ref);df_mini=df.tail(int(n/8));stats['rsi']=strengths.get('rsi');stats['buy_count']=int(df_mini[df_mini[signal_column].isin(['Buy','Hold'])].shape[0]);stats['sell_count']=int(df_mini[df_mini[signal_column].isin(['Sell','Hold'])].shape[0]);results_dict[symbol]=stats
				if df.empty:print('Resulting dataframe is empty after preprocessing.')
			except Exception as e:print(f"Error processing {symbol}: {e}")
	results_df=pd.DataFrame.from_dict(results_dict,orient='index');return results_df
def calculate_strengths(df):
	df=df.sort_index();df.index=pd.to_datetime(df.index)
	if df.empty:return{'rsi':None}
	current_time=df.index.max();window_start=current_time-pd.Timedelta(hours=8);window_df=df.loc[window_start:current_time]
	if window_df.empty or len(window_df)<16:return{'rsi':None}
	rsi_series=calculate_rsi(window_df['close'],period=14);return{'rsi':float(rsi_series.iloc[-1])}
def calculate_interval_trend(close_prices):
	y=close_prices.values;x=np.arange(len(y))
	if len(y)<2 or np.mean(y)==0:return .0
	slope,_,r_value,_,_=linregress(x,y);normalized_slope=slope/np.mean(y);raw_score=normalized_slope*r_value**2;scaling_factor=11000;scaled_score=raw_score*scaling_factor;return float(np.clip(scaled_score,-11,11))
def calculate_rsi(close_prices,period=14):
	if len(close_prices)<period+1:return pd.Series([np.nan]*len(close_prices),index=close_prices.index)
	delta=close_prices.diff().dropna()
	if delta.empty:return pd.Series([np.nan]*len(close_prices),index=close_prices.index)
	gain=delta.where(delta>0,0);loss=-delta.where(delta<0,0);avg_gain=gain.ewm(span=period,adjust=False).mean();avg_loss=loss.ewm(span=period,adjust=False).mean();rs=np.where(avg_loss!=0,avg_gain/avg_loss,0);rsi=100-100/(1+rs);result=pd.Series([np.nan]*len(close_prices),index=close_prices.index);result.iloc[1:]=rsi;return result
def calculate_adx(high,low,close,period=14):
	if len(high)<period+1:return pd.Series([np.nan]*len(high),index=high.index)
	prev_close=close.shift(1);high_low=high-low;high_close=(high-prev_close).abs();low_close=(low-prev_close).abs();tr=pd.concat([high_low,high_close,low_close],axis=1).max(axis=1);up_move=high.diff();down_move=-low.diff();plus_dm=pd.Series(np.where((up_move>down_move)&(up_move>0),up_move,0),index=high.index);minus_dm=pd.Series(np.where((down_move>up_move)&(down_move>0),down_move,0),index=high.index);tr_smooth=tr.ewm(span=period,adjust=False).mean();plus_dm_smooth=plus_dm.ewm(span=period,adjust=False).mean();minus_dm_smooth=minus_dm.ewm(span=period,adjust=False).mean();plus_di=np.where(tr_smooth!=0,100*plus_dm_smooth/tr_smooth,0);minus_di=np.where(tr_smooth!=0,100*minus_dm_smooth/tr_smooth,0);di_sum=plus_di+minus_di;dx=np.where(di_sum!=0,100*np.abs(plus_di-minus_di)/di_sum,0);dx=pd.Series(dx,index=high.index);adx=dx.ewm(span=period,adjust=False).mean();return adx
def calculate_dmi(high,low,close,period=14):
	if len(high)<period+1:return pd.DataFrame({'plus_di':[np.nan]*len(high),'minus_di':[np.nan]*len(high)},index=high.index)
	prev_close=close.shift(1);high_low=high-low;high_close=(high-prev_close).abs();low_close=(low-prev_close).abs();tr=pd.concat([high_low,high_close,low_close],axis=1).max(axis=1);up_move=high.diff();down_move=-low.diff();plus_dm=pd.Series(np.where((up_move>down_move)&(up_move>0),up_move,0),index=high.index);minus_dm=pd.Series(np.where((down_move>up_move)&(down_move>0),down_move,0),index=high.index);tr_smooth=tr.ewm(span=period,adjust=False).mean();plus_dm_smooth=plus_dm.ewm(span=period,adjust=False).mean();minus_dm_smooth=minus_dm.ewm(span=period,adjust=False).mean();plus_di=100*plus_dm_smooth/tr_smooth;minus_di=100*minus_dm_smooth/tr_smooth;plus_di=plus_di.replace([np.inf,-np.inf],np.nan).fillna(0);minus_di=minus_di.replace([np.inf,-np.inf],np.nan).fillna(0);return pd.DataFrame({'plus_di':plus_di,'minus_di':minus_di})
def compute_rolling_trends(sym_list,path,lookback_hours=8):
	import os,pandas as pd,numpy as np;sym_list=sym_list+['reference'];trend_data={};volume_data={};rsi_data={};adx_data={};plus_di_data={};minus_di_data={};dir=f"{path}/dataset/data";files=os.listdir(dir);print(f"[DEBUG] Processing {len(sym_list)} symbols")
	for symbol in sym_list:
		file_name=f"{symbol}.csv"
		if file_name in files:
			file_path=os.path.join(path,'dataset','data',file_name)
			try:
				if not os.path.isfile(file_path):print(f"[WARN] File not found: {file_path}");continue
				df=pd.read_csv(file_path,index_col=0,parse_dates=True);df.dropna(subset=['close'],inplace=True);df=df.sort_index();print(f"[DEBUG] {symbol}: Loaded {len(df)} rows, columns: {list(df.columns)}")
				if df.empty or not all(col in df for col in['high','low','close','volume']):print(f"[WARN] Missing data/columns for {symbol}");continue
				last_rows=min(144,len(df));timestamps_to_process=df.iloc[-last_rows:].index;print(f"[DEBUG] {symbol}: Processing {len(timestamps_to_process)} timestamps");first_timestamp=timestamps_to_process[0];extended_start=first_timestamp-pd.Timedelta(hours=lookback_hours+7);extended_df=df[df.index>extended_start]
				if len(extended_df)>=15:rsi_full=calculate_rsi(extended_df['close'],period=14);adx_full=calculate_adx(extended_df['high'],extended_df['low'],extended_df['close'],period=14);dmi_full=calculate_dmi(extended_df['high'],extended_df['low'],extended_df['close'],period=14)
				else:rsi_full=pd.Series([np.nan]*len(extended_df),index=extended_df.index);adx_full=pd.Series([np.nan]*len(extended_df),index=extended_df.index);dmi_full=pd.DataFrame({'plus_di':[np.nan]*len(extended_df),'minus_di':[np.nan]*len(extended_df)},index=extended_df.index)
				trend_series=[];volume_series=[];rsi_series=[];adx_series=[];plus_di_series=[];minus_di_series=[]
				for timestamp in timestamps_to_process:
					window_start=timestamp-pd.Timedelta(hours=lookback_hours);window_df=df[(df.index>window_start)&(df.index<=timestamp)]
					if window_df.empty:trend_strength=None;total_volume=None;rsi_value=None;adx_value=None;plus_di_val=None;minus_di_val=None
					else:trend_strength=calculate_interval_trend(window_df['close']);notional_volume=(window_df['volume']*window_df['close']).sum();total_volume=float(notional_volume);rsi_value=float(rsi_full.get(timestamp,np.nan))if timestamp in rsi_full.index else None;adx_value=float(adx_full.get(timestamp,np.nan))if timestamp in adx_full.index else None;plus_di_val=float(dmi_full['plus_di'].get(timestamp,np.nan))if timestamp in dmi_full.index else None;minus_di_val=float(dmi_full['minus_di'].get(timestamp,np.nan))if timestamp in dmi_full.index else None;rsi_value=None if pd.isna(rsi_value)else rsi_value;adx_value=None if pd.isna(adx_value)else adx_value;plus_di_val=None if pd.isna(plus_di_val)else plus_di_val;minus_di_val=None if pd.isna(minus_di_val)else minus_di_val
					trend_series.append((timestamp,trend_strength));volume_series.append((timestamp,total_volume));rsi_series.append((timestamp,rsi_value));adx_series.append((timestamp,adx_value));plus_di_series.append((timestamp,plus_di_val));minus_di_series.append((timestamp,minus_di_val))
				trend_data[symbol]=pd.Series(dict(trend_series),name=symbol);volume_data[symbol]=pd.Series(dict(volume_series),name=symbol);rsi_data[symbol]=pd.Series(dict(rsi_series),name=symbol);adx_data[symbol]=pd.Series(dict(adx_series),name=symbol);plus_di_data[symbol]=pd.Series(dict(plus_di_series),name=symbol);minus_di_data[symbol]=pd.Series(dict(minus_di_series),name=symbol);print(f"[DEBUG] {symbol}: Added {len(trend_series)} data points")
			except Exception as e:print(f"[ERROR] {symbol}: {e}");import traceback;traceback.print_exc()
	print(f"[DEBUG] Collected data for {len(trend_data)} symbols")
	if not trend_data:print('[WARN] No trend data computed.');return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
	trend_df=pd.DataFrame(trend_data);volume_df=pd.DataFrame(volume_data);rsi_df=pd.DataFrame(rsi_data);adx_df=pd.DataFrame(adx_data);plus_di_df=pd.DataFrame(plus_di_data);minus_di_df=pd.DataFrame(minus_di_data);print(f"[DEBUG] Initial DataFrames - Trend: {trend_df.shape}, Volume: {volume_df.shape}, RSI: {rsi_df.shape}, ADX: {adx_df.shape}")
	if'reference'in trend_df.columns:trend_df=trend_df.dropna(subset=['reference']).dropna(axis=1);volume_df=volume_df.reindex_like(trend_df);rsi_df=rsi_df.reindex_like(trend_df);adx_df=adx_df.reindex_like(trend_df);plus_di_df=plus_di_df.reindex_like(trend_df);minus_di_df=minus_di_df.reindex_like(trend_df);print(f"[DEBUG] After reference filtering - Trend: {trend_df.shape}")
	else:print('[DEBUG] reference not found, using all available data')
	print(f"[DEBUG] Final DataFrames - Trend: {trend_df.shape}, Volume: {volume_df.shape}, RSI: {rsi_df.shape}, ADX: {adx_df.shape}, +DI: {plus_di_df.shape}, -DI: {minus_di_df.shape}");return trend_df,volume_df,rsi_df,adx_df,plus_di_df,minus_di_df
def summarize_trends(trend_df,window_hours=8):
	trend_df=trend_df.sort_index();window_size=int(window_hours*60//30);recent_window=trend_df.iloc[-window_size:];avg_trends=recent_window.mean();top_strongest=avg_trends.sort_values(ascending=False).head(10);top_weakest=avg_trends.sort_values().head(10);strong_up_counts=(recent_window>=9).sum();strong_down_counts=(recent_window<=2).sum();top_uptrends=strong_up_counts.sort_values(ascending=False).head(10);top_downtrends=strong_down_counts.sort_values(ascending=False).head(10);momentum_scores={}
	for col in recent_window.columns:
		series=recent_window[col].dropna()
		if len(series)>=2:momentum=series.iloc[-1]-series.iloc[:-1].mean();momentum_scores[col]=momentum
	momentum_series=pd.Series(momentum_scores);top_momentum=momentum_series.sort_values(ascending=False).head(10);bottom_momentum=momentum_series.sort_values().head(10);avg_trend_per_time=pd.DataFrame({'mean_trend':trend_df.mean(axis=1),'median_trend':trend_df.median(axis=1)});summary={'Top 10 Strongest Symbols (8h avg)':top_strongest,'Top 10 Weakest Symbols (8h avg)':top_weakest,'Top 10 Symbols with Most Strong Uptrends (≥5)':top_uptrends,'Top 10 Symbols with Most Strong Downtrends (≤1)':top_downtrends,'Top 10 Symbols with Strongest Recent Momentum':top_momentum,'Top 10 Symbols with Weakest Recent Momentum':bottom_momentum};return summary,avg_trend_per_time
def summarize_volumes(volume_df,window_hours=8):
	volume_df=volume_df.sort_index().drop('reference',axis=1)
	if'BTCUSDT'in volume_df.columns:volume_df=volume_df.drop('BTCUSDT',axis=1)
	window_size=int(window_hours*60//30);recent_window=volume_df.iloc[-window_size:];avg_volume=recent_window.mean();top_volume=avg_volume.sort_values(ascending=False).head(10);bottom_volume=avg_volume.sort_values().head(10);high_threshold=recent_window.quantile(.75);high_volume_counts=(recent_window>high_threshold).sum();top_high_volume_counts=high_volume_counts.sort_values(ascending=False).head(10);volume_momentum={}
	for col in recent_window.columns:
		series=recent_window[col].dropna()
		if len(series)>=2:momentum=series.iloc[-1]-series.iloc[:-1].mean();volume_momentum[col]=momentum
	volume_momentum_series=pd.Series(volume_momentum);top_volume_momentum=volume_momentum_series.sort_values(ascending=False).head(10);bottom_volume_momentum=volume_momentum_series.sort_values().head(10);total_volume_per_time=pd.DataFrame({'total_volume':volume_df.sum(axis=1)});summary={'Top 10 Symbols by Avg Volume (8h)':top_volume,'Bottom 10 Symbols by Avg Volume (8h)':bottom_volume,'Top 10 Symbols with Frequent High Volumes':top_high_volume_counts,'Top 10 Symbols with Rising Volume Momentum':top_volume_momentum,'Top 10 Symbols with Falling Volume Momentum':bottom_volume_momentum};return summary,total_volume_per_time
def summarize_rsi(rsi_df,volume_df,window_hours=8):
	rsi_df=rsi_df.sort_index().sort_index().drop('reference',axis=1)
	if'BTCUSDT'in rsi_df.columns:rsi_df=rsi_df.drop('BTCUSDT',axis=1)
	volume_df=volume_df.sort_index().sort_index().drop('reference',axis=1)
	if'BTCUSDT'in volume_df.columns:volume_df=volume_df.drop('BTCUSDT',axis=1)
	window_size=int(window_hours*60//30);recent_rsi=rsi_df.iloc[-window_size:];recent_volume=volume_df.loc[recent_rsi.index];weighted_rsi=(recent_rsi*recent_volume).sum(axis=1)/recent_volume.sum(axis=1);mean_rsi=recent_rsi.mean(axis=1);rsi_metrics=pd.DataFrame({'volume_weighted_rsi':weighted_rsi,'mean_rsi':mean_rsi});last_rsi=recent_rsi.iloc[-1];top_rsi=last_rsi.sort_values(ascending=False).head(10);bottom_rsi=last_rsi.sort_values().head(10);summary={'Top 10 Symbols by Latest RSI':top_rsi,'Bottom 10 Symbols by Latest RSI':bottom_rsi};return summary,rsi_metrics
def summarize_adx(adx_df,volume_df,window_hours=8,top_n_by_volume=20):
	adx_df=adx_df.sort_index().sort_index().drop('reference',axis=1)
	if'BTCUSDT'in adx_df.columns:adx_df=adx_df.drop('BTCUSDT',axis=1)
	volume_df=volume_df.sort_index().sort_index().drop('reference',axis=1)
	if'BTCUSDT'in volume_df.columns:volume_df=volume_df.drop('BTCUSDT',axis=1)
	window_size=int(window_hours*60//30);recent_adx=adx_df.iloc[-window_size:];recent_volume=volume_df.loc[recent_adx.index];avg_volume=recent_volume.mean();top_symbols=avg_volume.sort_values(ascending=False).head(top_n_by_volume).index;top_adx=recent_adx[top_symbols];adx_metrics=pd.DataFrame({'mean_adx':top_adx.mean(axis=1),'median_adx':top_adx.median(axis=1)});latest_adx=top_adx.iloc[-1].sort_values(ascending=False).head(10);summary={f"Top 10 Symbols with Highest ADX (among Top {top_n_by_volume} by Volume)":latest_adx};return summary,adx_metrics
def summarize_plus_dmi(plus_dmi_df,volume_df,window_hours=8,top_n_by_volume=20):plus_dmi_df=plus_dmi_df.sort_index().drop(columns=['reference'],errors='ignore');plus_dmi_df=plus_dmi_df.drop(columns=['BTCUSDT'],errors='ignore');volume_df=volume_df.sort_index().drop(columns=['reference'],errors='ignore');volume_df=volume_df.drop(columns=['BTCUSDT'],errors='ignore');window_size=int(window_hours*60//30);recent_plus_dmi=plus_dmi_df.iloc[-window_size:];recent_volume=volume_df.loc[recent_plus_dmi.index];avg_volume=recent_volume.mean();top_symbols=avg_volume.sort_values(ascending=False).head(top_n_by_volume).index;top_plus_dmi=recent_plus_dmi[top_symbols];plus_dmi_metrics=pd.DataFrame({'mean_plus_dmi':top_plus_dmi.mean(axis=1),'median_plus_dmi':top_plus_dmi.median(axis=1)});latest_plus_dmi=top_plus_dmi.iloc[-1].sort_values(ascending=False).head(10);summary={f"Top 10 Symbols with Highest +DI (among Top {top_n_by_volume} by Volume)":latest_plus_dmi};return summary,plus_dmi_metrics
def summarize_minus_dmi(minus_dmi_df,volume_df,window_hours=8,top_n_by_volume=20):minus_dmi_df=minus_dmi_df.sort_index().drop(columns=['reference'],errors='ignore');minus_dmi_df=minus_dmi_df.drop(columns=['BTCUSDT'],errors='ignore');volume_df=volume_df.sort_index().drop(columns=['reference'],errors='ignore');volume_df=volume_df.drop(columns=['BTCUSDT'],errors='ignore');window_size=int(window_hours*60//30);recent_minus_dmi=minus_dmi_df.iloc[-window_size:];recent_volume=volume_df.loc[recent_minus_dmi.index];avg_volume=recent_volume.mean();top_symbols=avg_volume.sort_values(ascending=False).head(top_n_by_volume).index;top_minus_dmi=recent_minus_dmi[top_symbols];minus_dmi_metrics=pd.DataFrame({'mean_minus_dmi':top_minus_dmi.mean(axis=1),'median_minus_dmi':top_minus_dmi.median(axis=1)});latest_minus_dmi=top_minus_dmi.iloc[-1].sort_values(ascending=False).head(10);summary={f"Top 10 Symbols with Highest -DI (among Top {top_n_by_volume} by Volume)":latest_minus_dmi};return summary,minus_dmi_metrics
def directional_divergence_series(ref_a,a,baseline=3):
	ref_a=np.array(ref_a);a=np.array(a);moves={'ref':[baseline]*3*2,'a':[baseline]*3*2}
	for i in range(5,len(ref_a)):moves['ref'].append(ref_a[i]+moves['ref'][i-5]-3);moves['a'].append(a[i]+moves['a'][i-5]-3)
	diff=ref_a-a;ref_chg=np.diff(ref_a);a_chg=np.diff(a);divergence=[baseline]
	for i in range(len(a_chg)):delta=ref_chg[i]-a_chg[i];dir_val=0 if delta==0 else(a_chg[i]-ref_chg[i])/abs(delta);div=(abs(delta)+1)*abs(diff[i+1])*dir_val+baseline;divergence.append(div)
	divergence=pd.Series(divergence).ewm(span=6).mean();accumulation=[np.nan]*4
	for i in range(4,len(divergence)):acc_val=divergence[i-4:i].sum()/3-2;accumulation.append(acc_val)
	avg=(a+ref_a)/2;avg=pd.Series(avg).ewm(span=3).mean();s_a=pd.Series(a_chg);s_ref=pd.Series(ref_chg);rolling_corr=s_a.rolling(window=4).corr(s_ref);corr_scaled=[(val*4)**2 if pd.notna(val)else None for val in rolling_corr];corr=[3]
	for i in range(0,len(corr_scaled)):corr.append(corr_scaled[i])
	return divergence.tolist(),corr,accumulation,avg.tolist()
def plot_trend_comparison(ref_symbol,comp_symbol='mean_trend',window=15):
	ref_trend=trend_df[ref_symbol];trend=market_data['mean_trend']if comp_symbol=='mean_trend'else trend_df[comp_symbol];ref_volume=volume_df.get(ref_symbol);total_volume=volume_trend_df['total_volume']if comp_symbol=='mean_trend'else volume_df.get(comp_symbol);ref_rsi=rsi_df.get(ref_symbol);comp_rsi=rsi_df.mean(axis=1)if comp_symbol=='mean_trend'else rsi_df.get(comp_symbol);ref_adx=adx_df.get(ref_symbol);comp_adx=adx_df.mean(axis=1)if comp_symbol=='mean_trend'else adx_df.get(comp_symbol);ref_plus_dmi=plus_dmi.get(ref_symbol);ref_minus_dmi=minus_dmi.get(ref_symbol);comp_plus_dmi=plus_dmi.mean(axis=1)if comp_symbol=='mean_trend'else plus_dmi.get(comp_symbol);comp_minus_dmi=minus_dmi.mean(axis=1)if comp_symbol=='mean_trend'else minus_dmi.get(comp_symbol);ref_trend_ma=ref_trend.rolling(window=window).mean();trend_ma=trend.rolling(window=window).mean();ref_vol_ma=ref_volume.rolling(window=20).mean();vol_ma=total_volume.rolling(window=20).mean();fig=make_subplots(rows=5,cols=1,shared_xaxes=True,row_heights=[.25,.2,.2,.2,.15],vertical_spacing=.04,subplot_titles=[f"Trend Comparison: {ref_symbol} vs {comp_symbol}",'Volume Traded','RSI Comparison','ADX Comparison','Directional Movement Index (+DI / -DI)'],specs=[[{}],[{'secondary_y':True}],[{}],[{}],[{}]]);fig.add_trace(go.Scatter(x=ref_trend.index,y=ref_trend,name=ref_symbol,mode='lines',line=dict(color='orange'),opacity=.6),row=1,col=1);fig.add_trace(go.Scatter(x=trend.index,y=trend,name=comp_symbol,mode='lines',line=dict(color='purple'),opacity=.6),row=1,col=1);fig.add_trace(go.Scatter(x=ref_trend_ma.index,y=ref_trend_ma,name=f"{ref_symbol}_MA",line=dict(color='red')),row=1,col=1);fig.add_trace(go.Scatter(x=trend_ma.index,y=trend_ma,name=f"{comp_symbol}_MA",line=dict(color='black')),row=1,col=1)
	for(yval,label,color)in zip([0,5,-5],['y=0','y=5','y=-5'],['blue','grey','grey']):fig.add_trace(go.Scatter(x=ref_trend.index,y=[yval]*len(ref_trend),name=label,line=dict(color=color,dash='dash'),opacity=.4),row=1,col=1)
	fig.add_trace(go.Scatter(x=total_volume.index,y=total_volume,mode='lines',name='Total Market Volume',line=dict(color='purple')),row=2,col=1,secondary_y=False);fig.add_trace(go.Scatter(x=vol_ma.index,y=vol_ma,name=f"{comp_symbol}_vol_MA",line=dict(color='black')),row=2,col=1)
	if ref_volume is not None:fig.add_trace(go.Scatter(x=ref_volume.index,y=ref_volume,mode='lines',name=f"{ref_symbol} Volume",line=dict(color='orange')),row=2,col=1,secondary_y=True);fig.add_trace(go.Scatter(x=ref_vol_ma.index,y=ref_vol_ma,name=f"{ref_symbol}_vol_MA",line=dict(color='red')),row=2,col=1,secondary_y=True)
	if ref_rsi is not None:fig.add_trace(go.Scatter(x=ref_rsi.index,y=ref_rsi,mode='lines',name=f"{ref_symbol} RSI",line=dict(color='orange')),row=3,col=1)
	if comp_rsi is not None:fig.add_trace(go.Scatter(x=comp_rsi.index,y=comp_rsi,mode='lines',name=f"{comp_symbol} RSI",line=dict(color='purple')),row=3,col=1)
	for(level,color)in zip([70,30],['green','red']):fig.add_trace(go.Scatter(x=ref_trend.index,y=[level]*len(ref_trend),name=f"RSI={level}",line=dict(color=color,dash='dot'),opacity=.4),row=3,col=1)
	if ref_adx is not None:fig.add_trace(go.Scatter(x=ref_adx.index,y=ref_adx,mode='lines',name=f"{ref_symbol} ADX",line=dict(color='orange')),row=4,col=1)
	if comp_adx is not None:fig.add_trace(go.Scatter(x=comp_adx.index,y=comp_adx,mode='lines',name=f"{comp_symbol} ADX",line=dict(color='purple')),row=4,col=1)
	fig.add_trace(go.Scatter(x=ref_trend.index,y=[25]*len(ref_trend),name='ADX=25',line=dict(color='blue',dash='dot'),opacity=.4),row=4,col=1)
	if ref_plus_dmi is not None:fig.add_trace(go.Scatter(x=ref_plus_dmi.index,y=ref_plus_dmi,mode='lines',name=f"{ref_symbol} +DI",line=dict(color='green',dash='dot')),row=5,col=1)
	if comp_plus_dmi is not None:fig.add_trace(go.Scatter(x=comp_plus_dmi.index,y=comp_plus_dmi,mode='lines',name=f"{comp_symbol} +DI",line=dict(color='darkgreen')),row=5,col=1)
	if ref_minus_dmi is not None:fig.add_trace(go.Scatter(x=ref_minus_dmi.index,y=ref_minus_dmi,mode='lines',name=f"{ref_symbol} -DI",line=dict(color='red',dash='dot')),row=5,col=1)
	if comp_minus_dmi is not None:fig.add_trace(go.Scatter(x=comp_minus_dmi.index,y=comp_minus_dmi,mode='lines',name=f"{comp_symbol} -DI",line=dict(color='darkred')),row=5,col=1)
	fig.add_trace(go.Scatter(x=ref_trend.index,y=[25]*len(ref_trend),name='ADX=25',line=dict(color='blue',dash='dot'),opacity=.4),row=5,col=1);fig.update_layout(height=1700,width=2000,title_text=f"Trend, Volume, RSI, ADX, and DMI Comparison: {ref_symbol} vs {comp_symbol}",hovermode='x unified');fig.update_yaxes(title_text='Trend Value',row=1,col=1);fig.update_yaxes(title_text='Total Market Volume',row=2,col=1,secondary_y=False);fig.update_yaxes(title_text=f"{ref_symbol} Volume",row=2,col=1,secondary_y=True);fig.update_yaxes(title_text='RSI',row=3,col=1);fig.update_yaxes(title_text='ADX',row=4,col=1);fig.update_yaxes(title_text='+DI / -DI',row=5,col=1);fig.update_xaxes(title_text='Time',row=5,col=1);fig.show()
def pareto_filter(df,direction,maximize_cols=['high_pred'],minimize_cols=['next_week_low_%chg_pred_avg','dtw_week_avg'],target_col='week_high',desired_rows=20,method='layered',approximation_factor=.05,weights=None):
	if df.empty:return df
	if direction=='short':working_df=df[df['signal'].isin(['Buy','Sell','Hold'])].copy()
	else:working_df=df[df['signal'].isin(['Buy','Sell','Hold'])].copy()
	if working_df.empty:return working_df
	for col in maximize_cols+minimize_cols:
		min_val=working_df[col].min();max_val=working_df[col].max()
		if max_val>min_val:working_df[f"{col}_norm"]=(working_df[col]-min_val)/(max_val-min_val)
		else:working_df[f"{col}_norm"]=.0
	for col in minimize_cols:working_df[f"{col}_norm"]=1-working_df[f"{col}_norm"]
	cost_columns=[f"{col}_norm"for col in maximize_cols+minimize_cols];costs=working_df[cost_columns].values;result_df=pd.DataFrame()
	def is_pareto_efficient(costs):
		is_efficient=np.ones(costs.shape[0],dtype=bool)
		for(i,c)in enumerate(costs):
			if is_efficient[i]:dominates=np.all(costs<=c,axis=1)&np.any(costs<c,axis=1);is_efficient[i]=~np.any(dominates)
		return is_efficient
	def is_approximate_pareto_efficient(costs,tolerance=.05):
		is_efficient=np.ones(costs.shape[0],dtype=bool)
		for(i,c)in enumerate(costs):
			if is_efficient[i]:
				dominates_i=np.all(costs-c<=tolerance,axis=1)&np.any(costs-c<tolerance,axis=1)
				if np.any(dominates_i):is_efficient[i]=False
		return is_efficient
	if method=='strict':efficient_mask=is_pareto_efficient(costs);result_df=working_df[efficient_mask].copy()
	elif method=='layered':
		remaining_df=working_df.copy()
		while len(result_df)<desired_rows and not remaining_df.empty:remaining_costs=remaining_df[cost_columns].values;efficient_mask=is_pareto_efficient(remaining_costs);current_frontier=remaining_df[efficient_mask].copy();result_df=pd.concat([result_df,current_frontier]);remaining_df=remaining_df[~efficient_mask]
	elif method=='approximate':
		factor=approximation_factor
		while len(result_df)<desired_rows and factor<=.5:efficient_mask=is_approximate_pareto_efficient(costs,tolerance=factor);result_df=working_df[efficient_mask].copy();factor+=.05
	elif method=='weighted':
		if weights is None:weights={col:1./len(cost_columns)for col in cost_columns}
		else:total=sum(weights.values());weights={k:v/total for(k,v)in weights.items()}
		working_df['weighted_score']=sum(working_df[col]*weights.get(col,1./len(cost_columns))for col in cost_columns);result_df=working_df.sort_values('weighted_score',ascending=False).head(desired_rows)
	drop_cols=[f"{col}_norm"for col in maximize_cols+minimize_cols if f"{col}_norm"in result_df.columns];result_df.drop(columns=drop_cols,inplace=True,errors='ignore')
	if'weighted_score'in result_df.columns:result_df.drop(columns='weighted_score',inplace=True)
	return result_df.head(desired_rows)
def portfolio(analysis,longdf,shortdf,wallet,n_long,n_short,offset,liquidity=.0,long_ratio=.5):
	n_long=max(0,n_long);n_short=max(0,n_short);liquidity=min(max(liquidity,.0),1.);long_ratio=min(max(long_ratio,.0),1.);longdf=longdf.copy()if not longdf.empty else pd.DataFrame();shortdf=shortdf.copy()if not shortdf.empty else pd.DataFrame()
	if'positions'not in wallet:wallet['positions']={'LONG':{},'SHORT':{}}
	elif'LONG'not in wallet['positions']or'SHORT'not in wallet['positions']:wallet['positions']={'LONG':wallet['positions'].get('LONG',{}),'SHORT':wallet['positions'].get('SHORT',{})}
	if'cash'not in wallet:wallet['cash']=.0
	wallet.setdefault('commission_rate',.001);wallet.setdefault('tds_rate',.0);initial={'LONG':{},'SHORT':{}}
	for side in('LONG','SHORT'):
		for(sym,val)in wallet['positions'][side].items():initial[side][sym]=val
	def select_top(df,pred_col,n,invert=False):
		if df.empty or n<=0:return pd.DataFrame()
		if pred_col not in df.columns:print(f"Warning: '{pred_col}' column missing from DataFrame");return pd.DataFrame()
		if'sharpe+'not in df.columns:print(f"Warning: 'sharpe+' column missing from DataFrame");return pd.DataFrame()
		if'sharpe-'not in df.columns:print(f"Warning: 'sharpe-' column missing from DataFrame");return pd.DataFrame()
		corr_col='correlation_short'if invert else'correlation_long'
		if corr_col not in df.columns:print(f"Warning: '{corr_col}' column missing from DataFrame");return pd.DataFrame()
		preds=-df[pred_col]if invert else df[pred_col];preds=preds.clip(upper=100);sharpe=-df['sharpe-']if invert else df['sharpe+'];corr=df[corr_col];preds=preds.fillna(0);sharpe=sharpe.fillna(0);corr=corr.fillna(.5);score=preds*sharpe*corr;return df.assign(score=score).nlargest(min(n,len(df)),'score')
	desired_longs=[]
	if not longdf.empty and n_long>0:
		if'correlation_long'in longdf.columns:longdf.dropna(subset=['correlation_long'],inplace=True)
		desired_longs=list(set(select_top(longdf,'high_pred',n_long,invert=False).index))
	desired_shorts=[]
	if not shortdf.empty and n_short>0:
		if'correlation_short'in shortdf.columns:shortdf.dropna(subset=['correlation_short'],inplace=True)
		desired_shorts=list(set(select_top(shortdf,'low_pred',n_short,invert=True).index))
	for(side,close_sig)in[('LONG','nn'),('SHORT','nn')]:
		desired=desired_longs if side=='LONG'else desired_shorts
		for sym in list(wallet['positions'][side]):
			sig=analysis.at[sym,'signal']if sym in analysis.index else None
			if sym not in desired or sig==close_sig:val=wallet['positions'][side].pop(sym);fee=val*(wallet['commission_rate']+wallet['tds_rate']);wallet['cash']+=max(val-fee,0)
	cash_total=wallet['cash']+sum(v for pos in wallet['positions'].values()for v in pos.values());alloc_cash=cash_total*(1-liquidity)
	if long_ratio==0:long_cash=0;short_cash=alloc_cash
	elif long_ratio==1:long_cash=alloc_cash;short_cash=0
	else:long_cash=alloc_cash*long_ratio;short_cash=alloc_cash*(1-long_ratio)
	print(f"Total cash: ${cash_total:.2f}");print(f"Allocatable cash: ${alloc_cash:.2f}");print(f"Long allocation: ${long_cash:.2f} ({long_ratio*100:.1f}%)");print(f"Short allocation: ${short_cash:.2f} ({(1-long_ratio)*100:.1f}%)");print(f"Liquidity reserve: ${cash_total*liquidity:.2f} ({liquidity*100:.1f}%)");longs=pd.DataFrame()
	if desired_longs and long_cash>0:
		valid_longs=[sym for sym in desired_longs if sym in analysis.index]
		if valid_longs:
			longs=select_top(analysis.loc[valid_longs],'high_pred',len(valid_longs),invert=False);required_cols=['sharpe+','high_pred','correlation_long']
			if all(col in longs.columns for col in required_cols):longs['w']=(longs['sharpe+']*longs['high_pred'].clip(upper=100)/(longs['correlation_long'].abs()+.05)).clip(lower=0);total=longs['w'].sum();longs['w']=longs['w']/total if total>0 else 1/len(longs)if len(longs)else 0
			else:longs['w']=1/len(longs)if len(longs)else 0
	shorts=pd.DataFrame()
	if desired_shorts and short_cash>0:
		valid_shorts=[sym for sym in desired_shorts if sym in analysis.index]
		if valid_shorts:
			shorts=select_top(analysis.loc[valid_shorts],'low_pred',len(valid_shorts),invert=True);required_cols=['sharpe-','low_pred','correlation_short']
			if all(col in shorts.columns for col in required_cols):shorts['w']=(-shorts['sharpe-']*(-shorts['low_pred']).clip(upper=100)/(shorts['correlation_short'].abs()+.05)).clip(lower=0);total=shorts['w'].sum();shorts['w']=shorts['w']/total if total>0 else 1/len(shorts)if len(shorts)else 0
			else:shorts['w']=1/len(shorts)if len(shorts)else 0
	def allocate(df,initial_pos,alloc_cash,label):
		if df.empty or alloc_cash<=0:return{},alloc_cash
		df=df.copy();df['use_old']=False;remaining_cash=alloc_cash;final_alloc={}
		for(sym,row)in df.iterrows():
			curr_val=initial_pos.get(sym,.0);target_val=alloc_cash*row['w']
			if curr_val>0:
				diff=abs(target_val-curr_val)
				if diff<.4*curr_val:final_alloc[sym]=curr_val;df.at[sym,'use_old']=True
		adjusted=df[~df['use_old']].copy()
		if not adjusted.empty:
			adjusted['target_val']=alloc_cash*adjusted['w'];adjusted['cost']=adjusted['target_val']*(1+wallet['commission_rate']+wallet['tds_rate']);total_cost=adjusted['cost'].sum()
			if total_cost>alloc_cash:factor=alloc_cash/total_cost;adjusted['w']*=factor;adjusted['target_val']=alloc_cash*adjusted['w'];adjusted['cost']=adjusted['target_val']*(1+wallet['commission_rate']+wallet['tds_rate'])
			for(sym,row)in adjusted.iterrows():
				target_val=row['target_val'];cost=row['cost']
				if cost>remaining_cash:target_val=remaining_cash/(1+wallet['commission_rate']+wallet['tds_rate']);cost=remaining_cash
				if target_val>0 and cost>0:remaining_cash-=cost;final_alloc[sym]=target_val
		return final_alloc,remaining_cash
	final_longs,remaining_long_cash=allocate(longs,initial['LONG'],long_cash,'LONG');final_shorts,remaining_short_cash=allocate(shorts,initial['SHORT'],short_cash,'SHORT');unused_cash=remaining_long_cash+remaining_short_cash;wallet['cash']=unused_cash+cash_total*liquidity;wallet['positions']={'LONG':final_longs,'SHORT':final_shorts};print(f"Allocated to longs: ${long_cash-remaining_long_cash:.2f}");print(f"Allocated to shorts: ${short_cash-remaining_short_cash:.2f}");print(f"Unused cash returned: ${unused_cash:.2f}")
	def build_df(side):
		records=[];all_syms=set(initial[side].keys())|set(wallet['positions'][side].keys())
		if not all_syms:return pd.DataFrame(columns=['current_value','efficient_value','net_trade','stoploss','correlation','sharpe','momentum','signal','expected_return'])
		for sym in all_syms:
			curr=initial[side].get(sym,.0);eff=wallet['positions'][side].get(sym,.0);record={'asset':sym,'current_value':curr,'efficient_value':round(eff,1),'net_trade':round(eff-curr,2),'stoploss':0,'stoploss_actual':0,'correlation':0,'sharpe':0,'momentum':0,'signal':None,'expected_return':0}
			if sym in analysis.index:
				try:
					if'price'in analysis.columns and'atr'in analysis.columns:price=analysis.at[sym,'price'];atr=analysis.at[sym,'atr'];record['stoploss']=round((atr+offset)*1.7,2);record['stoploss_actual']=round(atr*1.5,2)
					if'signal'in analysis.columns:record['signal']=analysis.at[sym,'signal']
					if'uptrend'in analysis.columns:record['momentum']=analysis.at[sym,'uptrend']
					if side=='LONG':
						if'correlation_long'in analysis.columns:record['correlation']=round(analysis.at[sym,'correlation_long'],3)
						if'high_pred'in analysis.columns:record['expected_return']=analysis.at[sym,'high_pred']
						if'sharpe+'in analysis.columns:record['sharpe']=round(analysis.at[sym,'sharpe+'],2)
					else:
						if'correlation_short'in analysis.columns:record['correlation']=round(analysis.at[sym,'correlation_short'],3)
						if'low_pred'in analysis.columns:low_pred=analysis.at[sym,'low_pred'];record['expected_return']=abs(low_pred)if low_pred<0 else-low_pred
						if'sharpe-'in analysis.columns:record['sharpe']=round(analysis.at[sym,'sharpe-'],2)
				except Exception as e:print(f"Warning: Error accessing data for {sym}: {e}")
			records.append(record)
		if records:return pd.DataFrame(records).set_index('asset')
		else:return pd.DataFrame(columns=['current_value','efficient_value','net_trade','stoploss','stoploss_actual','correlation','sharpe','momentum','signal','expected_return'])
	long_df=build_df('LONG');short_df=build_df('SHORT');print('\nLong positions:',len(long_df[long_df['efficient_value']>0]));print('Short positions:',len(short_df[short_df['efficient_value']>0]));return long_df,short_df
def create_wallet(longdf,shortdf):
	wallet={'cash':1e2,'positions':{'LONG':{},'SHORT':{}},'commission_rate':.0045,'tds_rate':.01}
	for(sym,row)in longdf.iterrows():wallet['positions']['LONG'][sym]=round(row['efficient_value'],1)
	for(sym,row)in shortdf.iterrows():wallet['positions']['SHORT'][sym]=round(row['efficient_value'],1)
	return wallet
def reference_prediction(path,symbol):
	reference_test=pd.read_csv(f"{path}/dataset/predicted_data/{symbol}_preds.csv",index_col=0,parse_dates=True);reference_test.dropna(subset=['close'],inplace=True);reference_test=reference_test.copy();reference_test=soften_predictions(reference_test,adjusted_columns=True);reference_test=add_pressure_columns(reference_test);reference_test=add_volatility_columns(reference_test);reference_test=generate_trade_signals(reference_test);start_date='2025-05-01';end_date=reference_test.index.max();df_filtered=reference_test.loc[start_date:end_date];buy_signals=df_filtered[df_filtered['signal']=='Buy'];hold_signals=df_filtered[df_filtered['signal']=='Hold'];sell_signals=df_filtered[df_filtered['signal']=='Sell'];plt.figure(figsize=(15,7));plt.plot(df_filtered.index,df_filtered['close'],color='black',linewidth=1,label='Close Price');plt.scatter(buy_signals.index,buy_signals['close'],color='green',label='Buy Signal',marker='^',s=100);plt.scatter(sell_signals.index,sell_signals['close'],color='red',label='Sell Signal',marker='v',s=100);plt.xlabel('Date',fontsize=12);plt.ylabel('Close Price',fontsize=12);plt.title('Stock Close Price with Buy/Hold/Sell Signals',fontsize=14,fontweight='bold');plt.legend();plt.grid(True,linestyle='--',alpha=.6);plt.xticks(rotation=45);plt.show();columns_to_plot=['close','next_day_close_%chg_pred_value','next_week_close_%chg_pred_value'];pressure_columns=[];colors_primary=['black','blue','green','red','purple','orange','magenta'];colors_secondary=['purple','brown'];sns.set_style('whitegrid');fig,ax1=plt.subplots(figsize=(15,7))
	for(col,color)in zip(columns_to_plot,colors_primary):ax1.plot(df_filtered.index,df_filtered[col],label=col,color=color,linewidth=.8)
	ax2=ax1.twinx()
	for(col,color)in zip(pressure_columns,colors_secondary):ax2.plot(df_filtered.index,df_filtered[col],label=col,color=color,linewidth=.8,linestyle='dashed')
	ax1.set_xlabel('Date',fontsize=12);ax1.set_ylabel('Price',fontsize=12,color='black');ax1.tick_params(axis='y',labelcolor='black');ax2.set_ylim(1,2);ax2.set_ylabel('Pressure',fontsize=12,color='brown');ax2.tick_params(axis='y',labelcolor='brown');plt.title('Stock Price and Predictions with Pressure Indicators',fontsize=14,fontweight='bold');ax1.legend(fontsize=10,loc='upper left');ax2.legend(fontsize=10,loc='upper right');plt.xticks(rotation=45);ax1.grid(True,linestyle='--',alpha=.6);plt.show()
