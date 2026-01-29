import pandas_datareader.wb as wb
import pandas as pd
import streamlit as st
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import os
import requests
import json
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Connection Pooling for HTTP Requests ---
_http_session = None
_session_lock = threading.Lock()

def get_http_session():
    """Returns a shared requests.Session with connection pooling and retry logic."""
    global _http_session
    if _http_session is None:
        with _session_lock:
            if _http_session is None:
                session = requests.Session()
                retries = Retry(
                    total=2,
                    backoff_factor=0.3,
                    status_forcelist=[429, 500, 502, 503, 504]
                )
                adapter = HTTPAdapter(
                    max_retries=retries,
                    pool_connections=20,
                    pool_maxsize=50
                )
                session.mount('https://', adapter)
                session.mount('http://', adapter)
                _http_session = session
    return _http_session


# --- Visitor Counter (Thread-Safe) ---
VISITOR_FILE = "visits.txt"
_visitor_lock = threading.Lock()

def get_visitor_count():
    """Reads the current visitor count (thread-safe)."""
    with _visitor_lock:
        if os.path.exists(VISITOR_FILE):
            try:
                with open(VISITOR_FILE, "r") as f:
                    content = f.read().strip()
                    if content.isdigit():
                        return int(content)
            except:
                pass
    return 0

def get_and_update_visitor_count():
    """Increments and returns the visitor count (thread-safe)."""
    with _visitor_lock:
        count = get_visitor_count() + 1
        try:
            with open(VISITOR_FILE, "w") as f:
                f.write(str(count))
        except:
            pass
        return count



# --- Data Caching & Fetching ---
import warnings
# Suppress FutureWarning from pandas_datareader about errors='ignore'
warnings.simplefilter(action='ignore', category=FutureWarning)


INDICATORS = {
    "Tỷ lệ sinh ở vị thành niên (số ca sinh trên 1.000 phụ nữ tuổi 15-19)": "SP.ADO.TFRT",
    "Giá trị gia tăng nông, lâm, ngư nghiệp (% GDP)": "NV.AGR.TOTL.ZS",
    "Tổng lượng nước ngọt rút hàng năm (% nguồn nội bộ)": "ER.H2O.FWTL.ZS",
    "Tỷ lệ sinh được nhân viên y tế có kỹ năng đỡ đầu (% tổng số)": "SH.STA.BRTC.ZS",
    "Tỷ lệ sử dụng biện pháp tránh thai (% phụ nữ đã kết hôn tuổi 15-49)": "SP.DYN.CONU.ZS",
    "Tín dụng nội địa do khu vực tài chính cung cấp (% GDP)": "FS.AST.DOMS.GD.ZS",
    "Tiêu thụ điện năng (kWh trên đầu người)": "EG.USE.ELEC.KH.PC",
    "Sử dụng năng lượng (kg dầu quy đổi trên đầu người)": "EG.USE.PCAP.KG.OE",
    "Xuất khẩu hàng hóa và dịch vụ (% GDP)": "NE.EXP.GNFS.ZS",
    "Tổng nợ nước ngoài (DOD, USD hiện hành)": "DT.DOD.DECT.CD",
    "Tổng tỷ suất sinh (số con trên mỗi phụ nữ)": "SP.DYN.TFRT.IN",
    "Đầu tư trực tiếp nước ngoài, dòng vốn ròng (BoP, USD hiện hành)": "BX.KLT.DINV.CD.WD",
    "Diện tích rừng (km2)": "AG.LND.FRST.K2",
    "GDP (USD hiện hành)": "NY.GDP.MKTP.CD",
    "Tăng trưởng GDP (hàng năm %)": "NY.GDP.MKTP.KD.ZG",
    "GNI bình quân đầu người, phương pháp Atlas (USD hiện hành)": "NY.GNP.PCAP.CD",
    "GNI bình quân đầu người, PPP (đồng quốc tế hiện hành)": "NY.GNP.PCAP.PP.CD",
    "GNI, phương pháp Atlas (USD hiện hành)": "NY.GNP.ATLS.CD",
    "GNI, PPP (đồng quốc tế hiện hành)": "NY.GNP.MKTP.PP.CD",
    "Tổng tích lũy vốn (% GDP)": "NE.GDI.TOTL.ZS",
    "Xuất khẩu công nghệ cao (% xuất khẩu chế tạo)": "TX.VAL.TECH.MF.ZS",
    "Tiêm chủng, sởi (% trẻ em 12-23 tháng tuổi)": "SH.IMM.MEAS",
    "Nhập khẩu hàng hóa và dịch vụ (% GDP)": "NE.IMP.GNFS.ZS",
    "Tỷ trọng thu nhập của 20% thấp nhất": "SI.DST.FRST.20",
    "Giá trị gia tăng công nghiệp (% GDP)": "NV.IND.TOTL.ZS",
    "Lạm phát, chỉ số giảm phát GDP (hàng năm %)": "NY.GDP.DEFL.KD.ZG",
    "Tuổi thọ trung bình khi sinh (năm)": "SP.DYN.LE00.IN",
    "Thương mại hàng hóa (% GDP)": "TG.VAL.TOTL.GD.ZS",
    "Chi tiêu quân sự (% GDP)": "MS.MIL.XPND.GD.ZS",
    "Thuê bao di động (trên 100 người)": "IT.CEL.SETS.P2",
    "Tỷ lệ tử vong dưới 5 tuổi (trên 1.000 trẻ sinh sống)": "SH.DYN.MORT",
    "Chỉ số điều kiện thương mại hàng hóa (2015 = 100)": "TT.PRI.MRCH.XD.WD",
    "Di cư ròng": "SM.POP.NETM",
    "Viện trợ phát triển chính thức ròng và viện trợ chính thức nhận được (USD hiện hành)": "DT.ODA.ALLD.CD",
    "Kiều hối cá nhân nhận được (USD hiện hành)": "BX.TRF.PWKR.CD.DT",
    "Mật độ dân số (người trên km2 đất)": "EN.POP.DNST",
    "Tăng trưởng dân số (hàng năm %)": "SP.POP.GROW",
    "Tổng dân số": "SP.POP.TOTL",
    "Tỷ lệ nghèo ở mức 3.00 USD/ngày (2021 PPP) (% dân số)": "SI.POV.DDAY",
    "Tỷ lệ nghèo theo chuẩn nghèo quốc gia (% dân số)": "SI.POV.NAHC",
    "Tỷ lệ nhiễm HIV (% dân số tuổi 15-49)": "SH.DYN.AIDS.ZS",
    "Tỷ lệ thiếu cân (% trẻ em dưới 5 tuổi)": "SH.STA.MALN.ZS",
    "Tỷ lệ hoàn thành tiểu học (% nhóm tuổi liên quan)": "SE.PRM.CMPT.ZS",
    "Doanh thu, không bao gồm viện trợ (% GDP)": "GC.REV.XGRT.GD.ZS",
    "Tỷ lệ nhập học tiểu học (% gộp)": "SE.PRM.ENRR",
    "Chỉ số cân bằng giới tính nhập học tiểu học và trung học": "SE.ENR.PRSC.FM.ZS",
    "Tỷ lệ nhập học trung học (% gộp)": "SE.SEC.ENRR",
    "Diện tích bề mặt (km2)": "AG.SRF.TOTL.K2",
    "Doanh thu thuế (% GDP)": "GC.TAX.TOTL.GD.ZS",
    "Khu vực bảo tồn trên cạn và biển (% tổng diện tích lãnh thổ)": "ER.PTD.TOTL.ZS",
    "Tổng nghĩa vụ nợ (% xuất khẩu hàng hóa, dịch vụ và thu nhập sơ cấp)": "DT.TDS.DECT.EX.ZS"
}
@st.cache_data
def get_country_list():
    """Fetches a list of countries from World Bank API."""
    try:
        countries = wb.get_countries()
        # Filter for aggregates and actual countries if needed, but for now we take all valid ones
        # typically we want only actual countries, not regions, but the prompt mentioned Region/IncomeGroup
        return countries
    except Exception as e:
        st.error(f"Error fetching country list: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_worldbank_data(indicators, countries, start, end):
    """
    Downloads data from World Bank API with improved error handling.
    - First tries batch download for all countries
    - If batch fails, tries each country individually
    - Skips countries without data instead of failing entirely
    """
    all_dfs = []
    
    # Try batch download first (faster)
    try:
        df = wb.download(indicator=indicators, country=countries, start=start, end=end)
        if df is not None and not df.empty:
            df = df.reset_index()
            df['year'] = pd.to_numeric(df['year'])
            df = df.sort_values(by=['country', 'year'])
            return df
    except Exception as batch_error:
        # Batch failed, try individual countries
        pass
    
    # Fallback: Try each country individually
    for country_code in countries:
        try:
            single_df = wb.download(
                indicator=indicators, 
                country=[country_code], 
                start=start, 
                end=end
            )
            if single_df is not None and not single_df.empty:
                single_df = single_df.reset_index()
                all_dfs.append(single_df)
        except Exception as e:
            # Skip this country silently - no data available
            continue
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df['year'] = pd.to_numeric(combined_df['year'])
        combined_df = combined_df.sort_values(by=['country', 'year'])
        return combined_df
    
    return None

# --- Data Processing ---
def handle_missing_data(df, method, indicators):
    """Handles missing data based on the selected method."""
    df_clean = df.copy()
    
    # We apply the method group by country to avoid bleeding data between countries
    if method == 'Drop Missing':
        return df_clean.dropna(subset=indicators)
    
    # helper for group application
    def apply_method(group):
        if method == 'Forward Fill':
            return group.ffill()
        elif method == 'Backward Fill':
            return group.bfill()
        elif method == 'Linear Interpolation':
            return group.interpolate(method='linear')
        elif method == 'Mean Imputation':
            for col in indicators:
                group[col] = group[col].fillna(group[col].mean())
            return group
        elif method == 'Median Imputation':
            for col in indicators:
                group[col] = group[col].fillna(group[col].median())
            return group
        return group

    df_clean = df_clean.groupby('country', group_keys=False).apply(apply_method)
    
    # Some methods (like interpolation) might still leave NaNs at edges, 
    # user might want to fill those or drop them. For now we return as is.
    return df_clean

# --- Forecasting ---
def forecast_series(series, model_type, years=2):
    """
    Simple forecasting wrapper with 95% Confidence Intervals.
    Input: Series with DatetimeIndex or PeriodIndex would be best, but here we likely have simple Year int.
    We will assume annual frequency.
    Returns: 
        - forecast_map: dict {year: value}
        - ci_map: dict {year: [lower, upper]}
        - error: str or None
    """
    # Clean Series
    series = series.dropna()
    if len(series) < 3:
        return None, None, "Not enough data points"

    last_year = series.index.max()
    future_years = [int(last_year + i) for i in range(1, years + 1)]
    
    preds = []
    cis = [] # List of [lower, upper]
    
    try:
        if model_type == 'Linear Regression':
            # Use statsmodels OLS for prediction intervals
            import statsmodels.api as sm
            
            X = sm.add_constant(series.index.values) # Add intercept
            y = series.values
            
            model = sm.OLS(y, X)
            results = model.fit()
            
            # Future X
            X_future = sm.add_constant(future_years, has_constant='add')
            
            # get_prediction handles CIs
            predictions = results.get_prediction(X_future)
            pred_summary = predictions.summary_frame(alpha=0.05) # 95% CI
            
            preds = pred_summary['mean'].values
            cis = pred_summary[['mean_ci_lower', 'mean_ci_upper']].values
            
        elif model_type == 'ARIMA':
            # Needs simple index
            # Suppress warnings for frequency if possible
            model = ARIMA(series, order=(1,1,1)) 
            model_fit = model.fit()
            
            # Forecast with CI
            forecast_res = model_fit.get_forecast(steps=years)
            preds = forecast_res.predicted_mean.values
            
            ci_df = forecast_res.conf_int(alpha=0.05)
            # ci_df columns usually are lower/upper
            cis = ci_df.values
            
        elif model_type == 'Exponential Smoothing':
            model = ExponentialSmoothing(series, trend='add', seasonal=None)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=years)
            preds = forecast.values
            
            # Estimate CI manually based on residuals (Naive method)
            # CI = pred +/- 1.96 * std(residuals) * sqrt(h)
            residuals = model_fit.fittedvalues - series
            sigma = np.std(residuals)
            
            cis = []
            for i, p in enumerate(preds):
                h = i + 1
                margin = 1.96 * sigma * np.sqrt(h)
                cis.append([p - margin, p + margin])
            cis = np.array(cis)
            
        return dict(zip(future_years, preds)), dict(zip(future_years, cis.tolist())), None
        
    except Exception as e:
        return None, None, str(e)

# --- Ethereum Data ---
# Global cache for last successful ETH data (fallback when API fails)
_last_eth_blocks = None
_last_eth_price = None

@st.cache_data(ttl=60, show_spinner=False)  # Extended from 10s to 60s
def get_latest_eth_blocks(limit=6):
    """
    Fetches the latest 'limit' blocks from Ethereum mainnet.
    Uses connection pooling and returns cached fallback on complete failure.
    """
    global _last_eth_blocks
    
    rpc_urls = [
        "https://eth.public-rpc.com", 
        "https://1rpc.io/eth",
        "https://rpc.flashbots.net",
        "https://cloudflare-eth.com"
    ]
    
    session = get_http_session()
    headers = {'Content-Type': 'application/json'}
    
    for url in rpc_urls:
        try:
            # 1. Get Latest Block Number
            payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
            response = session.post(url, json=payload, headers=headers, timeout=2)
            
            if response.status_code != 200:
                continue
                
            res_json = response.json()
            if 'result' not in res_json:
                continue
                
            latest_block = int(res_json['result'], 16)
            
            # 2. Batch Fetch Previous Blocks
            batch_payload = []
            for i in range(limit):
                block_num = latest_block - i
                batch_payload.append({
                    "jsonrpc": "2.0",
                    "method": "eth_getBlockByNumber",
                    "params": [hex(block_num), False],
                    "id": block_num
                })
                
            batch_resp = session.post(url, json=batch_payload, headers=headers, timeout=3)
            if batch_resp.status_code != 200:
                continue
                
            results = batch_resp.json()
            
            blocks = []
            # Map by ID
            result_map = {r['id']: r['result'] for r in results if isinstance(r, dict) and 'result' in r and r['result']}
            
            for i in range(limit):
                b_num = latest_block - i
                b_data = result_map.get(b_num)
                if b_data:
                    gas_used = int(b_data['gasUsed'], 16)
                    gas_limit = int(b_data['gasLimit'], 16)
                    fullness = (gas_used / gas_limit) * 100 if gas_limit > 0 else 0
                    
                    blocks.append({
                        'number': b_num,
                        'gasUsed': gas_used,
                        'gasLimit': gas_limit,
                        'fullness': fullness
                    })
            
            if len(blocks) >= limit // 2:
                _last_eth_blocks = blocks  # Save for fallback
                return blocks
                
        except Exception:
            continue
    
    # Return last successful result if available
    if _last_eth_blocks:
        return _last_eth_blocks
            
    # Fallback to Mock Data if all RPCs fail and no cache
    import random
    import time
    
    base_block = 21000000 + int(time.time() / 12) 
    
    mock_blocks = []
    for i in range(limit):
        mock_blocks.append({
            'number': base_block - i,
            'gasUsed': 0,
            'gasLimit': 0,
            'fullness': random.uniform(40, 80)
        })
        
    return mock_blocks

@st.cache_data(ttl=60, show_spinner=False)  # Extended from 10s to 60s
def get_eth_price():
    """
    Fetches latest ETH price from Binance, falling back to CoinGecko.
    Uses connection pooling and returns cached fallback on failure.
    """
    global _last_eth_price
    
    session = get_http_session()
    
    # 1. Try Binance
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr?symbol=ETHUSDT"
        response = session.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            result = {
                'price': float(data['lastPrice']),
                'volume': float(data['quoteVolume']),
                'change': float(data['priceChangePercent'])
            }
            _last_eth_price = result  # Save for fallback
            return result
    except:
        pass

    # 2. Fallback: CoinGecko
    try:
        url_cg = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true"
        response = session.get(url_cg, timeout=2)
        if response.status_code == 200:
            data = response.json()
            eth = data.get('ethereum', {})
            result = {
                'price': float(eth.get('usd', 0)),
                'volume': float(eth.get('usd_24h_vol', 0)),
                'change': float(eth.get('usd_24h_change', 0))
            }
            _last_eth_price = result
            return result
    except:
        pass
    
    # Return last successful result if available
    if _last_eth_price:
        return _last_eth_price
        
    # Final fallback with default value
    return {'price': 3200.0, 'volume': 0, 'change': 0}

