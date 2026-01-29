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

# --- Visitor Counter ---
VISITOR_FILE = "visits.txt"

def get_visitor_count():
    """Reads the current visitor count."""
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
    """Increments and returns the visitor count."""
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

@st.cache_data
def fetch_worldbank_data(indicators, countries, start, end):
    """Downloads data from World Bank API."""
    try:
        df = wb.download(indicator=indicators, country=countries, start=start, end=end)
        df = df.reset_index()
        # Ensure Year is numeric and sorted
        df['year'] = pd.to_numeric(df['year'])
        df = df.sort_values(by=['country', 'year'])
        return df
    except Exception as e:
        # st.error(f"Error downloading data: {e}") # Let the app handle the alert
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

