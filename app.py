import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import wb_utils as utils
from PIL import Image
import textwrap
# --- Configuration ---

st.set_page_config(page_title="WorldBank Data Analytics Pro", layout="wide")

# --- Helper Fragments ---


# --- Header ---
col_header1, col_header2, col_header3 = st.columns([1, 4, 4], vertical_alignment="center") # Adjusted for ETH chart

with col_header1:
    st.image("Logo_HUB_New.png", width=300)

with col_header2:
    st.markdown("""
        <h1 style='text-align: center; vertical-align: middle; margin: 0; color: #003366;'>
            HỆ THỐNG GIÁM SÁT VÀ DỰ BÁO KINH TẾ VĨ MÔ
        </h1>
    """, unsafe_allow_html=True)

with col_header3:
    # Ethereum Blocks Visualization - REMOVED
    pass




# --- Sidebar / Data Ingestion ---
st.sidebar.title("Cấu hình")

# 1. Country Selection
st.sidebar.subheader("1. Chọn dữ liệu")

# Load countries from cache with spinner
with st.sidebar:
    with st.spinner("Đang tải danh sách quốc gia..."):
        all_countries_df = utils.get_country_list()

if not all_countries_df.empty:
    # Create country map {iso3c: name}
    if 'iso3c' in all_countries_df.columns:
        country_map = dict(zip(all_countries_df['iso3c'], all_countries_df['name']))
    elif 'id' in all_countries_df.columns:
        country_map = dict(zip(all_countries_df['id'], all_countries_df['name']))
    else:
        country_map = {}
else:
    country_map = {}

# Set default countries (only if they exist in the map)
default_countries = [c for c in ['VNM', 'CHN', 'USA'] if c in country_map]

selected_country_codes = st.sidebar.multiselect(
    "Chọn Quốc gia",
    options=list(country_map.keys()),
    format_func=lambda x: f"{x} - {country_map.get(x, '')}",
    default=default_countries
)

# 2. Indicator Selection
selected_indicators_names = st.sidebar.multiselect(
    "Chọn Chỉ số",
    options=list(utils.INDICATORS.keys()),
    default=list(utils.INDICATORS.keys())[:3]
)
selected_indicators_codes = [utils.INDICATORS[n] for n in selected_indicators_names]

# 3. Time Range
start_year, end_year = st.sidebar.slider("Chọn Giai đoạn", 1990, 2024, (2000, 2023))


# 4. Missing Data Strategy
st.sidebar.subheader("2. Xử lý dữ liệu")
missing_strategy = st.sidebar.selectbox(
    "Xử lý dữ liệu thiếu",
    ("Không", "Điền xuôi (Forward Fill)", "Điền ngược (Backward Fill)", "Nội suy tuyến tính (Linear Interpolation)", "Gán trung bình (Mean Imputation)", "Gán trung vị (Median Imputation)", "Bỏ qua (Drop Missing)")
)

# 5. Visitor Counter
st.sidebar.markdown("---")
st.sidebar.subheader("3. Thống kê lượt truy cập")
if "has_counted" not in st.session_state:
    utils.get_and_update_visitor_count()
    st.session_state["has_counted"] = True

current_count = utils.get_visitor_count()
st.sidebar.info(f"Tổng lượt truy cập: {current_count}")


# Fetch Data
if st.sidebar.button("Tải dữ liệu"):
    if not selected_country_codes:
        st.error("Vui lòng chọn ít nhất một quốc gia.")
    else:
        with st.spinner("Đang tải dữ liệu từ World Bank..."):
            raw_df = utils.fetch_worldbank_data(selected_indicators_codes, selected_country_codes, start=start_year, end=end_year)
            
            if raw_df is None or raw_df.empty:
                st.warning("Không tìm thấy dữ liệu cho các tham số đã chọn.")
            else:
                st.session_state['data'] = raw_df
                st.session_state['indicators_map'] = {v: k for k, v in utils.INDICATORS.items()} # Code -> Name
                st.markdown("<br>", unsafe_allow_html=True)
                st.success("Tải dữ liệu thành công!")

# Ensure data is in session state
if 'data' not in st.session_state:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Vui lòng chọn tham số và nhấn 'Tải dữ liệu' ở thanh bên.")
    st.stop()

df = st.session_state['data']

# Process Data
numeric_cols = [c for c in selected_indicators_codes if c in df.columns]
if missing_strategy != "Không":
    # Map missing strategy names back to English for logic if needed, or update logic to use Vietnamese
    # Actually utils.handle_missing_data expects English strings. Let's update the call or map.
    # We will map Vietnamese back to English for the utility function to keep it simple
    strat_map = {
        "Không": "None",
        "Điền xuôi (Forward Fill)": "Forward Fill",
        "Điền ngược (Backward Fill)": "Backward Fill",
        "Nội suy tuyến tính (Linear Interpolation)": "Linear Interpolation",
        "Gán trung bình (Mean Imputation)": "Mean Imputation",
        "Gán trung vị (Median Imputation)": "Median Imputation",
        "Bỏ qua (Drop Missing)": "Drop Missing"
    }
    df = utils.handle_missing_data(df, strat_map[missing_strategy], numeric_cols)


# 7. Visualization
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dữ liệu tổng quan", 
    "Trực quan hóa", 
    "Phân tích thống kê", 
    "Dự báo xu hướng",
    "Tăng trưởng kinh tế theo khu vực"
])

# --- Tab 1: Data Overview ---
with tab1:
    st.header("Dữ liệu tổng quan")
    st.dataframe(df)
    
    # Download Button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Tải xuống CSV",
        csv,
        "worldbank_data.csv",
        "text/csv",
        key='download-csv'
    )

# --- Tab 2: Visualization ---
with tab2:
    st.header("Bảng điều khiển trực quan hóa")
    
    def render_chart(key_suffix):
        if not selected_indicators_names:
            st.warning("Vui lòng chọn ít nhất một chỉ số để hiển thị biểu đồ.")
            return

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            chart_ind = st.selectbox("Chỉ số", selected_indicators_names, key=f"ind_{key_suffix}")
        
        # Get indicator code safely
        chart_ind_code = utils.INDICATORS.get(chart_ind)
        if not chart_ind_code:
             st.error(f"Lỗi: Không tìm thấy mã cho chỉ số '{chart_ind}'")
             return

        with col_ctrl2:
            chart_type_en = st.selectbox("Loại biểu đồ", ["Line", "Bar", "Area"], key=f"type_{key_suffix}")
        
        with col_ctrl3:
            chart_countries = st.multiselect("Quốc gia", selected_country_codes, default=selected_country_codes[:min(3, len(selected_country_codes))], key=f"ctry_{key_suffix}")
        
        # Check if indicator exists in DF
        if chart_ind_code not in df.columns:
            st.warning(f"Không có dữ liệu cho chỉ số: {chart_ind}")
            return
        
        # Filter data
        chart_df = df[df['country'].isin([country_map.get(c, c) for c in chart_countries])]
        
        if chart_df.empty:
            st.write("Không có dữ liệu cho lựa chọn này.")
            return

        title = f"{chart_ind} theo thời gian"
        if chart_type_en == "Line":
            fig = px.line(chart_df, x="year", y=chart_ind_code, color="country", title=title)
        elif chart_type_en == "Bar":
            fig = px.bar(chart_df, x="year", y=chart_ind_code, color="country", title=title, barmode='group')
        elif chart_type_en == "Area":
            fig = px.area(chart_df, x="year", y=chart_ind_code, color="country", title=title)
        
        st.plotly_chart(fig, key=f"chart_{key_suffix}")

    # 2x2 Grid
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.subheader("Biểu đồ 1")
        render_chart("1")
    with r1c2:
        st.subheader("Biểu đồ 2")
        render_chart("2")
        
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.subheader("Biểu đồ 3")
        render_chart("3")
    with r2c2:
        st.subheader("Biểu đồ 4")
        render_chart("4")

# --- Tab 3: Statistical Analysis ---
with tab3:
    st.header("Phân tích thống kê")
    
    if not selected_country_codes:
        st.warning("Vui lòng chọn ít nhất một quốc gia ở thanh cấu hình để xem thống kê.")
    else:
        st.subheader("Thống kê mô tả")
        if numeric_cols:
             st.dataframe(df[numeric_cols].describe())
        else:
             st.info("Vui lòng chọn chỉ số để xem thống kê.")
        
        st.subheader("Vị trí tứ phân vị")
        if not numeric_cols:
             st.stop()
        # Select specific year and indicator to compare
        c1, c2, c3 = st.columns(3)
        q_year = c1.selectbox("Chọn Năm", sorted(df['year'].unique(), reverse=True))
        q_ind_name = c2.selectbox("Chọn Chỉ số", selected_indicators_names)
        q_ind_code = utils.INDICATORS[q_ind_name]
        
        q_country_code = c3.selectbox("Chọn Quốc gia (để so sánh)", selected_country_codes)
        q_country_name = country_map[q_country_code]
        
        # Filter for year
        year_data = df[df['year'] == q_year]
        
        if not year_data.empty:
            if q_ind_code not in year_data.columns:
                 st.warning(f"Không có dữ liệu cho chỉ số: {q_ind_name}")
            else:
                # Get target value
                target_val = year_data[year_data['country'] == q_country_name][q_ind_code]
            
                if not target_val.empty:
                    val = target_val.values[0]
                    # Get percentiles
                    percentiles = year_data[q_ind_code].quantile([0.25, 0.5, 0.75]).to_dict()
                    
                    st.metric(f"Giá trị của {q_country_name}", f"{val:,.2f}")
                    
                    col_q1, col_q2, col_q3 = st.columns(3)
                    col_q1.metric("25% (Q1)", f"{percentiles[0.25]:,.2f}")
                    col_q2.metric("50% (Median)", f"{percentiles[0.5]:,.2f}")
                    col_q3.metric("75% (Q3)", f"{percentiles[0.75]:,.2f}")
                    
                    # Box Plot Context
                    # Horizontal Box Plot
                    # We use a dummy y-axis category "" to keep it clean and align the highlight point
                    fig_box = px.box(
                        year_data, 
                        x=q_ind_code, 
                        y=[""] * len(year_data),
                        points="all", 
                        title=f"Phân phối {q_ind_name} năm {q_year}", 
                        orientation='h'
                    )
                    
                    # Highlight selected country
                    fig_box.add_scatter(
                        x=[val], 
                        y=[""], 
                        mode='markers', 
                        marker=dict(color='red', size=12, symbol='diamond'), # Diamond shape for better visibility
                        name=q_country_name
                    )
                    
                    fig_box.update_yaxes(title="") # Hide dummy y-title
                    st.plotly_chart(fig_box)
                    
                    # --- Commentary ---
                    st.subheader("Nhận xét vị trí phân vị")
                    p25 = percentiles[0.25]
                    p50 = percentiles[0.5]
                    p75 = percentiles[0.75]
                    
                    comment = ""
                    if val < p25:
                        comment = f"**{q_country_name}** nằm trong nhóm **25% thấp nhất** (thấp hơn mức {p25:,.2f}). Điều này cho thấy chỉ số này đang ở mức thấp so với các quốc gia khác trong dữ liệu."
                    elif p25 <= val < p50:
                        comment = f"**{q_country_name}** nằm ở nhóm **dưới trung bình** (từ {p25:,.2f} đến {p50:,.2f}). Giá trị này cao hơn 25% các quốc gia thấp nhất nhưng vẫn thấp hơn mức trung vị."
                    elif p50 <= val < p75:
                        comment = f"**{q_country_name}** nằm ở nhóm **trên trung bình** (từ {p50:,.2f} đến {p75:,.2f}). Quốc gia này có chỉ số cao hơn mức trung vị nhưng chưa lọt vào nhóm cao nhất."
                    else: # val >= p75
                        comment = f"**{q_country_name}** nằm trong nhóm **25% cao nhất** (cao hơn mức {p75:,.2f}). Đây là mức chỉ số rất cao so với mặt bằng chung."
                        
                    st.info(comment)
                else:
                    st.warning("Không có dữ liệu cho quốc gia này trong năm đã chọn.")
        else:
            st.warning("Không có dữ liệu cho năm này.")

# --- Tab 4: Forecasting ---
with tab4:
    st.header("Dự báo xu hướng")
    
    if not selected_country_codes or not selected_indicators_names:
        st.warning("Vui lòng chọn Quốc gia và Chỉ số ở thanh cấu hình.")
    else:
        f_c1, f_c2, f_c3 = st.columns(3)
        f_country_code = f_c1.selectbox("Quốc gia", selected_country_codes, key="fc_country")
        f_country_name = country_map[f_country_code]
        
        f_ind_name = f_c2.selectbox("Chỉ số", selected_indicators_names, key="fc_ind")
        f_ind_code = utils.INDICATORS[f_ind_name]
        
        if f_ind_code not in df.columns:
            st.warning(f"Không có dữ liệu cho chỉ số: {f_ind_name}")
            st.stop()
        
        
        f_model = f_c3.selectbox("Thuật toán", ["Hồi quy tuyến tính (Linear Regression)", "ARIMA", "San bằng mũ (Exponential Smoothing)"])
        # Map back to English for logic
        algo_map = {
            "Hồi quy tuyến tính (Linear Regression)": "Linear Regression",
            "ARIMA": "ARIMA",
            "San bằng mũ (Exponential Smoothing)": "Exponential Smoothing"
        }
        f_model_en = algo_map[f_model]
        
        # Split Data
        st.subheader("Huấn luyện & Đánh giá")
        # Filter series
        f_df = df[(df['country'] == f_country_name)].sort_values('year')
        series = f_df.set_index('year')[f_ind_code]
        
        if len(series.dropna()) > 5:
            split_index = int(len(series) * 0.8)
            train = series.iloc[:split_index]
            test = series.iloc[split_index:]
            
            st.write(f"Huấn luyện trên {len(train)} điểm, Kiểm tra trên {len(test)} điểm.")
            
            # --- Evaluate ---
            preds = []
            try:
                if f_model_en == 'Linear Regression':
                    X_train = train.index.values.reshape(-1, 1)
                    y_train = train.values
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(X_train, y_train)
                    preds = lr.predict(test.index.values.reshape(-1, 1))
                    
                elif f_model_en == 'ARIMA':
                    from statsmodels.tsa.arima.model import ARIMA
                    model_fit = ARIMA(train, order=(1,1,1)).fit()
                    preds = model_fit.forecast(steps=len(test))
                
                elif f_model_en == 'Exponential Smoothing':
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing
                    model_fit = ExponentialSmoothing(train, trend='add').fit()
                    preds = model_fit.forecast(steps=len(test))
                    
                # Metrics
                mae = mean_absolute_error(test, preds)
                rmse = np.sqrt(mean_squared_error(test, preds))
                
                c_met1, c_met2 = st.columns(2)
                c_met1.metric("MAE (Sai số tuyệt đối trung bình)", f"{mae:.4f}")
                c_met2.metric("RMSE (Căn bậc hai sai số toàn phương trung bình)", f"{rmse:.4f}")
                
                # Plot Train/Test/Pred
                fig_eval = go.Figure()
                fig_eval.add_trace(go.Scatter(x=train.index, y=train.values, name='Huấn luyện (Train)'))
                fig_eval.add_trace(go.Scatter(x=test.index, y=test.values, name='Kiểm tra (Test)'))
                fig_eval.add_trace(go.Scatter(x=test.index, y=preds, name='Dự báo (Prediction)', line=dict(dash='dash')))
                st.plotly_chart(fig_eval, key="forecast_eval_chart")
                
            except Exception as e:
                st.error(f"Lỗi đánh giá mô hình: {e}")
                preds = []

            # --- Future Forecast ---
            st.subheader("Dự báo tương lai (2025-2026)")
            # Refit on ALL data
            future_years_count = 2
            future_map, ci_map, err = utils.forecast_series(series, f_model_en, years=future_years_count)
            
            if future_map:
                # Prepare data for plotting
                fut_years = list(future_map.keys())
                fut_values = list(future_map.values())
                
                # Extract CI
                if ci_map:
                    lower_bounds = [ci_map[y][0] for y in fut_years]
                    upper_bounds = [ci_map[y][1] for y in fut_years]
                else:
                    lower_bounds = fut_values
                    upper_bounds = fut_values

                # Plot
                fig_future = go.Figure()
                
                # Historical Data
                fig_future.add_trace(go.Scatter(x=series.index, y=series.values, name='Dữ liệu thực tế', line=dict(color='blue')))
                
                # Confidence Interval (Shaded Area)
                # Upper Bound (Hidden line)
                fig_future.add_trace(go.Scatter(
                    x=fut_years, 
                    y=upper_bounds,
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False,
                    name='Upper Bound'
                ))
                
                # Lower Bound (Filled to Upper)
                fig_future.add_trace(go.Scatter(
                    x=fut_years, 
                    y=lower_bounds,
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    name='Khoảng tin cậy 95%'
                ))
                
                # Forecast Line
                fig_future.add_trace(go.Scatter(
                    x=fut_years, 
                    y=fut_values, 
                    name='Dự báo', 
                    line=dict(color='red', dash='dot')
                ))
                
                fig_future.update_layout(title=f"Dự báo cho {f_country_name} (kèm Khoảng tin cậy 95%)", xaxis_title='Năm', yaxis_title=f_ind_name)
                st.plotly_chart(fig_future, key="forecast_future_chart")
                
                st.write("Giá trị dự báo:")
                
                # Create a DataFrame for nicer display
                disp_df = pd.DataFrame({
                    "Năm": fut_years,
                    "Dự báo": fut_values,
                    "Cận dưới (95% CI)": lower_bounds,
                    "Cận trên (95% CI)": upper_bounds
                })
                st.dataframe(disp_df)
                
            else:
                st.error(f"Dự báo thất bại: {err}")
                
        else:
            st.warning("Không đủ dữ liệu để chia tập huấn luyện/kiểm tra (Cần > 5 điểm).")

# --- Tab 5: Regional Economic Growth Map ---
with tab5:
    st.header("Tăng trưởng kinh tế theo khu vực")
    st.markdown("Số liệu về tăng trưởng kinh tế được tính theo giá cố định, các số liệu thu thập và dự báo được lấy từ World Economic Outlook (WEO) của IMF")
    
    import geopandas as gpd
    import matplotlib.pyplot as plt

    # 1. Define Regions
    REGION_COUNTRIES = {
        "Đông Nam Á": ["Vietnam", "Thailand", "Indonesia", "Malaysia", "Philippines", "Singapore", "Cambodia", "Laos", "Myanmar", "Brunei"],
        "Châu Á": ["Vietnam", "Thailand", "Indonesia", "Malaysia", "Philippines", "Singapore", "China", "Japan", "India", "South Korea", "Cambodia", "Laos", "Myanmar", "Brunei"],
        "EU": ["Germany", "France", "Italy", "Spain", "Netherlands", "Belgium", "Sweden", "Poland", "Austria", "Denmark", "Finland", "Portugal", "Greece", "Ireland", "Czech Republic", "Hungary"],
        "Bắc Mỹ": ["United States", "Canada", "Mexico"]
    }

    # 2. Select Region
    selected_region = st.selectbox("Chọn Khu vực", list(REGION_COUNTRIES.keys()))
    target_countries = REGION_COUNTRIES[selected_region]

    # 3. Load & Process Dataset
    try:
        # Load dataset.csv directly
        dataset_df = pd.read_csv("dataset.csv")
        
        target_indicator = "Gross domestic product (GDP), Constant prices, Percent change"
        
        # Filter rows
        region_df = dataset_df[
            (dataset_df['INDICATOR'] == target_indicator) & 
            (dataset_df['COUNTRY'].isin(target_countries))
        ]
        
        if region_df.empty:
            st.warning("Không tìm thấy dữ liệu cho chỉ tiêu tăng trưởng GDP trong file dataset.csv cho khu vực này.")
        else:
            # 4. Select Year
            year_cols = [c for c in region_df.columns if c.isdigit()]
            if not year_cols:
                st.error("Không tìm thấy cột năm trong dữ liệu.")
            else:
                selected_year = st.selectbox("Chọn Năm", year_cols, index=len(year_cols)-1) # Default to latest
                
                # Prepare data for plotting: Country, Value
                plot_data = region_df[['COUNTRY', selected_year]].rename(columns={selected_year: 'GrowthRate'})
                
                # 5. Load Map with better caching
                @st.cache_data(ttl=86400)  # Cache for 24 hours
                def load_map_data():
                    try:
                        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
                        return gpd.read_file(url)
                    except Exception:
                        # Try alternative URL
                        alt_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
                        return gpd.read_file(alt_url)

                try:
                    with st.spinner("Đang tải bản đồ..."):
                        world = load_map_data()
                except Exception as e:
                     st.error(f"Không thể tải dữ liệu bản đồ từ internet: {e}")
                     st.stop()
                
                # Normalize column names if needed
                if 'NAME' in world.columns:
                    world['name'] = world['NAME']
                
                # Name corrections
                name_corrections = {
                    "United States": "United States of America",
                    "Vietnam": "Vietnam"
                }
                
                plot_data['MapName'] = plot_data['COUNTRY'].replace(name_corrections)
                map_target_names = [name_corrections.get(c, c) for c in target_countries]
                
                filtered_world = world[world['name'].isin(map_target_names)]
                
                if filtered_world.empty:
                    st.error("Không tìm thấy dữ liệu bản đồ cho các quốc gia này.")
                else:
                    # Merge
                    final_map = filtered_world.merge(plot_data, left_on='name', right_on='MapName', how='left')
                    
                    # 6. Plot
                    st.subheader(f"Bản đồ Tăng trưởng GDP - {selected_region} ({selected_year})")
                    
                    # Switch to Plotly for interactivity
                    # We need to set the CRS to 4326 for Plotly if it's not already
                    if final_map.crs != "EPSG:4326":
                        final_map = final_map.to_crs("EPSG:4326")

                    # Create Choropleth
                    fig = px.choropleth(
                        final_map,
                        geojson=final_map.geometry,
                        locations=final_map.index,
                        color="GrowthRate",
                        hover_name="name",
                        hover_data=["GrowthRate"],
                        color_continuous_scale="RdYlGn",
                        projection="natural earth",
                        title=f"Tăng trưởng GDP (%) - {selected_region} {selected_year}"
                    )
                    
                    # Zoom to selected features
                    fig.update_geos(fitbounds="locations", visible=False)
                    
                    # Increase size
                    fig.update_layout(height=700, margin={"r":0,"t":40,"l":0,"b":0})
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.write("Dữ liệu chi tiết:")
                    st.dataframe(plot_data[['COUNTRY', 'GrowthRate']].sort_values('GrowthRate', ascending=False))

    except Exception as e:
        st.error(f"Có lỗi xảy ra khi xử lý dữ liệu bản đồ: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #003366; font-size: 1.1em; font-weight: bold; margin-top: 20px; margin-bottom: 20px;'>
        Hệ thống được xây dựng bởi Nhóm Nghiên cứu Kinh tế vĩ mô Việt Nam - Trường Đại học Ngân hàng Tp. Hồ Chí Minh
    </div>
""", unsafe_allow_html=True)
