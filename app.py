import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error as mape
from sktime.performance_metrics.forecasting import mean_squared_error as mse
from sktime.performance_metrics.forecasting import mean_absolute_error as mae

import warnings
warnings.filterwarnings("ignore")

# 1. 페이지 기본 설정
st.set_page_config(page_title="자동 시계열 예측 웹앱", layout="wide")
st.title("📈 임의의 단변량 시계열 자동 예측 대시보드")
st.write("CSV 파일을 업로드하면 데이터를 분석하여 평가지표와 예측 결과를 시각화합니다.")

# 2. 사이드바: 파라미터 조절
st.sidebar.header("⚙️ 분석 파라미터 설정")
# 과제 요구사항: 시평(Horizon) 파라미터화
horizon = st.sidebar.slider("예측 시평 (Horizon / Test Size)", min_value=1, max_value=60, value=24)
# (보너스) 데이터 주기를 선택할 수 있게 하여 임의의 파일에 대응
freq_option = st.sidebar.selectbox("데이터 측정 주기", options=['월별 (Monthly)', '일별 (Daily)'])
freq_dict = {'월별 (Monthly)': 'M', '일별 (Daily)': 'D'}
sp_dict = {'월별 (Monthly)': 12, '일별 (Daily)': 7} # 계절성 주기

# 3. 메인 화면: 파일 업로드
uploaded_file = st.file_uploader("분석할 시계열 CSV 파일을 업로드하세요. (첫 열: 날짜, 두 번째 열: 값)", type=["csv"])

if uploaded_file is not None:
    try:
        # [데이터 전처리]
        df = pd.read_csv(uploaded_file)
        date_col = df.columns[0]
        val_col = df.columns[1]
        
        # 결측치 선형 보간
        df[val_col] = df[val_col].interpolate(method='linear')
        
        # sktime 포맷 변환 (선택된 주기에 맞춰서)
        ts_data = df.set_index(date_col)[val_col]
        ts_data.index = pd.to_datetime(ts_data.index)
        ts_data.index = ts_data.index.to_period(freq_dict[freq_option])
        
        # [모델링 및 예측]
        # 슬라이더에서 설정한 시평(horizon)만큼 테스트 데이터로 분할
        y_train, y_test = temporal_train_test_split(ts_data, test_size=horizon)
        
        # Holt-Winter's 모델 학습 (사이드바 주기 연동)
        forecaster = ExponentialSmoothing(trend="add", seasonal="mul", sp=sp_dict[freq_option])
        forecaster.fit(y_train)
        
        fh = np.arange(1, len(y_test) + 1)
        y_pred = forecaster.predict(fh=fh)
        
        # [평가지표 계산]
        result_mae = mae(y_test, y_pred)
        result_rmse = np.sqrt(mse(y_test, y_pred))
        result_mape = mape(y_test, y_pred) * 100
        
        # --- 화면 출력(대시보드) ---
        st.divider()
        st.subheader("📊 1. 모델 예측 성능 지표")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{result_mae:.2f}")
        col2.metric("RMSE", f"{result_rmse:.2f}")
        col3.metric("MAPE", f"{result_mape:.2f}%")
        
        st.subheader("📈 2. 시계열 예측 결과 시각화")
        # Plotly 시각화 (Datetime 포맷으로 복구)
        train_dates = y_train.index.to_timestamp().astype(str)
        test_dates = y_test.index.to_timestamp().astype(str)
        pred_dates = y_pred.index.to_timestamp().astype(str)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_dates, y=y_train.values, mode='lines', name='Train Data', line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=test_dates, y=y_test.values, mode='lines', name='Actual (Test)', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=pred_dates, y=y_pred.values, mode='lines', name='Prediction', line=dict(color='#ff7f0e', width=2)))

        fig.update_layout(hovermode='x unified', template='plotly_white', xaxis_title=date_col, yaxis_title=val_col)
        st.plotly_chart(fig, use_container_width=True)
        
        # 원본 데이터 확인
        with st.expander("원본 데이터 확인하기"):
            st.dataframe(df)

    except Exception as e:
        st.error(f"데이터를 분석하는 중 오류가 발생했습니다. 파일 형식을 확인해주세요. (오류 내용: {e})")

else:
    st.info("👆 위에 있는 버튼을 눌러 시계열 데이터(CSV)를 업로드해 보세요.")
