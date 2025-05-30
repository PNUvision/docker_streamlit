import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import io
import matplotlib.pyplot as plt
from scipy.signal import correlate
# import pdb

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from matplotlib import font_manager, rc
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)  
plt.rcParams['axes.unicode_minus'] = False



st.set_page_config(layout="wide")  # 넓은 레이아웃

# ---------- 함수 정의 ----------
def normalized_cross_correlation(data, template):
    data_mean = np.mean(data)
    template_mean = np.mean(template)
    data_normalized = data - data_mean
    template_normalized = template - template_mean
    correlation = correlate(data_normalized, template_normalized, mode='valid')
    data_std = np.std(data)
    template_std = np.std(template)
    ncc = correlation / (data_std * template_std * len(template))
    return ncc

def group_consecutive(values, max_diff=50):
    if len(values) == 0:
        return []
    values = sorted(values)
    groups = []
    group = [values[0]]
    for current, next_ in zip(values, values[1:]):
        if next_ <= current + max_diff:            
            group.append(next_)
        else:
            groups.append(group)
            group = [next_]
    groups.append(group)
    return groups

# ---------- 데이터 로딩 ----------
st.title("🚀 신호 매치 및 추출 앱")


# 사용자로부터 feather 파일 업로드 받기
uploaded_file = st.file_uploader("📂 Feather (.ftr) 파일을 선택하세요", type=["ftr"])

if uploaded_file is not None:
    try:
        # 업로드된 feather 파일 읽기
        first_df = pd.read_feather(uploaded_file)
        cp_df = first_df.copy()
        # print(cp_df.head())
        
        # 사용자가 데이터 속성을 선택할 수 있는 드롭다운 메뉴 추가
        selected_column = st.selectbox(
            "매칭할 신호 데이터 컬럼을 선택하세요:",
            options=cp_df.columns.tolist(),
            index=cp_df.columns.tolist().index('GT FUEL CONSUMPTION') if 'GT FUEL CONSUMPTION' in cp_df.columns else 0
        )
        
        # 선택한 컬럼이 존재하는지 확인
        if selected_column in cp_df.columns:
            signal = cp_df[selected_column].values
            st.success(f"✅ '{selected_column}' 컬럼이 성공적으로 로드되었습니다.")
        else:
            st.error(f"❗ '{selected_column}' 컬럼이 존재하지 않습니다.")
            st.stop()

    except Exception as e:
        st.error(f"❗ 파일을 읽는 도중 오류 발생: {e}")
        st.stop()

else:
    st.warning("⏳ Feather 파일을 업로드해주세요.")
    st.stop()
# ------------------------------


# 사이드바 최상단: 템플릿 업로드
st.sidebar.markdown("🧬 **Template 파일 업로드 (.npy)**")

uploaded_template_1 = st.sidebar.file_uploader("📂 기동 시작 템플릿 (Template 1)", type=["npy"], key="t1")
uploaded_template_2 = st.sidebar.file_uploader("📂 기동 종료 템플릿 (Template 2)", type=["npy"], key="t2")

# 기본값으로 로드
template_1 = np.load('fuel_temp_st.npy')
template_2 = np.load('fuel_temp_et.npy')

# 업로드가 있다면 덮어쓰기
if uploaded_template_1 is not None:
    try:
        template_1 = np.load(uploaded_template_1)
        st.sidebar.success("✅ Template 1 업로드 완료")
    except:
        st.sidebar.error("❗ Template 1 로드 실패")

if uploaded_template_2 is not None:
    try:
        template_2 = np.load(uploaded_template_2)
        st.sidebar.success("✅ Template 2 업로드 완료")
    except:
        st.sidebar.error("❗ Template 2 로드 실패")


# ------------------------------------------------------------------------------------------
# 데이터 준비 및 전처리
def preprocess_signal(signal2):
    # NaN 및 Inf 값 확인
    nan_mask = np.isnan(signal2)
    inf_mask = np.isinf(signal2)
    
    if np.any(nan_mask) or np.any(inf_mask):
        print(f"Found {np.sum(nan_mask)} NaN values and {np.sum(inf_mask)} Inf values")
        
        # NaN/Inf 값 제거를 위한 복사본 생성
        clean_signal = signal2.copy()
        
        # 단순한 방법: NaN 및 Inf 값을 이웃 값의 평균으로 대체
        bad_indices = np.where(nan_mask | inf_mask)[0]
        for idx in bad_indices:
            # 좌우 10개 샘플 내에서 유효한 값을 찾아 평균 계산
            window_start = max(0, idx - 10)
            window_end = min(len(signal2), idx + 11)
            window = signal2[window_start:window_end]
            valid_values = window[~(np.isnan(window) | np.isinf(window))]
            
            if len(valid_values) > 0:
                clean_signal[idx] = np.mean(valid_values)
            else:
                # 주변에 유효한 값이 없으면 0으로 대체
                clean_signal[idx] = 0
        
        return clean_signal
    
    return signal2

# 전처리 적용
signal = preprocess_signal(signal)
template_1 = preprocess_signal(template_1)
template_2 = preprocess_signal(template_2)
# ------------------------------------------------------------------------------------------


# ---------- 사용자 입력 ----------
st.sidebar.header("🔧 매치 설정")
with st.sidebar:
    st.markdown("📉 **Template 1 (기동 시작)**")
    fig_t1, ax1 = plt.subplots(figsize=(3, 1.5))
    ax1.plot(template_1, linewidth=0.8)
    # ax1.set_xticks([]), ax1.set_yticks([])
    ax1.set_title("시작 템플릿", fontsize=10)
    st.pyplot(fig_t1)
    plt.close(fig_t1)

    st.markdown("📈 **Template 2 (기동 종료)**")
    fig_t2, ax2 = plt.subplots(figsize=(3, 1.5))
    ax2.plot(template_2, linewidth=0.8, color='orange')
    # ax2.set_xticks([]), ax2.set_yticks([])
    ax2.set_title("종료 템플릿", fontsize=10)
    st.pyplot(fig_t2)
    plt.close(fig_t2)

    st.markdown("---")


with st.sidebar:
    st.header("🔄 매칭기 설정")
    max_diff = st.selectbox(
        "연속으로 간주할 최대 차이값",
        options=[1, 10, 50, 100, 200, 500, 1000],
        index=2,  # 기본값을 50으로 설정 (index 2)
        help="두 값 사이의 차이가 이 값 이하이면 연속으로 간주합니다."
    )
    st.markdown("---")



# ---------- 사용자 입력 ----------
with st.sidebar:
    st.markdown("🧠 **회사명:** ㈜파시디엘")
    st.markdown("🏫 **연구실:** VisLAB PNU")
    st.markdown("👨‍💻 **제작자:** (C) DJKang")
    st.markdown("🛠️ **버전:** V.1.0 (04-22-2025)")
    st.markdown("---")


with st.form(key="matching_form"):
    st_thres = st.slider("기동 시작 NCC Threshold", 0.0, 1.0, 0.2, 0.01)
    st_low = st.number_input("기동 시작 신호 최소값", value=0.0)
    st_high = st.number_input("기동 시작 신호 최대값", value=1.0)
    offset_1 = st.number_input("기동 시작 offset", value=1000)

    et_thres = st.slider("기동 종료 NCC Threshold", 0.0, 1.0, 0.2, 0.01)
    et_low = st.number_input("기동 종료 신호 최소값", value=5.0)
    et_high = st.number_input("기동 종료 신호 최대값", value=8.0)
    offset_2 = st.number_input("기동 종료 offset", value=800)

    remove_st_idx = st.text_input("기동 시작부 제거할 그룹 인덱스 (쉼표로 구분)", value="0,5,17")
    remove_et_idx = st.text_input("기동 종료부 제거할 그룹 인덱스 (쉼표로 구분)", value="")

    submitted = st.form_submit_button("▶️ 매치 수행")

if submitted:
    # ---------- 시작부 매칭 ----------
    # ncc_start[0]은 템플릿이 signal[0:20]과 정렬될 때의 상관관계 값
    # ncc_start[10]은 템플릿이 signal[10:30]과 정렬될 때의 상관관계 값
    ncc_start = normalized_cross_correlation(signal, template_1)
    # true_idx_st = np.where((ncc_start > st_thres) & (signal[:len(ncc_start)] > st_low) & (signal[:len(ncc_start)] < st_high))[0]
    st_ncc_above_threshold = np.where(ncc_start > st_thres)[0]
    # 그 인덱스에서 signal 값이 범위 내에 있는지 확인합니다
    true_idx_st = st_ncc_above_threshold[
        (signal[st_ncc_above_threshold] > st_low) & 
        (signal[st_ncc_above_threshold] < st_high)
    ]    
    st_groups = group_consecutive(true_idx_st)

    for idx in sorted([int(i) for i in remove_st_idx.split(',') if i.strip().isdigit()], reverse=True):
        if 0 <= idx < len(st_groups):
            del st_groups[idx]

    means_start = [np.mean(signal[grp]) for grp in st_groups]
    st.subheader(f"🟢 기동 시작: 그룹 수 = {len(st_groups)}")
    with st.expander("기동 시작 그룹 평균값 (전체 표시)", expanded=True):
        st.markdown(
            f"<div style='max-height: 300px; overflow-y: auto; border:1px solid #ccc; padding:10px;'>"
            + "<br>".join([f"그룹 {i}: 평균 = {v:.4f}" for i, v in enumerate(means_start)])
            + "</div>",
            unsafe_allow_html=True
        )

    # ---------- 종료부 매칭 ----------
    ncc_end = normalized_cross_correlation(signal, template_2)
    # true_idx_et = np.where((ncc_end > et_thres) & (signal[:len(ncc_end)] > et_low) & (signal[:len(ncc_end)] < et_high))[0]
    et_ncc_above_threshold = np.where(ncc_end > et_thres)[0]
    true_idx_et = et_ncc_above_threshold[
        (signal[et_ncc_above_threshold] > et_low) & 
        (signal[et_ncc_above_threshold] < et_high)
    ]    
    et_groups = group_consecutive(true_idx_et)

    for idx in sorted([int(i) for i in remove_et_idx.split(',') if i.strip().isdigit()], reverse=True):
        if 0 <= idx < len(et_groups):
            del et_groups[idx]

    means_end = [np.mean(signal[grp]) for grp in et_groups]
    st.subheader(f"🔴 기동 종료: 그룹 수 = {len(et_groups)}")
    with st.expander("\ud68c\ubcf5 \uc885\ub8cc \ud3c9\uade0\uac12 (전체 표시)", expanded=True):
        st.markdown(
            f"<div style='max-height: 300px; overflow-y: auto; border:1px solid #ccc; padding:10px;'>"
            + "<br>".join([f"그룹 {i}: 평균 = {v:.4f}" for i, v in enumerate(means_end)])
            + "</div>",
            unsafe_allow_html=True
        )


    # -------------------------------------------------------------------
    # 샘플링 비율 선택 위젯 추가 (기본값: 10)
    sampling_rate = st.slider("샘플링 비율 선택", min_value=1, max_value=50, value=10, step=1)

    # 신호 샘플링 함수 정의
    def downsample(data, rate):
        return data[::rate]

    # 샘플링된 신호와 인덱스 생성
    sampled_signal = downsample(signal, sampling_rate)
    sampled_indices = list(range(0, len(signal), sampling_rate))

    # 기동 시작 매칭 시각화
    fig1 = go.Figure()

    # 메인 신호 플롯 (샘플링 적용)
    fig1.add_trace(
        go.Scatter(
            x=sampled_indices,
            y=sampled_signal,
            mode='lines',
            name='Signal',
            line=dict(color='blue', width=1)
        )
    )

    # 매칭 위치 표시 (샘플링 적용하지 않음 - 정확한 위치 유지)
    for i, grp in enumerate(st_groups):
        x = grp[0] - offset_1
        fig1.add_trace(
            go.Scatter(
                x=[x, x],
                y=[min(sampled_signal), max(sampled_signal)],
                mode='lines',
                name=f'Match {i}',
                line=dict(color='red', width=1, dash='dash')
            )
        )
        # 텍스트 레이블 추가
        fig1.add_annotation(
            x=x,
            y=max(sampled_signal) * 0.9,
            text=f"{i}",
            showarrow=False,
            font=dict(color='red', size=15)
        )

    # 레이아웃 설정
    fig1.update_layout(
        title=f'Template 1 Matching (Start) - 샘플링 비율: 1/{sampling_rate}',
        xaxis_title='Sample Index',
        yaxis_title='Signal Value',
        height=600,
        hovermode='closest',
        showlegend=False
    )

    # 플롯 표시
    st.plotly_chart(fig1, use_container_width=True)

    # 기동 종료 매칭 시각화
    fig2 = go.Figure()

    # 메인 신호 플롯 (샘플링 적용)
    fig2.add_trace(
        go.Scatter(
            x=sampled_indices,
            y=sampled_signal,
            mode='lines',
            name='Signal',
            line=dict(color='blue', width=1)
        )
    )

    # 매칭 위치 표시 (샘플링 적용하지 않음 - 정확한 위치 유지)
    for i, grp in enumerate(et_groups):
        x = grp[0] + offset_2
        fig2.add_trace(
            go.Scatter(
                x=[x, x],
                y=[min(sampled_signal), max(sampled_signal)],
                mode='lines',
                name=f'Match {i}',
                line=dict(color='red', width=1, dash='dash')
            )
        )
        # 텍스트 레이블 추가
        fig2.add_annotation(
            x=x,
            y=max(sampled_signal) * 0.9,
            text=f"{i}",
            showarrow=False,
            font=dict(color='red', size=15)
        )

    # 레이아웃 설정
    fig2.update_layout(
        title=f'Template 2 Matching (End) - 샘플링 비율: 1/{sampling_rate}',
        xaxis_title='Sample Index',
        yaxis_title='Signal Value',
        height=600,
        hovermode='closest',
        showlegend=False
    )

    # 플롯 표시
    st.plotly_chart(fig2, use_container_width=True)
    # -------------------------------------------------------------------


# ---- 1. 추출 실행 버튼 상태 관리 ----
if "run_extraction" not in st.session_state:
    st.session_state.run_extraction = False

if st.button("✂️ 기동 신호 추출 및 시각화"):
    st.session_state.run_extraction = True

# ---- 2. 추출 실행 조건 (제출 완료 후, 별도 실행) ----
if st.session_state.run_extraction:
    if 'st_groups' not in locals() or 'et_groups' not in locals():
        st.warning("먼저 좌측에서 매치 수행 후 추출할 수 있습니다.")
    elif len(st_groups) == 0 or len(et_groups) == 0:
        st.warning("시작 또는 종료 그룹이 비어 있어 추출할 수 없습니다.")
    else:
        st.success("✅ 추출 및 시각화 실행 중...")
        pairs = []
        for st_grp, et_grp in zip(st_groups, et_groups):
            st_pt = max(0, st_grp[0] - offset_1)
            et_pt = min(len(signal), et_grp[0] + offset_2)
            if st_pt < et_pt:
                pairs.append((st_pt, et_pt))

        st.write(f"총 추출 구간 수: {len(pairs)}")


        # ✅ 추가된 코드
        # save_folder = "saved_crops"  # docker에서는 /app로
        saved_root = "/app/data" if os.path.exists("/app/data") else os.getcwd()
        save_folder = os.path.join(saved_root, 'saved_crops')
        os.makedirs(save_folder, exist_ok=True)

        for i, (st_pt, et_pt) in enumerate(pairs):
            # 1. 다변량 crop
            crop_df = cp_df.iloc[st_pt:et_pt]  # 모든 컬럼에 대해 crop

            # 2. numpy로 변환
            crop_np = crop_df.to_numpy()  # shape: (crop_len, feature_dim)

            # crop_np.head()
            # 3. 저장 파일명 생성 (날짜+시간 기반)
            if 'timestamp' in crop_df.columns:  # 'timestamp' 컬럼이 datetime 형태라면
                # st_pt 위치의 날짜/시간을 기반으로 파일명 생성
                timestamp = pd.to_datetime(crop_df['timestamp'].iloc[0])
                timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            else:
                timestamp_str = f"crop_{i}"

            save_path = os.path.join(save_folder, f"{timestamp_str}_crop.npy")

            # 4. npy 저장
            np.save(save_path, crop_np)

            # 5. crop 시각화 (특징 하나만 간단히 예시)
            fig_crop, ax_crop = plt.subplots(figsize=(8, 3))
            # ax_crop.plot(crop_df.iloc[:, 0])   
            ax_crop.plot(signal[st_pt:et_pt])   
            ax_crop.set_title(f"추출 신호 {i} (len={len(crop_df)})")
            st.pyplot(fig_crop)
            plt.close(fig_crop)

            st.info(f"✅ Saved: {save_path}")        


        # 👉 추출 완료 후 자동 리셋 (선택)
        st.session_state.run_extraction = False


# ====================================================================== 
# 서버의 /app/data에 저장된 npy파일들을 client로 압축해서 다운로드
def get_npy_files_in_data_dir():
    """
    컨테이너 내의 /app/data 디렉토리에 있는 모든 .npy 파일 찾기
    (이 디렉토리는 호스트의 /home/pashidl/streamlit/dashboard에 매핑됨)
    """
    default_root = "/app/data" if os.path.exists("/app/data") else os.getcwd()
    data_dir = os.path.join(default_root, 'saved_crops')
    # data_dir = "/app/data"
    npy_files = []
    
    # 디렉토리 내의 모든 .npy 파일 찾기
    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            npy_files.append(os.path.join(data_dir, file))
    
    return npy_files

def create_download_link_for_all_files(npy_files):
    """모든 .npy 파일을 zip으로 압축하여 다운로드 링크 생성"""
    # 메모리에 ZIP 파일 생성
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in npy_files:
            file_name = os.path.basename(file_path)
            # 각 .npy 파일을 zip에 추가
            zip_file.write(file_path, file_name)
    
    zip_buffer.seek(0)
    
    # ZIP 파일 다운로드 버튼 생성
    st.download_button(
        label="모든 NPY 파일 다운로드",
        data=zip_buffer,
        file_name="all_npy_files.zip",
        mime="application/zip"
    )

# 다운로드 섹션을 UI에 추가
st.header("NPY 파일 다운로드")

# 다운로드 버튼 표시
if st.button("NPY 파일 검색 및 다운로드 준비"):
    # 버튼이 클릭되었을 때만 아래 코드 실행
    
    # .npy 파일 찾기
    npy_files = get_npy_files_in_data_dir()
    
    if not npy_files:
        st.warning("디렉토리에 .npy 파일이 없습니다.")
    else:
        # 파일 목록 표시
        st.write(f"총 {len(npy_files)}개의 .npy 파일을 찾았습니다:")
        
        # 파일 목록을 표시
        for file_path in npy_files:
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB 단위로 변환
            st.write(f"- **{file_name}** ({file_size:.2f} MB)")
        
        # 구분선 추가
        st.divider()
        
        # 전체 파일 ZIP으로 다운로드 옵션
        create_download_link_for_all_files(npy_files)