import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import zipfile
import matplotlib.pyplot as plt
from scipy.signal import correlate
import plotly.graph_objects as go

from matplotlib import font_manager, rc
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)  
plt.rcParams['axes.unicode_minus'] = False


st.set_page_config(layout="wide")  # 넓은 레이아웃

st.title("🔍 신호 추출을 위한 템플릿 설계")

# 📌 사이드바에 템플릿 업로드 영역 추가
with st.sidebar:
    st.header("📂 기존 템플릿 업로드")
    uploaded_template = st.file_uploader("npy 템플릿 업로드", type=["npy"])

    if uploaded_template:
        try:
            template_array = np.load(uploaded_template)
            st.success(f"템플릿 shape: {template_array.shape}")

            # 템플릿 시각화
            fig_template = go.Figure()
            fig_template.add_trace(go.Scatter(
                y=template_array,
                mode='lines',
                name='Uploaded Template'
            ))
            fig_template.update_layout(
                title="📈 업로드된 템플릿 시각화",
                height=250,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_template, use_container_width=True)
        except Exception as e:
            st.error(f"템플릿 파일 로드 실패: {e}")


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
    st.markdown("---")
    st.markdown("🧠 **회사명:** ㈜파시디엘")
    st.markdown("🏫 **연구실:** visLAB@PNU")
    st.markdown("👨‍💻 **제작자:** (C)Dong2")
    st.markdown("🛠️ **버전:** V.1.2 (05-20-2025)")
    st.markdown("---")



# 1. Feather 파일 업로드
uploaded_file = st.file_uploader("ftr(feather) 파일 업로드", type=["ftr", "feather"])

if uploaded_file:
    # 2. 파일 읽기
    df = pd.read_feather(uploaded_file)

    # 3. 첫 2개 컬럼 제거
    # df = df.drop(columns=df.columns[:2])

    st.success(f"파일 로드 완료! 현재 shape: {df.shape}")

    # 4. 컬럼 선택
    selected_col = st.selectbox("그래프를 그릴 컬럼을 선택하세요", df.columns.tolist())

    if selected_col:
        # 5. 다운샘플링 비율 입력
        st.subheader("그래프 표시용 다운샘플 비율 설정")
        downsample_rate = st.slider("다운샘플 비율 (1/N)", 1, 50, 10)

        # 6. 다운샘플링된 데이터 생성 (표시용)
        display_df = df[selected_col].iloc[::downsample_rate].reset_index(drop=True)

        # 7. Plotly WebGL 그래프 생성
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=np.arange(len(display_df)),  # 다운샘플된 인덱스
            y=display_df,
            mode='lines',
            name=f"{selected_col} (1/{downsample_rate} 다운샘플)"
        ))

        fig.update_layout(
            title="Plotly WebGL 그래프 (다운샘플 적용, Zoom/Pan 가능)",
            dragmode="zoom",
            xaxis=dict(rangeslider=dict(visible=False)),
            height=500
        )

        # 6. 그래프 출력
        st.plotly_chart(fig, use_container_width=True)

        # 7. 수직선 좌표 입력
        st.subheader("수직선 추가할 x좌표를 입력하세요")
        x1 = st.number_input("첫 번째 수직선 x좌표", min_value=0, max_value=len(df)-1, value=100)
        x2 = st.number_input("두 번째 수직선 x좌표", min_value=0, max_value=len(df)-1, value=200)

        # 8. 템플릿 파일명 입력
        template_filename = st.text_input("저장할 템플릿 파일명 (확장자 제외)", value="template")

        # 9. 버튼
        if st.button("수직선 추가 및 템플릿 추출/저장"):
            # 수직선 추가
            fig.add_shape(
                type="line",
                x0=x1, y0=0, x1=x1, y1=1,
                xref='x', yref='paper',
                line=dict(color="red", width=2, dash="dash")
            )

            fig.add_shape(
                type="line",
                x0=x2, y0=0, x1=x2, y1=1,
                xref='x', yref='paper',
                line=dict(color="blue", width=2, dash="dash")
            )

            # 수직선 추가된 그래프 다시 출력
            st.plotly_chart(fig, use_container_width=True)

            # 10. 수직선 사이의 데이터 추출
            start_idx = min(x1, x2)
            end_idx = max(x1, x2)

            start_idx_original = int(start_idx * downsample_rate)
            end_idx_original = int(end_idx * downsample_rate)
            template_data = df[selected_col].iloc[start_idx_original:end_idx_original+1].to_numpy()
            # print(start_idx_original, end_idx_original, len(df), selected_col)

            # 11. 템플릿 npy로 저장
            temp_path = os.path.join("/app/data",f"{template_filename}.npy")
            np.save(temp_path, template_data)

            st.success(f"✅ 템플릿이 {template_filename}.npy 로 저장되었습니다!")
else:
    st.info("먼저 ftr 파일을 업로드하세요.")



# ---------- 함수 정의 ----------
def normalized_cross_correlation(data, template):
    data = np.array(data, dtype=np.float64)
    template = np.array(template, dtype=np.float64)

    data_mean = np.nanmean(data)
    template_mean = np.nanmean(template)
    data_std = np.nanstd(data)
    template_std = np.nanstd(template)

    data_normalized = data - data_mean
    template_normalized = template - template_mean

    data_normalized = np.nan_to_num(data_normalized)
    template_normalized = np.nan_to_num(template_normalized)

    correlation = correlate(data_normalized, template_normalized, mode='valid')
    denominator = data_std * template_std * len(template)
    if denominator == 0:
        return np.zeros_like(correlation)
    ncc = correlation / denominator
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



# 위에서 선택된 template를 이용하여 선택된 signal을 스캔하면서 매칭을 수행하고 결과를 출력

with st.form(key="matching_form"):
    st_thres = st.slider("NCC Threshold", 0.0, 1.0, 0.2, 0.01)
    st_low = st.number_input("매칭부 신호 최소값", value=0.0)
    st_high = st.number_input("매칭부 신호 최대값", value=1.0)
    offset_1 = st.number_input("표시 위치 offset", value=1000)

    # remove_st_idx = st.text_input("기동 시작부 제거할 그룹 인덱스 (쉼표로 구분)", value="0,5,17")
    # remove_et_idx = st.text_input("기동 종료부 제거할 그룹 인덱스 (쉼표로 구분)", value="")

    submitted = st.form_submit_button("▶️ 매치 수행")

if submitted:
    signal = df[selected_col].to_numpy()

    # ---------- Template 선택 ----------
    if 'template_array' in locals():
        template = template_array
    elif 'template_data' in locals():
        template = template_data
    else:
        st.error("❌ 사용할 템플릿이 없습니다. 템플릿을 업로드하거나 새로 생성하세요.")
        st.stop()    

    # ---------- 시작부 매칭 ----------
    ncc_start = normalized_cross_correlation(signal, template)
    # signal값이 st_low~st_high 사이이면서 ncc값이 st_thres값보다 큰 위치를 모두 True로 감지
    # true_idx_st = np.where((ncc_start > st_thres) & (signal[:len(ncc_start)] > st_low) & (signal[:len(ncc_start)] < st_high))[0]
    ncc_above_threshold = np.where(ncc_start > st_thres)[0]
    true_idx_st = ncc_above_threshold[
        (signal[ncc_above_threshold] > st_low) & 
        (signal[ncc_above_threshold] < st_high)
    ]    
    st_groups = group_consecutive(true_idx_st)

    means_start = [np.mean(signal[grp]) for grp in st_groups]
    st.subheader(f"🟢 매칭 그룹 수 = {len(st_groups)}")
    with st.expander("매칭 그룹 평균값 (전체 표시)", expanded=True):
        st.markdown(
            f"<div style='max-height: 300px; overflow-y: auto; border:1px solid #ccc; padding:10px;'>"
            + "<br>".join([f"그룹 {i}: 평균 = {v:.4f}" for i, v in enumerate(means_start)])
            + "</div>",
            unsafe_allow_html=True
        )

    # ------------------- 시각화 ----------------------
    # 샘플링 비율 설정 (사용자가 조정 가능하도록)
    sampling_rate = st.slider("샘플링 비율 선택", min_value=1, max_value=50, value=10, step=1, 
                            help="데이터 포인트를 줄여 시각화 속도를 개선합니다. 값이 클수록 더 적은 데이터를 표시합니다.")

    # 신호 데이터 샘플링
    sampled_indices = list(range(0, len(signal), sampling_rate))
    sampled_signal = signal[sampled_indices]

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

    # 매칭 위치 표시 (정확한 위치 유지)
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
        title=f'Template Matching (정합 위치) - 샘플링 비율: 1/{sampling_rate}',
        xaxis_title='Sample Index',
        yaxis_title='Signal Value',
        height=600,
        hovermode='closest',
        showlegend=False
    )

    # 플롯 표시
    st.plotly_chart(fig1, use_container_width=True)    
    # ------------------------------------------------


# =================================================================================
# 다채널 신호 동시 관찰
if uploaded_file:
    st.title("🚀 신호 관찰 및 상호 관계 보기")

    df = pd.read_feather(uploaded_file)
    st.success(f"✅ Feather 로딩 완료! Shape: {df.shape}")

    selected_cols = st.multiselect("Plot할 컬럼을 선택하세요", df.columns.tolist())

    if selected_cols:
        # ✅ 다운샘플링 비율 설정
        st.subheader("📉 다운샘플 비율 설정 (1/N)")
        downsample_rate = st.slider("다운샘플 비율", min_value=1, max_value=100, value=10)

        # crosshair = st.button("▶️ 십자선 Hover 활성화")
        crosshair = st.checkbox("▶️ 십자선 Hover 활성화", value=True)

        fig = go.Figure()

        for col in selected_cols:
            y = df[col].iloc[::downsample_rate]
            x = df.index[::downsample_rate]
            fig.add_trace(go.Scattergl(
                x=x,
                y=y,
                mode='lines',
                name=col,
                showlegend=True,
                hoverinfo='x',
                hovertemplate=''
            ))

        fig.update_layout(
            title="📊 Plotly 그래프 (다운샘플링 적용)",
            dragmode="zoom",
            xaxis=dict(
                rangeslider=dict(visible=False)
            ),
            height=600
        )

        if crosshair:
            fig.update_layout(
                hovermode="x",
                xaxis=dict(
                    showspikes=True,
                    spikemode='across',
                    spikesnap='cursor',
                    spikecolor="red",
                    spikethickness=1
                ),
                yaxis=dict(
                    showspikes=True,
                    spikemode='across',
                    spikesnap='cursor',
                    spikecolor="blue",
                    spikethickness=1
                )
            )

        st.plotly_chart(fig, use_container_width=True)


# ====================================================================== 
# 서버의 /app/data에 저장된 npy파일들을 client로 압축해서 다운로드
def get_npy_files_in_data_dir():
    """
    컨테이너 내의 /app/data 디렉토리에 있는 모든 .npy 파일 찾기
    (이 디렉토리는 호스트의 /home/pashidl/streamlit/dashboard에 매핑됨)
    """
    data_dir = "/app/data"
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