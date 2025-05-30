import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from datetime import datetime
import time
import stat
import shutil
import gc 
import matplotlib.pyplot as plt 
import glob
# import pdb
# import io  # ✅ io 모듈 명시적 import 확인
# import zipfile


# 한글 폰트 설정
try:
    from matplotlib import font_manager, rc
    # Windows 환경
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        # Linux 환경
        from matplotlib import font_manager, rc
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)  
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # 폰트 로드 실패 시 기본 폰트 사용
        plt.rcParams['axes.unicode_minus'] = False

# matplotlib 경고 제거를 위한 설정
plt.rcParams['figure.max_open_warning'] = 50



# ✅ WebSocket 오류 방지를 위한 설정
st.set_page_config(
    page_title="Excel → Feather 변환기", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ FutureWarning 해결을 위한 설정 (더 안전한 방법으로 수정)
try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    # 구버전 pandas에서는 이 옵션이 없을 수 있음
    pass

# UI 입력
st.title("📊 대용량 Excel → Feather 변환기")
uploaded_file = st.file_uploader("📂 Excel 파일 업로드 (.xlsx)", type=["xlsx", "xls"])

st.sidebar.header("🔧 Excel 읽기 설정")

# ✅ 1. 시트 선택 기능 추가 (1개 시트 오류 해결)
sheet_name = None
available_sheets = []

if uploaded_file is not None:
    try:
        # Excel 파일의 시트명 목록 가져오기
        excel_file = pd.ExcelFile(uploaded_file)
        available_sheets = excel_file.sheet_names
        
        if len(available_sheets) == 1:
            # ✅ 2. 시트가 1개인 경우 자동으로 선택 (오류 방지)
            sheet_name = available_sheets[0]
            st.sidebar.success(f"📋 시트 자동 선택: {sheet_name}")
        elif len(available_sheets) > 1:
            # 시트가 여러개인 경우 선택 옵션 제공
            st.sidebar.subheader("📋 시트 선택")
            sheet_name = st.sidebar.selectbox(
                "읽을 시트를 선택하세요:",
                options=available_sheets,
                index=0
            )
            st.sidebar.info(f"총 {len(available_sheets)}개 시트 중 '{sheet_name}' 선택됨")
        else:
            st.sidebar.error("❌ 시트를 찾을 수 없습니다.")
            
    except Exception as e:
        st.sidebar.error(f"❌ 시트 정보를 읽을 수 없습니다: {e}")
        st.sidebar.info("💡 파일이 손상되었거나 지원되지 않는 형식일 수 있습니다.")

usecols = st.sidebar.text_input("읽을 컬럼 범위 (usecols)", value="A:CY")
date_column = st.sidebar.text_input(
    "날짜 컬럼명", 
    value="Description",
    help="Excel 파일에서 날짜로 변환할 컬럼명을 입력하세요 (이 컬럼은 문자-숫자 변환에서 자동 제외됩니다)"
)
skiprows = st.sidebar.number_input(
    "건너뛸 행 수 (skiprows)", 
    min_value=0, 
    value=3,
    help="Excel 파일 상단에서 헤더와 데이터를 읽기 전에 건너뛸 행 수 (메타 정보 등)"
)
skip_next = st.sidebar.number_input(
    "헤드행 다음 건너뛸 수 (skip_next)", 
    min_value=0, 
    value=2,
    help="Excel 파일 상단에서 헤더와 데이터를 읽기 전에 건너뛸 행 수 (메타 정보 등)"
)
nrows = st.sidebar.number_input(
    "읽을 행 수 (nrows)", 
    min_value=1000, 
    max_value=10**7, 
    step=10000, 
    value=3000, #518400,
    help="큰 값을 입력하면 시트의 마지막 행까지 자동으로 읽습니다"
)

# ✅ 날짜 컬럼을 더 효과적으로 식별하는 함수 추가
def identify_date_columns(df, date_column_name):
    """날짜 컬럼들을 식별하는 함수 - 명시적 지정과 자동 감지"""
    date_columns = set()
    
    # 1. 사용자가 명시적으로 지정한 날짜 컬럼
    if date_column_name and date_column_name in df.columns:
        date_columns.add(date_column_name)
    
    # 2. 자동 감지: 컬럼명에 날짜 관련 키워드가 있는 경우
    date_keywords = ['date', 'time', 'datetime', 'timestamp', '날짜', '시간', '일시']
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in date_keywords):
            date_columns.add(col)
    
    # 3. 자동 감지: 데이터 타입이 datetime인 경우
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.add(col)
    
    return list(date_columns)

# ✅ 2. 문자-숫자 변환 설정을 위한 세션 상태 초기화
if 'text_mapping' not in st.session_state:
    st.session_state.text_mapping = {}
if 'text_frequency' not in st.session_state:
    st.session_state.text_frequency = {}
if 'text_columns' not in st.session_state:
    st.session_state.text_columns = {}
if 'date_columns' not in st.session_state:
    st.session_state.date_columns = []

# ✅ 3. 문자값 추출 및 매핑 함수 (날짜 컬럼 완전 제외)
def extract_unique_text_values(df, date_columns):
    """DataFrame에서 모든 문자값들과 빈도를 추출하는 함수 - 날짜 컬럼 제외"""
    text_frequency = {}
    text_columns_dict = {}  # 각 문자값이 어느 컬럼에서 나타났는지 추적
    
    # 날짜 컬럼들을 제외한 컬럼들만 처리
    columns_to_process = [col for col in df.columns if col not in date_columns]
    
    st.info(f"🗓️ 날짜 컬럼으로 식별되어 문자 변환에서 제외된 컬럼: {date_columns}")
    st.info(f"🔤 문자-숫자 변환 대상 컬럼 수: {len(columns_to_process)}/{len(df.columns)}")
    
    for column in columns_to_process:
        # 각 컬럼에서 문자열 값들 찾기
        text_values = df[column].dropna().astype(str)
        
        # 숫자가 아닌 값들만 추출
        for value in text_values:
            # 숫자로 변환 가능한지 확인
            try:
                float(value)
            except (ValueError, TypeError):
                # 숫자가 아닌 경우에만 추가
                if value not in ['nan', 'None', '']:
                    # 빈도 카운트
                    if value not in text_frequency:
                        text_frequency[value] = 0
                        text_columns_dict[value] = set()
                    
                    # 해당 값의 빈도와 컬럼 정보 업데이트
                    value_count = (text_values == value).sum()
                    text_frequency[value] += value_count
                    text_columns_dict[value].add(column)
    
    return text_frequency, text_columns_dict

# ✅ 날짜 정보 추출 함수 추가 (한글 제거)
def extract_date_info_from_data(df, date_column):
    """데이터에서 날짜 정보를 추출하여 파일명용 문자열 생성 (한글 제거)"""
    try:
        if date_column in df.columns:
            date_series = pd.to_datetime(df[date_column], errors='coerce')
            date_series = date_series.dropna()
            
            if len(date_series) > 0:
                start_date = date_series.min()
                end_date = date_series.max()
                
                # ✅ 1. 한글 제거: 24시간 형식 사용, 영문 로케일 설정
                # 날짜 형식: YYYYMMDD_HHMMSS (시간 정보도 포함)
                try:
                    # 시간 정보 포함한 상세 형식
                    start_str = start_date.strftime("%Y%m%d_%H%M")
                    end_str = end_date.strftime("%Y%m%d_%H%M")
                    
                    if start_str == end_str:
                        # 같은 날짜/시간인 경우
                        return f"_{start_str}"
                    else:
                        # 날짜/시간 범위인 경우 (너무 길어지지 않도록 시작-끝 날짜만)
                        start_date_only = start_date.strftime("%Y%m%d")
                        end_date_only = end_date.strftime("%Y%m%d")
                        
                        if start_date_only == end_date_only:
                            # 같은 날이지만 시간이 다른 경우
                            return f"_{start_date_only}"
                        else:
                            # 다른 날짜 범위
                            return f"_{start_date_only}_{end_date_only}"
                            
                except Exception:
                    # strftime 오류 시 기본 형식 사용
                    start_str = start_date.strftime("%Y%m%d")
                    end_str = end_date.strftime("%Y%m%d")
                    
                    if start_str == end_str:
                        return f"_{start_str}"
                    else:
                        return f"_{start_str}_{end_str}"
        
        return ""  # 날짜 정보 없음
    except Exception:
        return ""  # 오류 발생 시 빈 문자열 반환

# ✅ 4. 문자-숫자 변환 적용 함수 (날짜 컬럼 완전 보호)
def apply_text_mapping(df, mapping_dict, date_columns):
    """DataFrame에 문자-숫자 매핑을 적용하는 함수 - 날짜 컬럼 보호"""
    df_converted = df.copy()
    
    # 날짜 컬럼들을 제외한 컬럼들만 처리
    columns_to_process = [col for col in df_converted.columns if col not in date_columns]
    
    for column in columns_to_process:
        # 각 셀을 확인하여 매핑 적용
        for text, number in mapping_dict.items():
            # ✅ FutureWarning 해결: 더 안전한 방법으로 수정
            try:
                # 최신 pandas 방식
                df_converted[column] = df_converted[column].replace(text, number).infer_objects(copy=False)
            except (AttributeError, TypeError):
                # 구버전 pandas 또는 호환성 문제 시 기본 방식 사용
                df_converted[column] = df_converted[column].replace(text, number)
        
        # 최종적으로 숫자로 변환 시도
        try:
            df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
        except:
            pass
    
    # 날짜 컬럼들은 그대로 유지하되, datetime 타입으로 확실히 변환
    for date_col in date_columns:
        if date_col in df_converted.columns:
            try:
                df_converted[date_col] = pd.to_datetime(df_converted[date_col], errors='coerce')
            except:
                pass  # 변환 실패 시 원본 유지
    
    return df_converted

# 캐시된 Excel 로더: # 청크 단위로 처리하는 경우의 함수
@st.cache_data(show_spinner=False, max_entries=3, ttl=300)  # ✅ show_spinner=False로 변경
def load_excel(file, sheet_name, usecols, nrows, date_column, skiprows, skip_next=0):
    """Excel 파일을 읽는 함수 - 단일 시트 처리 개선"""
    
    # ✅ 파일 유효성 검사 추가
    if file is None:
        raise ValueError("파일이 None입니다.")
    
    if sheet_name is None or sheet_name == "":
        raise ValueError("시트명이 유효하지 않습니다.")


    def count_rows():    
        if ':' in usecols:
            first_col = usecols.split(':')[0]  # 'A'
        else:
            first_col = usecols  # 이미 단일 컬럼인 경우

        range_col = f"{first_col}:{first_col}"
        temp_df = pd.read_excel(
            file,
            sheet_name=sheet_name,
            skiprows=skiprows + 1 + skip_next,
            header=None,
            usecols=range_col,  
            engine='openpyxl',
        )
        mask = temp_df.iloc[:, 0].notna() & (temp_df.iloc[:, 0].astype(str).str.strip() != '')
        temp_count = mask.sum()
        print(f"데이터 {sheet_name} 행 개수: {temp_count}")   
        final_nrows = temp_count - 10 if temp_count > 10 else temp_count
        return final_nrows

    
    try:
        if skip_next > 0:
            # 헤더만 먼저 읽기
            header_df = pd.read_excel(
                file,
                sheet_name=sheet_name,
                skiprows=skiprows,
                nrows=1,
                usecols=usecols,
                engine='openpyxl'
            )   
            
            # 헤더 컬럼명 추출
            column_names = header_df.columns.tolist()
            
            # ✅ 실제 데이터 행 수 확인 (간소화)
            try:
                final_nrows = count_rows()
                # final_nrows = nrows
            except Exception:
                final_nrows = nrows
            
            # 실제 데이터 읽기
            data_df = pd.read_excel(
                file,
                sheet_name=sheet_name,
                skiprows=skiprows + 1 + skip_next,
                header=None,
                usecols=usecols,
                nrows=final_nrows,
                engine='openpyxl',
                na_values=['I/O Timeout', 'Configure', 'Not Connect', 'Bad', 'Comm Fail']
            )
            
            # 추출한 헤더명 적용
            data_df.columns = column_names
            
        else:
            # ✅ skip_next가 0인 경우 단순화된 처리
            try:
                final_nrows = count_rows()
            except Exception:
                final_nrows = nrows

            data_df = pd.read_excel(
                file,
                sheet_name=sheet_name,
                usecols=usecols,
                nrows=final_nrows, # nrows,
                skiprows=skiprows,
                header=0,
                engine='openpyxl',
                na_values=['I/O Timeout', 'Configure', 'Not Connect', 'Bad', 'Comm Fail']
            )
        
        # ✅ 날짜 컬럼 변환
        if date_column and date_column in data_df.columns:
            try:
                data_df[date_column] = pd.to_datetime(data_df[date_column], errors='coerce')
            except Exception:
                pass  # 날짜 변환 실패 시 무시
        
        # ✅ 메모리 정리
        gc.collect()
        
        return data_df
        
    except Exception as e:
        # ✅ 메모리 정리
        gc.collect()
        raise RuntimeError(f"Excel 읽기 실패: {str(e)}")

# 세션 상태 초기화
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None

# Excel 로딩 부분 - 디버깅과 타임아웃 추가
if uploaded_file is not None and sheet_name is not None and st.button("Excel 읽기"):
    # ✅ 연결 상태 확인 및 진행 상황 관리
    progress_container = st.container()
    status_container = st.container()
    
    try:
        with progress_container:
            # ✅ 단계별 진행 상황 표시 (WebSocket 부하 최소화)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🚀 Excel 파일 읽기 시작...")
            progress_bar.progress(10)
            time.sleep(0.1)  # UI 업데이트 시간 확보
            
            status_text.text(f"📖 시트 '{sheet_name}' 처리 중...")
            progress_bar.progress(30)
            
            # 실제 파일 읽기
            df = load_excel(
                file=uploaded_file,
                sheet_name=sheet_name,
                usecols=usecols,
                nrows=nrows,
                date_column=date_column,
                skiprows=skiprows,   
                skip_next=skip_next   
            )
            
            progress_bar.progress(50)
            status_text.text("🗓️ 날짜 컬럼 식별 중...")
            
            # ✅ 날짜 컬럼들 식별
            date_columns = identify_date_columns(df, date_column)
            st.session_state.date_columns = date_columns
            
            progress_bar.progress(60)
            status_text.text("🔍 데이터 분석 중...")
            
            # 처리 완료 확인
            if df is not None and len(df) > 0:
                # ✅ 원본 데이터 저장
                st.session_state.raw_df = df.copy()
                
                progress_bar.progress(80)
                status_text.text("🔤 문자값 추출 중...")
                
                # ✅ 문자값들과 빈도 추출 (날짜 컬럼 제외)
                text_frequency, text_columns_dict = extract_unique_text_values(df, date_columns)
                st.session_state.text_frequency = text_frequency
                st.session_state.text_columns = text_columns_dict
                
                # 세션 상태에 저장
                st.session_state.df = df
                st.session_state.file_loaded = True
                
                progress_bar.progress(90)
                status_text.text("📝 파일명 생성 중...")
                
                # 파일 업로드 시 파일명 저장
                base_filename = os.path.splitext(uploaded_file.name)[0]
                
                # ✅ 1. 날짜 정보 추출 및 파일명 생성
                date_info = extract_date_info_from_data(df, date_column)
                sheet_info = f"_{sheet_name}" if sheet_name != "Sheet1" else ""
                
                # 최종 파일명: 원본파일명_시트명_날짜정보
                enhanced_filename = f"{base_filename}{sheet_info}{date_info}"
                st.session_state.last_filename = enhanced_filename
                
                progress_bar.progress(100)
                status_text.text("✅ 완료!")
                
                # ✅ 완료 후 진행률 제거하고 결과 표시 (WebSocket 부하 감소)
                time.sleep(0.5)
                progress_container.empty()
                
        with status_container:
            st.success(f"🎉 Excel 파일 읽기 완료!")
            st.info(f"📊 데이터 크기: {len(df):,}행 × {len(df.columns)}열")
            
            # ✅ 날짜 컬럼 정보 표시
            if date_columns:
                st.success(f"🗓️ 식별된 날짜 컬럼: {date_columns}")
                st.info("📌 이 컬럼들은 문자-숫자 변환에서 자동으로 제외됩니다.")
            
            # ✅ 발견된 문자값들과 빈도 표시
            if text_frequency:
                st.warning(f"🔤 발견된 문자값들: {list(text_frequency.keys())}")
                st.info("👇 아래에서 각 문자값에 대응할 숫자를 설정하세요.")
            else:
                st.success("✅ 모든 데이터가 이미 숫자 형태이거나 날짜 형태입니다.")
                
                # ✅ 메모리 정리
                gc.collect()
                
    except Exception as e:
        # ✅ 에러 발생 시 진행률 제거
        progress_container.empty()
        
        st.error(f"❌ 파일 로딩 중 오류 발생!")
        st.error(f"🔍 오류 상세: {str(e)}")
        
        # 캐시 클리어 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 캐시 클리어"):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if st.button("🔄 페이지 새로고침"):
                st.rerun()

# ✅ 문자-숫자 매핑 설정 UI (날짜 컬럼 보호 강화)
if st.session_state.file_loaded and st.session_state.text_frequency:
    st.markdown("---")
    st.subheader("🔢 문자-숫자 변환 설정")
    
    # ✅ 날짜 컬럼 보호 상태 표시
    if st.session_state.date_columns:
        st.info(f"🛡️ 보호되는 날짜 컬럼: {st.session_state.date_columns}")
        st.caption("이 컬럼들은 문자-숫자 변환에서 자동으로 제외되어 원본 형태를 유지합니다.")
    
    # ✅ 1. 빈도 정보와 컬럼 정보 표시
    st.write("**발견된 문자값들과 빈도 정보:**")
    
    # 빈도 정보를 테이블로 표시
    frequency_data = []
    for text, freq in st.session_state.text_frequency.items():
        columns_list = list(st.session_state.text_columns[text])
        frequency_data.append({
            "문자값": text,
            "빈도": f"{freq:,}",
            "출현 컬럼": ", ".join(columns_list) if len(columns_list) <= 3 else f"{', '.join(columns_list[:3])}... (총 {len(columns_list)}개)"
        })
    
    # 빈도 기준 정렬 (낮은 빈도부터)
    frequency_df = pd.DataFrame(frequency_data).sort_values('빈도')
    st.dataframe(frequency_df, use_container_width=True)
    
    # 빈도가 낮은 값들 (10개 미만) 별도 표시
    low_freq_items = [(text, freq, st.session_state.text_columns[text]) 
                      for text, freq in st.session_state.text_frequency.items() if freq < 10]
    
    if low_freq_items:
        st.warning("⚠️ 빈도가 낮은 문자값들 (10회 미만):")
        for text, freq, columns in low_freq_items:
            st.write(f"• **{text}** ({freq}회) → 컬럼: {list(columns)}")
    
    # 숫자 선택 옵션
    number_options = [0, 1, 5, 10, 25, 50, 75, 100]
    
    st.markdown("---")
    st.write("**숫자 매핑 설정:**")
    
    # 2열로 배치하여 매핑 설정
    mapping_dict = {}
    
    # 문자값들을 빈도 기준으로 정렬하여 표시
    sorted_texts = sorted(st.session_state.text_frequency.keys(), 
                         key=lambda x: st.session_state.text_frequency[x], reverse=True)
    
    cols = st.columns(2)
    for i, text in enumerate(sorted_texts):
        with cols[i % 2]:
            # 기본값 설정 (일반적인 매핑)
            if text.upper() in ['OFF', 'STOP', 'FALSE', '0']:
                default_idx = 0  # 0
            elif text.upper() in ['ON', 'RUNNING', 'TRUE', '1']:
                default_idx = 1  # 1
            else:
                default_idx = 1  # 기본값은 1
            
            freq = st.session_state.text_frequency[text]
            selected_number = st.selectbox(
                f"{text} ({freq}회) → ",
                options=number_options,
                index=default_idx,
                key=f"mapping_{text}"
            )
            mapping_dict[text] = selected_number
    
    # 매핑 적용 버튼
    if st.button("🔄 문자-숫자 변환 적용"):
        try:
            # 원본 데이터에 매핑 적용 (날짜 컬럼 보호)
            converted_df = apply_text_mapping(st.session_state.raw_df, mapping_dict, st.session_state.date_columns)
            st.session_state.df = converted_df
            st.session_state.text_mapping = mapping_dict
            
            st.success("✅ 문자-숫자 변환이 완료되었습니다!")
            
            # 변환 결과 요약
            st.write("**적용된 매핑:**")
            for text, number in mapping_dict.items():
                st.write(f"• {text} → {number}")
            
            # 변환 후 데이터 타입 확인
            numeric_columns = st.session_state.df.select_dtypes(include=[np.number]).columns
            datetime_columns = st.session_state.df.select_dtypes(include=['datetime64']).columns
            
            st.info(f"✅ 숫자 컬럼 수: {len(numeric_columns)}/{len(st.session_state.df.columns)}")
            st.info(f"🗓️ 날짜 컬럼 수: {len(datetime_columns)}/{len(st.session_state.df.columns)}")
            
        except Exception as e:
            st.error(f"❌ 변환 중 오류: {e}")

# 메인 데이터 처리 섹션
if st.session_state.file_loaded and st.session_state.df is not None:
    df = st.session_state.df
    st.markdown("---")
    st.success("✅ Excel 로딩 완료!")

    # 컬럼 리스트 스크롤 박스
    st.subheader("🧾 컬럼 리스트 (스크롤 박스)")
    with st.expander(f"전체 컬럼 보기 (총 {len(df.columns)}개)", expanded=True):
        # 컬럼 타입별로 분류하여 표시
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        col_info = []
        for i, col in enumerate(df.columns):
            if col in numeric_cols:
                col_type = "🔢 숫자"
            elif col in datetime_cols:
                col_type = "🗓️ 날짜"
            elif col in object_cols:
                col_type = "🔤 텍스트"
            else:
                col_type = "❓ 기타"
            
            col_info.append(f"{i+1}. {col} ({col_type})")
        
        # HTML을 활용한 스크롤 박스
        st.markdown(
            f"""
            <div style='max-height: 300px; overflow-y: scroll; padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9'>
            {"<br>".join(col_info)}
            </div>
            """,
            unsafe_allow_html=True
        )        

    # 타입 체크 및 보호된 날짜 컬럼 표시
    object_cols = df.columns[df.dtypes.eq(object)]
    if len(object_cols) > 0:
        st.markdown("🧪 `object` 타입인 컬럼 (추가 처리 필요할 수 있음):")
        st.write(object_cols.tolist())
    
    # 보호된 날짜 컬럼 표시
    if st.session_state.date_columns:
        protected_cols = [col for col in st.session_state.date_columns if col in df.columns]
        if protected_cols:
            st.success(f"🛡️ 보호된 날짜 컬럼: {protected_cols}")

    # ✅ 추가: 제거할 컬럼 선택 기능 (날짜 컬럼 보호)
    st.subheader("🗑️ 제거할 컬럼 선택")
    
    # 날짜 컬럼은 기본적으로 제거 대상에서 제외
    removable_columns = [col for col in df.columns.tolist() if col not in st.session_state.date_columns]
    
    if st.session_state.date_columns:
        st.info(f"🛡️ 날짜 컬럼 {st.session_state.date_columns}은(는) 보호되어 제거 옵션에서 제외됩니다.")
    
    cols_to_drop = st.multiselect(
        "데이터프레임에서 제거할 컬럼을 선택하세요",
        removable_columns,
        help="날짜 컬럼은 자동으로 보호되어 선택 목록에서 제외됩니다."
    )

    # ✅ 추가: 컬럼 제거 버튼
    if st.button("선택한 컬럼 제거하기"):
        if cols_to_drop:
            # 날짜 컬럼이 실수로 포함되지 않았는지 재확인
            safe_cols_to_drop = [col for col in cols_to_drop if col not in st.session_state.date_columns]
            
            if safe_cols_to_drop:
                df = df.drop(columns=safe_cols_to_drop)
                st.session_state.df = df  # 세션 상태 업데이트
                st.success(f"✅ 선택한 {len(safe_cols_to_drop)}개 컬럼 제거 완료!")
                st.info(f"남은 컬럼 수: {len(df.columns)}개")
                
                if len(safe_cols_to_drop) != len(cols_to_drop):
                    protected_count = len(cols_to_drop) - len(safe_cols_to_drop)
                    st.warning(f"🛡️ {protected_count}개 날짜 컬럼은 보호되어 제거되지 않았습니다.")
            else:
                st.warning("🛡️ 선택한 모든 컬럼이 보호된 날짜 컬럼입니다.")
        else:
            st.warning("❗ 제거할 컬럼을 선택하세요.")

    # 변환 전/후 비교 (문자값이 있었던 경우)
    if st.session_state.text_frequency and 'raw_df' in st.session_state:
        st.markdown("---")
        st.subheader("👀 데이터 변환 비교")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**변환 전 (원본):**")
            st.dataframe(st.session_state.raw_df.head(), use_container_width=True)
        
        with col2:
            st.write("**변환 후 (숫자/날짜):**")
            st.dataframe(df.head(), use_container_width=True)
            
        # 날짜 컬럼 보호 상태 확인
        if st.session_state.date_columns:
            st.info("🛡️ 날짜 컬럼들은 원본 형태를 유지하며 변환되지 않았습니다.")

    # 저장 설정
    st.markdown("---")
    st.subheader("💾 Feather 저장 설정")
    default_root = "/app/data" if os.path.exists("/app/data") else os.getcwd()

    # ✅ 1. 세션 상태에서 날짜 정보가 포함된 파일명 가져오기
    default_filename = st.session_state.get('last_filename', 'data')
    save_name = st.text_input(
        "📄 저장 파일명 (확장자 제외)", 
        value=default_filename,
        help="날짜 정보와 시트명이 자동으로 포함됩니다"
    )    

    # FTR 저장 버튼
    if st.button("💾 Feather로 저장하기"):
        save_path = os.path.join(default_root, save_name + ".ftr")
        try:
            # DataFrame 유효성 검사 및 안전한 저장
            if df is None or df.empty:
                st.error("❌ 저장할 데이터가 비어있습니다.")
            else:
                df_to_save = df.reset_index(drop=True)
                df_to_save.to_feather(save_path)
                st.success(f"✅ Feather 파일로 저장 완료:\n`{save_path}`")
                st.info(f"📊 저장된 데이터: {len(df_to_save)}행 × {len(df_to_save.columns)}열")
                
                # 날짜 컬럼 보존 확인
                if st.session_state.date_columns:
                    preserved_date_cols = [col for col in st.session_state.date_columns if col in df_to_save.columns]
                    if preserved_date_cols:
                        st.success(f"🗓️ 보존된 날짜 컬럼: {preserved_date_cols}")
                        
        except Exception as e:
            st.error(f"❌ 저장 실패: {e}")
            st.error(f"디버깅 정보: DataFrame shape={df.shape if df is not None else 'None'}")

    # Feather 다운로드 기능
    st.subheader("💾 Feather 파일 다운로드")

    # ✅ 2. 파일명 입력 - 날짜 정보가 포함된 파일명 사용
    default_download_filename = st.session_state.get('last_filename', 'ftr_data')
        
    download_name = st.text_input(
        "📄 다운로드할 파일명 (확장자 제외)", 
        value=default_download_filename,
        help="날짜 정보와 시트명이 자동으로 포함됩니다"
    )

    try:
        # ✅ io 모듈 오류 해결: 모듈 재import 및 안전한 처리
        import io as io_module  # 명시적 import
        
        # 메모리에 임시로 파일 저장
        buffer = io_module.BytesIO()
        
        # DataFrame 유효성 검사
        if df is None or df.empty:
            st.error("❌ 데이터가 비어있습니다.")
        else:
            # Feather 형식으로 저장
            df_to_save = df.reset_index(drop=True)
            df_to_save.to_feather(buffer)
            buffer.seek(0)
            
            # 다운로드 버튼
            st.download_button(
                label="📥 Feather 파일 다운로드",
                data=buffer.getvalue(),  # getvalue() 사용으로 더 안전하게
                file_name=f"{download_name}.ftr",
                mime="application/octet-stream"
            )
            
            # 성공 메시지
            st.success(f"✅ 다운로드 준비 완료: {download_name}.ftr")
            
            # 날짜 컬럼 보존 확인
            if st.session_state.date_columns:
                preserved_date_cols = [col for col in st.session_state.date_columns if col in df_to_save.columns]
                if preserved_date_cols:
                    st.info(f"🗓️ 다운로드 파일에 보존된 날짜 컬럼: {preserved_date_cols}")
        
    except ImportError:
        st.error("❌ io 모듈을 불러올 수 없습니다. Python 환경을 확인하세요.")
    except Exception as e:
        st.error(f"❌ 다운로드 준비 실패: {e}")
        # 상세한 디버깅 정보 추가
        st.error(f"디버깅 정보:")
        st.error(f"- DataFrame shape: {df.shape if df is not None else 'None'}")
        st.error(f"- DataFrame columns: {len(df.columns) if df is not None else 'N/A'}")
        st.error(f"- DataFrame dtypes: {df.dtypes.to_dict() if df is not None else 'N/A'}")
        st.error(f"- 오류 타입: {type(e).__name__}")
        
        # 대안 제시
        st.info("💡 대안: 서버 저장 기능을 사용하거나 CSV 형식으로 다운로드해보세요.")

    # 고속 Plotly 시각화 (GPU 필요)
    st.markdown("---")
    st.subheader("⚡ Plotly 고속 시각화 (WebGL)")
    num_cols = df.select_dtypes(include='number').columns
    
    if len(num_cols) > 0:
        # 날짜 컬럼이 아닌 숫자 컬럼들만 시각화 대상으로 제공
        visualizable_columns = [col for col in df.columns.tolist() 
                               if col not in st.session_state.date_columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if visualizable_columns:
            selected_columns = st.multiselect(
                '시각화할 숫자 컬럼을 선택하세요',
                visualizable_columns,
                default=visualizable_columns[:2] if len(visualizable_columns) >= 2 else visualizable_columns[:1]
            )            
            
            if selected_columns:
                downsample_rate = st.slider("📉 다운샘플 비율 (1/N)", 1, 50, 10)
                
                # 여러 컬럼에 대한 트레이스를 각각 추가
                fig = go.Figure()
                
                for col in selected_columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        downsampled_y = df[col][::downsample_rate]
                        fig.add_trace(go.Scattergl(
                            y=downsampled_y,
                            mode='lines',
                            name=str(col)  # 각 컬럼명을 문자열로 변환
                        ))
                
                fig.update_layout(
                    title=f"선택한 컬럼 시각화 (1/{downsample_rate} 다운샘플링)",
                    xaxis=dict(rangeslider=dict(visible=False)),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 날짜 컬럼이 있는 경우 X축 옵션 제공
                if st.session_state.date_columns:
                    available_date_cols = [col for col in st.session_state.date_columns if col in df.columns]
                    if available_date_cols:
                        st.info("💡 시간축 시각화를 원하면 아래에서 날짜 컬럼을 X축으로 선택할 수 있습니다.")
                        
                        use_date_axis = st.checkbox("🗓️ 날짜를 X축으로 사용")
                        if use_date_axis:
                            date_col_for_x = st.selectbox("X축으로 사용할 날짜 컬럼 선택:", available_date_cols)
                            
                            if date_col_for_x and selected_columns:
                                # 날짜를 X축으로 하는 시각화
                                fig_time = go.Figure()
                                
                                for col in selected_columns:
                                    if pd.api.types.is_numeric_dtype(df[col]):
                                        downsampled_df = df.iloc[::downsample_rate]  # 전체 행을 다운샘플링
                                        fig_time.add_trace(go.Scattergl(
                                            x=downsampled_df[date_col_for_x],
                                            y=downsampled_df[col],
                                            mode='lines',
                                            name=str(col)
                                        ))
                                
                                fig_time.update_layout(
                                    title=f"시간축 시각화 (X: {date_col_for_x})",
                                    xaxis_title=date_col_for_x,
                                    yaxis_title="값",
                                    margin=dict(l=20, r=20, t=40, b=20),
                                    height=400
                                )
                                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("시각화할 숫자 컬럼을 선택하세요.")
        else:
            st.info("시각화 가능한 숫자 컬럼이 없습니다. (날짜 컬럼은 제외됨)")
    else:
        st.info("숫자형 컬럼이 없습니다.")

    # ====================================================================== 
    # ✅ 1. 모든 시트에 일괄 적용 기능 추가 (날짜 컬럼 보호 강화)
    st.markdown("---")
    st.header("🔄 모든 시트에 일괄 적용")
    
    if uploaded_file is not None:
        # 현재 설정 요약 표시
        st.subheader("📋 현재 적용할 설정 요약")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔢 문자-숫자 매핑 설정:**")
            if st.session_state.text_mapping:
                for text, number in st.session_state.text_mapping.items():
                    st.write(f"• {text} → {number}")
            else:
                st.write("설정된 매핑이 없습니다.")
                
            st.write("**🛡️ 보호되는 날짜 컬럼:**")
            if st.session_state.date_columns:
                for col in st.session_state.date_columns:
                    st.write(f"• {col}")
            else:
                st.write("식별된 날짜 컬럼이 없습니다.")
        
        with col2:
            st.write("**🗑️ 제거할 컬럼 설정:**")
            if 'cols_to_drop' in locals() and cols_to_drop:
                for col in cols_to_drop:
                    st.write(f"• {col}")
            else:
                st.write("제거할 컬럼이 없습니다.")
        
        # 사용 가능한 시트 목록 표시
        if len(available_sheets) > 0:  # ✅ 2. 시트 목록이 있는 경우에만 실행
            st.write(f"**📋 처리 대상 시트 ({len(available_sheets)}개)**")
            # for i, sheet in enumerate(available_sheets, 1):
            #     status = "✅ 현재 처리됨" if sheet == sheet_name else "⏳ 대기중"
            #     st.write(f"{i}. {sheet} {status}")
        else:
            st.error("❌ 처리할 수 있는 시트가 없습니다.")
            
        # 일괄 처리 실행 버튼
        if len(available_sheets) > 1:  # 시트가 2개 이상인 경우에만 표시
            st.markdown("---")
            if st.button("🚀 모든 시트에 동일한 처리 적용 및 저장", type="primary"):
                
                # 현재 설정 저장
                current_mapping = st.session_state.text_mapping.copy()
                current_cols_to_drop = cols_to_drop.copy() if 'cols_to_drop' in locals() else []
                current_date_columns = st.session_state.date_columns.copy()
                
                st.info(f"🔄 {len(available_sheets)}개 시트에 일괄 처리를 시작합니다...")
                
                # 진행률 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                success_files = []
                error_files = []

                default_root = "/app/data" if os.path.exists("/app/data") else os.getcwd()
                # 이미 파일이 존재 시 모두 제거
                for file_path in glob.glob(os.path.join(default_root, "*")):
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'삭제 실패 {file_path}: {e}')

                
                for idx, target_sheet in enumerate(available_sheets):   #### Kang [:3]
                    try:
                        status_text.text(f"처리 중: {target_sheet} ({idx+1}/{len(available_sheets)})")
                        
                        # 각 시트 읽기
                        sheet_df = load_excel(
                            file=uploaded_file,
                            sheet_name=target_sheet,
                            usecols=usecols,
                            nrows=nrows,
                            date_column=date_column,
                            skiprows=skiprows,
                            skip_next=skip_next
                        )
                        
                        # 각 시트별로 날짜 컬럼 재식별 (시트마다 구조가 다를 수 있음)
                        sheet_date_columns = identify_date_columns(sheet_df, date_column)
                        
                        # 문자-숫자 매핑 적용 (날짜 컬럼 보호)
                        if current_mapping:
                            sheet_df = apply_text_mapping(sheet_df, current_mapping, sheet_date_columns)
                        
                        # 컬럼 제거 적용 (날짜 컬럼 보호)
                        if current_cols_to_drop:
                            # 존재하는 컬럼 중에서 날짜 컬럼이 아닌 것만 제거
                            cols_to_remove = [col for col in current_cols_to_drop 
                                            if col in sheet_df.columns and col not in sheet_date_columns]
                            if cols_to_remove:
                                sheet_df = sheet_df.drop(columns=cols_to_remove)



                        # ✅ 1. 파일명 생성 (한글 제거된 날짜 정보 사용)
                        base_filename = os.path.splitext(uploaded_file.name)[0]
                        date_info = extract_date_info_from_data(sheet_df, date_column)
                        sheet_info = f"_{target_sheet}" if target_sheet != "Sheet1" else ""

                        # 파일명에서 한글, 특수문자 및 공백 제거 (영문, 숫자, -, _ 만 허용)
                        def remove_korean_and_special_chars(text):
                            # 한글 범위: ㄱ-ㅎ, ㅏ-ㅣ, 가-힣
                            import re
                            # 영문자, 숫자, 하이픈, 언더스코어만 허용
                            return re.sub(r'[^a-zA-Z0-9\-_]', '', text)

                        safe_base = remove_korean_and_special_chars(base_filename)
                        safe_sheet = remove_korean_and_special_chars(sheet_info)
                        safe_date_info = remove_korean_and_special_chars(date_info)

                        timestamp_suffix = str(int(time.time() * 1000))[-6:]  # 마지막 8자리만 사용
                        # final_filename = f"{safe_sheet}{safe_date_info}_{timestamp_suffix}"
                        final_filename = f"{safe_date_info}_{timestamp_suffix}"
                        
                        # Feather 파일 저장
                        # default_root = "/app/data" if os.path.exists("/app/data") else os.getcwd()
                        save_path = os.path.join(default_root, final_filename + ".ftr")
                        
                        sheet_df.reset_index(drop=True).to_feather(save_path)
                        success_files.append((target_sheet, save_path, len(sheet_df), sheet_date_columns))
                        
                    except Exception as e:
                        error_files.append((target_sheet, str(e)))
                    
                    # 진행률 업데이트
                    progress_bar.progress((idx + 1) / len(available_sheets))
                
                # 결과 요약
                status_text.text("처리 완료!")
                
                if success_files:
                    st.success(f"✅ {len(success_files)}개 시트 처리 완료!")
                    
                    # 성공한 파일들 정보 표시
                    success_data = []
                    for sheet, path, rows, date_cols in success_files:
                        success_data.append({
                            "시트명": sheet,
                            "행 수": f"{rows:,}",
                            "보호된 날짜 컬럼": ", ".join(date_cols) if date_cols else "없음",
                            "파일 경로": path
                        })
                    
                    st.dataframe(pd.DataFrame(success_data), use_container_width=True)
                
                if error_files:
                    st.error(f"❌ {len(error_files)}개 시트 처리 실패:")
                    for sheet, error in error_files:
                        st.write(f"• {sheet}: {error}")
        
        else:
            st.info("💡 시트가 1개뿐이므로 일괄 처리 기능이 필요하지 않습니다.")
    
    # ====================================================================== 
    # ✅ 2. 저장된 모든 Feather 파일 다운로드 기능
    st.markdown("---")
    st.header("📦 저장된 Feather 파일 일괄 다운로드")
    
    def get_feather_files_in_directory():
        """저장 디렉토리에서 모든 .ftr 파일 찾기"""
        default_root = "/app/data" if os.path.exists("/app/data") else os.getcwd()
        feather_files = []
        
        try:
            for file in os.listdir(default_root):
                if file.endswith(".ftr"):
                    file_path = os.path.join(default_root, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    feather_files.append({
                        "파일명": file,
                        "경로": file_path,
                        "크기(MB)": f"{file_size:.2f}"
                    })
        except Exception as e:
            st.error(f"파일 목록을 가져올 수 없습니다: {e}")
        
        return feather_files
    
    def create_zip_download(file_paths):
        """여러 파일을 ZIP으로 압축하여 다운로드 준비"""
        import zipfile
        import io as io_module
        
        zip_buffer = io_module.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    file_name = os.path.basename(file_path)
                    zip_file.write(file_path, file_name)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    # Feather 파일 목록 조회 버튼
    if st.button("🔍 저장된 Feather 파일 조회"):
        feather_files = get_feather_files_in_directory()
        
        if feather_files:
            st.write(f"**📁 발견된 Feather 파일 ({len(feather_files)}개):**")
            
            # 파일 목록을 데이터프레임으로 표시
            files_df = pd.DataFrame(feather_files)
            st.dataframe(files_df, use_container_width=True)
            
            # 전체 크기 계산
            total_size = sum(float(f["크기(MB)"]) for f in feather_files)
            st.info(f"📊 총 파일 크기: {total_size:.2f} MB")
            
            # ZIP 다운로드 버튼
            if len(feather_files) > 0:
                try:
                    file_paths = [f["경로"] for f in feather_files]
                    zip_data = create_zip_download(file_paths)
                    
                    # ✅ 1. 파일명에 타임스탬프 추가 (한글 제거)
                    from datetime import datetime
                    # 24시간 형식 사용하여 한글(오전/오후) 제거
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    zip_filename = f"feather_files_{timestamp}.zip"
                    
                    st.download_button(
                        label=f"📥 모든 Feather 파일 다운로드 ({len(feather_files)}개 파일)",
                        data=zip_data,
                        file_name=zip_filename,
                        mime="application/zip"
                    )
                    
                    st.success(f"✅ {len(feather_files)}개 파일이 ZIP으로 압축되어 다운로드 준비됨")
                    st.info("🛡️ 모든 파일의 날짜 컬럼이 원본 형태로 보존되었습니다.")
                    
                except Exception as e:
                    st.error(f"❌ ZIP 생성 실패: {e}")
        else:
            st.warning("📂 저장된 Feather 파일이 없습니다.")
            st.info("💡 먼저 'Feather로 저장하기' 또는 '모든 시트에 동일한 처리 적용'을 실행하세요.")

else:
    st.info("📂 .xlsx 파일을 먼저 업로드해주세요.")
    






def safe_rmtree(path):
    def handle_readonly(func, path, exc):
        os.chmod(path, stat.S_IWRITE)  # 읽기전용 해제
        func(path)
    
    if os.path.exists(path):
        try:
            shutil.rmtree(path, onerror=handle_readonly)
        except OSError:
            # 실패시 폴더 내용만 삭제
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        os.chmod(file_path, stat.S_IWRITE)
                        os.unlink(file_path)
                    except:
                        pass


# ======================================================================
st.title("📊 CSV 파일 병합 및 Feather 변환 도구")
st.write("여러 CSV 파일을 읽어서 시간 인덱스를 조정하고, 노이즈를 제거한 후 Feather 파일로 변환합니다.")

# 파일 업로드 UI
st.subheader("📁 CSV 파일 업로드")
st.write("처리할 CSV 파일들을 선택하여 업로드하세요. (Ctrl 또는 Shift 키를 사용해 여러 파일 선택 가능)")

uploaded_files = st.file_uploader("CSV 파일 선택", type="csv", accept_multiple_files=True)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)}개의 CSV 파일이 업로드되었습니다.")
    
    # 업로드된 파일 목록 표시
    with st.expander("업로드된 파일 목록", expanded=False):
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size} bytes)")
    

    # 임시 폴더 생성 (이미 존재하면 제거 후 새로 생성)
    folder_path = os.path.join(os.getcwd(), "data", "temp_uploads")
    if os.path.exists(folder_path):
        # shutil.rmtree(folder_path)  # 기존 폴더 삭제
        safe_rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)  # 새 폴더 생성
    
    for uploaded_file in uploaded_files:
        with open(os.path.join(folder_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    st.info(f"업로드된 파일이 임시 폴더에 저장되었습니다: {folder_path}")
    
    # 이후 처리를 위해 파일 목록 준비
    csv_files = [file.name for file in uploaded_files]


    #-----------------------(20250523 추가 - 시작)------------------------------------
    # CSV 파일 분석
    st.subheader("📊 CSV 파일 분석")
    
    # 파일 선택
    selected_file = st.selectbox(
        "헤더를 확인할 파일 선택",
        csv_files,
        help="선택한 파일의 헤더와 미리보기를 표시합니다"
    )
    
    if selected_file:
        try:
            # 선택된 파일 읽기
            selected_file_path = os.path.join(folder_path, selected_file)
            df_preview = pd.read_csv(selected_file_path, nrows=100)  # 미리보기용으로 100행만 읽기
            
            st.write(f"**선택된 파일: {selected_file}**")
            
            # 헤더 정보 표시
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**📋 컬럼 헤더:**")
                headers_df = pd.DataFrame({
                    "컬럼명": df_preview.columns,
                    "데이터 타입": df_preview.dtypes.astype(str),
                    "샘플 값": [str(df_preview[col].iloc[0]) if not df_preview[col].empty else "N/A" 
                              for col in df_preview.columns]
                })
                st.dataframe(headers_df, use_container_width=True)
            
            with col2:
                st.write("**📈 파일 정보:**")
                st.write(f"- 총 컬럼 수: {len(df_preview.columns)}")
                st.write(f"- 미리보기 행 수: {len(df_preview)}")
                st.write(f"- 파일 크기: {os.path.getsize(selected_file_path):,} bytes")
            
            # 데이터 미리보기
            st.write("**🔍 데이터 미리보기 (상위 5행):**")
            st.dataframe(df_preview.head(), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ 파일 읽기 오류: {str(e)}")



    # ===================(시간 설정부 입력 - 시작)=====================================
    df_head = None
    for csv_file in csv_files:
        try:
            file_path = os.path.join(folder_path, csv_file)
            
            # 파일의 첫 100행과 마지막 100행을 읽어서 시간 범위 확인
            df_head = pd.read_csv(file_path, nrows=100)
            break
        except Exception as e:
            st.warning(f"⚠️ 파일 읽기 중 오류 발생: {csv_file}\n{str(e)}")   

    if df_head is not None:
        # Streamlit 입력 받기
        st.subheader("시간 컬럼 설정")
        col1, col2, col3 = st.columns(3)

        with col1:
            year_col = st.selectbox("Year 컬럼명", options=[None] + list(df_head.columns), index=0 if "year" not in df_head.columns else list(df_head.columns).index("year")+1)
            if year_col is None:
                year_col = st.text_input("Year 컬럼명 직접 입력", value="year")
            month_col = st.selectbox("Month 컬럼명", options=[None] + list(df_head.columns), index=0 if "month" not in df_head.columns else list(df_head.columns).index("month")+1)
            if month_col is None:
                month_col = st.text_input("Month 컬럼명 직접 입력", value="month")

        with col2:
            day_col = st.selectbox("Day 컬럼명", options=[None] + list(df_head.columns), index=0 if "day" not in df_head.columns else list(df_head.columns).index("day")+1)
            if day_col is None:
                day_col = st.text_input("Day 컬럼명 직접 입력", value="day")
            hour_col = st.selectbox("Hour 컬럼명", options=[None] + list(df_head.columns), index=0 if "hour" not in df_head.columns else list(df_head.columns).index("hour")+1)
            if hour_col is None:
                hour_col = st.text_input("Hour 컬럼명 직접 입력", value="hour")

        with col3:
            minute_col = st.selectbox("Minute 컬럼명", options=[None] + list(df_head.columns), index=0 if "minute" not in df_head.columns else list(df_head.columns).index("minute")+1)
            if minute_col is None:
                minute_col = st.text_input("Minute 컬럼명 직접 입력", value="minute")
            second_col = st.selectbox("Second 컬럼명", options=[None] + list(df_head.columns), index=0 if "second" not in df_head.columns else list(df_head.columns).index("second")+1)
            if second_col is None:
                second_col = st.text_input("Second 컬럼명 직접 입력", value="second")
                
        time_columns = {
            'year': year_col,
            'month': month_col, 
            'day': day_col,
            'hour': hour_col,
            'minute': minute_col,
            'second': second_col
        }
    else:
        st.error("읽을 수 있는 CSV 파일이 없습니다.")
        st.stop()
    # ===================(시간 설정부 입력 - 끝)=====================================            

else:
    st.warning("⚠️ CSV 파일을 업로드해주세요.")



# 업로드된 파일 처리 옵션
with st.sidebar:
    st.markdown("---")
    st.sidebar.header("📁 CSV 데이터 처리 옵션")
    sampling_rate = st.sidebar.selectbox("리샘플링 주기:", ["1s", "5s", "10s", "30s", "1min"], index=1)
    sampling_method = st.sidebar.selectbox("리샘플링 방법:", ["median", "mean", "min", "max"], index=0)
    remove_spikes = st.sidebar.checkbox("스파이크 노이즈(999) 제거", value=True)

    # 파일명 설정
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    output_filename = st.sidebar.text_input("출력 파일명:", f"processed_data_{timestamp}.ftr")


# ---------- 사용자 입력 ----------
with st.sidebar:
    st.markdown("---")
    st.markdown("🧠 **회사명:** ㈜파시디엘")
    st.markdown("🏫 **연구실:** visLAB@PNU")
    st.markdown("👨‍💻 **제작자:** (C)Dong2")
    st.markdown("🛠️ **버전:** V.1.1 (05-20-2025)")
    st.markdown("---")




# ======================================================================       
# 함수 정의
@st.cache_data(show_spinner=False)
def replace_999_with_neighbors_mean(df):
   df = df.copy()
   for col in df.columns:
       values = df[col].values
       for i in range(1, len(values) - 1):
           if values[i] == 999:
               if values[i - 1] != 999 and values[i + 1] != 999:
                   values[i] = (values[i - 1] + values[i + 1]) / 2
               else:
                   values[i] = np.nan  # 앞뒤도 999면 보류
       df[col] = values
   return df




# ======================================================================       
# 탭 분리
tab1, tab2 = st.tabs(["🔄 데이터 처리", "📊 데이터 시각화"])

with tab1:
    # 처리 완료 상태 확인
    if st.session_state.get('processing_complete', False):
        # ✅ 이미 처리 완료된 경우 - 항상 동일한 화면 표시
        
        # 재처리 버튼 (맨 위에 표시)
        if st.button("🔄 다시 처리", help="새로운 데이터로 다시 처리"):
            # 세션 상태 초기화
            keys_to_clear = ['processing_complete', 'resampled_df', 'processing_results']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("---")  # 구분선
        
        # 처리 결과 항상 표시 (처리 버튼 상태와 무관)
        if 'processing_results' in st.session_state:
            results = st.session_state['processing_results']
            
            # 처리 완료 메시지
            st.success(f"🎉 Feather 파일이 성공적으로 저장되었습니다!")
            st.write(f"📁 저장 경로: {results['file_path']}")
            st.write(f"📊 파일 크기: {results['file_size']:.2f} MB")
            st.write(f"📈 데이터 행 수: {results['resampled_rows']:,} (원본 대비 {results['reduction_ratio']:.1f}%)")
            
            # 🔥 핵심: 다운로드 버튼을 항상 표시 (버튼 상태와 무관)
            if os.path.exists(results['file_path']):
                with open(results['file_path'], 'rb') as f:
                    st.download_button(
                        label="📥 Feather 파일 다운로드",
                        data=f,
                        file_name=results['filename'],
                        mime="application/octet-stream",
                        help="처리된 데이터를 Feather 형식으로 다운로드",
                        type="primary",
                        key="download_persistent_state"
                    )
            else:
                st.error("⚠️ 파일을 찾을 수 없습니다. 다시 처리해주세요.")
            
            # 처리된 데이터 샘플 표시
            with st.expander("📋 처리된 데이터 샘플 (처음 5행)", expanded=False):
                if 'resampled_df' in st.session_state:
                    st.dataframe(st.session_state['resampled_df'].head())
                else:
                    st.error("데이터를 불러올 수 없습니다.")
            
            st.info("💡 '데이터 시각화' 탭에서 데이터를 시각화할 수 있습니다!")
    
    else:
        # ❌ 아직 처리되지 않은 경우 - 처리 시작
        st.info("📂 데이터 처리를 시작하려면 아래 버튼을 클릭하세요.")
        
        if st.button("🔄 처리 시작", type="primary"):
            try:
                # 입력 폴더 유효성 검사
                if not os.path.exists(folder_path):
                    st.error(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
                    st.stop()
                
                # CSV 파일 찾기
                csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
                
                if not csv_files:
                    st.error("❌ 지정된 폴더에 CSV 파일이 없습니다.")
                    st.stop()
                
                st.info(f"📂 총 {len(csv_files)} 개의 CSV 파일을 발견했습니다.")
                
                # 진행 상황 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 파일 처리 시작
                all_data = []

                for i, file in enumerate(csv_files):
                    status_text.text(f"처리 중: {file} ({i+1}/{len(csv_files)})")
                    progress_bar.progress((i + 1) / len(csv_files))
                    
                    try:
                        df = pd.read_csv(os.path.join(folder_path, file), low_memory=False)
                        
                        # time_columns에서 None이 아닌 값들만 필터링
                        valid_time_columns = {k: v for k, v in time_columns.items() if v is not None}
                        
                        # 유효한 시간 컬럼이 있는 경우에만 시간 인덱스 조정
                        if valid_time_columns:
                            # 시간 컬럼 존재 여부 확인
                            missing_columns = []
                            for key, col_name in valid_time_columns.items():
                                if col_name not in df.columns:
                                    missing_columns.append(col_name)

                            if missing_columns:
                                st.warning(f"파일 {file}에서 다음 시간 컬럼이 없습니다: {missing_columns}")
                                continue
                            else:
                                # year 컬럼이 있는 경우에만 값 조정
                                if 'year' in valid_time_columns:
                                    if df[valid_time_columns['year']].max() < 100:
                                        df[valid_time_columns['year']] = df[valid_time_columns['year']] + 2000
                                
                                # timestamp 생성을 위한 컬럼 리스트
                                time_col_list = [valid_time_columns[col] for col in ['year', 'month', 'day', 'hour', 'minute', 'second'] 
                                                if col in valid_time_columns]
                                
                                df['timestamp'] = pd.to_datetime(df[time_col_list])
                                df = df.set_index('timestamp')
                                df.drop(columns=list(valid_time_columns.values()), inplace=True)

                        all_data.append(df)
                        
                    except Exception as e:
                        st.warning(f"⚠️ 파일 처리 중 오류 발생: {file}\n{str(e)}")    
                
                # 처리 결과 확인
                if not all_data:
                    st.error("❌ 처리할 수 있는 데이터가 없습니다.")
                    st.stop()
                
                status_text.text("파일 병합 중...")
                
                # 파일 병합 
                merged_df = pd.concat(all_data)
                # print(merged_df.index)

                
                # 시간순 정렬 및 중복 제거
                status_text.text("데이터 정렬 및 중복 제거 중...")
                merged_df = merged_df.sort_index()
                merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
                
                # 스파이크 노이즈 제거
                if remove_spikes:
                    status_text.text("스파이크 노이즈(999) 제거 중...")
                    merged_df = replace_999_with_neighbors_mean(merged_df)
                
                # 리샘플링
                status_text.text(f"{sampling_rate} 주기로 리샘플링 중...")
                if sampling_method == "median":
                    resampled_df = merged_df.resample(sampling_rate).median()
                elif sampling_method == "mean":
                    resampled_df = merged_df.resample(sampling_rate).mean()
                elif sampling_method == "min":
                    resampled_df = merged_df.resample(sampling_rate).min()
                else:  # max
                    resampled_df = merged_df.resample(sampling_rate).max()
                
                # Feather 파일 저장
                status_text.text("Feather 파일 저장 중...")
                
                save_dir = os.path.dirname(os.path.join(folder_path, output_filename))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                # 파일 저장
                resampled_df_save = resampled_df.reset_index()
                save_path = os.path.join(folder_path, output_filename)
                resampled_df_save.to_feather(save_path)
                
                # 파일 크기 계산
                file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB 단위
                
                # ✅ 모든 처리 결과를 세션 상태에 저장
                st.session_state['resampled_df'] = resampled_df
                st.session_state['processing_complete'] = True
                st.session_state['processing_results'] = {
                    'merged_rows': len(merged_df),
                    'merged_cols': len(merged_df.columns),
                    'time_range': f"{merged_df.index.min()} ~ {merged_df.index.max()}",
                    'resampled_rows': len(resampled_df),
                    'reduction_ratio': len(resampled_df)/len(merged_df)*100,
                    'file_path': save_path,
                    'filename': output_filename,
                    'file_size': file_size
                }
                
                # 처리 완료 메시지 및 즉시 결과 표시
                status_text.text("✅ 처리 완료!")
                st.success(f"🎉 Feather 파일이 성공적으로 저장되었습니다!")
                st.write(f"📁 저장 경로: {save_path}")
                st.write(f"📊 파일 크기: {file_size:.2f} MB")
                st.write(f"📈 데이터 행 수: {len(resampled_df):,} (원본 대비 {len(resampled_df)/len(merged_df)*100:.1f}%)")
                
                # 즉시 다운로드 버튼 제공
                if os.path.exists(save_path):
                    with open(save_path, 'rb') as f:
                        st.download_button(
                            label="📥 Feather 파일 다운로드",
                            data=f,
                            file_name=output_filename,
                            mime="application/octet-stream",
                            help="처리된 데이터를 Feather 형식으로 다운로드",
                            type="primary",
                            key="download_after_processing"
                        )
                
                # 처리된 데이터 샘플 표시
                with st.expander("📋 처리된 데이터 샘플 (처음 5행)", expanded=False):
                    st.dataframe(resampled_df.head())
                
                st.info("💡 '데이터 시각화' 탭에서 데이터를 시각화할 수 있습니다!")
                
            except Exception as e:
                st.error(f"❌ 전체 처리 중 오류 발생: {str(e)}")
                # 오류 발생시 세션 상태 정리
                keys_to_clear = ['processing_complete', 'resampled_df', 'processing_results']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]

# ======================================================================       

# 데이터 시각화 부분 (독립적으로 처리)
with tab2:
    if st.session_state.get('processing_complete', False):
        st.subheader("📊 데이터 시각화")
        
        resampled_df = st.session_state['resampled_df']
        
        # 시각화할 컬럼 선택
        vis_columns = st.multiselect(
            "시각화할 컬럼 선택:",
            options=resampled_df.columns.tolist(),
            default=resampled_df.columns[:3].tolist(),  # 기본적으로 처음 3개 컬럼 선택
            key="vis_columns_selector"
        )
        
        if vis_columns:
            # 플로팅할 데이터 샘플링 (시각화 속도 개선)
            sampling_vis = st.slider(
                "시각화 샘플링 비율", 
                1, 50, 10,
                key="vis_sampling_slider",
                help="큰 값일수록 적은 데이터 포인트 표시 (속도 향상)"
            )
            sampled_df = resampled_df.iloc[::sampling_vis]
            
            # 통계 정보 표시
            st.write(f"📈 총 {len(resampled_df):,}개 중 {len(sampled_df):,}개 포인트 표시")
            
            # Plotly로 시각화
            fig = go.Figure()
            
            for col in vis_columns:
                fig.add_trace(
                    go.Scatter(
                        x=sampled_df.index,
                        y=sampled_df[col],
                        mode='lines',
                        name=col,
                        line=dict(width=1.5)
                    )
                )
            
            fig.update_layout(
                title=f"선택한 컬럼 시각화 (1/{sampling_vis} 샘플링)",
                xaxis_title="시간",
                yaxis_title="값",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 컬럼별 기본 통계
            with st.expander("📊 선택된 컬럼 통계", expanded=False):
                stats_df = resampled_df[vis_columns].describe()
                st.dataframe(stats_df)
                
        else:
            st.info("👆 시각화할 컬럼을 선택해주세요.")
    else:
        st.info("🔄 먼저 '데이터 처리' 탭에서 데이터를 처리해주세요.")








