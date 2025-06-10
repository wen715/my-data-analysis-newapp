import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import requests
import time
from pandasai import SmartDatalake
from pandasai.llm.base import LLM

# --- 核心功能函数区域 ---

# 1. 强力数据清洗函数
def clean_and_convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """将DataFrame中看起来像数字的文本列强制转换为数值类型"""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # 尝试移除千位分隔符等非数字字符，然后转换
            try:
                cleaned_series = df_clean[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                # 强制转换为数值，无法转换的值会变成NaN（Not a Number）
                numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                # 只有当转换后至少有一个有效数字时，才替换原始列
                if not numeric_series.isna().all():
                    df_clean[col] = numeric_series
            except Exception:
                # 如果在转换中发生任何意外，则保持该列不变
                pass
    return df_clean

# 2. 最终版的数据回填函数
def fill_template_final(template_df: pd.DataFrame, source_dfs: list, key_columns: list) -> pd.DataFrame:
    """根据多个关键列匹配，对数值列求和，并回填到模板中"""
    
    # 首先对所有输入文件进行数据清洗
    cleaned_template_df = clean_and_convert_to_numeric(template_df)
    cleaned_source_dfs = [clean_and_convert_to_numeric(df) for df in source_dfs]

    # 验证关键列是否存在
    if not all(key in cleaned_template_df.columns for key in key_columns):
        missing_keys = [key for key in key_columns if key not in cleaned_template_df.columns]
        raise ValueError(f"错误：关键列 {missing_keys} 在模板文件中不存在。")

    # 统一关键列的数据类型为字符串，以确保匹配成功
    for df in [cleaned_template_df] + cleaned_source_dfs:
        for key in key_columns:
            if key in df.columns:
                df[key] = df[key].astype(str).str.strip()

    # 合并所有有效的数据源
    all_sources = [df.copy() for df in cleaned_source_dfs if all(key in df.columns for key in key_columns)]
    if not all_sources:
        raise ValueError("错误：所有数据源文件中都不完整包含您输入的全部关键列。")
    
    combined_sources = pd.concat(all_sources, ignore_index=True)
    
    # 定义聚合规则：数值列求和，其他列取第一个
    agg_functions = {}
    for col in combined_sources.columns:
        if col not in key_columns:
            if pd.api.types.is_numeric_dtype(combined_sources[col]):
                agg_functions[col] = 'sum'
            else:
                agg_functions[col] = 'first'
    
    if not agg_functions: return cleaned_template_df
    
    # 按多关键列分组并聚合
    aggregated_source = combined_sources.groupby(key_columns).agg(agg_functions).reset_index()
    source_data_lookup = aggregated_source.set_index(key_columns)

    # 迭代并填充模板
    filled_df = cleaned_template_df.copy()
    for index, row in filled_df.iterrows():
        key_values = tuple(str(row.get(key, '')) for key in key_columns)
        
        if key_values in source_data_lookup.index:
            source_row = source_data_lookup.loc[key_values]
            for col_name in filled_df.columns:
                is_empty_or_zero = pd.isna(row[col_name]) or (np.isscalar(row[col_name]) and isinstance(row[col_name], (int, float, np.number)) and row[col_name] == 0)
                if is_empty_or_zero:
                    if col_name in source_row.index and not pd.isna(source_row[col_name]):
                        filled_df.loc[index, col_name] = source_row[col_name]
    return filled_df

# 3. AI 调用类 (保持不变)
class DeepSeekLLM(LLM):
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key; self.model = model; self.api_base = "https://api.deepseek.com/v1"
    def call(self, prompt: str, *args, **kwargs) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": [{"role": "user", "content": str(prompt)}], "temperature": 0.0}
        try:
            response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"API请求或网络发生错误: {e}"
    @property
    def type(self) -> str: return "deepseek-llm"


# --- Streamlit 应用主界面 ---
st.set_page_config(page_title="智能数据处理与分析平台", page_icon="📊", layout="wide")
st.title("📊 智能数据处理与分析平台")

with st.sidebar:
    st.header("🔑 API 密钥")
    api_key = st.text_input("DeepSeek API密钥", type="password", help="从DeepSeek官网获取API密钥")

tab1, tab2 = st.tabs(["✍️ 数据回填 (最终版)", "🧠 智能分析"])

# 数据回填分页
with tab1:
    st.header("将源数据回填到模板文件")
    st.info("此最终版功能会自动深度清洗数据，并根据多个关键列匹配，将数值求和后填充。")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        source_files = st.file_uploader("1. 上传一个或多个数据源文件", type=["xlsx", "xls"], accept_multiple_files=True, key="source_files")
    with col2:
        template_file = st.file_uploader("2. 上传一个空白的模板文件", type=["xlsx", "xls"], key="template_file")
    
    key_column_input = st.text_input("3. 请输入一个或多个关键列名，用英文逗号隔开", "部门,费用项目", help="例如：部门,费用项目")

    if st.button("开始回填", type="primary", use_container_width=True, key="start_fill_data"):
        if not source_files or not template_file or not key_column_input.strip():
            st.warning("请确保已上传数据源、模板文件，并已输入关键列名。")
        else:
            with st.spinner("正在深度清洗、分组求和并回填数据..."):
                try:
                    key_columns_list = [key.strip() for key in key_column_input.split(',')]
                    source_dfs = [pd.read_excel(f) if f.name.endswith('xlsx') else pd.read_csv(f) for f in source_files]
                    template_df = pd.read_excel(template_file) if template_file.name.endswith('xlsx') else pd.read_csv(template_file)
                    
                    filled_df = fill_template_final(template_df, source_dfs, key_columns_list)
                    
                    st.success("数据回填成功！")
                    st.dataframe(filled_df)
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        filled_df.to_excel(writer, index=False, sheet_name="Final_Filled_Template")
                    excel_data = output.getvalue()
                    
                    st.download_button(label="📥 下载已回填的模板文件", data=excel_data, file_name=f"final_filled_{template_file.name}")
                except Exception as e:
                    st.error(f"回填过程中发生错误: {e}")

# 智能分析分页
with tab2:
    st.header("使用 AI 进行通用的、探索性的数据分析")
    if not api_key or not api_key.startswith("sk-"): st.warning("请输入有效的DeepSeek API密钥以启用“智能分析”功能。")
    ai_uploaded_files_tab2 = st.file_uploader("上传用于AI分析的Excel文件", type=["xlsx", "xls"], accept_multiple_files=True, key="ai_uploader_tab2")
    if ai_uploaded_files_tab2 and api_key.startswith("sk-"):
        analysis_prompt = st.text_area("您的分析需求", height=100, key="analysis_input")
        if st.button("开始智能分析", type="primary", use_container_width=True, key="start_ai_analysis"):
            if analysis_prompt.strip():
                with st.spinner("AI正在思考中..."):
                    try:
                        llm = DeepSeekLLM(api_key=api_key)
                        data_frames = [pd.read_excel(f) for f in ai_uploaded_files_tab2]
                        lake = SmartDatalake(data_frames, config={"llm": llm})
                        response = lake.chat(analysis_prompt)
                        st.subheader("分析结果")
                        st.write(response)
                    except Exception as e: st.error(f"分析过程中发生严重错误：\n\n{e}")