import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import requests
import time
from pandasai import SmartDatalake
from pandasai.llm.base import LLM

# 自定义 DeepSeek LLM 类
class DeepSeekLLM(LLM):
    def __init__(self, api_key: str, model: str = "deepseek-chat", api_base: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key; self.model = model; self.api_base = api_base; self.max_retries = 3; self.retry_delay = 2

    def call(self, prompt: str, *args, **kwargs) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": [{"role": "user", "content": str(prompt)}], "temperature": 0.0, "max_tokens": 4096}
        for attempt in range(self.max_retries):
            try:
                response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    error_msg = f"API错误 ({response.status_code}): {response.text}"; time.sleep(self.retry_delay)
            except requests.exceptions.RequestException as e:
                error_msg = f"网络请求异常: {str(e)}"; time.sleep(self.retry_delay)
        return f"API请求失败: {error_msg}"

    @property
    def type(self) -> str: return "deepseek-llm"

# --- 页面设置 ---
st.set_page_config(page_title="智能数据分析助理 (DeepSeek版)", layout="wide")

# (update_template_file 函数保持不变)
def update_template_file(template_df: pd.DataFrame, data_dfs: list, key_column: str) -> pd.DataFrame | str:
    if key_column not in template_df.columns: return f"错误：关键列 '{key_column}' 不存在于您的模板文件中。"
    for df in data_dfs:
        if key_column not in df.columns: return f"错误：关键列 '{key_column}' 不存在于其中一个数据源文件中。"
    if not data_dfs: return template_df
    try:
        source_data = pd.concat(data_dfs, ignore_index=True).drop_duplicates(subset=[key_column], keep='last').set_index(key_column)
    except Exception as e: return f"设置索引时发生错误: {e}"
    updated_df = template_df.copy()
    for index, row in updated_df.iterrows():
        key_value = row[key_column]
        if key_value in source_data.index:
            source_row = source_data.loc[key_value]
            for col_name in updated_df.columns:
                if pd.isna(row[col_name]):
                    if col_name in source_row.index and not pd.isna(source_row[col_name]):
                        updated_df.loc[index, col_name] = source_row[col_name]
    return updated_df

# --- 主应用界面 ---
st.title("智能数据分析助理 🚀 (DeepSeek 驱动)")

# --- 安全获取 API 密钥 (本地测试版) ---
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # <--- 在这里直接粘贴您的 DeepSeek 密钥
st.session_state.api_key = DEEPSEEK_API_KEY
st.sidebar.warning("注意：当前为本地测试模式")
if not st.session_state.api_key or "sk-" not in st.session_state.api_key:
    st.error("请在代码中设置您的 DeepSeek API 密钥。"); st.stop()

# --- 初始化自定义 DeepSeek LLM ---
llm = DeepSeekLLM(api_key=st.session_state.api_key)

# --- 功能分页 ---
tab1, tab2 = st.tabs(["🧠 智能分析", "🎯 模板精确更新"])

with tab1:
    st.header("上传文件，提出问题")
    ai_uploaded_files = st.file_uploader("请上传相关 Excel 文件 (.xlsx)", type=["xlsx"], accept_multiple_files=True, key="ai_uploader")
    if ai_uploaded_files:
        try:
            ai_dataframes = [pd.read_excel(file) for file in ai_uploaded_files]
            user_prompt = st.text_area("请输入您的分析指令：", key="user_prompt", height=200)

            if st.button("🧠 开始智能分析", type="primary"):
                if user_prompt:
                    with st.spinner("DeepSeek AI 正在生成分析代码..."):
                        
                        # --- 最终版、高强度、格式化的指令模板 ---
                        expert_system_prompt = f"""
你是一名世界顶级的Python数据分析专家。你的任务是只编写Python代码来回答用户的问题。

[工作规则]
1. 你必须只通过生成Python代码来回答问题。
2. 你生成的代码将被 `exec()` 函数执行。
3. 代码的最终结果必须保存在一个名为 `result` 的变量中。
4. 你的整个回答必须且只能是一个Python代码块，像这样：```python ... ```。
5. **不要**在代码块之外添加任何解释、评论或文字。
6. 如果需要计算的列名可能不存在，请使用 `df.get('列名', 0)` 的方式来避免错误。

[用户的问题]
{user_prompt}

[你的回答]
"""
                        
                        lake = SmartDatalake(ai_dataframes, config={"llm": llm})
                        result = lake.chat(expert_system_prompt)
                        st.session_state.result = result
        except Exception as e:
            st.error(f"处理文件时发生错误: {e}")

# (tab2 和后续的显示代码保持不变)
with tab2:
    st.header("将多个文件的数据，更新到一个模板文件中")
    # ... (此部分代码无需改动)
    st.info("此功能会保持模板文件的行列顺序不变，仅填补其中的空白单元格。")
    st.subheader("① 上传您的模板文件"); template_file = st.file_uploader("请上传您要更新的目标模板文件", type=["xlsx"], accept_multiple_files=False, key="template_uploader")
    st.subheader("② 上传您的数据源文件"); data_source_files = st.file_uploader("请上传包含更新信息的一个或多个数据文件", type=["xlsx"], accept_multiple_files=True, key="data_source_uploader")
    st.subheader("③ 输入关键列名"); key_column = st.text_input("请输入用于匹配的字段名称", help="此字段必须同时存在于所有文件中。")
    if st.button("⚙️ 开始精确更新", type="primary"):
        if not template_file: st.warning("请上传模板文件。")
        elif not data_source_files: st.warning("请上传至少一个数据源文件。")
        elif not key_column: st.warning("请输入关键列名。")
        else:
            with st.spinner("正在执行精确更新..."):
                try:
                    template_df = pd.read_excel(template_file); data_dfs = [pd.read_excel(file) for file in data_source_files]
                    result = update_template_file(template_df, data_dfs, key_column)
                    st.session_state.result = result
                except Exception as e: st.error(f"更新过程中发生严重错误: {e}")

st.markdown("---")
if "result" in st.session_state and st.session_state.result is not None:
    st.subheader("📈 处理结果：")
    result_data = st.session_state.result
    if isinstance(result_data, pd.DataFrame):
        st.dataframe(result_data)
    elif isinstance(result_data, str):
        st.code(result_data)
    else:
        st.write(result_data)
st.markdown("---"); st.markdown("由 DeepSeek, PandasAI, and Streamlit 驱动")