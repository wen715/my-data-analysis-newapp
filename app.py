import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import requests
import time
from pandasai import SmartDatalake
from pandasai.llm.base import LLM

# 自定义 DeepSeek LLM 类，包含了您编写的所有优秀特性
class DeepSeekLLM(LLM):
    def __init__(self, api_key: str, model: str = "deepseek-chat", api_base: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.max_retries = 3
        self.retry_delay = 2  # 秒

    def call(self, prompt: str, *args, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": str(prompt)}],
            "temperature": 0.0,
            "max_tokens": 4096
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    error_msg = f"API错误 ({response.status_code}): {response.text}"
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        return f"API请求失败: {error_msg}"
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return f"网络请求异常: {str(e)}"
        return "未知错误: 所有重试失败"

    @property
    def type(self) -> str:
        return "deepseek-llm"

# --- 页面设置 ---
st.set_page_config(page_title="智能数据分析助理 (部署版)", layout="wide")

# --- 主应用界面 ---
st.title("智能数据分析助理 🚀 (DeepSeek 驱动)")
st.markdown("本应用已配置为安全模式，可供多用户使用。")

# --- 从 Streamlit Secrets 安全获取 DeepSeek API 密钥 (部署版) ---
api_key = None
try:
    # 用于 Streamlit Cloud 部署
    if "DEEPSEEK_API_KEY" in st.secrets and st.secrets["DEEPSEEK_API_KEY"]:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
        st.sidebar.success("API 密钥已成功加载", icon="✅")
    else:
        st.error("DeepSeek API 密钥未在应用的 Secrets 中正确设置。")
        st.info("如果您是此应用的所有者，请前往应用的 Settings > Secrets 添加您的 DeepSeek API 密钥。")
        st.stop()
except FileNotFoundError:
    # 用于本地测试
    st.error("在本地运行此应用时，请在项目根目录下创建一个 .streamlit/secrets.toml 文件来存放您的 DeepSeek API 密钥。")
    st.code('# 在 .streamlit/secrets.toml 文件中这样写:\nDEEPSEEK_API_KEY = "sk-..."')
    st.stop()
except Exception as e:
    st.error(f"加载 API 密钥时发生未知错误: {e}")
    st.stop()

# --- 初始化自定义 DeepSeek LLM ---
llm = DeepSeekLLM(api_key=api_key)

# --- 智能分析主功能 ---
st.header("提出您的数据分析请求")
ai_uploaded_files = st.file_uploader("请上传一个或多个相关 Excel 文件 (.xlsx)", type=["xlsx"], accept_multiple_files=True, key="ai_uploader")

if ai_uploaded_files:
    try:
        ai_dataframes = [pd.read_excel(file) for file in ai_uploaded_files]
        user_prompt = st.text_area("请输入您的分析指令（可以是一个复杂的多步计算）：", key="user_prompt", height=200)

        if st.button("🧠 开始智能分析", type="primary"):
            if user_prompt:
                with st.spinner("AI 正在深度思考并生成执行方案..."):
                    # --- 终极指令模板 ---
                    expert_system_prompt = f"""
<ROLE>
你是一名世界顶级的Python数据分析专家。你的唯一任务是编写一段可执行的Python代码来回答用户的问题。
</ROLE>
<INSTRUCTIONS>
你必须严格遵守以下规则：
1. **分析任务**: 首先，理解 <USER_TASK> 中的用户目标。
2. **编写代码**: 编写一段Python代码来完成这个任务。
3. **最终变量**: 代码的最终结果必须存储在一个名为 `result` 的变量中。
4. **代码封装**: 你的整个回答必须且只能是一个Python代码块，以 ```python 开头，以 ``` 结尾。
5. **禁止解释**: 不要在代码块之外添加任何解释、评论或文字。
6. **代码健壮性**: 如果需要计算的列名可能不存在，请使用 `df.get('列名', 0)` 的方式来避免错误。
</INSTRUCTIONS>
<USER_TASK>
用户的请求是: "{user_prompt}"
</USER_TASK>
<YOUR_ANSWER>
"""
                    
                    lake = SmartDatalake(ai_dataframes, config={"llm": llm})
                    max_retries = 3
                    result = None
                    
                    for attempt in range(max_retries):
                        try:
                            result = lake.chat(expert_system_prompt)
                            if result: break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                st.error(f"分析失败(尝试 {attempt + 1}/{max_retries}): {str(e)}")
                            else:
                                st.warning(f"分析尝试 {attempt + 1}/{max_retries} 失败，正在重试...")
                                time.sleep(2)
                    
                    st.subheader("📈 分析结果：")
                    if result is None:
                        st.error("AI未返回有效结果，请尝试：")
                        st.markdown("""
                        - 简化您的指令
                        - 检查数据是否包含所需信息
                        - 稍后再试
                        """)
                    elif isinstance(result, (pd.DataFrame, pd.Series)):
                        st.dataframe(result)
                        # 添加下载按钮
                        csv = result.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="下载结果为CSV",
                            data=csv,
                            file_name='analysis_result.csv',
                            mime='text/csv'
                        )
                    elif isinstance(result, str) and "```python" in result:
                        # 提取并显示代码
                        code_block = result.split("```python")[1].split("```")[0].strip()
                        st.success("AI生成的执行代码:")
                        st.code(code_block, language='python')
                        
                        # 添加执行按钮
                        if st.button("执行代码"):
                            try:
                                local_vars = {'df': ai_dataframes[0]} if ai_dataframes else {}
                                exec(code_block, globals(), local_vars)
                                if 'result' in local_vars:
                                    st.dataframe(local_vars['result'])
                                else:
                                    st.warning("代码执行完成但未生成result变量")
                            except Exception as e:
                                st.error(f"代码执行错误: {str(e)}")
                    elif isinstance(result, (str, int, float)):
                        st.metric(label="结果", value=result)
                    else:
                        st.warning("AI返回了非标准格式的响应:")
                        st.write(result)

    except Exception as e:
        st.error(f"处理文件时发生意外错误: {e}")

st.markdown("---")
st.markdown("由 DeepSeek, PandasAI, and Streamlit 驱动")