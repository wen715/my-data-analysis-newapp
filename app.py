import streamlit as st
import pandas as pd
import requests
import time
from pandasai import SmartDatalake
from pandasai.llm.base import LLM

class DeepSeekLLM(LLM):
    """优化的DeepSeek LLM集成类"""
    def __init__(self, api_key: str, model: str = "deepseek-chat", temperature: float = 0.3):
        super().__init__()
        if not api_key or not isinstance(api_key, str) or not api_key.startswith("sk-"):
            raise ValueError("无效的API密钥格式")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.api_base = "https://api.deepseek.com/v1"
        self.max_retries = 3
        self.timeout = 30
        self.last_response = None

    def call(self, prompt: str, *args, **kwargs) -> str:
        """兼容父类LLM的call方法签名"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 从kwargs中获取可选参数或使用默认值
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": str(prompt)}],
            "temperature": float(temperature),
            "max_tokens": kwargs.get("max_tokens", 3000)
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"API请求失败: {str(e)}")
                time.sleep(2 ** attempt)
    
    @property
    def type(self) -> str:
        return "deepseek-llm"

# 应用配置
st.set_page_config(
    page_title="智能数据分析专家",
    page_icon="🧠",
    layout="wide"
)

# 安全获取API密钥
def get_api_key():
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            return st.secrets["DEEPSEEK_API_KEY"]
        return None
    except:
        return None

# 主应用界面
def main():
    st.title("📊 智能数据分析专家")
    st.markdown("上传Excel文件，获取专业的分析结果")
    
    # 模型选择
    analysis_method = st.sidebar.radio(
        "选择分析引擎",
        options=["DeepSeek API", "本地开源模型", "基础数据分析"],
        help="根据您的资源选择分析方式"
    )
    
    # 初始化api_key变量
    api_key = None
    
    # DeepSeek API配置
    if analysis_method == "DeepSeek API":
        api_key = get_api_key() or st.text_input(
            "DeepSeek API密钥",
            type="password",
            help="从DeepSeek官网获取API密钥"
        )
        
        if not api_key or not api_key.startswith("sk-"):
            st.warning("请输入有效的DeepSeek API密钥")
            st.stop()
    
    # 文件上传
    uploaded_files = st.file_uploader(
        "上传Excel文件",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        help="支持多个文件同时上传"
    )
    
    if not uploaded_files:
        st.info("请上传Excel文件以开始分析")
        st.stop()
    
    # 读取数据
    try:
        data_frames = []
        for file in uploaded_files:
            df = pd.read_excel(file)
            data_frames.append(df)
            st.success(f"成功读取: {file.name} (行数: {len(df)})")
    except Exception as e:
        st.error(f"读取文件失败: {str(e)}")
        st.stop()
    
    # 分析请求
    analysis_prompt = st.text_area(
        "您的分析请求",
        height=150,
        placeholder="例如: 计算各产品的销售总额并排序"
    )
    
    if not analysis_prompt.strip():
        st.warning("请输入分析请求")
        st.stop()
    
    # 执行分析
    if st.button("开始分析", type="primary"):
        with st.spinner("正在分析数据..."):
            try:
                if analysis_method == "DeepSeek API":
                    llm = DeepSeekLLM(
                        api_key=api_key,
                        model="deepseek-chat",
                        temperature=0.3
                    )
                else:
                    st.warning("当前选择的分析模式不支持智能代码生成")
                    st.stop()
                # 增强型数据预处理和验证
                st.session_state.debug_info = []
                for i, df in enumerate(data_frames):
                    debug_msg = f"处理数据框 {i+1}/{len(data_frames)}: {len(df)}行×{len(df.columns)}列"
                    st.session_state.debug_info.append(debug_msg)
                    
                    # 验证并转换所有列
                    for col in df.columns:
                        # 记录列类型信息
                        col_type = str(df[col].dtype)
                        st.session_state.debug_info.append(f"列 '{col}': 类型 {col_type}")
                        
                        # 处理数值列
                        if pd.api.types.is_numeric_dtype(df[col]):
                            # 填充空值并确保可迭代
                            if df[col].isnull().any():
                                df[col] = df[col].fillna(0)
                            # 确保不是单个float值
                            if len(df[col]) == 1:
                                df[col] = pd.Series([df[col].iloc[0]], dtype='float64')
                        
                        # 处理非数值列
                        else:
                            # 确保所有值都是可迭代的
                            df[col] = df[col].astype(str)
                            if len(df[col]) == 1:
                                df[col] = pd.Series([df[col].iloc[0]])
                    
                    st.session_state.debug_info.append(f"处理后验证: {df.info()}")
                
                lake = SmartDatalake(
                    data_frames,
                    config={
                        "llm": llm,
                        "verbose": True,
                        "max_retries": 3,
                        "enable_cache": False  # 禁用缓存以避免类型问题
                    }
                )
                
                # 优化的系统提示
                system_prompt = f"""
您是一位资深数据分析师，请根据以下要求生成Python代码:
1. 只返回可执行的Python代码块
2. 最终结果存储在'result'变量中
3. 处理可能的数据异常
4. 使用pandas最佳实践

分析任务: {analysis_prompt}
数据: {len(data_frames)}个DataFrame
"""
                response = lake.chat(system_prompt)
                
                # 结果展示
                if "```python" in response:
                    code = response.split("```python")[1].split("```")[0]
                    st.subheader("分析代码")
                    st.code(code, language="python")
                    
                    # 执行代码
                    if st.button("执行代码"):
                        try:
                            local_vars = {"df": data_frames[0]}
                            exec(code, globals(), local_vars)
                            result = local_vars.get("result")
                            
                            if result is not None:
                                st.subheader("分析结果")
                                if isinstance(result, pd.DataFrame):
                                    st.dataframe(result)
                                    st.download_button(
                                        "下载结果",
                                        result.to_csv().encode("utf-8"),
                                        "result.csv"
                                    )
                                else:
                                    st.write(result)
                            else:
                                st.warning("代码执行未返回结果")
                        except Exception as e:
                            st.error(f"执行错误: {str(e)}")
                else:
                    st.warning("未获取到有效代码")
                    st.text(response)
                    
            except Exception as e:
                error_msg = str(e)
                st.error(f"分析失败: {error_msg}")
                
                if "配额不足" in error_msg or "402" in error_msg:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        **DeepSeek API解决方案:**
                        - [检查账户余额](https://platform.deepseek.com)
                        - [升级订阅计划](https://platform.deepseek.com/pricing)
                        - [申请教育优惠](https://platform.deepseek.com/edu)
                        """)
                    
                    with col2:
                        if st.button("自动切换到本地模型"):
                            st.session_state.analysis_method = "本地开源模型"
                            st.rerun()
                    
                    st.markdown("""
                    **或手动切换到:**
                    - 侧边栏选择"本地开源模型"
                    - 侧边栏选择"基础数据分析"
                    """)
                elif "HTTP 4" in error_msg:
                    st.info("建议检查API密钥是否正确或服务是否可用")

    # 本地开源模型选项
    elif analysis_method == "本地开源模型":
        with st.expander("🛠️ 本地模型配置", expanded=True):
            st.markdown("""
            **本地模型使用说明:**
            1. [下载安装Ollama](https://ollama.com)
            2. 运行模型: `ollama pull llama3`
            3. 安装Python包: `pip install llama-cpp-python`
            """)
            
            model_ready = st.checkbox("我已完成上述安装配置")
            
            if model_ready and st.button("启动本地模型分析"):
                with st.spinner("正在初始化本地模型..."):
                    try:
                        from llama_cpp import Llama
                        llm = Llama(
                            model_path="~/.ollama/models/blobs/llama3",
                            n_ctx=2048,
                            n_threads=4
                        )
                        st.session_state.local_llm = llm
                        st.success("本地模型已就绪!")
                        
                        # 简单问答测试
                        test_response = llm.create_chat_completion(
                            messages=[{"role": "user", "content": "你好"}]
                        )
                        st.info(f"模型响应测试: {test_response['choices'][0]['message']['content']}")
                    except Exception as e:
                        st.error(f"本地模型加载失败: {str(e)}")
                        st.info("请检查: 1) Ollama是否运行 2) 模型是否下载 3) 依赖是否安装")
    
    # 基础数据分析选项
    else:
        st.info("""
        **基础数据分析功能:**
        - 描述性统计
        - 数据可视化
        - 简单计算
        """)
        if st.button("显示基础分析"):
            try:
                stats = data_frames[0].describe()
                st.dataframe(stats)
            except Exception as e:
                st.error(f"基础分析失败: {str(e)}")

if __name__ == "__main__":
    main()
