import streamlit as st
import pandas as pd
import requests
import time
from pandasai import SmartDatalake
from pandasai.llm.base import LLM

# --------------------------------------------------------------------------
# 1. 自定义 DeepSeek LLM 类
# --------------------------------------------------------------------------
class DeepSeekLLM(LLM):
    """优化的DeepSeek LLM集成类"""
    def __init__(self, api_key: str, model: str = "deepseek-chat", temperature: float = 0.3):
        super().__init__()
        if not api_key or not isinstance(api_key, str):
            raise ValueError("无效的API密钥格式，请检查。")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.api_base = "https://api.deepseek.com/v1"
        self.max_retries = 3
        self.timeout = 60  # 延长超时时间以应对复杂查询
        self.last_response = None

    def call(self, prompt: str, *args, **kwargs) -> str:
        """兼容父类LLM的call方法签名，调用DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": str(prompt)}],
            "temperature": self.temperature,
            "max_tokens": 4096 # 增加max_tokens以获得更完整的代码或分析
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status() # 如果HTTP状态码是4xx或5xx，则抛出异常
                self.last_response = response.json()
                return self.last_response["choices"][0]["message"]["content"]
            
            except requests.exceptions.HTTPError as e:
                 # 特别处理402付款错误
                if e.response.status_code == 402:
                    raise Exception("API请求失败: 402 Client Error. 付款失败或账户余额不足。请检查您的DeepSeek账户。")
                # 其他HTTP错误
                if attempt == self.max_retries - 1:
                    raise Exception(f"API请求失败: {str(e)}")
                time.sleep(2 ** attempt)

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"网络或API请求失败: {str(e)}")
                time.sleep(2 ** attempt)
    
    @property
    def type(self) -> str:
        return "deepseek-llm"

# --------------------------------------------------------------------------
# 2. Streamlit 应用配置
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="🧠 智能数据分析专家",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------------------------------
# 3. 模板管理功能
# --------------------------------------------------------------------------
def manage_templates():
    """管理Excel模板文件"""
    if not os.path.exists("templates"):
        os.makedirs("templates")
    
    with st.sidebar:
        st.header("📁 模板管理")
        
        # 上传新模板
        new_template = st.file_uploader(
            "上传新模板", 
            type=["xlsx"],
            key="template_uploader"
        )
        
        if new_template:
            template_name = st.text_input("模板名称", value=new_template.name.split(".")[0])
            if st.button("保存模板"):
                template_path = os.path.join("templates", f"{template_name}.xlsx")
                with open(template_path, "wb") as f:
                    f.write(new_template.getbuffer())
                st.success(f"模板 '{template_name}' 保存成功!")

        # 模板列表
        templates = [f for f in os.listdir("templates") if f.endswith(".xlsx")]
        if templates:
            st.subheader("可用模板")
            selected_template = st.selectbox("选择模板", templates)
            return selected_template
        return None

# --------------------------------------------------------------------------
# 4. 主应用界面与逻辑
# --------------------------------------------------------------------------
def main():
    st.title("📊 智能Excel处理专家")
    st.markdown("""
    **使用说明:**
    1. 上传多个Excel数据文件
    2. 选择或上传模板文件
    3. 用自然语言描述处理需求
    4. 获取处理结果并下载
    """)
    st.markdown("---")

    # --- 初始化 ---
    if not os.path.exists("processed"):
        os.makedirs("processed")

    # --- 模板管理 ---
    selected_template = manage_templates()
    
    # --- API配置 ---
    api_key = st.sidebar.text_input(
        "DeepSeek API密钥",
        type="password",
        help="从DeepSeek官网获取API密钥"
    )
    
    if not api_key or not api_key.startswith("sk-"):
        st.warning("请输入有效的DeepSeek API密钥")
        st.stop()

    # --- 文件上传 ---
    uploaded_files = st.file_uploader(
        "上传一个或多个Excel文件",
        type=["xlsx", "xls"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("请上传Excel文件以开始分析。")
        st.stop()
    
    # --- 数据读取与预览 ---
    try:
        data_frames_dict = {file.name: pd.read_excel(file).fillna(0) for file in uploaded_files}
        data_frames = list(data_frames_dict.values())
        
        st.success(f"成功读取 {len(data_frames)} 个文件。所有空值已自动替换为0。")
        for name, df in data_frames_dict.items():
            with st.expander(f"预览文件: `{name}` (前5行)"):
                st.dataframe(df.head())
    except Exception as e:
        st.error(f"读取文件时发生错误: {e}")
        st.stop()
    
    # --- 用户输入分析请求 ---
    st.subheader("📝 分析指令")
    
    # 示例指令选择
    example_instructions = {
        "基础分析": "计算各产品的销售总额并排序",
        "高级计算": "计算各产品的月环比增长率，并筛选增长率>10%的产品",
        "数据合并": "按客户ID合并两个表格，计算每个客户的总消费金额",
        "条件筛选": "筛选出销售额大于100万且利润率高于20%的产品",
        "时间序列": "按周汇总销售额，并计算4周移动平均值"
    }
    
    selected_example = st.selectbox("选择示例指令", list(example_instructions.keys()))
    st.text_area(
        "或自定义您的分析需求",
        height=100,
        value=example_instructions[selected_example],
        key="analysis_prompt"
    )
    
    # 高级选项
    with st.expander("⚙️ 高级计算设置"):
        precision = st.slider("计算精度(小数位数)", 0, 6, 2)
        use_advanced = st.checkbox("启用高级分析模式", help="支持更复杂的计算和自定义函数")

    # --- 执行分析按钮 ---
    if st.button("🚀 开始分析", type="primary", use_container_width=True):
        if not analysis_prompt.strip():
            st.warning("请输入您的分析需求。")
            st.stop()

        if not api_key or not api_key.startswith("sk-"):
            st.error("请输入一个有效的DeepSeek API密钥。")
            st.stop()

        with st.spinner("🧠 AI正在思考中，请稍候..."):
            try:
                # 初始化 LLM
                llm = DeepSeekLLM(api_key=api_key)

                # 初始化 PandasAI 的 SmartDatalake
                # 它会自动处理多个DataFrame
                lake = SmartDatalake(
                    data_frames,
                    config={
                        "llm": llm,
                        "verbose": True,        # 在终端打印详细日志，方便调试
                        "enable_cache": False   # 禁用缓存，确保每次都是实时分析
                    }
                )
                
                # 直接将用户的自然语言问题传递给 chat 方法
                response = lake.chat(analysis_prompt)
                
                st.subheader("💡 分析结果")
                st.markdown("---")

                # --- 智能结果展示 ---
                if response is None:
                    st.warning("分析未能返回有效结果，请尝试调整您的问题。")
                
                elif isinstance(response, pd.DataFrame):
                    st.dataframe(response)
                    # 提供下载按钮
                    csv = response.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 下载结果 (CSV)",
                        data=csv,
                        file_name="analysis_result.csv",
                        mime="text/csv",
                    )
                
                elif isinstance(response, (str, int, float)):
                    st.markdown(f"### {response}")

                elif isinstance(response, dict) and response.get("type") == "plot":
                    st.image(response["value"], caption="分析图表")
                
                else:
                    # 对于其他未知类型的返回结果，直接以文本形式输出
                    st.text("未能识别的返回类型，原始输出如下：")
                    st.code(str(response))
                    
            except Exception as e:
                st.error(f"分析过程中发生严重错误：\n\n{e}")

# --------------------------------------------------------------------------
# 4. 启动应用
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()