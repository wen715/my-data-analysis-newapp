import streamlit as st
import pandas as pd
import requests
import time
import os
from builtins import ValueError  # 确保ValueError可用
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
            from builtins import ValueError
            raise ValueError("无效的API密钥格式，请检查。")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.api_base = "https://api.deepseek.com/v1"
        self.max_retries = 3
        self._timeout = 60  # 内部超时变量
        self.last_response = None

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._timeout = value

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
    
    # --- 数据预处理和预览 ---
    @st.cache_data(show_spinner="正在预处理数据...")
    def preprocess_data(uploaded_files):
        """统一数据预处理函数"""
        data_frames_dict = {}
        for file in uploaded_files:
            try:
                # 读取Excel文件
                df = pd.read_excel(file)
                
                # 处理所有类型的空值
                df = df.fillna(0)  # 处理NaN
                df = df.replace(r'^\s*$', 0, regex=True)  # 处理空字符串和纯空格
                df = df.replace('', 0)  # 处理空字符串
                
                # 确保数值类型正确
                for col in df.columns:
                    if df[col].dtype == object:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        except:
                            pass
                
                data_frames_dict[file.name] = df
                
            except Exception as e:
                st.error(f"处理文件 {file.name} 时出错: {str(e)}")
                st.stop()
        
        return data_frames_dict

    # 执行预处理
    data_frames_dict = preprocess_data(uploaded_files)
    data_frames = list(data_frames_dict.values())
    
    # 显示预处理结果
    st.success(f"✅ 成功预处理 {len(data_frames)} 个文件")
    with st.expander("🔍 数据质量报告"):
        for name, df in data_frames_dict.items():
            st.write(f"**文件**: {name}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("总行数", len(df))
                st.metric("数值列", len(df.select_dtypes(include=['number']).columns))
            with col2:
                st.metric("总列数", len(df.columns))
                st.metric("文本列", len(df.select_dtypes(include=['object']).columns))
            st.dataframe(df.head(3))
    
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
    analysis_prompt = st.text_area(
        "或自定义您的分析需求",
        height=100,
        value=example_instructions[selected_example],
        key="analysis_input"
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
                    
                    # 结果回填选项 - 更醒目的UI
                    st.markdown("---")
                    st.markdown("### 📤 结果输出方式")
                    with st.container():
                        col1, col2 = st.columns([3,7])
                        with col1:
                            fill_original = st.checkbox(
                                "🔁 回填到原文件", 
                                value=True,
                                help="将分析结果回填到原始Excel文件，保留所有格式和样式"
                            )
                        with col2:
                            if fill_original:
                                st.success("已启用回填功能 - 结果将保存回原始文件")
                            else:
                                st.info("将生成新的CSV文件")
                    st.markdown("---")
                    
                    if fill_original:
                        from openpyxl import load_workbook
                        from io import BytesIO
                        
                        with st.status("正在处理回填操作...", expanded=True) as status:
                            # 处理每个原始文件
                            for file_idx, file in enumerate(uploaded_files):
                                try:
                                    st.write(f"🔧 正在处理文件: {file.name}...")
                                    
                                    # 读取原始文件
                                    file.seek(0)
                                    wb = load_workbook(file)
                                    ws = wb.active
                                    
                                    # 获取原文件列名和格式
                                    original_columns = []
                                    column_formats = {}
                                    for col in range(1, ws.max_column+1):
                                        col_name = ws.cell(row=1, column=col).value
                                        original_columns.append(col_name)
                                        # 记录列格式(从第二行获取)
                                        if ws.max_row >= 2:
                                            sample_cell = ws.cell(row=2, column=col)
                                            column_formats[col_name] = {
                                                'number_format': sample_cell.number_format,
                                                'alignment': sample_cell.alignment,
                                                'font': sample_cell.font,
                                                'fill': sample_cell.fill,
                                                'border': sample_cell.border
                                            }
                                    
                                    # 准备结果数据
                                    result_df = response.copy()
                                    
                                    # 列名匹配检查
                                    missing_cols = [col for col in original_columns if col not in result_df.columns]
                                    if missing_cols:
                                        st.warning(f"⚠️ 原文件中有 {len(missing_cols)} 列在结果中不存在，将保留为空列")
                                    
                                    # 重新排列结果列以匹配原文件顺序
                                    result_df = result_df.reindex(columns=original_columns, fill_value=None)
                                    
                                    # 写入数据
                                    st.write("📝 正在回填数据...")
                                    for row_idx, row_data in enumerate(result_df.itertuples(index=False), start=2):
                                        for col_idx, value in enumerate(row_data, start=1):
                                            cell = ws.cell(row=row_idx, column=col_idx, value=value)
                                            # 应用原格式
                                            col_name = original_columns[col_idx-1]
                                            if col_name in column_formats:
                                                fmt = column_formats[col_name]
                                                cell.number_format = fmt['number_format']
                                                cell.alignment = fmt['alignment']
                                                cell.font = fmt['font']
                                                cell.fill = fmt['fill']
                                                cell.border = fmt['border']
                                    
                                    # 保存文件
                                    st.write("💾 正在保存文件...")
                                    output = BytesIO()
                                    wb.save(output)
                                    output.seek(0)
                                    
                                    status.update(label=f"✅ 文件 {file.name} 回填完成!", state="complete")
                                    
                                    # 提供下载
                                    st.download_button(
                                        label=f"📥 下载回填后的文件: {file.name}",
                                        data=output,
                                        file_name=f"updated_{file.name}",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        help="点击下载已回填结果的Excel文件，将保留原文件的所有格式和公式"
                                    )
                                    
                                    # 显示回填摘要
                                    with st.expander(f"🔍 回填摘要: {file.name}"):
                                        st.write(f"原文件列数: {len(original_columns)}")
                                        st.write(f"结果数据列数: {len(response.columns)}")
                                        st.write(f"匹配列数: {len(set(original_columns) & set(response.columns))}")
                                        st.dataframe(result_df.head(3))
                                    
                                except Exception as e:
                                    st.error(f"❌ 回填文件 {file.name} 时出错: {str(e)}")
                                    st.error("""
                                    常见解决方法:
                                    1. 检查原文件是否受保护或损坏
                                    2. 确保分析结果包含必要的列
                                    3. 尝试简化分析指令
                                    """)
                    else:
                        # 提供CSV下载
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