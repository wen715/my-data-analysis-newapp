import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import requests
import time
from pandasai.llm.base import LLM
from pandasai import SmartDatalake

# -------------------------------------------------------------------
# 模块一: 自定义的 AI 模型接口 (保持不变)
# -------------------------------------------------------------------
class DeepSeekLLM(LLM):
    """优化的DeepSeek LLM集成类"""
    def __init__(self, api_key: str, model: str = "deepseek-chat", temperature: float = 0.1):
        super().__init__()
        self.api_key = api_key; self.model = model; self.temperature = temperature; self.api_base = "https://api.deepseek.com/v1"
    def call(self, prompt: str, *args, **kwargs) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": [{"role": "user", "content": str(prompt)}], "temperature": self.temperature, "max_tokens": 4096}
        try:
            response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"API请求或网络发生错误: {e}"
    @property
    def type(self) -> str: return "deepseek-llm"

# -------------------------------------------------------------------
# 模块二: 最终版的数据回填函数（增加专家级清洗和调试）
# -------------------------------------------------------------------
def fill_template_final(template_df: pd.DataFrame, source_dfs: list, key_columns: list, columns_to_fill: list, 
                      template_path: str = None, inplace: bool = False) -> pd.DataFrame:
    """
    根据多个关键列匹配，对指定列进行求和并回填。
    参数:
        template_path: 模板文件路径，如果提供且inplace=True则直接修改该文件
        inplace: 是否直接修改模板文件
    """
    
    # 专家级数据清洗函数
    def expert_clean_df(df):
        df_clean = df.copy()
        for col in df_clean.columns:
            # 只对字符串类型执行字符串操作
            if pd.api.types.is_string_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].astype(str).str.strip()
            # 尝试转换为数值
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
            except:
                pass
        return df_clean

    cleaned_template_df = expert_clean_df(template_df)
    cleaned_source_dfs = [expert_clean_df(df) for df in source_dfs]

    # 改进关键列处理 - 保留原始格式用于匹配
    for df in [cleaned_template_df] + cleaned_source_dfs:
        for key in key_columns:
            if key in df.columns:
                # 保留原始值用于调试
                df['_original_'+key] = df[key]  
                # 标准化处理
                df[key] = df[key].fillna('N/A').astype(str).str.strip().str.lower().str.replace(r'\s+', '', regex=True)

    valid_sources = [df for df in cleaned_source_dfs if all(key in df.columns for key in key_columns)]
    if not valid_sources: raise ValueError("数据源文件中缺少部分或全部关键列。")
    
    combined_sources = pd.concat(valid_sources, ignore_index=True)
    
    # 改进聚合逻辑 - 保留文本列
    agg_functions = {}
    for col in columns_to_fill:
        if col in combined_sources.columns:
            if pd.api.types.is_numeric_dtype(combined_sources[col]):
                agg_functions[col] = 'sum'
            else:
                agg_functions[col] = 'first'  # 保留第一个非空值
    
    if not agg_functions: return cleaned_template_df
    
    # 识别并保留特殊行（如"项目"、"一"等）
    special_rows = combined_sources[
        combined_sources[key_columns[0]].astype(str).str.contains('项目|一|—', regex=True)
    ]
    
    # 执行常规聚合
    aggregated_source = combined_sources.groupby(key_columns, as_index=False).agg(agg_functions)
    
    # 合并特殊行和聚合结果
    aggregated_source = pd.concat([special_rows, aggregated_source]).drop_duplicates(subset=key_columns, keep='first')
    
    # --- 调试信息将直接显示 ---
    st.markdown("---")
    st.subheader("⚙️ 调试诊断信息 (请仔细比对以下两个表格)")
    st.write("**1. 模板文件关键列 (清洗后用于匹配的值):**")
    st.dataframe(cleaned_template_df[key_columns].head())
    st.write("**2. 源数据关键列 (清洗并聚合后用于匹配的值):**")
    st.dataframe(aggregated_source[key_columns].head())
    st.markdown("---")
    
    for key in key_columns:
        cleaned_template_df[key] = cleaned_template_df[key].astype(str)
        
    # 添加调试信息
    st.write("**合并前模板数据样本:**")
    st.dataframe(cleaned_template_df.head())
    st.write("**合并前分析结果样本:**")
    st.dataframe(aggregated_source.head())
    
    # 改进合并逻辑 - 处理非唯一键
    # 先为模板数据添加临时索引列
    cleaned_template_df['_temp_index'] = range(len(cleaned_template_df))
    
    # 执行合并，不验证一对一关系
    # 改进合并逻辑 - 添加调试信息
    st.write("**关键列匹配详情:**")
    st.write(f"模板数据关键列值: {cleaned_template_df[key_columns].head().to_dict()}")
    st.write(f"源数据关键列值: {aggregated_source[key_columns].head().to_dict()}")
    
    # 执行合并并显示未匹配记录
    merged_df = pd.merge(
        cleaned_template_df,
        aggregated_source,
        on=key_columns,
        how='left',
        suffixes=('', '_source'),
        indicator=True  # 添加合并标记
    )
    
    # 显示未匹配的记录
    unmatched = merged_df[merged_df['_merge'] == 'left_only']
    if not unmatched.empty:
        st.warning(f"发现 {len(unmatched)} 条未匹配记录")
        st.dataframe(unmatched[key_columns + ['_merge']])
    
    merged_df = merged_df.drop(columns=['_merge'])  # 移除合并标记
    
    # 按原始顺序恢复并删除临时索引
    merged_df = merged_df.sort_values('_temp_index').drop(columns=['_temp_index'])
    
    # 检查合并结果
    st.write("**合并后数据样本:**")
    st.dataframe(merged_df.head())

    for col in columns_to_fill:
        source_col_name = f"{col}_source"
        if source_col_name in merged_df.columns:
            # 改进的空白替换逻辑 - 放宽条件
            # 只要源数据有值且模板值为空/0/None/空字符串，就进行替换
            if pd.api.types.is_numeric_dtype(merged_df[col].dtype):
                mask_to_fill = (merged_df[col].isna()) | (merged_df[col] == 0) | (merged_df[col].astype(str) == "None") | (merged_df[col].astype(str) == "nan")
            else:
                mask_to_fill = (merged_df[col].isna()) | (merged_df[col] == "") | (merged_df[col].astype(str) == "None") | (merged_df[col].astype(str) == "nan")
            
            # 确保源数据有值时才替换
            mask_to_fill = mask_to_fill & (~merged_df[source_col_name].isna())
            
            # 执行替换
            merged_df[col] = np.where(
                mask_to_fill,
                merged_df[source_col_name],
                merged_df[col]
            )
            merged_df.drop(columns=[source_col_name], inplace=True)

    final_df = merged_df[template_df.columns]
    for col in final_df.select_dtypes(include=np.number).columns:
        final_df[col] = final_df[col].fillna(0)
    
    # 如果指定了模板路径且需要直接修改
    if template_path and inplace:
        try:
            # 获取原始文件路径
            original_path = os.path.abspath(template_path)
            # 创建临时副本(使用.xlsx扩展名)
            temp_path = os.path.splitext(original_path)[0] + "_temp.xlsx"
            
            # 写入修改后的数据到临时文件
            with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                final_df.to_excel(writer, index=False)
            
            # 替换原始文件
            if os.path.exists(original_path):
                os.remove(original_path)
            os.rename(temp_path, original_path)
            
            st.success(f"已成功修改本地文件: {original_path}")
        except Exception as e:
            st.error(f"直接修改文件失败: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    return final_df

# -------------------------------------------------------------------
# 主程序: Streamlit 界面逻辑 (简化版)
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="智能数据分析平台", page_icon="🧠", layout="wide")
    st.title("🧠 智能数据分析平台")

    with st.sidebar:
        st.header("🔑 API 密钥")
        api_key = st.text_input("DeepSeek API密钥", type="password", help="从DeepSeek官网获取API密钥")

    st.header("AI 智能分析")
    st.info("上传数据文件并用自然语言提问，AI会自动分析并可将结果填入指定模板。")
    
    # 文件上传区域
    col1, col2 = st.columns(2)
    with col1:
        data_files = st.file_uploader("上传数据文件", type=["xlsx", "xls"], accept_multiple_files=True)
    with col2:
        template_option = st.radio("模板文件来源", 
                                 ["上传模板文件", "指定本地文件路径"],
                                 help="选择直接修改本地文件或上传模板文件")
        
        if template_option == "上传模板文件":
            template_file = st.file_uploader("上传模板文件", type=["xlsx", "xls"])
            local_path = None
        else:
            local_path = st.text_input("本地模板文件路径", 
                                     placeholder="例如: e:/电工杯/templates/report_template.xlsx")
            template_file = None

    # 分析请求输入
    prompt = st.text_area("分析需求", height=100, 
                        placeholder="例如: 计算各产品的销售总额并填入模板")

    if st.button("开始分析", type="primary", use_container_width=True):
        if not api_key or not api_key.startswith("sk-"):
            st.warning("请输入有效的API密钥。")
        elif not data_files:
            st.warning("请上传至少一个数据文件。")
        elif not prompt.strip():
            st.warning("请输入您的分析需求。")
        else:
            with st.spinner("AI 正在分析中..."):
                try:
                    # 读取数据文件
                    dfs = [pd.read_excel(f) for f in data_files]
                    
                    # 初始化AI模型
                    llm = DeepSeekLLM(api_key=api_key)
                    lake = SmartDatalake(dfs, config={"llm": llm})
                    
                    # 执行分析
                    result = lake.chat(prompt)
                    
                    # 显示分析结果
                    st.subheader("分析结果")
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                        
                        # 处理模板文件
                        if template_option == "上传模板文件" and template_file:
                            template_df = pd.read_excel(template_file)
                            # 使用fill_template_final函数处理
                            result_df = fill_template_final(
                                template_df, [result], 
                                key_columns=template_df.columns.tolist()[:1],  # 使用第一列作为关键列
                                columns_to_fill=result.columns.tolist()
                            )
                            
                            # 提供下载
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                result_df.to_excel(writer, index=False)
                            st.download_button(
                                "下载填入模板的结果", 
                                output.getvalue(), 
                                "analysis_result_in_template.xlsx"
                            )
                        
                        # 处理本地文件路径
                        elif template_option == "指定本地文件路径" and local_path:
                            if os.path.exists(local_path):
                                template_df = pd.read_excel(local_path)
                                # 直接修改本地文件
                                result_df = fill_template_final(
                                    template_df, [result],
                                    key_columns=template_df.columns.tolist()[:1],
                                    columns_to_fill=result.columns.tolist(),
                                    template_path=local_path,
                                    inplace=True
                                )
                            else:
                                st.error("指定的本地文件路径不存在，请检查路径是否正确")
                    else:
                        st.write(result)
                        
                except Exception as e:
                    st.error(f"分析失败: {str(e)}")

if __name__ == "__main__":
    main()