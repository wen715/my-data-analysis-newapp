import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import requests
import time
from pandasai.llm.base import LLM
from pandasai import SmartDatalake
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re

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
    def type(self) -> str:
        return "deepseek-llm"

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
    if not valid_sources:
        raise ValueError("数据源文件中缺少部分或全部关键列。")
    
    combined_sources = pd.concat(valid_sources, ignore_index=True)
    
    # 改进聚合逻辑 - 处理工作簿6和工作簿2的特殊情况
    agg_functions = {}
    for col in columns_to_fill:
        if col in combined_sources.columns:
            if pd.api.types.is_numeric_dtype(combined_sources[col]):
                if len(source_dfs) == 2:  # 当有两个数据源时（工作簿6和工作簿2）
                    # 检查工作簿6和工作簿2的数据情况
                    df6_has_data = not source_dfs[0][col].isna().all()
                    df2_has_data = not source_dfs[1][col].isna().all()
                    
                    if df6_has_data and not df2_has_data:
                        # 工作簿6有数据而工作簿2没有数据，直接使用工作簿6的数据
                        agg_functions[col] = ('first', lambda x: x.iloc[0])
                    elif df6_has_data and df2_has_data:
                        # 两者都有数据，将两列数据相加
                        agg_functions[col] = ('sum', lambda x: x.sum())
                    else:
                        # 其他情况使用默认的sum
                        agg_functions[col] = 'sum'
                else:
                    # 默认情况使用sum
                    agg_functions[col] = 'sum'
            else:
                agg_functions[col] = 'first'  # 保留第一个非空值
    
    if not agg_functions:
        return cleaned_template_df
    
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
            
            # 检查模板文件是否存在
            if not os.path.exists(original_path):
                st.error(f"找不到模板文件: {original_path}")
                st.info("请检查文件路径是否正确")
                st.info("确保文件存在且具有读取权限")
                st.info("或者重新上传模板文件")
                return final_df
                
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
# 新增模块: 通用统计和预测功能
# -------------------------------------------------------------------
# 通用数据获取函数，支持链接和文件上传
def get_general_data(data_source, data_type, data_files=None):
    if data_type == "金属价格":
        if data_source:
            try:
                url = data_source
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                high_prices = []
                low_prices = []
                high_price_elements = soup.find_all('span', class_='high-price')
                low_price_elements = soup.find_all('span', class_='low-price')
                for high, low in zip(high_price_elements, low_price_elements):
                    high_prices.append(float(high.text.strip()))
                    low_prices.append(float(low.text.strip()))
                dates = pd.date_range(end=pd.Timestamp.now(), periods=len(high_prices))
                df = pd.DataFrame({
                    'Date': dates,
                    'High Price': high_prices,
                    'Low Price': low_prices
                })
                df['Average Price'] = (df['High Price'] + df['Low Price']) / 2
                return df
            except Exception as e:
                st.error(f"从链接获取数据失败: {str(e)}")
        elif data_files:
            try:
                dfs = [pd.read_excel(f) for f in data_files]
                df = pd.concat(dfs, ignore_index=True)
                if 'High Price' in df.columns and 'Low Price' in df.columns:
                    df['Average Price'] = (df['High Price'] + df['Low Price']) / 2
                    return df
                else:
                    st.error("上传文件缺少必要列（High Price 和 Low Price）")
            except Exception as e:
                st.error(f"从上传文件获取数据失败: {str(e)}")
    # 可以添加其他数据类型的获取逻辑
    st.error(f"暂不支持 {data_type} 数据类型的获取")
    return pd.DataFrame()

# 通用数据预测函数
def general_predict(df, target_column):
    model = SARIMAX(df[target_column], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit()
    future_steps_1day = 1
    future_steps_1month = 30
    future_steps_1quarter = 90
    future_steps_6months = 180
    forecast_1day = model_fit.get_forecast(steps=future_steps_1day)
    forecast_1month = model_fit.get_forecast(steps=future_steps_1month)
    forecast_1quarter = model_fit.get_forecast(steps=future_steps_1quarter)
    forecast_6months = model_fit.get_forecast(steps=future_steps_6months)
    return forecast_1day.predicted_mean, forecast_1month.predicted_mean, forecast_1quarter.predicted_mean, forecast_6months.predicted_mean

# 通用图表生成函数
def general_plot(df, forecasts, target_column, data_type):
    plt.figure(figsize=(15, 8))
    plt.plot(df['Date'], df[target_column], label=f'{data_type} 历史数据')
    future_dates = {}
    steps = [30, 90, 180]
    periods = ['1 Month', '1 Quarter', '6 Months']
    for i, forecast in enumerate(forecasts[1:]):
        future_dates[periods[i]] = pd.date_range(start=df['Date'].iloc[-1], periods=len(forecast) + 1, freq='D')[1:]
        plt.plot(future_dates[periods[i]], forecast, label=f'{periods[i]} Forecast', linestyle='--')

    changes = df[target_column].pct_change()
    for i in range(1, len(df)):
        if abs(changes.iloc[i]) > 0.1:
            event = find_event(df['Date'].iloc[i], data_type)
            plt.annotate(f'{event}',
                         xy=(df['Date'].iloc[i], df[target_column].iloc[i]),
                         xytext=(10, 10),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->'))

    plt.title(f'{data_type} 统计和预测')
    plt.xlabel('日期')
    plt.ylabel('数值')
    plt.legend()
    plt.grid(True)
    return plt

# 查找对应事件（示例，需根据实际情况调整）
def find_event(date, data_type):
    return f'{data_type} 事件于 {date.strftime("%Y-%m-%d")}'

# 可能的事件类型列表
def possible_general_events(data_type):
    if data_type == "金属价格":
        return [
            "全球经济形势变化",
            "金属供需关系改变",
            "政策法规调整",
            "自然灾害影响生产",
            "地缘政治冲突"
        ]
    # 可以添加其他数据类型的事件列表
    return [f"{data_type} 相关的市场变化", f"{data_type} 相关的政策调整"]

# -------------------------------------------------------------------
# 主程序: Streamlit 界面逻辑 (修改版)
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="智能数据分析平台", page_icon="🧠", layout="wide")
    st.title("🧠 智能数据分析平台")

    with st.sidebar:
        st.header("🔑 API 密钥")
        api_key = st.text_input("DeepSeek API密钥", type="password", help="从DeepSeek官网获取API密钥")

    st.header("AI 智能分析")
    st.info("上传数据文件并用自然语言提问，AI会自动分析并可将结果填入指定模板。")

    st.header("通用统计和预测")
    data_type = st.selectbox("选择数据类型", ["金属价格", "其他类型1", "其他类型2"])  # 可按需扩展
    data_source = st.text_input("输入数据来源链接", "https://example.chinametal.com")
    data_files = st.file_uploader("上传数据文件", accept_multiple_files=True)

    if st.button("开始统计和预测"):
        with st.spinner("正在获取数据和进行预测..."):
            try:
                df = get_general_data(data_source, data_type, data_files)
                if df.empty:
                    return
                if data_type == "金属价格":
                    target_column = "Average Price"
                else:
                    # 可根据不同数据类型指定目标列
                    target_column = "Value"  
                forecasts = general_predict(df, target_column)
                fig = general_plot(df, forecasts, target_column, data_type)
                st.pyplot(fig)
                st.write(f"第二天 {target_column} 预测:", forecasts[0].iloc[0])
                st.write(f"可能导致 {data_type} 大幅变化的事件类型:", possible_general_events(data_type))
                st.write("预测原理：使用SARIMAX时间序列模型进行预测，结合历史数据的趋势和季节性特征。")
            except Exception as e:
                st.error(f"统计和预测失败: {str(e)}")

    # 修复未定义变量问题
    data_files = st.file_uploader("上传分析用数据文件", accept_multiple_files=True)
    prompt = st.text_area("输入分析需求", "")
    template_option = st.selectbox("选择模板处理方式", ["无模板", "上传模板文件", "指定本地文件路径"])
    if template_option == "上传模板文件":
        template_file = st.file_uploader("上传模板文件", type=["xlsx", "xls"])
    elif template_option == "指定本地文件路径":
        local_path = st.text_input("输入本地模板文件路径")
    else:
        template_file = None
        local_path = None

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
                            
                            # 提供下载或直接保存到指定路径
                            if local_path:
                                try:
                                    # 检查路径是否存在并具有写入权限
                                    dir_path = os.path.dirname(local_path)
                                    if not os.path.exists(dir_path):
                                        os.makedirs(dir_path)
                                    
                                    # 创建备份文件
                                    backup_path = f"{local_path}.bak"
                                    if os.path.exists(local_path):
                                        os.rename(local_path, backup_path)
                                    
                                    # 写入结果到指定路径
                                    with pd.ExcelWriter(local_path, engine='openpyxl') as writer:
                                        result_df.to_excel(writer, index=False)
                                    st.success(f"结果已成功保存到: {local_path}")
                                    
                                    # 保留备份文件3秒后删除
                                    time.sleep(3)
                                    if os.path.exists(backup_path):
                                        os.remove(backup_path)
                                except Exception as e:
                                    st.error(f"保存到指定路径失败: {str(e)}")
                                    if os.path.exists(backup_path) and not os.path.exists(local_path):
                                        os.rename(backup_path, local_path)
                            else:
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
                            try:
                                # 规范化路径并检查存在性
                                normalized_path = os.path.normpath(local_path)
                                
                                # 检查路径是否存在
                                if not os.path.exists(normalized_path):
                                    # 尝试在templates目录下查找
                                    templates_path = os.path.join(os.path.dirname(__file__), 'templates', os.path.basename(normalized_path))
                                    if os.path.exists(templates_path):
                                        normalized_path = templates_path
                                    else:
                                        # 更友好的错误提示，包含可能的解决方案
                                        st.error(f"找不到模板文件: {normalized_path}\n\n解决方案:\n1. 请检查文件路径是否正确\n2. 确保文件已放入'templates'目录\n3. 或者重新上传模板文件")
                                        return
                                
                                # 检查文件扩展名
                                if not normalized_path.lower().endswith(('.xlsx', '.xls')):
                                    st.error(f"仅支持Excel文件(.xlsx/.xls)，当前文件: {os.path.splitext(normalized_path)[1]}")
                                    return
                                
                                template_df = pd.read_excel(normalized_path)
                                # 直接修改本地文件
                                result_df = fill_template_final(
                                    template_df, [result],
                                    key_columns=template_df.columns.tolist()[:1],
                                    columns_to_fill=result.columns.tolist(),
                                    template_path=normalized_path,
                                    inplace=True
                                )
                                st.success(f"成功处理文件: {normalized_path}")
                            except Exception as e:
                                st.error(f"处理文件时出错: {str(e)}\n请检查文件格式是否正确或是否被其他程序占用")
                    else:
                        st.write(result)
                        
                except Exception as e:
                    st.error(f"分析失败: {str(e)}")

if __name__ == "__main__":
    main()