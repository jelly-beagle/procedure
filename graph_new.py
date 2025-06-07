import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import networkx as nx # Import networkx for graph visualization
import matplotlib.pyplot as plt # Import matplotlib for plotting

# 忽略 openpyxl 相关的 UserWarning，因为读取 Excel 文件时可能产生
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# --- Configure Matplotlib for Chinese characters (if needed for titles/labels) ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # Fixes negative sign display

# ==================== 配置参数 ====================
INDEX_FILE = "index.xlsx"  # 项目股票代码列表文件路径
NEWS_DATA_PATH_TEMPLATE = r"D:\news\processed_results\dedup_processed_{year}_part{num}.csv" # 新闻数据文件模板
OUTPUT_BASE_DIR = r"D:\news_cooccurrence_graphs" # 新闻共现图的总输出目录

# 月度图配置 (保留，但默认不运行构建方法以节省资源)
MONTHLY_OUTPUT_SUBDIR = "monthly"
MONTHLY_START_YEAR_MONTH = "2018-12"
MONTHLY_END_YEAR_MONTH = "2023-12"

# 每周图配置 (保留，但默认不运行构建方法以节省资源)
WEEKLY_OUTPUT_SUBDIR = "weekly"
WEEKLY_START_DATE = "2018-12-24"
WEEKLY_END_DATE = "2023-12-31"
WEEKLY_WINDOW_SIZE_DAYS = 7
WEEKLY_STEP_DAYS = 7

# --- 新增：每日图配置 ---
DAILY_OUTPUT_SUBDIR = "daily" # 每日图数据保存的子目录名
DAILY_START_DATE = "2018-12-24" # 每日图处理的起始日期 (YYYY-MM-DD)，即第一个窗口的结束日期
DAILY_END_DATE = "2023-12-31" # 每日图处理的结束日期 (YYYY-MM-DD)
DAILY_WINDOW_SIZE_DAYS = 14 # 每日图的共现计算窗口大小 (t-13 到 t，共 14 天)
DAILY_STEP_DAYS = 1 # 每日更新，步长为 1 天

# ==================== 核心处理器 ====================
# 负责新闻数据的读取、预处理、共现权重计算、图构建和保存
class NewsGraphBuilder:
    # 类的初始化方法
    def __init__(self, index_file, news_data_template, output_base_dir,
                 # Visualization flags
                 plot_monthly_network=False, # 默认不绘制月度图
                 plot_weekly_network=False, # 默认不绘制每周图
                 plot_daily_network=True,   # 新增标志：是否生成每日网络的图可视化
                 filter_isolated_nodes_viz=True): # 新增标志：可视化时是否过滤孤立节点


        # 将传入的参数存储为类的实例属性
        self.index_file = index_file
        self.news_data_template = news_data_template
        self.output_base_dir = output_base_dir
        self.plot_monthly_network = plot_monthly_network
        self.plot_weekly_network = plot_weekly_network # 保存每周网络可视化标志
        self.plot_daily_network = plot_daily_network # 保存每日网络可视化标志
        self.filter_isolated_nodes_viz = filter_isolated_nodes_viz # 保存可视化过滤标志


        # 加载目标股票列表
        self.target_stocks = self._load_target_stocks()
        if not self.target_stocks:
            print("错误：未加载到目标股票列表，无法继续。")
            self.is_initialized = False # 初始化失败标志
            return # 如果加载失败，停止初始化过程

        print(f"\n已加载 {len(self.target_stocks)} 个目标股票代码。")
        # 将目标股票列表排序，作为图节点的固定顺序 (索引 0, 1, 2... 对应这个排序后的列表)
        self.nodes = sorted(list(self.target_stocks))
        # 创建股票代码到节点索引的映射字典，方便查找
        self.node2id = {node: i for i, node in enumerate(self.nodes)}
        self.is_initialized = True # 初始化成功标志

        # 确保总输出目录及各子目录存在
        base_output_path = Path(self.output_base_dir)
        base_output_path.mkdir(parents=True, exist_ok=True)
        Path(base_output_path / MONTHLY_OUTPUT_SUBDIR).mkdir(parents=True, exist_ok=True)
        Path(base_output_path / WEEKLY_OUTPUT_SUBDIR).mkdir(parents=True, exist_ok=True)
        Path(base_output_path / DAILY_OUTPUT_SUBDIR).mkdir(parents=True, exist_ok=True)


        # 预读取并过滤新闻数据到内存
        print("\n--- 正在预读取并过滤新闻数据 ---")
        self.all_news_df = self._read_and_preprocess_news_data()
        if self.all_news_df.empty:
            print("错误：未能加载或过滤任何新闻数据，无法继续。")
            self.is_initialized = False


    # --- 辅助方法：加载目标股票列表 ---
    def _load_target_stocks(self):
        """从Excel文件加载目标股票代码"""
        try:
            # 移除 header=None，让 Pandas 默认将第一行作为列头读取
            df = pd.read_excel(self.index_file)
            # 提取第一列数据 (现在第一行是列头，数据从第二行开始读)
            # 使用 df.iloc[:, 0] 仍然是安全的，它会选择第一列的所有数据行
            # .astype(str).str.zfill(6).tolist() 等后续操作不变
            symbols = set(df.iloc[:, 0].astype(str).str.zfill(6).tolist())
            return symbols # 返回目标股票代码集合
        except FileNotFoundError:
            print(f"错误：指数文件未找到于 {self.index_file}")
            return set() # 返回空集合
        except Exception as e:
            # 处理读取或处理文件时的其他异常
            print(f"加载指数文件时出错: {e}")
            return set() # 返回空集合


    # --- 辅助方法：读取和预处理新闻数据 ---
    def _read_and_preprocess_news_data(self):
        """
        读取指定年份范围内的所有新闻文件，过滤并预处理。
        返回包含 Reptime, codesnums, target_newscodes (filtered list of target stocks) 的DataFrame.
        """
        all_dfs = [] # 用于存储从各个文件读取并处理后的 DataFrame

        # 确定需要处理的年份范围，覆盖所有图的起始/结束年份
        # 并且需要包含构建第一个每日图窗口所需的足够早的数据
        start_year_config = int(min(MONTHLY_START_YEAR_MONTH[:4], WEEKLY_START_DATE[:4], DAILY_START_DATE[:4]))
        end_year_config = int(max(MONTHLY_END_YEAR_MONTH[:4], WEEKLY_END_DATE[:4], DAILY_END_DATE[:4]))

        # 计算需要读取数据的最早日期，以便覆盖每日图的第一个窗口
        earliest_date_needed_for_daily = datetime.strptime(DAILY_START_DATE, "%Y-%m-%d") - timedelta(days=DAILY_WINDOW_SIZE_DAYS - 1)
        # 确保新闻数据读取的起始年份足够早，包含最早需要的日期所在的年份
        start_year_actual = min(start_year_config, earliest_date_needed_for_daily.year)

        years_to_process = range(start_year_actual, end_year_config + 1)

        print(f"新闻数据实际读取年份范围: {start_year_actual} - {end_year_config}")


        # 遍历需要处理的每一年
        for year in tqdm(years_to_process, desc="Reading News Files"):
            part = 0 # 文件编号通常从 0 或 1 开始，根据您的文件命名调整
            # 设置一个合理的 max_part_number，避免因文件缺失导致无限循环或长时间等待
            # 假设一个年份最多有 10 个 part 文件，您可以根据实际情况调整
            max_part_number = 10 # Adjust based on your actual file structure

            while part <= max_part_number: # 尝试从 part 0 到 max_part_number
                # 尝试构建当前文件的完整路径
                file_path = self.news_data_template.format(year=year, num=part)

                # 检查文件是否存在
                if not os.path.exists(file_path):
                    # 如果当前 part 文件不存在，尝试下一个 part
                    part += 1
                    continue # 跳过当前文件，尝试下一个 part

                # 如果文件存在，尝试读取和处理
                try:
                    # 使用 pd.read_csv 读取，sep 根据您的文件格式设置 (之前确定是 ',')
                    df = pd.read_csv(file_path, sep=',', usecols=['Reptime', 'codesnums', 'newscodes'], dtype=str)
                    # 移除 Reptime 或 newscodes 列中包含空值的行
                    df.dropna(subset=['Reptime', 'newscodes'], inplace=True)

                    # 将 Reptime 列转换为 datetime 对象，errors='coerce' 将无法转换的值设为 NaT (Not a Time)
                    df['Reptime'] = pd.to_datetime(df['Reptime'], errors='coerce')
                    # 移除日期转换失败 (NaT) 的行
                    df.dropna(subset=['Reptime'], inplace=True)

                    # 定义一个内部函数用于过滤和清洗 newscodes
                    def filter_and_clean_codes(codes_str):
                        if pd.isna(codes_str): return [] # 如果 newscodes 是空值，返回空列表
                        # 将字符串转换为列表，注意可能的其他分隔符或格式问题，这里假定是空格分隔
                        codes = [code.strip() for code in codes_str.split()] # 按空格分割股票代码并去除首尾空白
                        # 填充前导零并过滤，只保留在 self.target_stocks 集合中的股票代码
                        filtered_codes = [code.zfill(6) for code in codes if code.zfill(6) in self.target_stocks]
                        # 返回去重并排序后的目标股票列表
                        return sorted(list(set(filtered_codes)))

                    # 对 df['newscodes'] 列应用 filter_and_clean_codes 函数，结果存储在新列 'target_newscodes' 中
                    df['target_newscodes'] = df['newscodes'].apply(filter_and_clean_codes)

                    # 移除没有目标股票共现的新闻 (target_newscodes 列表长度小于 2，因为至少需要 2 个股票才能形成一条边)
                    df = df[df['target_newscodes'].apply(len) > 1].copy() # 使用 copy() 避免 SettingWithCopyWarning

                    # 将 codesnums 列转换为数字，处理非数字情况为 NaN 并填充为 1，然后转为整数
                    df['codesnums'] = pd.to_numeric(df['codesnums'], errors='coerce').fillna(1).astype(int)

                    all_dfs.append(df) # 将处理后的 DataFrame 添加到列表中

                except Exception as e:
                    # 如果读取或处理文件时发生异常，打印警告并跳过当前文件
                    print(f"\nError reading or processing file {file_path} (part {part}, year {year}): {e}，跳过。")
                    part += 1 # 尝试下一个 part 文件
                    continue # 继续外层 while 循环

                # 如果成功读取并处理了一个文件，递增 part 并继续尝试下一个文件
                part += 1

            # 如果内层 while 循环因为 part > max_part_number 而退出，则尝试下一个年份

        # 检查是否成功读取或预处理了任何新闻文件
        if not all_dfs:
            print("未能加载或预处理任何新闻文件。")
            return pd.DataFrame() # 返回空 DataFrame

        # 将所有处理后的 DataFrame 合并成一个
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # 按新闻发布日期排序，方便按时间窗口切片处理
        combined_df.sort_values(by='Reptime', inplace=True)
        print(f"预处理并过滤后总新闻条数: {len(combined_df)}")
        return combined_df # 返回合并后的 DataFrame


    def _process_cooccurrence(self, news_df_window):
        """
        处理一个时间窗口内的新闻数据，计算股票对的共现权重。
        """
        cooccurrence_weights = defaultdict(float)

        if news_df_window.empty:
            return cooccurrence_weights

        # 在处理窗口内新闻时使用 tqdm 显示进度，这里不再使用 tqdm，因为它在每日循环内，会生成太多进度条
        # 直接循环处理
        for _, row in news_df_window.iterrows(): # No tqdm here
            target_codes_in_news = row['target_newscodes']
            num_target_codes = len(target_codes_in_news)

            if num_target_codes > 1:
                # Weight increment based on the number of target stocks in this news item
                weight_increment = 1.0 / (num_target_codes - 1)

                for code1, code2 in combinations(target_codes_in_news, 2):
                    ordered_pair = tuple(sorted((code1, code2)))
                    cooccurrence_weights[ordered_pair] += weight_increment

        return cooccurrence_weights


    def _build_and_save_graph(self, cooccurrence_weights, period_name, output_subdir):
        """
        将共现权重构建为图，并保存为包含节点列表的 PyTorch .pt 文件。
        """
        edge_index_list = []
        edge_weight_list = []

        # 将 cooccurrence_weights 转换为 DataFrame 方便处理和排序
        if cooccurrence_weights:
            weights_df = pd.DataFrame(list(cooccurrence_weights.items()), columns=['pair', 'weight'])
            # 按权重降序排序，以便可视化时可以选择最重要的边（如果需要）
            weights_df = weights_df.sort_values(by='weight', ascending=False)

            for index, row in weights_df.iterrows():
                 stock1, stock2 = row['pair']
                 weight = row['weight']

                 if stock1 in self.node2id and stock2 in self.node2id:
                    idx1 = self.node2id[stock1]
                    idx2 = self.node2id[stock2]

                    # 避免自环
                    if idx1 != idx2:
                        edge_index_list.append([idx1, idx2])
                        edge_weight_list.append(weight)
                        edge_index_list.append([idx2, idx1]) # Add reverse edge for undirected graph
                        edge_weight_list.append(weight)
        else:
             # 如果没有 cooccurrence_weights， edge_index_list 和 edge_weight_list 保持为空
             pass # The check below handles the empty case


        if not edge_index_list:
            #print(f"\n警告：时间段 {period_name} 没有生成任何边。将保存空的边信息。") # Too many warnings for daily empty graphs
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_weight_tensor = torch.empty((0,), dtype=torch.float)
        else:
            edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_weight_tensor = torch.tensor(edge_weight_list, dtype=torch.float)

        graph_data = {
            'nodes': self.nodes,
            'edge_index': edge_index_tensor,
            'edge_weight': edge_weight_tensor
        }

        output_path = Path(self.output_base_dir) / output_subdir / f"graph_cooccurrence_{period_name}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.save(graph_data, output_path)
            #print(f"Saved graph data for period {period_name} to {output_path} (Edges: {edge_index_tensor.shape[1] // 2} undirected edges)") # Too verbose for daily
            if edge_index_tensor.shape[1] > 0:
                print(f"  Saved graph for {period_name} (Edges: {edge_index_tensor.shape[1] // 2})")
            # else: print(f"  Saved empty graph for {period_name}")
            return graph_data # Return the saved data for potential visualization
        except Exception as e:
            print(f"\nError saving PyTorch file {output_path}: {e}")
            return None


    def _visualize_cooccurrence_network(self, graph_data, period_name, output_subdir, filter_isolated=False):
        """
        可视化新闻共现网络图。
        Args:
            graph_data (dict): Dictionary containing 'nodes', 'edge_index', 'edge_weight'.
            period_name (str): Time period name for the title and filename.
            output_subdir (str): Subdirectory for saving plots (e.g., "monthly", "daily").
            filter_isolated (bool): Whether to filter out isolated nodes (degree 0) from visualization.
        """
        if graph_data is None or graph_data['edge_index'].shape[1] == 0:
            print(f"时间段 {period_name} 没有边，跳过可视化。")
            # Optional: save a placeholder image indicating no graph/edges
            # self._save_empty_viz_placeholder(period_name, output_subdir, filter_isolated)
            return

        nodes = graph_data['nodes']
        edge_index = graph_data['edge_index'].cpu().numpy()
        edge_weight = graph_data['edge_weight'].cpu().numpy()

        G = nx.Graph()
        # Add nodes and edges from the graph_data
        # Using add_weighted_edges_from is efficient
        # Prepare list of tuples: (u_stock, v_stock, weight)
        weighted_edge_list = []
        for i in range(edge_index.shape[1]):
            u_idx, v_idx = edge_index[:, i]
            u_stock = nodes[u_idx]
            v_stock = nodes[v_idx]
            weight = edge_weight[i]
            weighted_edge_list.append((u_stock, v_stock, weight))

        G.add_weighted_edges_from(weighted_edge_list)

        # Remove self-loops (should already be handled in _build_and_save_graph but as a safety)
        G.remove_edges_from(nx.selfloop_edges(G))


        # --- Filter isolated nodes if requested ---
        title_suffix = ""
        if filter_isolated:
            initial_node_count = G.number_of_nodes()
            isolated_nodes = list(nx.isolates(G)) # Get a list of isolated nodes
            G.remove_nodes_from(isolated_nodes) # Remove them from the graph
            title_suffix = " (已过滤孤立节点)"

            if G.number_of_nodes() == 0:
                 #print(f"时间段 {period_name} 过滤孤立节点后没有节点，跳过可视化。") # Too many printouts for daily
                 return # Skip visualization if no nodes are left


        # print(f"\n可视化时间段 {period_name} 的共现网络图 ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)...") # Too verbose for daily

        plt.figure(figsize=(10, 10)) # Smaller figure size for potentially many daily plots

        # --- Layout ---
        # Use spring_layout only on the (potentially filtered) graph
        # Use a seed for reproducibility if plotting the same graph multiple times and not filtering
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42 if not filter_isolated else None)


        # --- Edge Drawing ---
        weights = np.array([d['weight'] for u, v, d in G.edges(data=True)])
        if weights.size > 0:
            min_w, max_w = weights.min(), weights.max()
            if max_w > min_w:
                 scaled_weights = 0.5 + 2.5 * (weights - min_w) / (max_w - min_w) # Scale to range [0.5, 3]
            else:
                 scaled_weights = np.full_like(weights, 1.0)

            nx.draw_networkx_edges(G, pos, width=scaled_weights, alpha=0.7, edge_color='gray')
        # else: no edges to draw


        # --- Node Drawing ---
        degrees = dict(G.degree())
        if degrees:
            max_degree = max(degrees.values())
            if max_degree > 0:
                # Scale node size based on degree in the (potentially filtered) graph
                node_size = [100 + 300 * (degrees[node] / max_degree) for node in G.nodes()] # Scale to [100, 400]
            else:
                 node_size = 200 # Default size if all degrees are zero (shouldn't happen if filtering isolated)
                 node_size = [node_size] * G.number_of_nodes()
        else:
             node_size = []


        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', alpha=0.9)

        # --- Labels ---
        # No labels drawn by default for clarity with many nodes
        nx.draw_networkx_labels(G, pos, font_size=8, labels=None)


        # --- Title and Saving ---
        title = f"新闻共现网络图 - {period_name}{title_suffix}"
        plt.title(title)

        plt.axis('off') # Hide axes
        plt.tight_layout() # Adjust layout

        viz_output_dir = Path(self.output_base_dir) / output_subdir / "network_visualizations"
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        save_path = viz_output_dir / f"network_{period_name}.png"

        try:
            plt.savefig(save_path, dpi=200, bbox_inches='tight') # Lower DPI for potentially many daily plots
            # print(f"Saved network graph to: {save_path}") # Too verbose for daily
        except Exception as e:
            print(f"\nError saving network graph {save_path}: {e}")
        plt.close()

    # Optional: Helper to save a placeholder if visualization is skipped (e.g., no edges)
    # def _save_empty_viz_placeholder(self, period_name, output_subdir, filter_isolated):
    #     viz_output_dir = Path(self.output_base_dir) / output_subdir / "network_visualizations"
    #     viz_output_dir.mkdir(parents=True, exist_ok=True)
    #     save_path = viz_output_dir / f"network_{period_name}.png"
    #     plt.figure(figsize=(10, 10))
    #     title_suffix = " (已过滤孤立节点)" if filter_isolated else ""
    #     plt.title(f"新闻共现网络图 - {period_name}{title_suffix}\n(无边或节点)")
    #     plt.text(0.5, 0.5, "No edges or nodes after filtering.", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    #     plt.axis('off')
    #     plt.savefig(save_path, dpi=100, bbox_inches='tight')
    #     plt.close()


    def build_monthly_graphs(self):
        """构建并保存月度新闻共现图"""
        if not self.is_initialized or self.all_news_df.empty:
            print("构建月度图失败：初始化未完成或无新闻数据。")
            return

        print("\n--- 开始构建月度新闻共现图 ---")

        start_date = datetime.strptime(MONTHLY_START_YEAR_MONTH + "-01", "%Y-%m-%d")
        end_date = datetime.strptime(MONTHLY_END_YEAR_MONTH + "-01", "%Y-%m-%d")
        if end_date.month == 12:
            end_date_inclusive = end_date.replace(year=end_date.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end_date_inclusive = end_date.replace(month=end_date.month + 1, day=1) - timedelta(days=1)

        # Ensure we only process data within the defined range
        news_df_processed_range = self.all_news_df[
            (self.all_news_df['Reptime'] >= start_date) &
            (self.all_news_df['Reptime'] <= end_date_inclusive)
        ].copy()

        current_month_start = start_date.replace(day=1) # Start from the beginning of the first month

        # Iterate while the current month's start date is less than or equal to the intended end month's start date
        end_month_start = datetime.strptime(MONTHLY_END_YEAR_MONTH + "-01", "%Y-%m-%d")
        while current_month_start <= end_month_start:
            year_month = current_month_start.strftime("%Y-%m")
            period_name = year_month # 用于文件名

            month_start = current_month_start
            if current_month_start.month == 12:
                month_end = current_month_start.replace(year=current_month_start.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                month_end = current_month_start.replace(month=current_month_start.month + 1, day=1) - timedelta(days=1)

            # Select news for the current month from the pre-processed data
            news_df_window = news_df_processed_range[
                (news_df_processed_range['Reptime'] >= month_start) &
                (news_df_processed_range['Reptime'] <= month_end)
            ].copy()

            print(f"\n处理月度窗口: {year_month} (新闻条数: {len(news_df_window)})")

            cooccurrence_weights = self._process_cooccurrence(news_df_window)
            saved_graph_data = self._build_and_save_graph(cooccurrence_weights, period_name, MONTHLY_OUTPUT_SUBDIR)

            # Visualize the monthly graph (if flag is True)
            if self.plot_monthly_network and saved_graph_data is not None:
                # Pass filter_isolated_nodes_viz to the visualization function
                self._visualize_cooccurrence_network(saved_graph_data, period_name, MONTHLY_OUTPUT_SUBDIR,
                                                     filter_isolated=self.filter_isolated_nodes_viz)

            # Move to the next month
            if current_month_start.month == 12:
                current_month_start = current_month_start.replace(year=current_month_start.year + 1, month=1, day=1)
            else:
                current_month_start = current_month_start.replace(month=current_month_start.month + 1, day=1)

        print("\n--- 月度新闻共现图构建完成 ---")


    def build_weekly_graphs(self):
        """构建并保存每周新闻共现图 (7天滑动，步长7天)"""
        if not self.is_initialized or self.all_news_df.empty:
            print("构建每周图失败：初始化未完成或无新闻数据。")
            return

        print("\n--- 开始构建每周新闻共现图 ---")

        current_window_start = datetime.strptime(WEEKLY_START_DATE, "%Y-%m-%d")
        end_process_date = datetime.strptime(WEEKLY_END_DATE, "%Y-%m-%d")

        # Ensure we only process data within the defined range
        # Need data up to the end date of the *last possible window* that starts by end_process_date
        last_possible_window_end = end_process_date + timedelta(days=WEEKLY_WINDOW_SIZE_DAYS - 1)
        news_df_processed_range = self.all_news_df[
            (self.all_news_df['Reptime'] >= current_window_start) &
            (self.all_news_df['Reptime'] <= last_possible_window_end)
        ].copy()


        while current_window_start <= end_process_date:
            current_window_end = current_window_start + timedelta(days=WEEKLY_WINDOW_SIZE_DAYS - 1)

            # Only process if the window has some overlap with the overall processing range
            # and the window start is not after the overall end date
            if current_window_start > end_process_date:
                 print("Window start date is after the overall end date. Stopping weekly processing.")
                 break # Stop processing if the window has moved past the end date

            period_name = f"{current_window_start.strftime('%Y%m%d')}_{current_window_end.strftime('%Y%m%d')}" # 用于文件名

            # Select news for the current window
            news_df_window = news_df_processed_range[
                (news_df_processed_range['Reptime'] >= current_window_start) &
                (news_df_processed_range['Reptime'] <= current_window_end)
            ].copy()

            print(f"\n处理每周窗口: {current_window_start.strftime('%Y-%m-%d')} to {current_window_end.strftime('%Y-%m-%d')} (新闻条数: {len(news_df_window)})")

            if news_df_window.empty:
                 print("Warning: No news data in the current weekly window, skipping.")
                 pass # Continue to the next window
            else:
                cooccurrence_weights = self._process_cooccurrence(news_df_window)
                saved_graph_data = self._build_and_save_graph(cooccurrence_weights, period_name, WEEKLY_OUTPUT_SUBDIR)
                # For weekly graphs, visualize if flag is True
                if self.plot_weekly_network and saved_graph_data is not None:
                    # Pass filter_isolated_nodes_viz to the visualization function
                    self._visualize_cooccurrence_network(saved_graph_data, period_name, WEEKLY_OUTPUT_SUBDIR,
                                                         filter_isolated=self.filter_isolated_nodes_viz)


            # Move to the next window's start date (step)
            current_window_start += timedelta(days=WEEKLY_STEP_DAYS)

        print("\n--- 每周新闻共现图构建完成 ---")


    # --- 新增：构建每日新闻共现图 ---
    def build_daily_graphs(self):
        """构建并保存每日新闻共现图 (14天窗口，步长1天)"""
        if not self.is_initialized or self.all_news_df.empty:
            print("构建每日图失败：初始化未完成或无新闻数据。")
            return

        print("\n--- 开始构建每日新闻共现图 ---")

        # 确定需要处理的日期范围
        start_process_date = datetime.strptime(DAILY_START_DATE, "%Y-%m-%d")
        end_process_date = datetime.strptime(DAILY_END_DATE, "%Y-%m-%d")

        # 确保预加载的数据范围包含第一个窗口的起始日期
        earliest_date_needed = start_process_date - timedelta(days=DAILY_WINDOW_SIZE_DAYS - 1)

        # 筛选出需要处理日期范围内的所有新闻数据（为了高效切片）
        # 需要包含到最后一个窗口的结束日期 (end_process_date)
        news_df_processed_range = self.all_news_df[
            (self.all_news_df['Reptime'] >= earliest_date_needed) &
            (self.all_news_df['Reptime'] <= end_process_date) # 只需包含到结束日期即可，因为窗口结束日期是 Daily_END_DATE
        ].copy()

        # 使用 tqdm 显示每日处理进度
        # current_day 从指定的开始日期开始
        for current_day in tqdm(pd.date_range(start=start_process_date, end=end_process_date, freq='D'), desc="Processing Daily Graphs"):
            # 计算当前窗口的开始日期 (current_day - 13 天) 和结束日期 (current_day)
            window_start_date = current_day - timedelta(days=DAILY_WINDOW_SIZE_DAYS - 1)
            window_end_date = current_day # 窗口结束日期就是当前处理的日期 t

            # 构建用于文件命名的时间段字符串 (YYYYMMDD)
            period_name = current_day.strftime('%Y%m%d')
            output_file_path = Path(self.output_base_dir) / DAILY_OUTPUT_SUBDIR / f"graph_cooccurrence_{period_name}.pt"

            # 可选：如果文件已存在，可以跳过处理
            if output_file_path.exists():
                # print(f"文件已存在：{output_file_path}，跳过。")
                continue # 跳过当前日期，处理下一个


            # 从筛选出的新闻数据中，选择当前 14 天窗口内的数据
            news_df_window = news_df_processed_range[
                (news_df_processed_range['Reptime'] >= window_start_date) &
                (news_df_processed_range['Reptime'] <= window_end_date)
            ].copy()

            # print(f"\n处理每日窗口: {window_start_date.strftime('%Y-%m-%d')} to {window_end_date.strftime('%Y-%m-%d')} (新闻条数: {len(news_df_window)})") # Too verbose for daily


            # 处理共现并构建图 (.pt 文件)
            cooccurrence_weights = self._process_cooccurrence(news_df_window)
            saved_graph_data = self._build_and_save_graph(cooccurrence_weights, period_name, DAILY_OUTPUT_SUBDIR)

            # 调用可视化方法 (如果可视化标志为 True)
            if self.plot_daily_network and saved_graph_data is not None:
                 # Pass filter_isolated_nodes_viz to the visualization function
                 self._visualize_cooccurrence_network(saved_graph_data, period_name, DAILY_OUTPUT_SUBDIR,
                                                      filter_isolated=self.filter_isolated_nodes_viz)


            # 每日循环自动由 pd.date_range 完成，步长为 1 天
            # current_day += timedelta(days=DAILY_STEP_DAYS) # Not needed with pd.date_range

        print("\n--- 每日新闻共现图构建完成 ---")


# ==================== 执行入口点 ====================
# 当脚本作为主程序运行时执行以下代码块
if __name__ == "__main__":
    print("--- 开始执行新闻共现图构建脚本 ---")

    # --- 图生成和可视化控制 ---
    # 设置为 True 会生成对应的图数据 (.pt 文件) 和网络可视化图片 (如果该类别的 plot_*_network 标志为 True)
    DO_BUILD_MONTHLY_GRAPHS = False
    DO_BUILD_WEEKLY_GRAPHS = False
    DO_BUILD_DAILY_GRAPHS = True # 新增：控制是否生成每日图

    # 控制是否为生成的图类别绘制网络可视化图片
    DO_PLOT_MONTHLY_NETWORK = False
    DO_PLOT_WEEKLY_NETWORK = False
    DO_PLOT_DAILY_NETWORK = True # 新增：控制是否绘制每日图的网络可视化

    # 控制可视化时是否过滤掉度为零的孤立节点
    FILTER_ISOLATED_NODES_VIZ = True


    # 创建 NewsGraphBuilder 类的实例
    builder = NewsGraphBuilder(
        index_file=INDEX_FILE, # 传递目标股票文件路径
        news_data_template=NEWS_DATA_PATH_TEMPLATE, # 传递新闻数据文件模板
        output_base_dir=OUTPUT_BASE_DIR, # 传递输出目录
        plot_monthly_network=DO_PLOT_MONTHLY_NETWORK, # 传递月度网络可视化标志
        plot_weekly_network=DO_PLOT_WEEKLY_NETWORK, # 传递每周网络可视化标志
        plot_daily_network=DO_PLOT_DAILY_NETWORK,   # 传递每日网络可视化标志
        filter_isolated_nodes_viz=FILTER_ISOLATED_NODES_VIZ # 传递可视化过滤标志
    )

    # 如果类初始化成功 (即成功加载了目标股票和新闻数据)
    if builder.is_initialized:
        # 根据标志调用相应的图构建方法
        if DO_BUILD_MONTHLY_GRAPHS:
             builder.build_monthly_graphs()

        if DO_BUILD_WEEKLY_GRAPHS:
             builder.build_weekly_graphs()

        if DO_BUILD_DAILY_GRAPHS:
             builder.build_daily_graphs()

    print("\n--- Script execution finished ---")