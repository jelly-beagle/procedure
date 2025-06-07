import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# --- 配置Matplotlib支持中文 ---
# (保持你原来的配置)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置参数 ====================
INDEX_FILE = "index.xlsx"
DATA_ROOT = r"C:\Users\佚名\Desktop\raw数据集\资金流向"
OUTPUT_DIR = r"D:\graph_high_freq" # 修改输出目录名以反映阈值

Path(OUTPUT_DIR).mkdir(exist_ok=True) # 确保输出目录存在

START_DATE = "20190102"
END_DATE = "20231229"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"全局检测到的计算设备: {DEVICE}")

# ==================== 核心处理器 ====================
class HighFrequencyGraphBuilder:
    # (保持 __init__, _load_index_symbols, _get_valid_dates 不变)
    def __init__(self, index_file, data_root, start_date, end_date, output_dir, device):
        self.index_file = index_file
        self.data_root = data_root
        self.start_date = start_date
        self.end_date = end_date
        self.OUTPUT_DIR = output_dir
        self.DEVICE = device

        self.raw_buy_sell_cols = [
            '特大单买入量(手)', '特大单卖出量(手)',
            '大单买入量(手)', '大单卖出量(手)',
            '中单买入量(手)', '中单卖出量(手)',
            '小单买入量(手)', '小单卖出量(手)'
        ]
        self.net_flow_features = ['LargeNetFlow', 'MediumNetFlow', 'SmallNetFlow']

        self.symbols = self._load_index_symbols()
        self.file_dates = self._get_valid_dates()



    def _load_index_symbols(self):
        """从Excel文件加载目标股票代码"""
        try:
            df = pd.read_excel(self.index_file)
            symbols = set(df.iloc[:, 0].astype(str).str.zfill(6).tolist())
            print(f"成功加载 {len(symbols)} 个目标股票代码。")
            return symbols
        except FileNotFoundError:
            print(f"错误：指数文件未找到于 {self.index_file}")
            return set()
        except Exception as e:
            print(f"加载指数文件时出错: {e}")
            return set()

    def _get_valid_dates(self):
        """获取在指定日期范围内，且对应数据文件存在并包含数据的交易日列表"""
        valid_dates = []
        print(f"\n正在扫描数据目录 {self.data_root} 以获取有效交易日...")
        all_files = sorted(os.listdir(self.data_root))

        for file in tqdm(all_files, desc="Scanning Dates"):
            if file.endswith(".csv"):
                date_str = file.split(".")[0]
                if not (self.start_date <= date_str <= self.end_date):
                    continue

                file_path = os.path.join(self.data_root, file)
                try:
                    # Read only necessary columns if possible, check header first
                    # Read just the header to check for columns quickly
                    header_df = pd.read_csv(file_path, encoding="utf-8", engine="python", nrows=0)
                    if not all(col in header_df.columns for col in self.raw_buy_sell_cols):
                        # print(f"文件 {file} 缺少必需的列，跳过。")
                        continue

                    # Check if file has more than header (basic check for non-empty)
                    # This is less robust than reading a row but faster for scanning
                    if os.path.getsize(file_path) > 100: # Arbitrary small size check
                         valid_dates.append(date_str)
                    # else:
                    #     print(f"文件 {file} 可能为空或只有表头，跳过。")

                except pd.errors.EmptyDataError:
                    # print(f"警告: 文件 {file_path} 为空，跳过。")
                    continue
                except Exception as e:
                    # print(f"警告: 文件 {file_path} 读取失败或处理异常: {e}，跳过。")
                    continue

        sorted_valid_dates = sorted(valid_dates)
        print(f"在 {self.start_date} 到 {self.end_date} 范围内找到 {len(sorted_valid_dates)} 个有效交易日。")
        return sorted_valid_dates

    # (保持 _preprocess_data, _calculate_pearson_correlation, _visualize_distribution 不变)
    def _preprocess_data(self, date_window):
        """
        处理滑动窗口内的数据，计算净流入特征（包含小户），并展平为每支股票的特征向量。
        """
        window_data = []
        # Use sorted list for consistent order if self.symbols is modified elsewhere (though unlikely here)
        sorted_symbols = sorted(list(self.symbols))
        template_df = pd.DataFrame({'代码': sorted_symbols})

        missing_cols_dates = set()

        for date in date_window: # No tqdm here, tqdm is in pipeline
            file_path = os.path.join(self.data_root, f"{date}.csv")
            try:
                df = pd.read_csv(file_path, encoding="utf-8", engine="python",
                                 usecols=['代码'] + self.raw_buy_sell_cols) # Read only needed cols
                if df.empty:
                    raise ValueError("空文件") # Will be caught and handled below

                # Check required columns again after loading
                if not all(col in df.columns for col in self.raw_buy_sell_cols):
                     missing_cols_dates.add(date)
                     raise ValueError("缺少资金流列") # Will be caught and handled below

                # Data Cleaning and Transformation
                df['代码'] = df['代码'].astype(str)
                # Ensure cleaning handles potential NaN or unexpected formats before regex
                df['代码'] = df['代码'].str.replace(r'\.\w+$', '', regex=True).str.zfill(6)
                df.dropna(subset=['代码'], inplace=True) # Remove rows where code became NaN

                # Calculate Net Flows
                df['LargeNetFlow'] = (df['特大单买入量(手)'] + df['大单买入量(手)']) - (df['特大单卖出量(手)'] + df['大单卖出量(手)'])
                df['MediumNetFlow'] = df['中单买入量(手)'] - df['中单卖出量(手)']
                df['SmallNetFlow'] = df['小单买入量(手)'] - df['小单卖出量(手)']

                df_net_flow = df[['代码'] + self.net_flow_features]

                # Merge with template to ensure all target stocks are present
                merged_df = template_df.merge(
                    df_net_flow,
                    on='代码', how='left'
                )
                # Fill missing stock data for this date with 0
                merged_df[self.net_flow_features] = merged_df[self.net_flow_features].fillna(0)

                merged_df['date'] = pd.to_datetime(date, format='%Y%m%d')
                window_data.append(merged_df)

            except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as e:
                # Handle cases where file is missing, empty, or lacks critical columns
                # print(f"信息：日期 {date} 文件处理异常或数据缺失 ({type(e).__name__})，将使用0填充该日数据。")
                fallback_df = template_df.copy()
                fallback_df[self.net_flow_features] = 0 # Fill features with 0
                fallback_df['date'] = pd.to_datetime(date, format='%Y%m%d')
                window_data.append(fallback_df)
            # except Exception as e: # Catch unexpected errors
                # print(f"警告：处理日期 {date} 时发生未知错误: {e}，将使用0填充该日数据。")
                # fallback_df = template_df.copy()
                # fallback_df[self.net_flow_features] = 0
                # fallback_df['date'] = pd.to_datetime(date, format='%Y%m%d')
                # window_data.append(fallback_df)

        # Report missing columns at the end of the window processing
        if missing_cols_dates:
            print(f"\n警告：窗口 {date_window[0]}-{date_window[-1]} 中以下日期的文件缺少必需资金流列: {sorted(list(missing_cols_dates))}")

        if not window_data:
             # This case should be rare now due to fallback logic, but kept for safety
             print(f"警告：窗口 {date_window} 内没有成功处理或回填的数据。")
             return np.array([]), []

        # Concatenate and sort
        full_df = pd.concat(window_data)
        full_df = full_df.sort_values(by=['代码', 'date']) # Sort crucial for correct flattening

        features_list = []
        symbols_list_in_order = [] # Keep track of the order for the output array

        # Group by stock code to flatten features
        grouped = full_df.groupby('代码')
        for code in sorted_symbols: # Iterate through the template order
             if code in grouped.groups:
                 stock_df = grouped.get_group(code)
                 # Ensure the stock has data for all days in the window (important!)
                 if len(stock_df) == len(date_window):
                      # Flatten the features for the window [F1_d1, F2_d1, F3_d1, F1_d2, F2_d2, F3_d2, ...]
                      stock_features = stock_df[self.net_flow_features].values.flatten()
                      features_list.append(stock_features)
                      symbols_list_in_order.append(code)
                 # else:
                      # print(f"信息：股票 {code} 在窗口 {date_window[0]}-{date_window[-1]} 数据不完整 ({len(stock_df)}/{len(date_window)} 天)，已跳过。")
             # else:
                  # This case means the stock was in template but had no data at all (merged as NaN, then 0)
                  # We should decide whether to include stocks with all-zero features.
                  # Current logic implicitly excludes them because the group won't exist if merge fails entirely,
                  # or len(stock_df) might be wrong if some merges failed. Let's stick to excluding incomplete data.
                  # print(f"信息：股票 {code} 在窗口 {date_window[0]}-{date_window[-1]} 未找到数据。")


        if not features_list:
             # print(f"警告：窗口 {date_window} 内没有股票数据满足所有天数的要求。")
             return np.array([]), []

        # Stack features into a NumPy array (Stocks x Flattened Features)
        feature_array = np.vstack(features_list)

        # Return the feature array and the list of symbols IN THE SAME ORDER as the rows of the array
        return feature_array, symbols_list_in_order

    def _calculate_pearson_correlation(self, tensor_data):
        """
        在指定的设备上计算输入张量（行是样本，列是特征）的行之间的皮尔逊相关系数矩阵。
        """
        tensor_data = tensor_data.float().to(self.DEVICE) # Ensure float and on correct device

        # Handle edge cases: less than 2 stocks or features
        if tensor_data.dim() != 2 or tensor_data.shape[0] < 2 or tensor_data.shape[1] < 1:
             # print("信息：输入数据不足 (少于2支股票或无特征)，返回单位矩阵。")
             identity_matrix = torch.eye(tensor_data.shape[0], device=self.DEVICE)
             # Return as numpy array as expected by downstream functions
             return identity_matrix.cpu().numpy()

        # --- Pearson Calculation using PyTorch ---
        # 1. Center the data (subtract mean)
        mean = torch.mean(tensor_data, dim=1, keepdim=True)
        centered_data = tensor_data - mean

        # 2. Calculate standard deviation (handle zero std)
        std = torch.std(tensor_data, dim=1, keepdim=True)
        # Avoid division by zero: where std is 0, keep centered_data as 0 (or std as 1)
        # Using safe division: add small epsilon or replace std=0 with 1
        std_safe = torch.where(std == 0, torch.ones_like(std), std)
        normalized_data = centered_data / std_safe

        # 3. Calculate covariance matrix (using matrix multiplication)
        # Corr(X, Y) = Cov(X_norm, Y_norm) = E[X_norm * Y_norm]
        # For rows (stocks) i, j: Corr_ij = (1/N) * sum(normalized_data[i, k] * normalized_data[j, k] for k in features)
        # This is equivalent to (1/N) * (normalized_data @ normalized_data.T)
        # We actually want Cov(Xi, Xj) / (std(Xi) * std(Xj))
        # Using normalized data (z-scores implicitly handles the std division)
        # Correlation = Z @ Z.T / (num_features - 1 if unbiased else num_features)
        # PyTorch `torch.corrcoef` handles this directly and efficiently.

        # Let's use torch.corrcoef for robustness and clarity
        try:
             correlation_matrix = torch.corrcoef(tensor_data) # tensor_data should be [stocks, features]
             # Handle potential NaNs resulting from zero variance rows after centering
             correlation_matrix = torch.nan_to_num(correlation_matrix, nan=0.0) # Replace NaN with 0
        except Exception as e:
             print(f"\n错误：使用 torch.corrcoef 计算相关性时出错: {e}。返回单位矩阵。")
             correlation_matrix = torch.eye(tensor_data.shape[0], device=self.DEVICE)


        # Clamp values just in case of floating point inaccuracies
        correlation_matrix = torch.clamp(correlation_matrix, -1.0, 1.0)

        # Return as numpy array
        return correlation_matrix.cpu().numpy()


    def _visualize_distribution(self, matrix, date):
        """可视化相关性分布直方图"""
        # Ensure output directory exists (might be redundant but safe)
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        # Extract non-diagonal elements
        non_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)].flatten()
        # Remove any NaNs or Infs that might have slipped through
        non_diagonal = non_diagonal[np.isfinite(non_diagonal)]

        if non_diagonal.size > 0:
             plt.hist(non_diagonal, bins=50, alpha=0.7, range=(-1, 1)) # Set range for consistency
             plt.title(f"资金流皮尔逊相关系数分布 (非对角线) - {date}")
             plt.xlabel("pearson")
             plt.ylabel("freq")
             plt.grid(axis='y', alpha=0.75)
             # Use Path object for saving
             save_path = Path(self.OUTPUT_DIR) / f"corr_dist_{date}.png"
             plt.savefig(save_path)
             plt.close() # Close the figure to free memory
        else:
             print(f"\n警告：日期 {date} 的相关系数矩阵（非对角线）没有有效数值进行可视化。")
             plt.close()

    # (保持 process_window 不变)
    def process_window(self, window_dates):
        """处理一个滑动窗口，生成相关性矩阵和对应的股票列表"""
        target_date = window_dates[-1] # Date for which the graph is generated
        # print(f"\n--- 处理窗口: {window_dates[0]} 到 {target_date} ---")
        try:
            # 1. Preprocess data for the window
            feature_array, symbols = self._preprocess_data(window_dates)

            # Check if preprocessing yielded valid results
            if feature_array.shape[0] < 2: # Need at least 2 stocks to calculate correlation
                 print(f"窗口 {target_date}: 预处理后股票数量不足 ({feature_array.shape[0]})，无法计算相关性。")
                 return None, None, []
            if feature_array.shape[1] < 1: # Need at least 1 feature dimension
                 print(f"窗口 {target_date}: 预处理后特征维度为0，无法计算相关性。")
                 return None, None, []

            # 2. Convert to Tensor
            # Device transfer happens inside _calculate_pearson_correlation
            feature_tensor = torch.tensor(feature_array).float() # Keep on CPU initially

            # 3. Calculate Correlation Matrix
            correlation_matrix = self._calculate_pearson_correlation(feature_tensor) # Handles device transfer

            # print(f"窗口 {target_date}: 成功计算了 {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]} 相关性矩阵。")
            return correlation_matrix, target_date, symbols # Return matrix, target date, and ordered symbols

        except Exception as e:
            # Catch any unexpected errors during window processing
            print(f"\n处理日期窗口 {window_dates[0]}-{target_date} 时发生严重错误: {str(e)}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            return None, None, []

    # --- MODIFIED save_graph_data ---
    def save_graph_data(self, matrix, date, symbols):
        """
        将相关性矩阵转换为GNN格式并保存，包含节点列表、edge_index和edge_weight。
        阈值设定为只保留相关性 > 0.6 的边，并移除自相关。
        """
        THRESHOLD = 0.6 # Define the correlation threshold

        # Check inputs
        if matrix is None or not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            print(f"\n警告：日期 {date} 的输入矩阵无效，跳过图数据保存。")
            return
        if not symbols or len(symbols) != matrix.shape[0]:
             print(f"\n警告：日期 {date} 的股票列表无效或与矩阵维度不匹配 ({len(symbols)} vs {matrix.shape[0]})，跳过图数据保存。")
             return
        if matrix.shape[0] < 2:
             print(f"\n信息：日期 {date} 的矩阵维度小于2 ({matrix.shape[0]})，无法形成边。")
             return


        # --- 将相关性矩阵转换为阈值化的邻接矩阵 ---
        # Create DataFrame for easier handling (optional but can be convenient)
        # corr_df = pd.DataFrame(matrix, index=symbols, columns=symbols)
        # adj_matrix = corr_df.values # Work directly with numpy array for efficiency

        adj_matrix = matrix.copy() # Work directly with numpy array

        # Apply threshold: set values <= THRESHOLD to 0
        adj_matrix[adj_matrix <= THRESHOLD] = 0

        # Remove self-loops (set diagonal to 0)
        np.fill_diagonal(adj_matrix, 0)

        # --- 提取边和权重 ---
        edge_index_list = []
        edge_weight_list = []
        # Use node indices directly, map symbols later if needed outside graph structure
        # node2id = {node: i for i, node in enumerate(symbols)} # Mapping is implicit by order

        rows, cols = adj_matrix.shape
        # Iterate through the upper triangle of the adjacency matrix
        for r in range(rows):
            for c in range(r + 1, cols): # Start from r+1 to avoid self-loops and duplicates
                weight = adj_matrix[r, c]
                if weight > 0: # If weight survived thresholding and is not diagonal
                    # Add edge in both directions for undirected graph representation in PyG?
                    # Typically PyG prefers directed edges [source, target]. If graph is undirected,
                    # edges (i, j) and (j, i) are often added unless specified otherwise.
                    # Let's add both pairs for standard undirected representation.
                    edge_index_list.append([r, c]) # Edge from r to c
                    edge_weight_list.append(weight)
                    edge_index_list.append([c, r]) # Edge from c to r (for undirected)
                    edge_weight_list.append(weight) # Same weight

        # --- 检查是否有边生成 ---
        if not edge_index_list:
            print(f"\n信息：日期 {date} 的资金流图没有边 (相关性未达到阈值 > {THRESHOLD})。")
            # Decide if you want to save a file with no edges
            # Option 1: Skip saving
            # return
            # Option 2: Save file with empty edge info but with nodes
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long) # Empty edge_index
            edge_weight_tensor = torch.empty((0,), dtype=torch.float) # Empty edge_weight
        else:
            # Convert to Tensors and move to device
            edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(self.DEVICE)
            edge_weight_tensor = torch.tensor(edge_weight_list, dtype=torch.float).to(self.DEVICE)

        # --- 打包数据并保存 ---
        graph_data = {
            'nodes': symbols,  # Save the list of stock codes corresponding to indices 0..N-1
            'edge_index': edge_index_tensor,
            'edge_weight': edge_weight_tensor
        }

        # Ensure output directory exists
        output_path = Path(self.OUTPUT_DIR) / f"graph_data_hf_{date}.pt"
        try:
             torch.save(graph_data, output_path)
             num_edges_saved = edge_index_tensor.shape[1] // 2 # Count unique undirected edges
             print(f"已保存日期 {date} 的资金流图数据 (.pt, 阈值 > {THRESHOLD}) 到 {output_path}。包含 {num_edges_saved} 条无向边 ({edge_index_tensor.shape[1]} 条有向表示)。")
        except Exception as e:
             print(f"\n错误：保存 PyTorch 文件 {output_path} 时出错: {e}")


    # (保持 pipeline 不变)
    def pipeline(self):
        """主处理流程，按滑动窗口处理数据并生成图"""
        if not self.symbols:
            print("错误：没有加载到目标股票代码，无法继续。")
            return
        if not self.file_dates:
             print("错误：没有找到指定日期范围内的有效交易日数据，无法继续。")
             return

        # Define sliding window size
        window_size = 7 # Example: 7-day window

        if len(self.file_dates) < window_size:
            print(f"错误：有效交易日数量 ({len(self.file_dates)}) 小于窗口大小 ({window_size})，无法构建图。")
            return

        num_windows = len(self.file_dates) - window_size + 1
        print(f"\n准备处理 {num_windows} 个滑动窗口 (大小={window_size})...")

        # Iterate through the end dates of the sliding windows
        for i in tqdm(range(window_size - 1, len(self.file_dates)), desc="Processing Time Windows"):
            # Define the current window dates
            window_dates = self.file_dates[i - window_size + 1 : i + 1]
            target_date = window_dates[-1] # The date for which the graph is generated

            # Process the window: calculate correlation matrix and get symbols
            correlation_matrix, matrix_date, symbols_for_matrix = self.process_window(window_dates)

            # Check if processing was successful
            if correlation_matrix is not None and matrix_date is not None and symbols_for_matrix:
                # 1. Visualize the distribution of correlations (optional)
                # Consider visualizing only occasionally to save time/disk space
                # if i % 20 == 0: # Example: visualize every 20 windows
                self._visualize_distribution(correlation_matrix, matrix_date)

                # 2. Save the graph data (nodes, edges, weights) based on threshold
                self.save_graph_data(correlation_matrix, matrix_date, symbols_for_matrix)
            else:
                print(f"信息：跳过日期 {target_date} 的图数据保存，因为窗口处理未成功或未生成有效矩阵/符号。")


        print(f"\n--- 资金流图构建管道完成 ({num_windows} 个窗口已处理) ✅ ---")


# ==================== 执行入口 ====================
if __name__ == "__main__":
    print("--- 开始执行高频资金流图构建脚本 ---")
    builder = HighFrequencyGraphBuilder(
        index_file=INDEX_FILE,
        data_root=DATA_ROOT,
        start_date=START_DATE,
        end_date=END_DATE,
        device=DEVICE,
        output_dir=OUTPUT_DIR # Pass the updated output directory
    )
    builder.pipeline()
    print("\n--- 脚本执行结束 ---")