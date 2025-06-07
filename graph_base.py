import pandas as pd
import numpy as np
import os
import torch # 导入torch库，用于处理张量和保存PyTorch Geometric所需格式
from sklearn.metrics.pairwise import cosine_similarity # 导入余弦相似度计算
import matplotlib.pyplot as plt # 导入matplotlib.pyplot库，用于可视化
import seaborn as sns # 导入seaborn库，用于增强可视化效果
from tqdm import tqdm # 导入tqdm库，用于显示进度条

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl") # 忽略openpyxl相关的UserWarning

# --- 配置Matplotlib支持中文 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 数据读取和过滤函数 ---
# 修改：返回目标股票列表
def read_and_filter_data(index_path, data_dir, num_files=5):
    """
    读取目标股票列表和基金持仓数据，并根据目标股票进行过滤。

    Args:
        index_path (str): 包含目标股票代码的Excel文件路径 (无header，第一列是股票代码)。
        data_dir (str): 存放基金持仓Excel文件的目录。
        num_files (int): 基金持仓文件的数量 (HLD_Fundhold0.xlsx, HLD_Fundhold1.xlsx, ...)。

    Returns:
        tuple: (pd.DataFrame: 包含目标股票过滤后的基金持仓数据, list: 排序后的目标股票代码列表).
               如果读取或过滤失败，返回 (pd.DataFrame(), []).
    """
    print("--- 读取目标股票列表 ---")
    try:
        # 读取目标股票列表，假定文件无header，股票代码在第一列
        target_stocks_df = pd.read_excel(index_path, names=["index"])
        # 将股票代码列转换为字符串列表
        target_stocks = target_stocks_df["index"].astype(str).tolist()
        # Ensure stock codes have leading zeros if needed (assuming 6 digits standard)
        target_stocks = [s.zfill(6) for s in target_stocks]
        # 去重并排序，确保节点顺序一致
        sorted_target_stocks = sorted(list(set(target_stocks)))
        print(f"已读取和排序 {len(sorted_target_stocks)} 个目标股票代码。")
    except FileNotFoundError:
        print(f"错误：目标股票文件未找到于 {index_path}")
        return pd.DataFrame(), [] # 文件未找到则返回空

    except Exception as e:
        print(f"读取目标股票文件时出错: {e}")
        return pd.DataFrame(), [] # 读取错误则返回空


    print("\n--- 合并基金持仓数据 ---")
    all_dfs = []
    # 使用tqdm显示文件读取进度
    for i in tqdm(range(num_files), desc="Reading Fund Files"):
        file_path = os.path.join(data_dir, f"HLD_Fundhold{i}.xlsx")
        if not os.path.exists(file_path):
            print(f"\n警告：基金持仓文件未找到：{file_path}，跳过。")
            continue
        try:
            # 读取Excel文件
            df = pd.read_excel(file_path)
            # 重命名列，确保列名一致
            # Make sure these column names EXACTLY match your excel files
            # Using generic names if specific ones aren't known for sure.
            # Assuming first 7 columns are the relevant ones in order:
            # Stkcd, Reptdt, Fundcd, Fundnm, Mconme, Fundhold, Holdperct
            # *** ADJUST THIS LIST BASED ON YOUR ACTUAL FILE COLUMNS IF NAMES ARE DIFFERENT ***
            expected_cols = ["Stkcd", "Reptdt", "Fundcd", "Fundnm", "Mconme", "Fundhold", "Holdperct"]
            if len(df.columns) >= len(expected_cols):
                 # Take only the first N columns as specified in expected_cols
                 df = df.iloc[:, :len(expected_cols)]
                 df.columns = expected_cols # Rename the columns
                 all_dfs.append(df)
            else:
                 print(f"\n警告：文件 {file_path} 的列数少于预期 ({len(df.columns)} < {len(expected_cols)})，跳过。")

        except Exception as e:
            print(f"\n读取或处理文件 {file_path} 时出错: {e}")
            continue

    if not all_dfs:
         print("未成功读取任何基金持仓文件。")
         return pd.DataFrame(), sorted_target_stocks # 返回空DataFrame和已加载的股票列表

    # 合并所有读取到的DataFrame
    print("正在合并所有基金持仓文件...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"合并后总行数: {len(combined_df)}")

    # Convert Stkcd to string and ensure formatting (e.g., leading zeros)
    # Adjust zfill(6) if your stock codes have a different standard length
    combined_df["Stkcd"] = combined_df["Stkcd"].astype(str).str.zfill(6)

    # --- Data Cleaning ---
    # Ensure 'Holdperct' is numeric, coercing errors to NaN
    combined_df['Holdperct'] = pd.to_numeric(combined_df['Holdperct'], errors='coerce')
    # Remove rows where Holdperct became NaN or is non-positive
    original_rows = len(combined_df)
    combined_df.dropna(subset=['Holdperct'], inplace=True)
    combined_df = combined_df[combined_df['Holdperct'] > 0].copy() # Use copy() to avoid SettingWithCopyWarning
    print(f"移除了 {original_rows - len(combined_df)} 行，因为 'Holdperct' 无效或非正。")


    # Filter data to keep only target stocks
    print(f"根据 {len(sorted_target_stocks)} 个目标股票过滤数据...")
    filtered_df = combined_df[combined_df["Stkcd"].isin(sorted_target_stocks)].copy() # Keep only rows related to target stocks
    print(f"过滤后行数: {len(filtered_df)}")

    if filtered_df.empty:
        print("过滤后没有目标股票相关的持仓数据。")
        return pd.DataFrame(), sorted_target_stocks # 返回空DataFrame和已加载的股票列表

    # 将报告日期转换为datetime对象
    print("转换报告日期并提取季度信息...")
    try:
        filtered_df["Reptdt"] = pd.to_datetime(filtered_df["Reptdt"], errors='coerce')
        filtered_df.dropna(subset=['Reptdt'], inplace=True) # Remove rows where date conversion failed
        # 提取季度信息，'Q'表示季度
        filtered_df["Quarter"] = filtered_df["Reptdt"].dt.to_period("Q")
        print("日期转换和季度提取完成。")
    except Exception as e:
        print(f"转换报告日期或提取季度时出错: {e}")
        return pd.DataFrame(), sorted_target_stocks # Return empty DataFrame and stock list if date processing fails


    return filtered_df, sorted_target_stocks # 返回过滤后的数据和完整的股票列表

# --- 可视化相似度矩阵热力图函数 ---
def plot_similarity(sim_matrix, quarter, threshold=0.15, output_dir="."):
    """
    绘制相似度矩阵热力图。

    Args:
        sim_matrix (pd.DataFrame): 股票相似度矩阵 (应为 N x N 矩阵)。
        quarter (str): 季度标识 (用于文件名和标题)。
        threshold (float): 用于可视化的相似度阈值，低于此阈值的区域可能被遮盖。
        output_dir (str): 图片保存目录。
    """
    plt.figure(figsize=(12, 10))
    # 创建遮罩，低于阈值的相似度值将被遮盖 (不会显示颜色)
    # 如果sim_matrix不是数值类型，或包含无穷大，heatmap可能会出错
    sim_matrix_numeric = sim_matrix.values.astype(float)
    sim_matrix_numeric[np.isinf(sim_matrix_numeric)] = np.nan # Replace inf with NaN
    mask = np.isnan(sim_matrix_numeric) | (sim_matrix_numeric < threshold)

    # 绘制热力图，annot=False不显示数值，xticklabels/yticklabels=False不显示股票代码标签
    sns.heatmap(sim_matrix_numeric, mask=mask, cmap="YlGnBu", annot=False,
                xticklabels=False, yticklabels=False)
    plt.title(f"股票持股相似度矩阵 - {quarter}")
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 保存图片
    plt.savefig(os.path.join(output_dir, f"heatmap_{quarter}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存季度 {quarter} 的相似度热力图。")


# --- 保存 PyTorch 格式的图数据 (包含节点) ---
# 修改：接受完整的节点列表，并在相似度矩阵上应用，确保输出维度一致
def save_graph_data_pytorch(sim_df, nodes, quarter, output_dir, top_k_percent=0.15):
    """
    将相似度矩阵转换为 PyTorch Geometric 的格式 (包含节点列表、edge_index 和 edge_weight) 并保存为单个 .pt 文件。
    基于保留前 top_k_percent 的强连接计算动态阈值。始终使用完整的节点列表构建图。

    Args:
        sim_df (pd.DataFrame): 原始的股票相似度矩阵 (行/列索引为 Stkcd，应包含所有目标股票)。
        nodes (list): 完整的、排序后的目标股票代码列表，将作为图的节点。
        quarter (str): 季度标识 (用于文件名)。
        output_dir (str): 文件保存目录。
        top_k_percent (float): 要保留的最强连接的比例 (例如 0.15 表示保留前 15%)。
    """
    # Ensure sim_df has the same index/columns as the full nodes list
    # This should already be the case if build_similarity_matrices reindexed correctly
    if not sim_df.index.equals(pd.Index(nodes)) or not sim_df.columns.equals(pd.Index(nodes)):
         print(f"\n错误：季度 {quarter} 的相似度矩阵索引或列名与完整的节点列表不匹配。无法保存图数据。")
         # Still save an empty graph to maintain file structure consistency
         num_nodes = len(nodes)
         edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
         edge_weight_tensor = torch.empty((0,), dtype=torch.float)
         graph_data = {'nodes': nodes, 'edge_index': edge_index_tensor, 'edge_weight': edge_weight_tensor}
         output_path = os.path.join(output_dir, f"graph_data_{quarter}.pt")
         torch.save(graph_data, output_path)
         print(f"保存季度 {quarter} 的图数据 (空边)，由于相似度矩阵维度错误。")
         return


    num_nodes = len(nodes) # 节点数量始终是完整的股票数量

    if num_nodes == 0:
         print(f"\n警告：节点列表为空，无法构建图数据。")
         return # Return if no nodes at all

    if num_nodes < 2:
        print(f"\n警告：节点数量少于2 ({num_nodes})，无法构建边。将保存包含节点和空边的文件。")
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_weight_tensor = torch.empty((0,), dtype=torch.float)
        graph_data = {'nodes': nodes, 'edge_index': edge_index_tensor, 'edge_weight': edge_weight_tensor}
        output_path = os.path.join(output_dir, f"graph_data_{quarter}.pt")
        torch.save(graph_data, output_path)
        print(f"保存季度 {quarter} 的图数据 (0边)。")
        return


    # 1. 提取非对角线的相似度值 (使用完整的 N x N 相似度矩阵)
    sim_matrix_np = sim_df.values
    mask_off_diagonal = ~np.eye(num_nodes, dtype=bool)
    off_diagonal_sim_values = sim_matrix_np[mask_off_diagonal]

    # 2. 过滤掉 NaN 和无穷大值，以及非正的相似度 (通常只关心正相关性作为强连接)
    # cosine_similarity 理论上结果在 [-1, 1]，nan_to_num 会处理 NaN/Inf
    # 关注大于0的相似度作为正相关连接
    finite_positive_off_diagonal_sim = off_diagonal_sim_values[np.isfinite(off_diagonal_sim_values) & (off_diagonal_sim_values > 0)]


    if finite_positive_off_diagonal_sim.size == 0 or top_k_percent <= 0:
        print(f"\n季度 {quarter}: 没有正的有效相似度值，或保留比例为零。将保存没有边的图数据。")
        # 保存包含节点列表和空边的文件
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_weight_tensor = torch.empty((0,), dtype=torch.float)
        graph_data = {'nodes': nodes, 'edge_index': edge_index_tensor, 'edge_weight': edge_weight_tensor}
        output_path = os.path.join(output_dir, f"graph_data_{quarter}.pt")
        torch.save(graph_data, output_path)
        print(f"保存季度 {quarter} 的图数据 (0边)。")
        return


    # 3. 计算阈值：保留前 top_k_percent 的正相关性连接
    # 计算正相关性值的 (1 - top_k_percent) 分位点
    threshold = np.quantile(finite_positive_off_diagonal_sim, 1.0 - top_k_percent)

    print(f"\n季度 {quarter}: 基于保留前 {top_k_percent*100:.2f}% 正相关性计算出的相似度阈值为: {threshold:.4f}")


    # 4. 根据计算出的阈值生成邻接矩阵 (DataFrame形式)
    adj_df = sim_df.copy()
    # 将小于计算出的阈值的相似度值置为 0
    adj_df[adj_df < threshold] = 0
    # 移除自环 (对角线置为 0)
    np.fill_diagonal(adj_df.values, 0)


    # 5. 提取边列表 (edge_index, edge_weight)
    edge_index_list = []
    edge_weight_list = []
    # node2id 映射基于完整的节点列表 nodes
    node2id = {node: i for i, node in enumerate(nodes)}

    adj_matrix_np = adj_df.values # 使用 numpy 数组加速遍历
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = adj_matrix_np[i, j]
            # 只处理大于0的边权重 (阈值化后 > 0 的就是保留的边) 并且只添加无向图的一侧边 (i < j)
            if weight > 0 and i < j:
                # 使用基于完整节点列表的 node2id 获取节点索引
                edge_index_list.append([node2id[nodes[i]], node2id[nodes[j]]])
                edge_weight_list.append(weight)

    # 检查是否有边生成
    if not edge_index_list:
        print(f"\n警告：季度 {quarter} 没有生成任何边 (所有正相似度均低于计算出的阈值 {threshold:.4f})。将保存空的边信息。")
        # 保存包含节点列表和空边的文件
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_weight_tensor = torch.empty((0,), dtype=torch.float)
    else:
        # 将边索引列表转换为 PyTorch 的 LongTensor，并转置使其形状为 [2, num_edges]
        # 确保使用 contiguous()
        edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        # 将边权重列表转换为 PyTorch 的 FloatTensor
        edge_weight_tensor = torch.tensor(edge_weight_list, dtype=torch.float)

    # --- 保存 PyTorch Geometric 数据结构 ---
    # 将节点列表 (Stkcd)、edge_index 和 edge_weight 组合到一个字典中
    # 这里的 'nodes' 列表是完整的目标股票列表，其顺序定义了 edge_index 中的节点索引
    graph_data = {
        'nodes': nodes,  # List of stock codes (Stkcd) corresponding to node indices 0, 1, 2...
        'edge_index': edge_index_tensor, # Shape [2, num_edges]
        'edge_weight': edge_weight_tensor # Shape [num_edges]
        # 你也可以选择性地保存 node2id 映射，如果之后需要的话:
        # 'node2id': node2id
    }

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存字典为单个 .pt 文件
    # 文件名格式例如: D:\graph_basic\fund_graph\graph_data_2019Q1.pt
    # 为了与 Dataset 加载逻辑兼容，建议将不同图源保存到不同子目录下
    # 例如： output_dir / 'fund_graph' / f"graph_data_{quarter}.pt"
    # 或者在 save 函数中指定图源名称作为文件名一部分
    # 比如 filename = f"fund_graph_{quarter}.pt"
    # 让调用者 (build_similarity_matrices) 指定文件名或子目录可能更灵活
    # 我们修改 build_similarity_matrices 来指定文件名

    output_path = os.path.join(output_dir, f"fund_graph_{quarter}.pt") # 使用 fund_graph_ 前缀标识图源
    try:
        torch.save(graph_data, output_path)
        print(f"Saved fund graph data for {quarter} to {output_path} (Edges: {edge_index_tensor.shape[1]})")
    except Exception as e:
        print(f"\n保存 PyTorch 文件 {output_path} 时出错: {e}")


# --- 构建相似度矩阵和邻接矩阵的主函数 ---
# 修改：接收完整的股票列表，并在构建矩阵时使用
def build_similarity_matrices(filtered_df, target_stocks, output_dir="output", top_k_percent=0.15, plot_heatmaps=False):
    """
    按季度构建股票持股相似度矩阵和邻接矩阵，并保存为包含节点信息的 GNN 可用格式。

    Args:
        filtered_df (pd.DataFrame): 过滤后的基金持仓数据。
        target_stocks (list): 完整的、排序后的目标股票代码列表。
        output_dir (str): 输出文件保存目录。
        top_k_percent (float): 用于生成邻接矩阵的阈值，保留前 top_k_percent 的强连接。
        plot_heatmaps (bool): 是否绘制并保存相似度热力图。
    """
    if filtered_df.empty:
        print("输入数据为空，无法构建相似度矩阵。")
        return

    if not target_stocks:
        print("目标股票列表为空，无法构建相似度矩阵。")
        return

    print(f"将基于 {len(target_stocks)} 个目标股票构建图。")

    # Ensure Quarter is string for filename safety
    filtered_df['Quarter_str'] = filtered_df['Quarter'].astype(str)

    os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在

    # 按季度对数据进行分组
    quarter_groups = filtered_df.groupby("Quarter_str") # Group by string representation

    print("\n--- 正在按季度构建基金持股相似度图数据文件 ---")
    # 使用tqdm显示季度处理进度
    # 将 quarter_groups 转换为列表，以便 tqdm 显示总进度
    quarter_group_list = list(quarter_groups)

    for quarter_str, group in tqdm(quarter_group_list, desc="Processing Quarters"):
        print(f"\n--- 处理季度: {quarter_str} ---")

        # 针对当前季度的数据，构建一个透视表
        # 行是股票代码(Stkcd)，列是基金代码(Fundcd)，值是持股比例(Holdperct)的总和
        # 只包含本季度实际有持股数据的股票和基金
        try:
            # Ensure only target stocks are in the pivot index initially if they had holdings
            # This pivot will naturally only include stocks and funds present in the 'group' DataFrame
            matrix = group.pivot_table(
                index="Stkcd",
                columns="Fundcd",
                values="Holdperct",
                aggfunc="sum", # 对同一基金持有同一股票的记录进行求和
                fill_value=0 # 持股比例填充0
            )
        except Exception as e:
            print(f"为季度 {quarter_str} 创建透视表时出错: {e}。跳过此季度。")
            # 即使出错，也尝试保存一个空的图数据文件以保持文件结构一致性
            save_graph_data_pytorch(pd.DataFrame(), target_stocks, quarter_str, output_dir, top_k_percent=0) # Pass empty dataframe but full nodes
            continue # 跳过当前季度

        # --- 关键修改：使用完整的 target_stocks 列表重新索引矩阵 ---
        # 这会引入在当前季度没有持股数据的目标股票，其持股数据将为 NaN
        # 然后用 0 填充这些 NaN
        try:
            # Reindex both index (stocks) and columns (funds) to include all possibilities
            # Reindexing columns might not be necessary if cosine_similarity handles different columns,
            # but reindexing index is crucial to include all target stocks.
            # Let's reindex only the index first.
            matrix = matrix.reindex(index=target_stocks, fill_value=0)

            # Check if any new columns were introduced by reindexing (shouldn't happen with index reindex)
            # and ensure column names are clean if needed.

            # Ensure all values are numeric after reindexing
            matrix = matrix.apply(pd.to_numeric, errors='coerce').fillna(0) # Fill any remaining NaN after reindex with 0


            print(f"季度 {quarter_str}: 原始透视表 shape {group.pivot_table(index='Stkcd', columns='Fundcd', values='Holdperct', aggfunc='sum', fill_value=0).shape} -> 重新索引后 shape {matrix.shape}.")
            print(f"确保所有目标股票 ({len(target_stocks)}) 都已包含。")

        except Exception as e:
            print(f"为季度 {quarter_str} 重新索引矩阵时出错: {e}。跳过此季度。")
            # 即使出错，也尝试保存一个空的图数据文件以保持文件结构一致性
            save_graph_data_pytorch(pd.DataFrame(), target_stocks, quarter_str, output_dir, top_k_percent=0) # Pass empty dataframe but full nodes
            continue # 跳过当前季度


        # 检查重新索引后的矩阵是否有效
        if matrix.empty:
             print(f"警告：季度 {quarter_str} 重新索引后的矩阵为空，跳过。")
             save_graph_data_pytorch(pd.DataFrame(), target_stocks, quarter_str, output_dir, top_k_percent=0) # Pass empty dataframe but full nodes
             continue # 跳过当前季度
        num_stocks_in_matrix = matrix.shape[0]
        if num_stocks_in_matrix != len(target_stocks):
             print(f"错误：季度 {quarter_str} 重新索引后的股票数量不等于目标股票数量 ({num_stocks_in_matrix} != {len(target_stocks)})。")
             save_graph_data_pytorch(pd.DataFrame(), target_stocks, quarter_str, output_dir, top_k_percent=0) # Pass empty dataframe but full nodes
             continue # 跳过当前季度


        # 计算股票之间的余弦相似度
        try:
            # Cosine similarity of the N x M matrix results in an N x N similarity matrix
            # Ensure matrix only contains numerical columns for similarity calculation
            matrix_numeric = matrix.select_dtypes(include=np.number)
            if matrix_numeric.empty:
                 print(f"警告：季度 {quarter_str} 矩阵中没有数值列用于相似度计算。")
                 save_graph_data_pytorch(pd.DataFrame(), target_stocks, quarter_str, output_dir, top_k_percent=0) # Pass empty dataframe but full nodes
                 continue # 跳过当前季度


            sim_matrix = cosine_similarity(matrix_numeric)

            # Check for NaN results (can happen if a stock has zero holdings across all funds, which reindex+fillna(0) should handle,
            # but sometimes numerical issues can occur). Replace NaN/Inf with 0.
            if np.isnan(sim_matrix).any() or np.isinf(sim_matrix).any():
                print(f"警告：季度 {quarter_str} 计算出的相似度矩阵包含 NaN 或 Inf 值。将替换为 0。")
                sim_matrix = np.nan_to_num(sim_matrix, nan=0.0, posinf=0.0, neginf=0.0)


            # 将相似度矩阵转换为DataFrame，行和列索引都是完整的股票代码列表，顺序与 matrix 相同
            sim_df = pd.DataFrame(sim_matrix, index=matrix.index, columns=matrix.index)

        except Exception as e:
            print(f"为季度 {quarter_str} 计算余弦相似度时出错: {e}。跳过此季度。")
            save_graph_data_pytorch(pd.DataFrame(), target_stocks, quarter_str, output_dir, top_k_percent=0) # Pass empty dataframe but full nodes
            continue # 跳过当前季度


        # --- 可视化热力图 (可选) ---
        if plot_heatmaps:
            plot_similarity(sim_df, quarter_str, threshold=0.15, output_dir=output_dir)


        # --- 生成并保存包含节点信息的 PyTorch 图数据文件 ---
        # 将计算出的 sim_df (200x200) 和完整的 target_stocks 列表传递给保存函数
        save_graph_data_pytorch(sim_df, target_stocks, quarter_str, output_dir, top_k_percent=top_k_percent)

    print("\n--- 全部处理完成 ✅ ---")

# --- 主执行块 ---
if __name__ == "__main__":
    # --- 配置参数 ---
    # 自定义你的文件路径
    index_path = "index.xlsx" # 替换为你保存目标股票代码的文件路径 (e.g., 200支股票)
    data_dir = r"C:\Users\佚名\Downloads\基金持股文件205826688(仅供耶鲁大学使用)" # 替换为你的基金持仓文件目录
    output_dir = r"D:\graph_basic" # 输出目录名 (保存基金持股图数据到此处)
    num_fund_files = 5 # *** 确认你的基金持仓文件数量 (HLD_Fundhold0 to HLD_FundholdN-1) ***

    # 设置要保留的最强连接的比例 (例如 0.05 表示保留前 5% 的边)
    # 对于 200 个节点，总可能的非自环边数量是 200 * 199 = 39800
    # 保留 5% 就是 39800 * 0.05 = 1990 条边 (这是无向图的双向边，实际保存单向边数量会是它的一半)
    edge_retention_percentage = 0.05 # 建议从较小的比例开始尝试

    generate_heatmaps = True # 是否生成并保存热力图 (设为 True 如果需要)

    # --- 执行步骤 ---
    print("--- 开始执行基金持股图数据生成脚本 ---")
    print(f"目标股票文件: {index_path}")
    print(f"基金数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"基金文件数量: {num_fund_files}")
    print(f"保留最强连接比例: {edge_retention_percentage*100:.2f}%")
    print(f"生成热力图: {generate_heatmaps}")

    # 步骤 A: 读取目标股票列表并过滤基金持仓数据
    # 修改：read_and_filter_data 现在返回过滤后的数据和完整的排序后目标股票列表
    filtered_df, sorted_target_stocks = read_and_filter_data(index_path, data_dir, num_files=num_fund_files)

    # 步骤 B: 按季度构建相似度矩阵、邻接矩阵和包含节点信息的 GNN 格式文件
    if filtered_df is not None and not filtered_df.empty and sorted_target_stocks:
        # 修改：将完整的 sorted_target_stocks 列表传递给 build_similarity_matrices
        build_similarity_matrices(
            filtered_df,
            sorted_target_stocks, # 传递完整的股票列表
            output_dir=output_dir,
            top_k_percent=edge_retention_percentage, # 将保留比例作为参数传递
            plot_heatmaps=generate_heatmaps
        )
    elif not sorted_target_stocks:
         print("\n未能加载目标股票列表，无法进行后续处理。")
    else:
        print("\n未能加载或过滤到目标股票相关的持仓数据，无法进行后续处理。请检查输入文件和路径。")

    print("\n--- 脚本执行结束 ---")