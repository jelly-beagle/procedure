import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations


# --- 配置参数 ---
DATA_PATH_TEMPLATE = r"D:\news\processed_results\dedup_processed_{year}_part{num}.csv"
INDEX_PATH = "code0.xlsx"  # 指数股票池文件路径
OUTPUT_DIR = r"D:\news\cooccurrence_graph_output" # 输出目录

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
MIN_COOCCURRENCE_FREQ = 1 # 最小共现频率：一对股票共现次数需达到此阈值，才能在共现图中建立边

# 最终希望选取的股票数量
TARGET_STOCK_COUNT = 200
# 混合筛选比例
HIGH_CENTRALITY_PERCENT = 0.80
LOW_CENTRALITY_PERCENT = 0.20

# 计算高中心性和低中心性需要选取的股票数量
num_high_centrality = int(TARGET_STOCK_COUNT * HIGH_CENTRALITY_PERCENT)
num_low_centrality = TARGET_STOCK_COUNT - num_high_centrality # 确保总数为TARGET_STOCK_COUNT


# --- 步骤 1: 数据处理和共现矩阵构建 (逐年累加) ---

print("--- 步骤 1: 数据处理和共现矩阵构建 ---")

edge_counter_by_year = defaultdict(lambda: defaultdict(int)) # 按年份存储边的共现次数
news_count_by_month = defaultdict(int) # 按年月统计新闻数量
for year in tqdm(YEARS, desc="Processing Years"):
    part = 1
    unique_news_this_year_processing = set() # 跟踪当前年份已处理过的唯一新闻实例
    while True:
        file_path = DATA_PATH_TEMPLATE.format(year=year, num=part)
        if not os.path.exists(file_path):
            break # 如果文件不存在，假设当前年份的part文件已处理完毕
        try:
            df = pd.read_csv(file_path, sep=',', usecols=['News1_ID', 'Reptime', 'codesnums', 'newscodes'], dtype=str)
        except FileNotFoundError:
             print(f"\n文件未找到，停止处理年份 {year} 的 part {part}")
             break
        except Exception as e:
            print(f"\n读取文件 {file_path} 时出错: {e}")
            part += 1
            continue
        df = df.drop_duplicates(subset=['News1_ID', 'Reptime', 'codesnums'])

        for _, row in df.iterrows():
            news_id = row['News1_ID']
            reptime_str = row['Reptime']
            codesnums_str = row['codesnums']
            news_identifier = (news_id, reptime_str, codesnums_str)
            if news_identifier not in unique_news_this_year_processing:
                 unique_news_this_year_processing.add(news_identifier)
                 # 更新新闻数量统计（按月）
                 try:
                     date = pd.to_datetime(reptime_str)
                     ym = f"{date.year}-{date.month:02d}"
                     news_count_by_month[ym] += 1
                 except Exception as date_e:
                     pass
                 # 构建共现矩阵
                 newscodes_str = row['newscodes']
                 if pd.notna(newscodes_str):
                    codes = [code.strip() for code in newscodes_str.split()]
                    unique_codes_in_news = sorted(list(set(codes)))

                    for code1, code2 in combinations(unique_codes_in_news, 2):
                        edge_counter_by_year[year][(code1, code2)] += 1
        part += 1

print("步骤 1 完成：数据处理和共现矩阵构建完毕。")


# --- 步骤 2: 新闻数量可视化 ---

print("\n--- 步骤 2: 新闻数量可视化 ---")
if news_count_by_month:
    month_df = pd.DataFrame.from_dict(news_count_by_month, orient='index', columns=['news_count'])
    month_df.index = pd.to_datetime(month_df.index)
    month_df = month_df.sort_index()

    plt.figure(figsize=(15, 7))
    sns.lineplot(data=month_df, x=month_df.index, y='news_count')
    plt.title("Monthly Unique News Volume Trend (2019-2023)")
    plt.xlabel("Date")
    plt.ylabel("Number of Unique News Articles")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    news_volume_plot_path = os.path.join(OUTPUT_DIR, "monthly_unique_news_trend.png")
    plt.savefig(news_volume_plot_path)
    plt.close()
    print(f"步骤 2 完成：已保存每月新闻数量趋势图到 {news_volume_plot_path}")
else:
    print("步骤 2 未完成：没有收集到新闻数量数据进行可视化。")


# --- 步骤 3: 逐年构建图、可视化最大连通分量并初步筛选股票池 ---

print("\n--- 步骤 3: 逐年构建图、可视化和初步筛选 ---")

graph_by_year = {} # 存储每年的图对象
largest_component_nodes_by_year = {} # 存储每年最大连通分量的节点

for year in tqdm(YEARS, desc="Building Yearly Graphs and Filtering"):
    G = nx.Graph()
    year_edges = edge_counter_by_year[year]

    for (c1, c2), weight in year_edges.items():
        if weight >= MIN_COOCCURRENCE_FREQ:
            G.add_edge(c1, c2, weight=weight)

    graph_by_year[year] = G

    # print(f"\n处理年份 {year} 的图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边.")

    if G.number_of_nodes() > 0:
        components = list(nx.connected_components(G))
        if components:
            components = sorted(components, key=len, reverse=True)
            largest_component = components[0]
            largest_component_nodes_by_year[year] = list(largest_component)
            G_largest_cc = G.subgraph(largest_component).copy()

            # print(f"  年份 {year} 最大连通分量: {G_largest_cc.number_of_nodes()} 节点.")

            if G_largest_cc.number_of_nodes() > 1:
                plt.figure(figsize=(12, 10))
                pos = nx.spring_layout(G_largest_cc, seed=42)
                degrees = dict(G_largest_cc.degree())
                node_size = [max(10, v * 50) for v in degrees.values()]
                node_color = [degrees[n] for n in G_largest_cc.nodes()]

                nx.draw(G_largest_cc, pos, with_labels=False, node_size=node_size, node_color=node_color,
                        cmap=plt.cm.viridis, edge_color='gray', alpha=0.7)
                plt.title(f"年份 {year} 股票共现最大连通分量", fontsize=16)
                plt.axis('off')
                graph_plot_path = os.path.join(OUTPUT_DIR, f"graph_{year}_largest_component.png")
                plt.savefig(graph_plot_path)
                plt.close()
                # print(f"  已保存年份 {year} 最大连通分量图到 {graph_plot_path}")
            # else:
                # print(f"  年份 {year} 最大连通分量只有 1 个节点，跳过可视化。")
        # else:
             # largest_component_nodes_by_year[year] = []
             # print(f"  年份 {year} 没有找到连通分量。")
    # else:
        # largest_component_nodes_by_year[year] = []
        # print(f"  年份 {year} 图中没有节点。")

# 合并各年最大连通分量中的节点，形成初步股票池 (在各年最大连通分量中)
preliminary_stock_pool_by_component = set()
for year, nodes in largest_component_nodes_by_year.items():
    preliminary_stock_pool_by_component.update(nodes)

print(f"步骤 3 完成：已合并各年最大连通分量中的 {len(preliminary_stock_pool_by_component)} 个独特股票。")

# --- 步骤 4: 加载指数股票池并与初步股票池取交集 (初步筛选结果) ---

print("\n--- 步骤 4: 加载指数股票池并取交集 ---")
index_stock_pool = set()
try:
    df_index = pd.read_excel(INDEX_PATH)
    if 'index' in df_index.columns:
        index_stock_pool = set(df_index['index'].astype(str).str.zfill(6).tolist())
        print(f"已从指数文件加载 {len(index_stock_pool)} 个股票代码。")
    else:
        print(f"警告：指数文件 {INDEX_PATH} 中没有找到 'symbol' 列，跳过指数加载。")
except FileNotFoundError:
    print(f"错误：指数文件未找到于 {INDEX_PATH}，跳过指数加载。")
except Exception as e:
    print(f"加载指数文件 {INDEX_PATH} 时出错: {e}")

# 与指数股票池取交集，得到最终用于混合筛选的股票池 (在各年最大连通分量中且在指数内)
if index_stock_pool:
    final_stock_pool = sorted(list(preliminary_stock_pool_by_component.intersection(index_stock_pool)))
    print(f"步骤 4 完成：初步筛选出 {len(final_stock_pool)} 个股票 (在各年最大连通分量中且在指数内)。")
else:
    # 如果指数未加载，则初步筛选结果就是各年最大连通分量的合并
    final_stock_pool = sorted(list(preliminary_stock_pool_by_component))
    print(f"步骤 4 完成：指数未加载，初步筛选结果为各年最大连通分量合并的 {len(final_stock_pool)} 个股票。")


# --- 步骤 5: 构建总体的共现网络 (仅包含初步筛选出的股票) ---

print(f"\n--- 步骤 5: 构建总体的共现网络 ({len(final_stock_pool)} 个股票) ---")

combined_G = nx.Graph() # 创建一个总体的无向图对象

# 累加所有年份的共现计数到总计数中
all_years_total_edges = defaultdict(int)
for year, edge_counts in edge_counter_by_year.items():
    for (c1, c2), count in edge_counts.items():
        all_years_total_edges[(c1, c2)] += count

# 仅将初步筛选出的股票之间的边添加到总图中
selected_stock_set = set(final_stock_pool)
for (c1, c2), total_count in all_years_total_edges.items():
    if c1 in selected_stock_set and c2 in selected_stock_set and total_count >= MIN_COOCCURRENCE_FREQ:
        combined_G.add_edge(c1, c2, weight=total_count)

print(f"步骤 5 完成：已构建包含 {combined_G.number_of_nodes()} 节点和 {combined_G.number_of_edges()} 边的总共现图 (已根据初步筛选池过滤)。")


# --- 步骤 6: 根据中心性进行混合筛选 ---

print("\n--- 步骤 6: 根据中心性进行混合筛选 ---")

final_selected_stocks_mixed = [] # 存储最终混合筛选出的股票代码

if combined_G.number_of_nodes() < TARGET_STOCK_COUNT:
    print(f"警告：总图节点数量 ({combined_G.number_of_nodes()}) 少于目标筛选数量 ({TARGET_STOCK_COUNT})。将选取总图中的所有节点。")
    final_selected_stocks_mixed = list(combined_G.nodes())

else:
    print("计算特征向量中心性进行混合筛选...")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(combined_G, max_iter=1000, tol=1e-6)
        print("特征向量中心性计算完成。")

        sorted_stocks_by_centrality = sorted(eigenvector_centrality.items(), key=lambda item: item[1], reverse=True)

        # 选取高中心性股票
        high_centrality_stocks = [stock for stock, centrality in sorted_stocks_by_centrality[:num_high_centrality]]

        # 选取低中心性股票
        low_centrality_stocks = [stock for stock, centrality in sorted_stocks_by_centrality[-num_low_centrality:]]

        final_selected_stocks_mixed = list(set(high_centrality_stocks + low_centrality_stocks))

        if len(final_selected_stocks_mixed) != TARGET_STOCK_COUNT:
             print(f"警告：混合筛选结果数量为 {len(final_selected_stocks_mixed)}，与目标数量 {TARGET_STOCK_COUNT} 不符。")

        print(f"步骤 6 完成：已成功混合筛选出 {len(final_selected_stocks_mixed)} 个股票。")

    except nx.PowerIterationFailedConvergence as e:
        print(f"计算特征向量中心性时出错: {e}")
        print("特征向量中心性计算未收敛，混合筛选失败。请检查图结构或增加 max_iter。")
    except Exception as e:
        print(f"混合筛选中心性计算时发生意外错误: {e}")


# --- 步骤 7: 输出最终混合筛选的股票池 ---

print("\n--- 步骤 7: 输出最终筛选的股票池 ---")
print("最终筛选股票池 (混合方法 - 仅代码):")
if final_selected_stocks_mixed:
    for code in final_selected_stocks_mixed:
        print(code)

    df_final_mixed = pd.DataFrame(final_selected_stocks_mixed, columns=['Symbol'])
    final_pool_output_path_mixed = os.path.join(OUTPUT_DIR, f"final_selected_stock_pool_mixed_top{TARGET_STOCK_COUNT}_scientific.xlsx")
    df_final_mixed.to_excel(final_pool_output_path_mixed, index=False)
    print(f"\n步骤 7 完成：已保存最终混合筛选股票池到 {final_pool_output_path_mixed}")
else:
    print("\n步骤 7 未完成：最终混合筛选股票池为空。")


# --- 步骤 8: 可视化最终筛选的股票之间的网络 ---

print(f"\n--- 步骤 8: 可视化最终筛选的 {len(final_selected_stocks_mixed)} 个股票之间的网络 ---")

# 从总图中创建包含最终选定股票的子图
sub_G = nx.Graph()
selected_stock_set_viz = set(final_selected_stocks_mixed)

# 遍历combined_G中的边，仅添加两个端点都在最终选定股票集合中的边
for u, v, data in combined_G.edges(data=True):
    if u in selected_stock_set_viz and v in selected_stock_set_viz:
        weight = data.get('weight', 0)
        # 可以选择在这里应用一个更高的阈值来简化可视化，例如：
        # MIN_DISPLAY_EDGE_WEIGHT = 5 # 示例阈值
        # if weight >= MIN_DISPLAY_EDGE_WEIGHT:
        sub_G.add_edge(u, v, weight=weight)


if sub_G.number_of_nodes() == 0:
    print("步骤 8 未完成：最终筛选股票子图中没有节点，跳过可视化。")
else:
    plt.figure(figsize=(15, 12))

    # 布局算法
    try:
        pos = nx.spring_layout(sub_G, k=0.5, iterations=100, seed=42) # 调整k和iterations，固定seed
    except Exception as e:
        print(f"布局计算出错: {e}，尝试使用随机布局。")
        pos = nx.random_layout(sub_G, seed=42) # 备用布局

    # 节点样式 (根据度数)
    degrees = dict(sub_G.degree())
    # 防止所有度数都为0导致max错误，或度数太小导致节点看不见
    if degrees:
         max_degree = max(degrees.values()) if max(degrees.values()) > 0 else 1
         node_size = [max(50, v / max_degree * 500) for v in degrees.values()] # 根据相对度数调整大小
         node_color = [degrees[n] for n in sub_G.nodes()] # 根据度数着色
         cmap = plt.cm.viridis
    else:
         node_size = 100 # 默认大小
         node_color = 'skyblue'
         cmap = None


    # 绘制节点
    nodes = nx.draw_networkx_nodes(sub_G, pos, node_size=node_size, node_color=node_color,
                                   cmap=cmap, alpha=0.9)


    # 绘制边
    # 可以根据边的权重调整透明度或粗细
    # edge_weights_list = [d.get('weight', 1) for u, v, d in sub_G.edges(data=True)]
    # if edge_weights_list:
    #     max_weight = max(edge_weights_list) if max(edge_weights_list) > 0 else 1
    #     edge_alpha = [w / max_weight * 0.7 + 0.2 for w in edge_weights_list] # 透明度根据权重变化 (0.2-0.9)
    # else:
    #     edge_alpha = 0.5

    nx.draw_networkx_edges(sub_G, pos, edge_color='gray', alpha=0.5, width=1.0)


    # 绘制节点标签 (可选，可能重叠)
    # 默认不绘制所有标签
    # nx.draw_networkx_labels(sub_G, pos, font_size=9, font_color='black')

    plt.title(f"最终筛选的 {len(final_selected_stocks_mixed)} 个股票共现网络", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    graph_output_path = os.path.join(OUTPUT_DIR, f"final_selected_stocks_network_{len(final_selected_stocks_mixed)}.png")
    plt.savefig(graph_output_path, dpi=300)
    plt.close()
    print(f"步骤 8 完成：已保存最终筛选股票网络可视化图到 {graph_output_path}")