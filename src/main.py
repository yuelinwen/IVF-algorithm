"""
Inverted File Index (IVF) 向量相似度搜索 - 学习用简化实现

核心思路:
1. 把所有向量用 K-Means 聚类，分成 K/nlist 个簇 (cluster)
2. 建立"倒排索引": 记录每个簇里有哪些向量
3. 搜索时，只在最近的几个簇里找，而不是扫描全部数据
   --> 这就是 IVF 比暴力搜索快的原因

比喻: 图书馆按主题分区，找书时只去相关区域，不用翻遍整个图书馆。
"""

import numpy as np
import random


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度，值越大越相似 (最大为 1.0)"""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return dot / norm


def find_nearest_cluster(vector, centroids):
    """找到离 vector 最近的聚类中心，返回其编号"""
    best_index = 0
    best_sim = -1
    for i, centroid in enumerate(centroids):
        sim = cosine_similarity(vector, centroid)
        if sim > best_sim:
            best_sim = sim
            best_index = i
    return best_index


# ──────────────────────────────────────────────
# 第一步: K-Means 聚类 (把向量分组)
# ──────────────────────────────────────────────

def kmeans(vectors, k, num_iterations=20):
    """
    简单 K-Means 聚类
    输入: 所有向量列表, 簇数量 k
    输出: k 个聚类中心 (centroids)
    """
    # 随机选 k 个向量作为初始中心
    centroids = random.sample(vectors, k)
    centroids = [np.array(c, dtype=float) for c in centroids]

    for iteration in range(num_iterations):
        # 把每个向量分配到最近的中心
        clusters = [[] for _ in range(k)]
        for vec in vectors:
            idx = find_nearest_cluster(vec, centroids)
            clusters[idx].append(vec)

        # 更新中心 = 簇内所有向量的平均值
        new_centroids = []
        for i in range(k):
            if clusters[i]:
                new_centroids.append(np.mean(clusters[i], axis=0))
            else:
                new_centroids.append(centroids[i])  # 空簇保留旧中心

        # 检查是否收敛 (中心不再移动)
        if all(np.allclose(new_centroids[i], centroids[i]) for i in range(k)):
            print(f"  K-Means 在第 {iteration+1} 轮收敛")
            break
        centroids = new_centroids

    return centroids


# ──────────────────────────────────────────────
# 第二步: 建立 IVF 索引
# ──────────────────────────────────────────────

def build_ivf_index(vectors, k=4):
    """
    建立 IVF 索引
    输入: 向量列表, 簇数量 k
    输出: (centroids, inverted_index)
      - centroids: k 个聚类中心
      - inverted_index: 字典，key=簇编号, value=[(向量编号, 向量), ...]
    """
    print(f"\n[建立索引] 对 {len(vectors)} 个向量做 K-Means 聚类，分成 {k} 个簇...")
    centroids = kmeans(vectors, k)

    # 倒排索引: 记录每个簇包含哪些向量
    inverted_index = {i: [] for i in range(k)}
    for vec_id, vec in enumerate(vectors):
        cluster_id = find_nearest_cluster(vec, centroids)
        inverted_index[cluster_id].append((vec_id, vec))

    print(f"[建立索引] 完成! 各簇大小: { {i: len(v) for i, v in inverted_index.items()} }")
    return centroids, inverted_index


# ──────────────────────────────────────────────
# 第三步: IVF 搜索
# ──────────────────────────────────────────────

def ivf_search(query, centroids, inverted_index, top_k=3, n_probe=2):
    """
    IVF 搜索
    输入:
      query          - 查询向量
      centroids      - 聚类中心列表
      inverted_index - 倒排索引
      top_k          - 返回最相似的前 k 个
      n_probe        - 搜索几个簇 (越大越准但越慢)
    输出: [(相似度, 向量编号), ...] 排序后的结果
    """
    # 找最近的 n_probe 个簇
    cluster_sims = [(cosine_similarity(query, c), i) for i, c in enumerate(centroids)]
    cluster_sims.sort(reverse=True)
    probe_clusters = [idx for _, idx in cluster_sims[:n_probe]]

    print(f"  查询向量最近的 {n_probe} 个簇: {probe_clusters}")

    # 只在这几个簇里搜索
    candidates = []
    for cluster_id in probe_clusters:
        for vec_id, vec in inverted_index[cluster_id]:
            sim = cosine_similarity(query, vec)
            candidates.append((sim, vec_id))

    # 按相似度排序，返回 top_k
    candidates.sort(reverse=True)
    return candidates[:top_k]


# ──────────────────────────────────────────────
# 对比: 暴力搜索 (Brute Force)
# ──────────────────────────────────────────────

def brute_force_search(query, vectors, top_k=3):
    """暴力搜索: 和每一个向量都计算相似度"""
    results = [(cosine_similarity(query, vec), i) for i, vec in enumerate(vectors)]
    results.sort(reverse=True)
    return results[:top_k]


# ──────────────────────────────────────────────
# 主程序: 演示
# ──────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  IVF 向量相似度搜索 演示")
    print("=" * 55)

    # 生成随机数据: 50 个 8 维向量
    np.random.seed(42)
    random.seed(42)
    num_vectors = 50
    dim = 8
    vectors = [np.random.randn(dim) for _ in range(num_vectors)]

    print(f"\n数据集: {num_vectors} 个 {dim} 维向量")

    # ── 建立 IVF 索引 ──
    centroids, inverted_index = build_ivf_index(vectors, k=5)

    # ── 生成一个查询向量 ──
    query = np.random.randn(dim)
    print(f"\n查询向量: {np.round(query, 2)}")

    # ── IVF 搜索 ──
    print("\n[IVF 搜索] (n_probe=2，只搜索最近的 2 个簇)")
    ivf_results = ivf_search(query, centroids, inverted_index, top_k=3, n_probe=2)
    print("  IVF 结果 (相似度, 向量编号):")
    for sim, vid in ivf_results:
        print(f"    向量 #{vid:2d}  相似度 = {sim:.4f}")

    # ── 暴力搜索对比 ──
    print("\n[暴力搜索] (扫描全部 50 个向量)")
    bf_results = brute_force_search(query, vectors, top_k=3)
    print("  暴力搜索结果 (相似度, 向量编号):")
    for sim, vid in bf_results:
        print(f"    向量 #{vid:2d}  相似度 = {sim:.4f}")

    # ── 对比准确率 ──
    ivf_ids = {vid for _, vid in ivf_results}
    bf_ids  = {vid for _, vid in bf_results}
    overlap = ivf_ids & bf_ids
    recall = len(overlap) / len(bf_ids)
    print(f"\n[对比] IVF 召回率 = {recall:.0%}  "
          f"(n_probe 越大召回越高，但速度越慢)")

    # ── 展示 IVF 的核心权衡 ──
    print("\n" + "=" * 55)
    print("  n_probe 对召回率的影响 (用全量簇数=5 对比)")
    print("=" * 55)
    bf_ids_set = {vid for _, vid in brute_force_search(query, vectors, top_k=5)}
    for n_probe in [1, 2, 3, 5]:
        results = ivf_search(query, centroids, inverted_index, top_k=5, n_probe=n_probe)
        ids = {vid for _, vid in results}
        r = len(ids & bf_ids_set) / len(bf_ids_set)
        scanned = sum(len(inverted_index[i]) for _, i in
                      sorted([(cosine_similarity(query, c), i)
                               for i, c in enumerate(centroids)], reverse=True)[:n_probe])
        print(f"  n_probe={n_probe}  召回率={r:.0%}  扫描向量数={scanned}/{num_vectors}")


if __name__ == "__main__":
    main()
