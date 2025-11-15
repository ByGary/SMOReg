import glob
import os
import shutil
import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score, rand_score, adjusted_rand_score, \
    homogeneity_score, completeness_score, v_measure_score, silhouette_score
from sklearn.neighbors import kneighbors_graph
import libpysal
import esda
import scanpy as sc
from config.defaults import get_default_config
from permetrics import ClusteringMetric


def load_config():
    config = get_default_config()
    config.merge_from_file('')
    config.freeze()
    return config


cfg = load_config()


def clustering(adata, clustering_algo, n_clusters, n_neighbors, random_state, key='hidden_emb', start=0.01, end=0.8, increment=0.01):

    if clustering_algo == 'mclust':
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=key)
        adata = mclust_R(adata, n_clusters, random_state, used_obsm=key)
    elif clustering_algo == 'leiden':
        res = search_res(adata, clustering_algo, n_clusters, n_neighbors, random_state, use_rep=key, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=random_state, resolution=res)
    elif clustering_algo == 'louvain':
        res = search_res(adata, clustering_algo, n_clusters, n_neighbors, random_state, use_rep=key, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=random_state, resolution=res)


def mclust_R(adata, num_cluster, random_state, used_obsm='hidden_emb', modelNames='EEE'):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_state)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    print("res:", res)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def search_res(adata, clustering_algo, n_clusters, n_neighbors, random_state, use_rep='hidden_emb', start=0.1, end=3.0, increment=0.01):
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if clustering_algo == 'leiden':
            sc.tl.leiden(adata, random_state=random_state, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            MI, NMI, AMI, RI, ARI, Homogeneity, completeness, V_measure, purity = supervised_metrics(adata.obs['cell_type'], adata.obs[clustering_algo])
        elif clustering_algo == 'louvain':
            sc.tl.louvain(adata, random_state=random_state, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
        if count_unique == n_clusters:
            label = 1
            break
    return res


# def search_res(adata, clustering_algo, n_clusters, n_neighbors, random_state, use_rep='hidden_emb', start=0.1, end=3.0, increment=0.01):
#     print('Searching resolution...')
#     label = 0
#     sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
#     for res in sorted(list(np.arange(start, end, increment)), reverse=True):
#         if clustering_algo == 'leiden':
#             sc.tl.leiden(adata, random_state=random_state, resolution=res)
#             count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
#             moran, silhouette_avg, dhi, bi, chi, hi = unsupervised_metrics(adata.obsm[use_rep],
#                                                                            adata.obsm['spatial'],
#                                                                            adata.obs['leiden'].astype(int))
#             # sc.pl.spatial(adata, color=[clustering_algo], spot_size=140, show=False, title=f"cluster:{count_unique} res: {res}")
#             # plt.show()
#         elif clustering_algo == 'louvain':
#             sc.tl.louvain(adata, random_state=random_state, resolution=res)
#             count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
#         if count_unique == n_clusters:
#             label = 1
#             break
#     assert label == 1, "Resolution is not found. Please try bigger range or smaller step!"
#
#     return res


def supervised_metrics(y_true, y_pred):
    MI = mutual_info_score(y_true, y_pred)
    NMI = normalized_mutual_info_score(y_true, y_pred)
    AMI = adjusted_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    RI = rand_score(y_true, y_pred)
    ARI = adjusted_rand_score(y_true, y_pred)
    Homogeneity = homogeneity_score(y_true, y_pred)
    completeness = completeness_score(y_true, y_pred)
    V_measure = v_measure_score(y_true, y_pred)

    def purity_score(y_true, y_pred):
        cm = contingency_matrix(y_true, y_pred)
        purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
        return purity

    purity = purity_score(y_true, y_pred)

    return MI, NMI, AMI, RI, ARI, Homogeneity, completeness, V_measure, purity


def unsupervised_metrics(hidden_emb, spatial_coords, cluster_label):
    coords = list(zip(spatial_coords[:, 0], spatial_coords[:, 1]))
    w = libpysal.weights.KNN.from_array(coords, k=10)
    w.transform = 'r'
    moran = esda.Moran(np.mean(hidden_emb, axis=1), w)
    silhouette_avg = silhouette_score(X=hidden_emb, labels=cluster_label, metric='euclidean')

    cm = ClusteringMetric(X=hidden_emb, y_pred=cluster_label.to_list())
    chi = cm.calinski_harabasz_index()

    return moran.I, silhouette_avg, chi


def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def pca(adata, use_reps=None, n_comps=10):
    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca



def save_all_regl(perm_matrices, gene2id, pro2id, result_folder):
    print("Saving all the regulatory networks...")
    regl_folder = os.path.join(result_folder, 'all_regular')
    if not os.path.exists(regl_folder):
        os.makedirs(regl_folder)
    id2gene = {v: k for k, v in gene2id.items()}
    id2pro = {v: k for k, v in pro2id.items()}

    for i, matrix in enumerate(perm_matrices):

        df = pd.DataFrame(matrix, index=[id2gene[idx] for idx in range(matrix.shape[0])],
                          columns=[id2pro[idx] for idx in range(matrix.shape[1])])
        csv_filename = os.path.join(regl_folder, f'Spot_{i}.csv')
        df.to_csv(csv_filename)

    print("All the regulatory networks of {} cells have been saved.".format(perm_matrices.shape[0]))


def save_all_regl_simulate(perm_matrices, obs_names, gene_names, pro_names, result_folder):
    print("Saving all the regulatory networks...")
    regl_folder = os.path.join(result_folder, 'all_regular')
    if not os.path.exists(regl_folder):
        os.makedirs(regl_folder)
    for i, matrix in enumerate(perm_matrices):
        df = pd.DataFrame(matrix, index=gene_names,
                          columns=pro_names)
        csv_filename = os.path.join(regl_folder, f'{obs_names[i]}.csv')
        df.to_csv(csv_filename)

    print("All the regulatory networks of {} cells have been saved.".format(perm_matrices.shape[0]))


def save_code(result_folder):

    source_dir = os.getcwd()
    target_dir = os.path.join(result_folder, 'code')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    python_files = glob.glob(os.path.join(source_dir, '*.py'))
    for file_path in python_files:
        shutil.copy(file_path, target_dir)
    print(f"All .py files have been copied to {target_dir}")

    shutil.copy(os.path.join(source_dir, 'config', 'exp.yaml'), os.path.join(result_folder, 'exp.yaml'))


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)