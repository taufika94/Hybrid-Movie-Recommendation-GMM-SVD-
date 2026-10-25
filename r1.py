import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import joblib
import warnings
import os
import time
import scipy.stats as stats
import random
import sys
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity
import pickle
warnings.filterwarnings('ignore')


# ====================== KONSTANTA PATH ======================
# ====================== KONSTANTA PATH ======================
# Folder tempat r1.py berada (otomatis menyesuaikan lokasi)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model folder: relatif terhadap r1.py
MODEL_DIR = os.path.join(BASE_DIR, "modell1")

def save_all_models(save_dir=None):
    """
    Menyimpan semua komponen model yang sudah di-train ke folder.
    """
    import os
    import joblib
    import pickle
    
    if save_dir is None:
        save_dir = MODEL_DIR
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    saved_files = []
    
    # 1. Simpan SVD model
    if st.session_state.get('svd') is not None:
        svd_path = os.path.join(save_dir, 'svd_model.pkl')
        joblib.dump(st.session_state.svd, svd_path)
        saved_files.append(('SVD Model', svd_path))
        print(f"SVD model saved to {svd_path}")
    
    # 2. Simpan komponen Hybrid model
    if st.session_state.get('hybrid') is not None:
        hybrid = st.session_state.hybrid
        
        if hasattr(hybrid, 'gmm_map') and hybrid.gmm_map is not None:
            gmm_map_path = os.path.join(save_dir, 'gmm_map.pkl')
            joblib.dump(hybrid.gmm_map, gmm_map_path)
            saved_files.append(('GMM Map', gmm_map_path))
            print(f"GMM Map saved: {len(hybrid.gmm_map)} items")
        
        if hasattr(hybrid, 'user_pref') and hybrid.user_pref is not None:
            user_pref_path = os.path.join(save_dir, 'user_pref.npy')
            np.save(user_pref_path, hybrid.user_pref)
            saved_files.append(('User Preference', user_pref_path))
            print(f"User Preference saved: shape {hybrid.user_pref.shape}")
        
        if hasattr(hybrid, 'item_vectors') and hybrid.item_vectors:
            item_vectors_path = os.path.join(save_dir, 'item_vectors.pkl')
            joblib.dump(hybrid.item_vectors, item_vectors_path)
            saved_files.append(('Item Vectors', item_vectors_path))
    
    # 3. Simpan gmm_map tambahan
    if st.session_state.get('gmm_map') is not None:
        gmm_map_full_path = os.path.join(save_dir, 'gmm_map_full.pkl')
        joblib.dump(st.session_state.gmm_map, gmm_map_full_path)
        saved_files.append(('GMM Map Full', gmm_map_full_path))
        print(f"GMM Map Full saved: {len(st.session_state.gmm_map)} items")
    
    # 4. Simpan trainset dan testset
    if st.session_state.get('trainset') is not None:
        trainset_path = os.path.join(save_dir, 'trainset.pkl')
        with open(trainset_path, 'wb') as f:
            pickle.dump(st.session_state.trainset, f)
        saved_files.append(('Trainset', trainset_path))
        print(f"Trainset saved")
    
    if st.session_state.get('testset') is not None:
        testset_path = os.path.join(save_dir, 'testset.pkl')
        with open(testset_path, 'wb') as f:
            pickle.dump(st.session_state.testset, f)
        saved_files.append(('Testset', testset_path))
        print(f"Testset saved")
    
    # 5. Simpan genre cols
    if st.session_state.get('genre_cols') is not None:
        genre_path = os.path.join(save_dir, 'genre_cols.pkl')
        joblib.dump(st.session_state.genre_cols, genre_path)
        saved_files.append(('Genre Columns', genre_path))
    
    # 6. Simpan cluster genre mapping
    if st.session_state.get('cluster_genre_mapping') is not None:
        cluster_path = os.path.join(save_dir, 'cluster_genre_mapping.pkl')
        joblib.dump(st.session_state.cluster_genre_mapping, cluster_path)
        saved_files.append(('Cluster Genre Mapping', cluster_path))
    
    # 7. Simpan metrik
    metrics = {
        'svd_rmse': st.session_state.get('svd_rmse', 0),
        'svd_mae': st.session_state.get('svd_mae', 0),
        'hybrid_rmse': st.session_state.get('hybrid_rmse', 0),
        'hybrid_mae': st.session_state.get('hybrid_mae', 0),
        'model_trained': st.session_state.get('model_trained', False),
        'n_factors': st.session_state.get('n_factors', 100),
        'n_epochs': st.session_state.get('n_epochs', 30),
        'max_components': st.session_state.get('max_components', 50),
        'smoothing': st.session_state.get('smoothing', 0.1)
    }
    
    metrics_path = os.path.join(save_dir, 'metrics.pkl')
    joblib.dump(metrics, metrics_path)
    saved_files.append(('Metrics', metrics_path))
    print(" Metrics saved")
    
    # 8. Simpan dataframe hasil prediksi
    if st.session_state.get('svd_predictions_df') is not None:
        svd_df_path = os.path.join(save_dir, 'svd_predictions_df.pkl')
        joblib.dump(st.session_state.svd_predictions_df, svd_df_path)
        saved_files.append(('SVD Predictions DF', svd_df_path))
    
    if st.session_state.get('hybrid_predictions_df') is not None:
        hybrid_df_path = os.path.join(save_dir, 'hybrid_predictions_df.pkl')
        joblib.dump(st.session_state.hybrid_predictions_df, hybrid_df_path)
        saved_files.append(('Hybrid Predictions DF', hybrid_df_path))
    
    if st.session_state.get('comparison_df') is not None:
        comparison_df_path = os.path.join(save_dir, 'comparison_df.pkl')
        joblib.dump(st.session_state.comparison_df, comparison_df_path)
        saved_files.append(('Comparison DF', comparison_df_path))
    
    # 9. Simpan data split
    if st.session_state.get('train_df') is not None:
        st.session_state.train_df.to_csv(os.path.join(save_dir, 'train_df.csv'), index=False)
        saved_files.append(('Train Data CSV', os.path.join(save_dir, 'train_df.csv')))
    
    if st.session_state.get('test_df') is not None:
        st.session_state.test_df.to_csv(os.path.join(save_dir, 'test_df.csv'), index=False)
        saved_files.append(('Test Data CSV', os.path.join(save_dir, 'test_df.csv')))
    
    # 10. Simpan train_movies dan test_movies
    if st.session_state.get('train_movies') is not None:
        train_movies_path = os.path.join(save_dir, 'train_movies.pkl')
        joblib.dump(list(st.session_state.train_movies), train_movies_path)
        saved_files.append(('Train Movies', train_movies_path))
    
    if st.session_state.get('test_movies') is not None:
        test_movies_path = os.path.join(save_dir, 'test_movies.pkl')
        joblib.dump(list(st.session_state.test_movies), test_movies_path)
        saved_files.append(('Test Movies', test_movies_path))
    
    # 11. Simpan grid results (Tab 2)
    if st.session_state.get('grid_results') is not None:
        grid_path = os.path.join(save_dir, 'grid_results.pkl')
        joblib.dump(st.session_state.grid_results, grid_path)
        saved_files.append(('Grid Results', grid_path))
        print("Grid results saved")
    
    # 12. Simpan CV results (Tab 4)
    if st.session_state.get('tab3_results') is not None:
        cv_path = os.path.join(save_dir, 'cv_results.pkl')
        joblib.dump(st.session_state.tab3_results, cv_path)
        saved_files.append(('CV Results', cv_path))
        print("CV results saved")
    
    # 13. Simpan split info
    split_info = {
        'split_done': st.session_state.get('split_done', False),
        'train_movies': list(st.session_state.get('train_movies', set())) if st.session_state.get('train_movies') else [],
        'test_movies': list(st.session_state.get('test_movies', set())) if st.session_state.get('test_movies') else []
    }
    split_path = os.path.join(save_dir, 'split_info.pkl')
    joblib.dump(split_info, split_path)
    saved_files.append(('Split Info', split_path))
    
    # 14. Simpan tab3_processed flag
    if st.session_state.get('tab3_processed', False):
        tab3_processed_path = os.path.join(save_dir, 'tab3_processed.pkl')
        joblib.dump(True, tab3_processed_path)
        saved_files.append(('Tab3 Processed', tab3_processed_path))
    
    # 15. Simpan best params
    if st.session_state.get('grid_results') is not None and st.session_state.grid_results.get('best_params'):
        best_params_path = os.path.join(save_dir, 'best_params.pkl')
        joblib.dump(st.session_state.grid_results['best_params'], best_params_path)
        saved_files.append(('Best Parameters', best_params_path))
    
    print(f"\nTotal {len(saved_files)} komponen berhasil disimpan ke {save_dir}")
    return saved_files




def reevaluate_and_display_models():
    """
    Evaluasi ulang model dan tampilkan tabel (dipanggil saat user request).
    """
    try:
        trainset = st.session_state.get('trainset')
        testset = st.session_state.get('testset')
        svd = st.session_state.get('svd')
        hybrid = st.session_state.get('hybrid')
        movies_df = st.session_state.get('movies')
        ratings_df = st.session_state.get('ratings')
        
        if testset is None or svd is None or hybrid is None:
            st.error("Data atau model tidak lengkap untuk evaluasi ulang")
            return False
        
        if movies_df is None:
            st.error("Data film tidak tersedia")
            return False
        
        with st.spinner("Melakukan evaluasi ulang model..."):
            svd_errors = []
            hybrid_errors = []
            
            svd_data = []
            hybrid_data = []
            comparison_data = []
            
            total_test = len(testset)
            progress_bar = st.progress(0)
            
            for idx, (uid, iid, true_r) in enumerate(testset):
                if idx % 100 == 0:
                    progress_bar.progress(min(idx / total_test, 1.0))
                
                # SVD prediction
                svd_pred = svd.predict(uid, iid).est
                svd_error = abs(svd_pred - true_r)
                svd_errors.append(svd_error)
                
                # Hybrid prediction
                hybrid_pred, gmm_used = hybrid.predict(uid, iid)
                hybrid_error = abs(hybrid_pred - true_r)
                hybrid_errors.append(hybrid_error)
                
                # Get user info
                try:
                    u_inner = trainset.to_inner_uid(str(uid))
                    n_user_ratings = len(trainset.ur[u_inner])
                except:
                    n_user_ratings = 0
                
                # Get movie info
                try:
                    movie_id_int = int(iid)
                    movie_info = movies_df[movies_df['movieId'] == movie_id_int]
                    if not movie_info.empty:
                        title = movie_info['title'].values[0]
                        genres = movie_info['genres'].values[0]
                    else:
                        title = f"Movie {iid}"
                        genres = "Unknown"
                except:
                    title = f"Movie {iid}"
                    genres = "Unknown"
                
                # SVD data
                svd_data.append({
                    'No': idx + 1,
                    'User ID': uid,
                    'Film': title[:50] + '...' if len(title) > 50 else title,
                    'Genre': genres[:30] + '...' if len(genres) > 30 else genres,
                    'Jml Rating': n_user_ratings,
                    'Actual': f"{true_r:.1f}",
                    'Prediksi SVD': f"{svd_pred:.3f}",
                    'Error': f"{svd_error:.3f}"
                })
                
                # GMM info
                gmm_cluster = None
                gmm_prob = None
                iid_str = str(iid)
                gmm_map = st.session_state.get('gmm_map', {})
                if iid_str in gmm_map:
                    gmm_vector = gmm_map[iid_str]
                    gmm_cluster = np.argmax(gmm_vector)
                    gmm_prob = gmm_vector[gmm_cluster]
                
                # Hybrid data
                hybrid_data.append({
                    'No': idx + 1,
                    'User ID': uid,
                    'Film': title[:50] + '...' if len(title) > 50 else title,
                    'Genre': genres[:30] + '...' if len(genres) > 30 else genres,
                    'Jml Rating': n_user_ratings,
                    'GMM Cluster': gmm_cluster if gmm_cluster is not None else '-',
                    'GMM Prob': f"{gmm_prob:.4f}" if gmm_prob is not None else '-',
                    'Actual': f"{true_r:.1f}",
                    'Prediksi Hybrid': f"{hybrid_pred:.3f}",
                    'Error': f"{hybrid_error:.3f}"
                })
                
                # Comparison data
                error_diff = svd_error - hybrid_error
                improvement = (error_diff / svd_error) * 100 if svd_error > 0 else 0
                
                if svd_error < hybrid_error:
                    winner = "SVD"
                elif hybrid_error < svd_error:
                    winner = "Hybrid"
                else:
                    winner = "Seri"
                
                comparison_data.append({
                    'No': idx + 1,
                    'User ID': uid,
                    'Jml Rating': n_user_ratings,
                    'Actual': f"{true_r:.1f}",
                    'SVD Pred': f"{svd_pred:.3f}",
                    'SVD Error': f"{svd_error:.3f}",
                    'Hybrid Pred': f"{hybrid_pred:.3f}",
                    'Hybrid Error': f"{hybrid_error:.3f}",
                    'Δ Error': f"{error_diff:+.3f}",
                    'Improvement': f"{improvement:+.1f}%",
                    'Winner': winner
                })
            
            progress_bar.progress(1.0)
            
            # Hitung RMSE dan MAE
            svd_rmse = np.sqrt(np.mean(np.array(svd_errors)**2))
            svd_mae = np.mean(svd_errors)
            hybrid_rmse = np.sqrt(np.mean(np.array(hybrid_errors)**2))
            hybrid_mae = np.mean(hybrid_errors)
            
            # Update session state
            st.session_state.svd_rmse = svd_rmse
            st.session_state.svd_mae = svd_mae
            st.session_state.hybrid_rmse = hybrid_rmse
            st.session_state.hybrid_mae = hybrid_mae
            
            st.session_state.svd_predictions_df = pd.DataFrame(svd_data)
            st.session_state.hybrid_predictions_df = pd.DataFrame(hybrid_data)
            st.session_state.comparison_df = pd.DataFrame(comparison_data)
            
            st.success(f"""
            Evaluasi ulang selesai!
            - SVD RMSE: {svd_rmse:.6f}
            - Hybrid RMSE: {hybrid_rmse:.6f}
            - Improvement: {((svd_rmse - hybrid_rmse) / svd_rmse * 100):+.2f}%
            """)
            
            return True
            
    except Exception as e:
        st.error(f"Evaluasi ulang gagal: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False


def safe_has_session_state():
    """
    Mengecek apakah session_state tersedia.
    """
    try:
        import streamlit as st
        return hasattr(st, 'session_state') and st.session_state is not None
    except:
        return False


def safe_session_state_get(key, default=None):
    """
    Mengakses session_state dengan aman.
    
    Parameter:
        key: Key yang ingin diakses
        default: Nilai default jika session_state tidak tersedia atau key tidak ada
        
    Returns:
        Nilai dari session_state atau default    """
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and st.session_state is not None:
            return st.session_state.get(key, default)
    except:
        pass
    return default

# ====================== FUNGSI SAVE DAN LOAD MODEL ======================

def check_saved_models(load_dir=None):
    if load_dir is None:
        load_dir = MODEL_DIR
    """
    Memeriksa apakah model sudah tersimpan di folder.
    
    Parameter:
        load_dir (str): Direktori penyimpanan model
        
    Returns:
        dict: Status keberadaan file-file model
    """
    import os
    
    if not os.path.exists(load_dir):
        return {'exists': False, 'files': {}}
    
    required_files = [
        'svd_model.pkl',
        'gmm_map.pkl',
        'trainset.pkl',
        'genre_cols.pkl',
        'metrics.pkl'
    ]
    
    file_status = {}
    for file in required_files:
        file_path = os.path.join(load_dir, file)
        file_status[file] = os.path.exists(file_path)
    
    exists = all(file_status.values())
    
    return {'exists': exists, 'files': file_status}

def safe_session_state_set(key, value):
    """
    Menyimpan ke session_state dengan aman.
    
    Parameter:
        key: Key untuk menyimpan
        value: Nilai yang akan disimpan
    """
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and st.session_state is not None:
            st.session_state[key] = value
            return True
    except:
        pass
    return False


# ====================== KONFIGURASI PATH DATA ======================
# ====================== KONFIGURASI PATH DATA ======================
# Data folder: taruh ratings.csv & movies.csv di subfolder "data" di samping r1.py
DATA_DIR = os.path.join(BASE_DIR, "data")
PATH_RATINGS = os.path.join(DATA_DIR, "ratings.csv")
PATH_MOVIES = os.path.join(DATA_DIR, "movies.csv")

# ====================== KONFIGURASI VISUALISASI ======================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# ====================== GRID SEARCH UNTUK OPTIMASI PARAMETER ======================
def run_grid_search_cv(ratings_df, movies_df, param_grid, n_folds=5, random_state=42):
    """
    Melakukan grid search dengan 5-fold cross-validation untuk mencari parameter optimal.
    """
    import itertools
    from tqdm import tqdm
    
    # Generate semua kombinasi parameter
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    results = []
    best_score = float('inf')
    best_params = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_combinations = len(combinations)
    
    for idx, combination in enumerate(combinations):
        params = dict(zip(param_keys, combination))
        
        status_text.text(f"Proses kombinasi {idx+1}/{total_combinations}: {params}")
        
        # Ekstrak parameter SVD dan GMM
        svd_params = {
            'n_factors': params.get('n_factors', 100),
            'n_epochs': params.get('n_epochs', 30),
            'lr_all': params.get('lr_all', 0.005),
            'reg_all': params.get('reg_all', 0.02)
        }
        
        gmm_params = {
            'max_components': params.get('max_components', 50),
            'smoothing': params.get('smoothing', 0.1)
        }
        
        try:
            # Jalankan cross-validation untuk kombinasi parameter
            cv_results = run_hybrid_cross_validation_with_save(
                ratings_df=ratings_df,
                movies_df=movies_df,
                output_key=f"grid_search_{idx}",
                n_folds=n_folds,
                test_size=0.2,
                random_state=random_state + idx,
                svd_params=svd_params,
                gmm_params=gmm_params,
                show_tables=False
            )
            
            if cv_results and 'summary' in cv_results:
                avg_rmse = cv_results['summary']['hybrid_rmse_mean']
                
                rmse_std = 0
                if 'fold_results' in cv_results and not cv_results['fold_results'].empty:
                    if 'hybrid_rmse' in cv_results['fold_results'].columns:
                        rmse_std = cv_results['fold_results']['hybrid_rmse'].std()
                
                results.append({
                    **params,
                    'avg_rmse': avg_rmse,
                    'avg_mae': cv_results['summary']['hybrid_mae_mean'],
                    'rmse_std': rmse_std,
                    'improvement': cv_results['summary'].get('rmse_improvement_mean', 0)
                })
                
                if avg_rmse < best_score:
                    best_score = avg_rmse
                    best_params = params.copy()
                    
        except Exception as e:
            st.warning(f"Error pada kombinasi {params}: {e}")
            continue
        
        progress_bar.progress((idx + 1) / total_combinations)
    
    progress_bar.empty()
    status_text.empty()
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('avg_rmse')
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'results_df': results_df,
        'n_combinations': len(results)
    }

# ====================== FUNGSI UNTUK MENYIMPAN OUTPUT STREAMLIT ======================

def save_output(container, key, content, output_type="info", title=None):
    """
    Menyimpan output Streamlit ke session_state untuk ditampilkan nanti.
    
    Parameter:
        container: Container Streamlit (st.empty() atau st.container())
        key: Key untuk menyimpan di session_state
        content: Konten yang akan ditampilkan
        output_type: Jenis output (info, success, error, warning, text, markdown, dataframe, metric, plot)
        title: Judul untuk output (opsional)
    """
    if 'saved_outputs' not in st.session_state:
        st.session_state.saved_outputs = {}
    
    if key not in st.session_state.saved_outputs:
        st.session_state.saved_outputs[key] = []
    
    st.session_state.saved_outputs[key].append({
        'content': content,
        'type': output_type,
        'title': title,
        'timestamp': time.time()
    })
    
    with container:
        if title:
            if output_type == "markdown":
                st.markdown(f"**{title}**")
            else:
                st.subheader(title)
        
        if output_type == "info":
            st.info(content)
        elif output_type == "success":
            st.success(content)
        elif output_type == "error":
            st.error(content)
        elif output_type == "warning":
            st.warning(content)
        elif output_type == "text":
            st.text(content)
        elif output_type == "markdown":
            st.markdown(content)
        elif output_type == "dataframe":
            st.dataframe(content, use_container_width=True, height=400)
        elif output_type == "dataframe_full":
            st.dataframe(content, use_container_width=True, height=600)
        elif output_type == "metric":
            col1, col2, col3, col4 = st.columns(4)
            metrics = content
            if len(metrics) >= 8:
                with col1:
                    st.metric(metrics[0], metrics[1])
                with col2:
                    st.metric(metrics[2], metrics[3])
                with col3:
                    st.metric(metrics[4], metrics[5])
                with col4:
                    st.metric(metrics[6], metrics[7])
        elif output_type == "plot":
            st.pyplot(content)
            plt.close(content)


def get_svd_params(svd_model):
    """
    Mendapatkan parameter SVD dari model atau session state.
    
    Returns:
        dict: Dictionary berisi parameter SVD
    """
    default_params = {
        'n_factors': 100,
        'n_epochs': 30,
        'lr_all': 0.005,
        'reg_all': 0.02,
        'random_state': 42
    }
    
    if 'svd_params' in st.session_state:
        return st.session_state.svd_params
    
    if svd_model is not None:
        params = {}
        if hasattr(svd_model, 'n_factors'):
            params['n_factors'] = svd_model.n_factors
        if hasattr(svd_model, 'n_epochs'):
            params['n_epochs'] = svd_model.n_epochs
        if hasattr(svd_model, 'random_state'):
            params['random_state'] = svd_model.random_state
        
        for key in default_params:
            if key not in params:
                params[key] = default_params[key]
        
        return params
    
    return default_params


def display_saved_outputs(key, container=None):
    """
    Menampilkan semua output yang telah disimpan untuk key tertentu.
    """
    if 'saved_outputs' not in st.session_state or key not in st.session_state.saved_outputs:
        return
    
    outputs = st.session_state.saved_outputs[key]
    outputs.sort(key=lambda x: x['timestamp'])
    
    display_func = st if container is None else container
    
    for output in outputs:
        if output.get('title'):
            if output['type'] == "markdown":
                display_func.markdown(f"**{output['title']}**")
            else:
                display_func.subheader(output['title'])
        
        if output['type'] == "info":
            display_func.info(output['content'])
        elif output['type'] == "success":
            display_func.success(output['content'])
        elif output['type'] == "error":
            display_func.error(output['content'])
        elif output['type'] == "warning":
            display_func.warning(output['content'])
        elif output['type'] == "text":
            display_func.text(output['content'])
        elif output['type'] == "markdown":
            display_func.markdown(output['content'])
        elif output['type'] == "dataframe":
            display_func.dataframe(output['content'], use_container_width=True, height=400)
        elif output['type'] == "dataframe_full":
            display_func.dataframe(output['content'], use_container_width=True, height=600)
        elif output['type'] == "metric":
            metrics = output['content']
            if len(metrics) >= 8:
                col1, col2, col3, col4 = display_func.columns(4)
                with col1:
                    col1.metric(metrics[0], metrics[1])
                with col2:
                    col2.metric(metrics[2], metrics[3])
                with col3:
                    col3.metric(metrics[4], metrics[5])
                with col4:
                    col4.metric(metrics[6], metrics[7])
        elif output['type'] == "plot":
            display_func.pyplot(output['content'])
            plt.close(output['content'])


def clear_saved_outputs(key):
    """
    Menghapus output yang tersimpan untuk key tertentu.
    """
    if 'saved_outputs' in st.session_state and key in st.session_state.saved_outputs:
        del st.session_state.saved_outputs[key]

# ====================== FUNGSI UNTUK MENGHITUNG BOBOT ADAPTIF ======================
def calculate_adaptive_weights(user_data, trainset=None, n_ratings=None, 
                               user_rating_std=None, user_rating_mean=None,
                               user_embedding_norm=None, gmm_confidence=None,
                               genre_diversity=None, rating_entropy=None,
                               movies_df=None, genre_cols=None, 
                               ratings_df=None, gmm_map=None, 
                               svd_model=None, hybrid_model=None,
                               is_cold_start=False, fav_genres=None):
    
    # Tentukan jumlah rating
    if n_ratings is None:
        if trainset is not None:
            try:
                u_inner = trainset.to_inner_uid(str(user_data))
                n_ratings = len(trainset.ur[u_inner])
            except:
                n_ratings = 0
        else:
            n_ratings = 0
    
    # Rule-based alpha
    if n_ratings < 5:
        cf_weight = 0.20
    elif n_ratings < 10:
        cf_weight = 0.30
    elif n_ratings < 20:
        cf_weight = 0.35
    elif n_ratings < 50:
        cf_weight = 0.40
    elif n_ratings < 100:
        cf_weight = 0.45
    else:
        cf_weight = 0.50
    
    # Fine-tuning berdasarkan std rating
    if user_rating_std is not None and user_rating_std > 1.2:
        cf_weight = max(0.35, cf_weight * 0.95)
    
    # Fine-tuning berdasarkan genre diversity
    if genre_diversity is not None and genre_diversity > 0.7:
        cf_weight = min(0.75, cf_weight * 1.05)
    
    cf_weight = np.clip(cf_weight, 0.20, 0.75)
    
    return {
        'cf_weight': cf_weight,
        'cb_weight': 1.0 - cf_weight,
        'n_ratings': n_ratings,
        'method': 'rule_based',
        'is_cold_start': (n_ratings < 5)
    }

# ====================== FUNGSI UNTUK MENGHITUNG GENRE DIVERSITY ======================

def calculate_genre_diversity(user_id, trainset, movies_df, genre_cols):
    """
    Menghitung keragaman genre dari film yang dirating oleh user.
    Nilai 0 menunjukkan homogen (semua genre sama), nilai 1 menunjukkan sangat beragam.
    
    Parameter:
        user_id: ID user
        trainset: Surprise trainset
        movies_df: DataFrame film
        genre_cols: Daftar genre
        
    Returns:
        float: Skor keragaman genre antara 0 dan 1
    """
    try:
        u_inner = trainset.to_inner_uid(str(user_id))
        rated_items = [trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[u_inner]]
        
        if len(rated_items) < 2:
            return 0.5
        
        all_genres = []
        for mid in rated_items[:50]:
            try:
                mid_int = int(mid)
                movie_info = movies_df[movies_df['movieId'] == mid_int]
                if not movie_info.empty:
                    genres = str(movie_info['genres'].values[0]).split('|')
                    all_genres.extend(genres)
            except:
                continue
        
        if not all_genres:
            return 0.5
        
        unique_genres = len(set(all_genres))
        total_genres = len(genre_cols)
        
        if total_genres == 0:
            return 0.5
        
        diversity_ratio = unique_genres / total_genres
        
        from collections import Counter
        genre_counts = Counter(all_genres)
        total = sum(genre_counts.values())
        
        if total == 0:
            return 0.5
        
        probs = [count/total for count in genre_counts.values()]
        entropy = -sum(p * np.log(p) for p in probs)
        max_entropy = np.log(len(genre_counts)) if len(genre_counts) > 0 else 1
        
        if max_entropy > 0:
            entropy_norm = entropy / max_entropy
        else:
            entropy_norm = 0.5
        
        diversity_score = 0.5 * diversity_ratio + 0.5 * entropy_norm
        
        return diversity_score
        
    except Exception as e:
        return 0.5

# ====================== FUNGSI UNTUK MENGHITUNG RATING ENTROPY ======================

def calculate_rating_entropy(user_id, trainset):
    """
    Menghitung entropy dari distribusi rating user.
    Entropy tinggi menunjukkan user memberikan rating dengan variasi yang tinggi.
    
    Parameter:
        user_id: ID user
        trainset: Surprise trainset
        
    Returns:
        float: Skor entropy ternormalisasi antara 0 dan 1
    """
    try:
        u_inner = trainset.to_inner_uid(str(user_id))
        ratings = [r for (_, r) in trainset.ur[u_inner]]
        
        if len(ratings) < 2:
            return 0.5
        
        unique_ratings = sorted(set(ratings))
        rating_counts = {r: ratings.count(r) for r in unique_ratings}
        
        total = len(ratings)
        probs = [count/total for count in rating_counts.values()]
        
        entropy = -sum(p * np.log(p) for p in probs)
        
        max_entropy = np.log(min(len(unique_ratings), len(ratings)))
        
        if max_entropy > 0:
            entropy_norm = entropy / max_entropy
        else:
            entropy_norm = 0.5
        
        return entropy_norm
        
    except Exception as e:
        return 0.5

# ====================== FUNGSI KONVERSI GMM KE GENRE VECTOR ======================

def build_gmm_to_genre_mapping(gmm_model, movies_enc, feature_cols, genre_cols):
    """
    Membangun mapping dari cluster GMM ke genre preference vector.
    Setiap cluster GMM direpresentasikan sebagai vektor preferensi genre.
    
    Parameter:
        gmm_model: Trained GMM model
        movies_enc: DataFrame film dengan encoding
        feature_cols: Kolom fitur GMM
        genre_cols: Daftar genre
        
    Returns:
        dict: Mapping cluster index ke genre vector
    """
    if gmm_model is None or movies_enc is None:
        return {}
    
    X = movies_enc[feature_cols].values
    cluster_probs = gmm_model.predict_proba(X)
    
    n_clusters = gmm_model.n_components
    n_genres = len(genre_cols)
    
    cluster_genre_vectors = {}
    
    for cluster_idx in range(n_clusters):
        cluster_assignments = np.argmax(cluster_probs, axis=1)
        cluster_mask = cluster_assignments == cluster_idx
        
        if np.sum(cluster_mask) > 0:
            cluster_movies = movies_enc.iloc[cluster_mask]
            
            genre_vector = np.zeros(n_genres)
            
            for _, row in cluster_movies.iterrows():
                genres_str = row.get('genres', '')
                if isinstance(genres_str, str) and genres_str != '(no genres listed)':
                    movie_genres = genres_str.split('|')
                    for g in movie_genres:
                        if g in genre_cols:
                            idx = genre_cols.index(g)
                            genre_vector[idx] += 1
            
            if np.sum(genre_vector) > 0:
                genre_vector = genre_vector / np.sum(genre_vector)
            
            cluster_genre_vectors[cluster_idx] = genre_vector
        else:
            cluster_genre_vectors[cluster_idx] = np.zeros(n_genres)
    
    return cluster_genre_vectors

# ====================== UTILITY FUNCTIONS ======================

def format_p_value(p_val):
    """
    Memformat p-value untuk tampilan yang lebih baik.
    """
    if p_val is None or np.isnan(p_val):
        return "N/A"
    elif p_val < 0.000001:
        return "< 0.000001"
    elif p_val < 0.00001:
        return f"{p_val:.6f}"
    elif p_val < 0.0001:
        return f"{p_val:.5f}"
    elif p_val < 0.001:
        return f"{p_val:.4f}"
    else:
        return f"{p_val:.6f}"


def search_movies(movies_df, query, top_n=20):
    """
    Mencari film berdasarkan judul atau genre.
    
    Parameter:
        movies_df (DataFrame): Data film
        query (str): Kata kunci pencarian
        top_n (int): Jumlah maksimal hasil
        
    Returns:
        DataFrame: Film yang cocok dengan query
    """
    if not query or query.strip() == "":
        return movies_df.head(top_n)
    
    query = query.strip().lower()
    
    title_matches = movies_df[
        movies_df['title'].str.lower().str.contains(query, na=False)
    ]
    
    genre_matches = movies_df[
        movies_df['genres'].str.lower().str.contains(query, na=False)
    ]
    
    results = pd.concat([title_matches, genre_matches]).drop_duplicates()
    
    return results.sort_values('title').head(top_n)


def get_page_number(user_id, total_pages, default=1):
    """
    Mendapatkan nomor halaman untuk pagination.
    
    Parameter:
        user_id: ID user untuk session key
        total_pages: Jumlah total halaman
        default: Halaman default
        
    Returns:
        int: Nomor halaman yang valid
    """
    session_key = f'page_{user_id}'
    
    if session_key not in st.session_state:
        st.session_state[session_key] = default
    
    current_page = st.session_state[session_key]
    
    if current_page < 1:
        current_page = 1
        st.session_state[session_key] = 1
    elif current_page > total_pages:
        current_page = total_pages
        st.session_state[session_key] = total_pages
    
    return current_page


def set_page_number(user_id, page):
    """
    Mengatur nomor halaman dengan aman.
    """
    session_key = f'page_{user_id}'
    st.session_state[session_key] = page

# ====================== FUNGSI LOAD DATA DENGAN SAVE OUTPUT ======================

def display_cleaned_data_summary(ratings_df, movies_df, stats_dict):
    """
    Menampilkan ringkasan jumlah hasil setelah pembersihan data.
    """
    st.markdown("## Hasil Setelah Pembersihan Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Jumlah User", f"{stats_dict['n_users']:,}")
    with col2:
        st.metric("Jumlah Film", f"{stats_dict['n_items']:,}")
    with col3:
        st.metric("Jumlah Rating", f"{stats_dict['n_ratings']:,}")
    with col4:
        st.metric("Sparsity", f"{stats_dict['sparsity']:.2%}")
    
    # Tampilkan informasi duplikat yang dihapus
    duplicates_rating = ratings_df.duplicated(subset=['userId', 'movieId', 'rating']).sum()
    duplicates_movie = movies_df.duplicated(subset=['movieId']).sum()
    
    if duplicates_rating > 0 or duplicates_movie > 0:
        st.info(f"Duplikat yang dihapus: {duplicates_rating:,} baris rating, {duplicates_movie:,} film")
    
    st.success(f"Total data bersih: {stats_dict['n_ratings']:,} rating dari {stats_dict['n_users']:,} user dan {stats_dict['n_items']:,} film")

def show_data_summary():
    """
    Menampilkan ringkasan jumlah data setelah pembersihan.
    """
    if st.session_state.get('ratings') is not None and st.session_state.get('movies') is not None:
        ratings = st.session_state.ratings
        movies = st.session_state.movies
        
        n_users = ratings['userId'].nunique()
        n_items = ratings['movieId'].nunique()
        n_ratings = len(ratings)
        sparsity = 1 - n_ratings / (n_users * n_items) if n_users > 0 and n_items > 0 else 1.0
        
        st.markdown("## Hasil Setelah Pembersihan Data")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Jumlah User", f"{n_users:,}")
        with col2:
            st.metric("Jumlah Film", f"{n_items:,}")
        with col3:
            st.metric("Jumlah Rating", f"{n_ratings:,}")
        with col4:
            st.metric("Sparsity", f"{sparsity:.2%}")
        
        # Informasi duplikat yang dihapus
        duplicates_rating = ratings.duplicated(subset=['userId', 'movieId', 'rating']).sum()
        duplicates_movie = movies.duplicated(subset=['movieId']).sum()
        
        if duplicates_rating > 0 or duplicates_movie > 0:
            st.info(f"Duplikat yang dihapus: {duplicates_rating:,} baris rating, {duplicates_movie:,} film")
        
        st.success(f"Total data bersih: {n_ratings:,} rating dari {n_users:,} user dan {n_items:,} film")
        
        return True
    return False

def load_and_filter_data_with_save(path_ratings, path_movies, output_key, show_tables=True):
    """
    Memuat dan memfilter data rating dan film dari file CSV dengan menyimpan output.
    
    Parameter:
        path_ratings (str): Path ke file ratings.csv
        path_movies (str): Path ke file movies.csv
        output_key (str): Key untuk menyimpan output
        show_tables (bool): Tampilkan output
        
    Returns:
        tuple: (ratings_df, movies_df, stats_dict)
    """
    output_container = st.empty()
    
    save_output(output_container, output_key, "LANGKAH 1: LOADING DAN PREPROCESSING DATA", "info")
    start_time = time.time()

    try:
        ratings = pd.read_csv(path_ratings)
        movies = pd.read_csv(path_movies)
    except Exception as e:
        save_output(output_container, output_key, f"Error loading files: {e}", "error")
        return None, None, None
    
    movies['movieId'] = movies['movieId'].astype(int)
    ratings['movieId'] = ratings['movieId'].astype(int)
    ratings['userId'] = ratings['userId'].astype(int)

    save_output(output_container, output_key, 
                f"Data mentah berhasil dimuat. ({time.time() - start_time:.2f} detik)", "success")
    
    save_output(output_container, output_key, "Analisis dan Penghapusan Duplikasi", "markdown")
    
    duplicates_rating = ratings.duplicated(subset=['userId', 'movieId', 'rating']).sum()
    ratings_before = len(ratings)
    if duplicates_rating > 0:
        ratings = ratings.drop_duplicates(subset=['userId', 'movieId', 'rating'], keep='first')
        ratings = ratings.drop_duplicates(keep='first')
        ratings_after = len(ratings)
        save_output(output_container, output_key, 
                   f"Duplikasi rating dihapus: {ratings_before - ratings_after:,} baris", "success")
    else:
        ratings_after = ratings_before
    
    duplicates_movies = movies.duplicated(subset=['movieId']).sum()
    movies_before = len(movies)
    if duplicates_movies > 0:
        movies = movies.drop_duplicates(subset=['movieId'], keep='first')
        movies = movies.drop_duplicates(keep='first')
        movies_after = len(movies)
        save_output(output_container, output_key,
                   f"Duplikasi film dihapus: {movies_before - movies_after:,} baris", "success")
    else:
        movies_after = movies_before

    n_users = ratings['userId'].nunique()
    n_items = ratings['movieId'].nunique()
    n_ratings = len(ratings)
    sparsity = 1 - n_ratings / (n_users * n_items) if n_users > 0 and n_items > 0 else 1.0
    
    stats_dict = {
        'n_users': n_users,
        'n_items': n_items,
        'n_ratings': n_ratings,
        'sparsity': sparsity,
        'density': 1 - sparsity,
        'rating_mean': ratings['rating'].mean(),
        'rating_std': ratings['rating'].std()
    }

    # TAMPILKAN RINGKASAN HASIL PEMBERSIHAN
    display_cleaned_data_summary(ratings, movies, stats_dict)

    if show_tables:
        save_output(output_container, output_key, "Visualisasi Distribusi Rating", "markdown")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(ratings['rating'], bins=10, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_title('Distribusi Rating', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Rating')
        axes[0].set_ylabel('Frekuensi')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].boxplot(ratings['rating'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
        axes[1].set_title('Boxplot Rating', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Rating')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_output(output_container, output_key, fig, "plot")
    
    duration = time.time() - start_time
    save_output(output_container, output_key, 
               f"Data setelah preprocessing selesai. Total waktu: {duration:.2f} detik", "success")
    
    return ratings, movies, stats_dict
# ====================== FUNGSI WARM-START SPLIT DENGAN SAVE OUTPUT ======================

def warm_start_item_split_with_save(ratings_df, movies_df, output_key, test_size=0.2, random_state=42, show_tables=True):
    """
    Warm-start item split dengan jaminan tidak ada overlap.
    SETIAP FILM HANYA MASUK KE SATU SET (TRAINING ATAU TEST).
    
    Parameter:
        ratings_df (DataFrame): Data rating
        movies_df (DataFrame): Data film
        output_key (str): Key untuk menyimpan output
        test_size (float): Proporsi film untuk test set
        random_state (int): Random seed
        show_tables (bool): Tampilkan output
        
    Returns:
        tuple: (train_df, test_df, train_movies, test_movies)
    """
    output_container = st.empty()
    
    save_output(output_container, output_key, "LANGKAH 2: WARM-START ITEM SPLIT", "info")
    
    ratings_df = ratings_df.reset_index(drop=True).copy()
    
    movies_with_ratings = set(ratings_df['movieId'].unique())
    valid_movies = list(movies_with_ratings)
    
    if len(valid_movies) == 0:
        save_output(output_container, output_key, "Tidak ada film dengan rating!", "error")
        return None, None, None, None
    
    np.random.seed(random_state)
    n_test_movies = int(len(valid_movies) * test_size)
    
    test_movie_ids = set(np.random.choice(valid_movies, n_test_movies, replace=False))
    train_movie_ids = set(valid_movies) - test_movie_ids
    
    save_output(output_container, output_key, f"Split Configuration:", "markdown")
    save_output(output_container, output_key, 
               f"- Total film dengan rating: {len(valid_movies):,}", "text")
    save_output(output_container, output_key,
               f"- Film di TEST set (eksklusif): {len(test_movie_ids):,} ({len(test_movie_ids)/len(valid_movies)*100:.1f}%)", "text")
    save_output(output_container, output_key,
               f"- Film di TRAINING set: {len(train_movie_ids):,} ({len(train_movie_ids)/len(valid_movies)*100:.1f}%)", "text")
    
    train_df = ratings_df[ratings_df['movieId'].isin(train_movie_ids)].copy()
    test_df = ratings_df[ratings_df['movieId'].isin(test_movie_ids)].copy()
    
    train_movies_final = train_movie_ids
    test_movies_final = test_movie_ids
    
    overlap_movies = train_movies_final.intersection(test_movies_final)
    if overlap_movies:
        save_output(output_container, output_key, f"Terdapat {len(overlap_movies)} film yang sama antara train dan test!", "error")
        return None, None, None, None
    
    train_pairs = set(zip(train_df['userId'], train_df['movieId']))
    test_pairs = set(zip(test_df['userId'], test_df['movieId']))
    overlap_pairs = train_pairs.intersection(test_pairs)
    
    if overlap_pairs:
        save_output(output_container, output_key, f"Terdapat {len(overlap_pairs)} pasangan (user, item) yang sama!", "error")
        return None, None, None, None
    
    save_output(output_container, output_key, 
               f"Split berhasil! Train: {len(train_df):,} rating, Test: {len(test_df):,} rating", "success")
    save_output(output_container, output_key, f"Film dalam TRAINING set (final): {len(train_movies_final):,} film", "text")
    save_output(output_container, output_key, f"Film dalam TEST set (final): {len(test_movies_final):,} film", "text")
    
    return train_df, test_df, train_movies_final, test_movies_final


# ====================== TAMBAHKAN DI BAGIAN FUNGSI SAVE DAN LOAD ======================

def save_training_outputs(save_dir=None):
    """
    Menyimpan semua output training (One-Hot Encoding, GMM, dll) ke file.
    """
    if save_dir is None:
        save_dir = MODEL_DIR
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    saved_outputs = []
    
    # 1. Simpan output dari session_state.saved_outputs
    if 'saved_outputs' in st.session_state and 'tab2_output' in st.session_state.saved_outputs:
        training_outputs = []
        for output in st.session_state.saved_outputs['tab2_output']:
            # Hanya simpan data yang bisa di-serialize (bukan plot)
            if output['type'] != 'plot':
                output_copy = output.copy()
                if 'content' in output_copy and hasattr(output_copy['content'], 'to_dict'):
                    output_copy['content'] = output_copy['content'].to_dict()
                training_outputs.append(output_copy)
        
        output_path = os.path.join(save_dir, 'training_outputs.pkl')
        joblib.dump(training_outputs, output_path)
        saved_outputs.append(('Training Outputs', output_path))
        print(f"Training outputs saved: {len(training_outputs)} items")
    
    # 2. Simpan data utama training
    training_data = {
        'movies_enc': st.session_state.get('movies_enc', None),
        'genre_cols': st.session_state.get('genre_cols', None),
        'feature_cols': st.session_state.get('feature_cols', None),
        'gmm_map': st.session_state.get('gmm_map', None),
        'gmm_model': st.session_state.get('gmm_model', None),
        'cluster_genre_mapping': st.session_state.get('cluster_genre_mapping', None),
        'n_components': st.session_state.get('n_components', None),
        'optimal_n': st.session_state.get('optimal_n', None),
        'gmm_bics': st.session_state.get('gmm_bics', None),
        'n_components_range': st.session_state.get('n_components_range', None)
    }
    
    # Konversi DataFrame ke format yang bisa disimpan
    if training_data['movies_enc'] is not None:
        training_data['movies_enc'] = training_data['movies_enc'].to_dict()
    
    training_data_path = os.path.join(save_dir, 'training_data.pkl')
    joblib.dump(training_data, training_data_path)
    saved_outputs.append(('Training Data', training_data_path))
    
    return saved_outputs


def load_training_outputs(load_dir=None):
    """
    Memuat kembali output training yang tersimpan.
    """
    if load_dir is None:
        load_dir = MODEL_DIR
    
    if not os.path.exists(load_dir):
        return False
    
    loaded_items = []
    
    # 1. Load training outputs
    output_path = os.path.join(load_dir, 'training_outputs.pkl')
    if os.path.exists(output_path):
        try:
            training_outputs = joblib.load(output_path)
            
            # Restore ke session_state.saved_outputs
            if 'saved_outputs' not in st.session_state:
                st.session_state.saved_outputs = {}
            
            if 'tab2_output' not in st.session_state.saved_outputs:
                st.session_state.saved_outputs['tab2_output'] = []
            
            for output in training_outputs:
                # Konversi kembali dictionary ke DataFrame jika perlu
                if output['type'] in ['dataframe', 'dataframe_full'] and isinstance(output['content'], dict):
                    output['content'] = pd.DataFrame(output['content'])
                st.session_state.saved_outputs['tab2_output'].append(output)
            
            loaded_items.append('Training Outputs')
            print(f"Training outputs loaded: {len(training_outputs)} items")
        except Exception as e:
            print(f"Error loading training outputs: {e}")
    
    # 2. Load training data
    training_data_path = os.path.join(load_dir, 'training_data.pkl')
    if os.path.exists(training_data_path):
        try:
            training_data = joblib.load(training_data_path)
            
            # Restore ke session_state
            if training_data.get('movies_enc'):
                st.session_state.movies_enc = pd.DataFrame(training_data['movies_enc'])
            if training_data.get('genre_cols'):
                st.session_state.genre_cols = training_data['genre_cols']
            if training_data.get('feature_cols'):
                st.session_state.feature_cols = training_data['feature_cols']
            if training_data.get('gmm_map'):
                st.session_state.gmm_map = training_data['gmm_map']
            if training_data.get('gmm_model'):
                st.session_state.gmm_model = training_data['gmm_model']
            if training_data.get('cluster_genre_mapping'):
                st.session_state.cluster_genre_mapping = training_data['cluster_genre_mapping']
            if training_data.get('n_components'):
                st.session_state.n_components = training_data['n_components']
            if training_data.get('optimal_n'):
                st.session_state.optimal_n = training_data['optimal_n']
            if training_data.get('gmm_bics'):
                st.session_state.gmm_bics = training_data['gmm_bics']
            if training_data.get('n_components_range'):
                st.session_state.n_components_range = training_data['n_components_range']
            
            loaded_items.append('Training Data')
            print(f"Training data loaded")
        except Exception as e:
            print(f"Error loading training data: {e}")
    
    return len(loaded_items) > 0


# ====================== MODIFIKASI FUNGSI GMM_SOFT_CLUSTERING_WITH_SAVE ======================

def gmm_soft_clustering_with_save(movies_enc, feature_cols, output_key, max_components=50, random_state=42, show_tables=True):
    """
    Soft clustering menggunakan Gaussian Mixture Model dengan menyimpan output.
    """
    output_container = st.empty()
    
    save_output(output_container, output_key, "LANGKAH 4: CONTENT-BASED CLUSTERING DENGAN GMM", "info")
    start_time = time.time()
    
    X = movies_enc[feature_cols].values
    n_samples = X.shape[0]
    
    max_components_dynamic = min(max_components, max(5, n_samples // 10))
    
    save_output(output_container, output_key, f"Parameter GMM:", "markdown")
    save_output(output_container, output_key, f"Jumlah film (TRAINING): {n_samples:,}", "text")
    save_output(output_container, output_key, f"Dimensi fitur: {len(feature_cols)}", "text")
    save_output(output_container, output_key, f"Maksimum komponen: {max_components_dynamic} (dinamis dari n_samples//10)", "text")
    
    if n_samples > 50:
        n_components_range = list(range(3, max_components_dynamic, max(1, max_components_dynamic//20)))
    else:
        n_components_range = list(range(3, min(max_components_dynamic, 10)))
    
    bics = []
    
    # SIMPAN KE SESSION STATE
    st.session_state.n_components_range = n_components_range
    st.session_state.gmm_bics = []
    
    if show_tables and len(n_components_range) > 1:
        progress_bar = st.progress(0)
        for idx, n in enumerate(n_components_range):
            progress_bar.progress((idx + 1) / len(n_components_range))
            gmm = GaussianMixture(n_components=n, covariance_type='diag',
                                random_state=random_state, max_iter=100, n_init=3)
            gmm.fit(X)
            bic_val = gmm.bic(X)
            bics.append(bic_val)
            st.session_state.gmm_bics.append(bic_val)
        progress_bar.empty()
    else:
        for idx, n in enumerate(n_components_range):
            gmm = GaussianMixture(n_components=n, covariance_type='diag',
                                random_state=random_state, max_iter=100, n_init=3)
            gmm.fit(X)
            bic_val = gmm.bic(X)
            bics.append(bic_val)
            st.session_state.gmm_bics.append(bic_val)
    
    optimal_idx = np.argmin(bics)
    optimal_n = n_components_range[optimal_idx]
    
    # SIMPAN KE SESSION STATE
    st.session_state.optimal_n = optimal_n
    st.session_state.n_components = optimal_n
    
    if show_tables and len(n_components_range) > 1:
        save_output(output_container, output_key, "Visualisasi Nilai BIC untuk Pemilihan Jumlah Komponen GMM", "markdown")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(list(n_components_range), bics, 'b-o', linewidth=2, markersize=6, label='Nilai BIC')
        ax.plot(optimal_n, bics[optimal_idx], 'r*', markersize=20, label=f'Optimal: {optimal_n} komponen')
        ax.axvline(x=optimal_n, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Jumlah Komponen GMM', fontsize=12)
        ax.set_ylabel('Nilai BIC', fontsize=12)
        ax.set_title('Kurva BIC untuk Seleksi Jumlah Komponen GMM (Training Set)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        save_output(output_container, output_key, fig, "plot")
    
    gmm = GaussianMixture(n_components=optimal_n, covariance_type='diag',
                        random_state=random_state, max_iter=100, n_init=3)
    gmm.fit(X)
    probs = gmm.predict_proba(X)
    
    gmm_map = {str(int(mid)): probs[i] for i, mid in enumerate(movies_enc.index)}
    
    # SIMPAN GMM MODEL KE SESSION STATE
    st.session_state.gmm_model = gmm
    
    duration = time.time() - start_time
    save_output(output_container, output_key, 
               f"GMM selesai! Komponen optimal: {optimal_n}, Total film (TRAINING): {len(gmm_map):,} "
               f"({duration:.2f} detik)", "success")
    
    if show_tables:
        save_output(output_container, output_key, f"TABEL: SEMUA Hasil Clustering GMM (Total: {len(gmm_map)} Film)", "markdown")
        
        cluster_assignments = []
        for i, (mid, probs_array) in enumerate(gmm_map.items()):
            max_cluster = np.argmax(probs_array)
            max_prob = probs_array[max_cluster]
            
            prob_str = ", ".join([f"C{c}:{p:.3f}" for c, p in enumerate(probs_array[:5])])
            
            cluster_assignments.append({
                'No': i + 1,
                'movieId': mid,
                'Title': movies_enc.loc[int(mid), 'title'] if int(mid) in movies_enc.index else f'Movie {mid}',
                'Main Cluster': max_cluster,
                'Probability Main Cluster': f"{max_prob:.4f}",
                'Top 5 Probabilities': prob_str + ("..." if optimal_n > 5 else ""),
                'Jumlah Cluster': optimal_n,
                'Entropy': f"{-np.sum(probs_array * np.log(probs_array + 1e-10)):.4f}"
            })
        
        cluster_assignments_df = pd.DataFrame(cluster_assignments)
        save_output(output_container, output_key, cluster_assignments_df, "dataframe_full")
        
        save_output(output_container, output_key, "Distribusi Film per Cluster (Training Set)", "markdown")
        all_clusters = [np.argmax(probs) for probs in gmm_map.values()]
        cluster_counts = pd.Series(all_clusters).value_counts().sort_index()
        
        cluster_dist_df = pd.DataFrame({
            'Cluster': cluster_counts.index,
            'Jumlah Film': cluster_counts.values,
            'Persentase': [f"{c/len(gmm_map)*100:.1f}%" for c in cluster_counts.values]
        })
        save_output(output_container, output_key, cluster_dist_df, "dataframe")
        
        save_output(output_container, output_key, f"TABEL: SEMUA Film per Cluster", "markdown")
        
        for cluster in sorted(cluster_counts.index)[:5]:
            save_output(output_container, output_key, f"Cluster {cluster}", "markdown")
            cluster_films = []
            for mid, probs_array in gmm_map.items():
                if np.argmax(probs_array) == cluster:
                    title = movies_enc.loc[int(mid), 'title'] if int(mid) in movies_enc.index else f'Movie {mid}'
                    cluster_films.append({
                        'movieId': mid,
                        'Title': title,
                        f'P(C{cluster})': f"{probs_array[cluster]:.4f}"
                    })
            
            if cluster_films:
                cluster_df = pd.DataFrame(cluster_films)
                save_output(output_container, output_key, cluster_df, "dataframe")
    
    return gmm_map, gmm


# ====================== MODIFIKASI FUNGSI ONE_HOT_ENCODE_GENRES_WITH_SAVE ======================

def one_hot_encode_genres_with_save(movies_df, train_movie_ids, output_key, smoothing_factor=0.1, show_tables=True):
    """
    One-hot encoding untuk genre film dengan Laplace smoothing.
    """
    output_container = st.empty()
    
    save_output(output_container, output_key, "LANGKAH 3: FEATURE EXTRACTION - ENCODING GENRE DENGAN SMOOTHING", "info")
    start_time = time.time()
    
    train_movie_ids = set([int(mid) for mid in train_movie_ids])
    movies_df = movies_df.copy()
    movies_df['movieId'] = movies_df['movieId'].astype(int)
    
    train_movies = movies_df[movies_df['movieId'].isin(train_movie_ids)].copy()
        
    missing_movies = train_movie_ids - set(train_movies['movieId'])
    if missing_movies:
        save_output(output_container, output_key, 
                   f"{len(missing_movies)} film training tidak ditemukan di movies_df, menambahkan dengan genre default", "warning")
        
        for movie_id in missing_movies:
            new_row = pd.DataFrame([{
                'movieId': movie_id,
                'title': f'Unknown Movie {movie_id}',
                'genres': '(no genres listed)'
            }])
            train_movies = pd.concat([train_movies, new_row], ignore_index=True)
    
    save_output(output_container, output_key, f"Final jumlah film untuk encoding: {len(train_movies):,}", "text")
    
    all_genres = set()
    for genres in train_movies['genres'].dropna():
        if str(genres) != '(no genres listed)':
            all_genres.update(str(genres).split('|'))
    
    genre_cols = sorted(all_genres)
    
    # SIMPAN KE SESSION STATE
    st.session_state.genre_cols = genre_cols
    
    for genre in genre_cols:
        train_movies[f'genre_{genre}'] = smoothing_factor
    
    for idx, row in train_movies.iterrows():
        genres_str = str(row['genres'])
        if genres_str != '(no genres listed)' and genres_str != 'nan':
            movie_genres = genres_str.split('|')
            n_genres = len(movie_genres)
            
            genre_value = 1.0 - smoothing_factor
            
            for genre in movie_genres:
                col_name = f'genre_{genre}'
                if col_name in train_movies.columns:
                    train_movies.at[idx, col_name] = genre_value
    
    movies_enc = train_movies.set_index('movieId')
    movies_enc.index = movies_enc.index.astype(int)
    feature_cols = [f'genre_{genre}' for genre in genre_cols]
    
    # SIMPAN KE SESSION STATE
    st.session_state.feature_cols = feature_cols
    st.session_state.movies_enc = movies_enc
    
    duration = time.time() - start_time
    save_output(output_container, output_key, 
               f"Encoding selesai! {len(train_movies)} film, {len(feature_cols)} fitur genre, "
               f"smoothing={smoothing_factor} ({duration:.2f} detik)", "success")
    
    if show_tables:
        save_output(output_container, output_key, "TABEL ONE-HOT ENCODING GENRE", "markdown")
        
        display_df = movies_enc.reset_index()
        display_df = display_df.merge(
            movies_df[['movieId', 'title', 'genres']], 
            on='movieId', 
            how='left'
        )
        
        available_cols = ['movieId', 'title', 'genres'] + feature_cols
        display_cols = [col for col in available_cols if col in display_df.columns]
        
        if len(display_cols) >= 3:
            table_df = display_df[display_cols].copy()
            
            for col in feature_cols:
                if col in table_df.columns:
                    table_df[col] = table_df[col].apply(lambda x: f"{x:.4f}")
            
            rename_dict = {}
            for col in feature_cols:
                if col in table_df.columns:
                    new_name = col.replace('genre_', '')
                    rename_dict[col] = new_name
            
            if rename_dict:
                table_df = table_df.rename(columns=rename_dict)
            
            save_output(output_container, output_key, f"TABEL FILM DENGAN ONE-HOT ENCODING (TOTAL: {len(display_df)} FILM)", "markdown")
            save_output(output_container, output_key, table_df, "dataframe_full")
        
        save_output(output_container, output_key, "STATISTIK DISTRIBUSI GENRE (TRAINING SET)", "markdown")
        genre_freq = {}
        for genre in genre_cols:
            count = sum(1 for genres in train_movies['genres'] if genre in str(genres))
            genre_freq[genre] = count
        
        genre_stats = pd.DataFrame({
            'Genre': list(genre_freq.keys()),
            'Jumlah Film (Training)': list(genre_freq.values()),
            'Persentase Training': [f"{count/len(train_movies)*100:.1f}%" for count in genre_freq.values()]
        }).sort_values('Jumlah Film (Training)', ascending=False)
        
        save_output(output_container, output_key, genre_stats, "dataframe")
    
    return movies_enc, feature_cols, genre_cols


# ====================== MODIFIKASI FUNGSI LOAD_ALL_MODELS ======================

def load_all_models(load_dir=None):
    """
    Memuat semua model dari folder dengan merekonstruksi Hybrid model.
    """
    import os
    import joblib
    import pickle
    import numpy as np
    import pandas as pd
    
    if load_dir is None:
        load_dir = MODEL_DIR
    
    if not os.path.exists(load_dir):
        st.error(f"Folder tidak ditemukan: {load_dir}")
        return False
    
    loaded_files = []
    errors = []
    
    try:
        # 1. Load SVD model
        svd_path = os.path.join(load_dir, 'svd_model.pkl')
        if os.path.exists(svd_path):
            st.session_state.svd = joblib.load(svd_path)
            loaded_files.append('SVD Model')
            print("SVD model loaded")
        else:
            st.warning(f"SVD model not found at {svd_path}")
        
        # 2. Load GMM map
        gmm_map_path = os.path.join(load_dir, 'gmm_map.pkl')
        gmm_map = None
        if os.path.exists(gmm_map_path):
            gmm_map = joblib.load(gmm_map_path)
            st.session_state.gmm_map = gmm_map
            loaded_files.append('GMM Map')
            print(f"GMM Map loaded: {len(gmm_map)} items")
        else:
            gmm_map_full_path = os.path.join(load_dir, 'gmm_map_full.pkl')
            if os.path.exists(gmm_map_full_path):
                gmm_map = joblib.load(gmm_map_full_path)
                st.session_state.gmm_map = gmm_map
                loaded_files.append('GMM Map Full')
                print(f" GMM Map Full loaded: {len(gmm_map)} items")
        
        # 3. Load user_pref
        user_pref_path = os.path.join(load_dir, 'user_pref.npy')
        user_pref = None
        if os.path.exists(user_pref_path):
            user_pref = np.load(user_pref_path, allow_pickle=True)
            loaded_files.append('User Preference')
            print(f"User Preference loaded: shape {user_pref.shape if user_pref is not None else 'None'}")
        
        # 4. Load item_vectors
        item_vectors_path = os.path.join(load_dir, 'item_vectors.pkl')
        item_vectors = None
        if os.path.exists(item_vectors_path):
            item_vectors = joblib.load(item_vectors_path)
            loaded_files.append('Item Vectors')
            print(f"Item Vectors loaded: {len(item_vectors) if item_vectors else 0} items")
        
        # 5. Rekonstruksi Hybrid model
        if st.session_state.get('svd') is not None and gmm_map is not None:
            hybrid = HybridGMM_SVD(st.session_state.svd, gmm_map)
            if user_pref is not None:
                hybrid.user_pref = user_pref
            if item_vectors is not None:
                hybrid.item_vectors = item_vectors
            if st.session_state.get('trainset') is not None:
                hybrid.trainset = st.session_state.trainset
            
            st.session_state.hybrid = hybrid
            loaded_files.append('Hybrid Model (Reconstructed)')
            print(" Hybrid model reconstructed")
        
        # 6. Load training outputs (One-Hot Encoding, GMM BIC, dll)
        training_outputs_loaded = load_training_outputs(load_dir)
        if training_outputs_loaded:
            loaded_files.append('Training Outputs (One-Hot Encoding, GMM)')
            print("Training outputs loaded successfully")
        
        # 7. Load trainset
        trainset_path = os.path.join(load_dir, 'trainset.pkl')
        if os.path.exists(trainset_path):
            with open(trainset_path, 'rb') as f:
                st.session_state.trainset = pickle.load(f)
            loaded_files.append('Trainset')
            print(f"Trainset loaded: {st.session_state.trainset.n_users} users")
        
        # 8. Load testset
        testset_path = os.path.join(load_dir, 'testset.pkl')
        if os.path.exists(testset_path):
            with open(testset_path, 'rb') as f:
                st.session_state.testset = pickle.load(f)
            loaded_files.append('Testset')
            print(f"Testset loaded: {len(st.session_state.testset)} ratings")
        
        # 9. Load genre cols
        genre_path = os.path.join(load_dir, 'genre_cols.pkl')
        if os.path.exists(genre_path):
            st.session_state.genre_cols = joblib.load(genre_path)
            loaded_files.append('Genre Columns')
        
        # 10. Load cluster genre mapping
        cluster_path = os.path.join(load_dir, 'cluster_genre_mapping.pkl')
        if os.path.exists(cluster_path):
            st.session_state.cluster_genre_mapping = joblib.load(cluster_path)
            loaded_files.append('Cluster Genre Mapping')
        
        # 11. Load metrics
        metrics_path = os.path.join(load_dir, 'metrics.pkl')
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
            for key, value in metrics.items():
                st.session_state[key] = value
            loaded_files.append('Metrics')
            print(f"Metrics loaded: SVD RMSE={st.session_state.get('svd_rmse', 0):.6f}")
        
        # 12. Load dataframe hasil prediksi
        svd_df_path = os.path.join(load_dir, 'svd_predictions_df.pkl')
        if os.path.exists(svd_df_path):
            st.session_state.svd_predictions_df = joblib.load(svd_df_path)
            loaded_files.append('SVD Predictions DF')
            print(f"SVD Predictions DF loaded: {len(st.session_state.svd_predictions_df)} rows")
        
        hybrid_df_path = os.path.join(load_dir, 'hybrid_predictions_df.pkl')
        if os.path.exists(hybrid_df_path):
            st.session_state.hybrid_predictions_df = joblib.load(hybrid_df_path)
            loaded_files.append('Hybrid Predictions DF')
        
        comparison_df_path = os.path.join(load_dir, 'comparison_df.pkl')
        if os.path.exists(comparison_df_path):
            st.session_state.comparison_df = joblib.load(comparison_df_path)
            loaded_files.append('Comparison DF')
        
        # 13. Load grid results (Tab 2)
        grid_path = os.path.join(load_dir, 'grid_results.pkl')
        if os.path.exists(grid_path):
            st.session_state.grid_results = joblib.load(grid_path)
            loaded_files.append('Grid Results')
            print("Grid results loaded")

        # 14. Load CV results (Tab 4)
        cv_path = os.path.join(load_dir, 'cv_results.pkl')
        if os.path.exists(cv_path):
            st.session_state.tab3_results = joblib.load(cv_path)
            loaded_files.append('CV Results')
            print("CV results loaded")

        # 15. Load tab3_processed flag
        tab3_processed_path = os.path.join(load_dir, 'tab3_processed.pkl')
        if os.path.exists(tab3_processed_path):
            st.session_state.tab3_processed = joblib.load(tab3_processed_path)
            loaded_files.append('Tab3 Processed')
            print("Tab3 processed flag loaded")

        # 16. Load best params
        best_params_path = os.path.join(load_dir, 'best_params.pkl')
        if os.path.exists(best_params_path):
            st.session_state.best_params = joblib.load(best_params_path)
            loaded_files.append('Best Parameters')
            print("Best parameters loaded")
        
        # 17. Load data split
        train_df_path = os.path.join(load_dir, 'train_df.csv')
        if os.path.exists(train_df_path):
            st.session_state.train_df = pd.read_csv(train_df_path)
            loaded_files.append('Train Data')
        
        test_df_path = os.path.join(load_dir, 'test_df.csv')
        if os.path.exists(test_df_path):
            st.session_state.test_df = pd.read_csv(test_df_path)
            loaded_files.append('Test Data')
        
        # 18. Load train_movies dan test_movies
        train_movies_path = os.path.join(load_dir, 'train_movies.pkl')
        if os.path.exists(train_movies_path):
            st.session_state.train_movies = set(joblib.load(train_movies_path))
            loaded_files.append('Train Movies')
        
        test_movies_path = os.path.join(load_dir, 'test_movies.pkl')
        if os.path.exists(test_movies_path):
            st.session_state.test_movies = set(joblib.load(test_movies_path))
            loaded_files.append('Test Movies')
        
        # 19. Load split info
        split_path = os.path.join(load_dir, 'split_info.pkl')
        if os.path.exists(split_path):
            split_info = joblib.load(split_path)
            st.session_state.split_done = split_info.get('split_done', False)
            loaded_files.append('Split Info')
        
        # Set flags
        if 'svd' in st.session_state and 'hybrid' in st.session_state:
            st.session_state.model_trained = True
       
        if loaded_files:
            st.success(f"Berhasil memuat {len(loaded_files)} komponen model")
            
            # Tampilkan output training yang sudah di-load di Tab 3
            if training_outputs_loaded:
                with st.expander("📊 Output Proses Training (One-Hot Encoding, GMM, dll) - Dimuat Ulang", expanded=True):
                    st.info("Berikut adalah output dari proses training sebelumnya:")
                    # Panggil fungsi display untuk menampilkan output
                    if 'saved_outputs' in st.session_state and 'tab2_output' in st.session_state.saved_outputs:
                        display_saved_outputs('tab2_output')
                    else:
                        st.warning("Tidak ada output training yang tersimpan")
            
            return True
        else:
            st.error("Tidak ada komponen model yang berhasil dimuat")
            return False
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False


# ====================== MODIFIKASI FUNGSI AUTO_SAVE_ALL_COMPONENTS ======================

def auto_save_all_components():
    """
    Menyimpan semua komponen yang ada di session state secara otomatis.
    """
    import os
    import joblib
    import pickle  
    import numpy as np

    save_dir = MODEL_DIR
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    saved_count = 0
    saved_items = []
    
    # 1. Simpan data split (Tab 1)
    if st.session_state.get('split_done', False):
        if st.session_state.get('train_df') is not None:
            st.session_state.train_df.to_csv(os.path.join(save_dir, 'train_df.csv'), index=False)
            saved_items.append('train_df.csv')
            saved_count += 1
        
        if st.session_state.get('test_df') is not None:
            st.session_state.test_df.to_csv(os.path.join(save_dir, 'test_df.csv'), index=False)
            saved_items.append('test_df.csv')
            saved_count += 1
        
        if st.session_state.get('train_movies') is not None:
            joblib.dump(list(st.session_state.train_movies), os.path.join(save_dir, 'train_movies.pkl'))
            saved_items.append('train_movies.pkl')
            saved_count += 1
        
        if st.session_state.get('test_movies') is not None:
            joblib.dump(list(st.session_state.test_movies), os.path.join(save_dir, 'test_movies.pkl'))
            saved_items.append('test_movies.pkl')
            saved_count += 1
        
        if st.session_state.get('ratings') is not None:
            st.session_state.ratings.to_csv(os.path.join(save_dir, 'ratings.csv'), index=False)
            saved_items.append('ratings.csv')
            saved_count += 1
        
        if st.session_state.get('movies') is not None:
            st.session_state.movies.to_csv(os.path.join(save_dir, 'movies.csv'), index=False)
            saved_items.append('movies.csv')
            saved_count += 1
    
    # 2. Simpan hasil Grid Search (Tab 2)
    if st.session_state.get('grid_results') is not None:
        joblib.dump(st.session_state.grid_results, os.path.join(save_dir, 'grid_results.pkl'))
        saved_items.append('grid_results.pkl')
        saved_count += 1
    
    # 3. Simpan model utama (Tab 3)
    if st.session_state.get('svd') is not None:
        joblib.dump(st.session_state.svd, os.path.join(save_dir, 'svd_model.pkl'))
        saved_items.append('svd_model.pkl')
        saved_count += 1
    
    if st.session_state.get('hybrid') is not None:
        hybrid = st.session_state.hybrid
        if hasattr(hybrid, 'gmm_map') and hybrid.gmm_map is not None:
            joblib.dump(hybrid.gmm_map, os.path.join(save_dir, 'gmm_map.pkl'))
            saved_items.append('gmm_map.pkl')
            saved_count += 1
        
        if hasattr(hybrid, 'user_pref') and hybrid.user_pref is not None:
            np.save(os.path.join(save_dir, 'user_pref.npy'), hybrid.user_pref)
            saved_items.append('user_pref.npy')
            saved_count += 1
        
        if hasattr(hybrid, 'item_vectors') and hybrid.item_vectors:
            joblib.dump(hybrid.item_vectors, os.path.join(save_dir, 'item_vectors.pkl'))
            saved_items.append('item_vectors.pkl')
            saved_count += 1
    
    if st.session_state.get('gmm_map') is not None:
        joblib.dump(st.session_state.gmm_map, os.path.join(save_dir, 'gmm_map_full.pkl'))
        saved_items.append('gmm_map_full.pkl')
        saved_count += 1
    
    # 4. Simpan data training (Tab 3)
    if st.session_state.get('trainset') is not None:
        with open(os.path.join(save_dir, 'trainset.pkl'), 'wb') as f:
            pickle.dump(st.session_state.trainset, f)
        saved_items.append('trainset.pkl')
        saved_count += 1
    
    if st.session_state.get('testset') is not None:
        with open(os.path.join(save_dir, 'testset.pkl'), 'wb') as f:
            pickle.dump(st.session_state.testset, f)
        saved_items.append('testset.pkl')
        saved_count += 1
    
    if st.session_state.get('genre_cols') is not None:
        joblib.dump(st.session_state.genre_cols, os.path.join(save_dir, 'genre_cols.pkl'))
        saved_items.append('genre_cols.pkl')
        saved_count += 1
    
    if st.session_state.get('cluster_genre_mapping') is not None:
        joblib.dump(st.session_state.cluster_genre_mapping, os.path.join(save_dir, 'cluster_genre_mapping.pkl'))
        saved_items.append('cluster_genre_mapping.pkl')
        saved_count += 1
    
    # 5. Simpan hasil prediksi (Tab 3)
    if st.session_state.get('svd_predictions_df') is not None:
        joblib.dump(st.session_state.svd_predictions_df, os.path.join(save_dir, 'svd_predictions_df.pkl'))
        saved_items.append('svd_predictions_df.pkl')
        saved_count += 1
    
    if st.session_state.get('hybrid_predictions_df') is not None:
        joblib.dump(st.session_state.hybrid_predictions_df, os.path.join(save_dir, 'hybrid_predictions_df.pkl'))
        saved_items.append('hybrid_predictions_df.pkl')
        saved_count += 1
    
    if st.session_state.get('comparison_df') is not None:
        joblib.dump(st.session_state.comparison_df, os.path.join(save_dir, 'comparison_df.pkl'))
        saved_items.append('comparison_df.pkl')
        saved_count += 1
    
    # 6. Simpan training outputs (One-Hot Encoding, GMM, dll)
    training_outputs_saved = save_training_outputs(save_dir)
    if training_outputs_saved:
        saved_count += len(training_outputs_saved)
        for name, path in training_outputs_saved:
            saved_items.append(f"{name} - {os.path.basename(path)}")
    
    # 7. Simpan metrik
    metrics = {
        'svd_rmse': st.session_state.get('svd_rmse', 0),
        'svd_mae': st.session_state.get('svd_mae', 0),
        'hybrid_rmse': st.session_state.get('hybrid_rmse', 0),
        'hybrid_mae': st.session_state.get('hybrid_mae', 0),
        'model_trained': st.session_state.get('model_trained', False),
        'split_done': st.session_state.get('split_done', False),
        'n_factors': st.session_state.get('n_factors', 100),
        'n_epochs': st.session_state.get('n_epochs', 30),
        'max_components': st.session_state.get('max_components', 50),
        'smoothing': st.session_state.get('smoothing', 0.1),
        'test_size': st.session_state.get('test_size', 0.2)
    }
    joblib.dump(metrics, os.path.join(save_dir, 'metrics.pkl'))
    saved_items.append('metrics.pkl')
    saved_count += 1
    
    # 8. Simpan hasil CV (Tab 4)
    if st.session_state.get('tab3_results') is not None:
        joblib.dump(st.session_state.tab3_results, os.path.join(save_dir, 'cv_results.pkl'))
        saved_items.append('cv_results.pkl')
        saved_count += 1
    
    if st.session_state.get('tab3_processed', False):
        joblib.dump(True, os.path.join(save_dir, 'tab3_processed.pkl'))
        saved_items.append('tab3_processed.pkl')
        saved_count += 1
    
    # 9. Simpan parameter terbaik dari grid search
    if st.session_state.get('grid_results') is not None and st.session_state.grid_results.get('best_params'):
        best_params = st.session_state.grid_results['best_params']
        joblib.dump(best_params, os.path.join(save_dir, 'best_params.pkl'))
        saved_items.append('best_params.pkl')
        saved_count += 1
    
    # 10. Simpan split info
    split_info = {
        'split_done': st.session_state.get('split_done', False),
        'train_movies': list(st.session_state.get('train_movies', set())) if st.session_state.get('train_movies') else [],
        'test_movies12': list(st.session_state.get('test_movies', set())) if st.session_state.get('test_movies') else [],
        'train_df_shape': st.session_state.get('train_df', pd.DataFrame()).shape if st.session_state.get('train_df') is not None else (0, 0),
        'test_df_shape': st.session_state.get('test_df', pd.DataFrame()).shape if st.session_state.get('test_df') is not None else (0, 0)
    }
    joblib.dump(split_info, os.path.join(save_dir, 'split_info.pkl'))
    saved_items.append('split_info.pkl')
    saved_count += 1
    
    print(f"\nAuto-save completed! Saved {saved_count} components to {save_dir}")
    return saved_count, saved_items

# ====================== FUNGSI TRANSFORM FILM TEST DENGAN GMM ======================

def transform_test_movies_with_gmm(movies_df, test_movie_ids, gmm_model, feature_cols, smoothing_factor=0.1):
    """
    Mentransformasi film test set menggunakan model GMM yang sudah dilatih.
    
    Parameter:
        movies_df (DataFrame): Data film
        test_movie_ids (set): ID film dalam test set
        gmm_model: Trained GMM model
        feature_cols (list): Nama kolom fitur
        smoothing_factor (float): Faktor smoothing untuk one-hot encoding
        
    Returns:
        dict: Mapping film test ke probabilitas cluster GMM
    """
    test_movie_ids = set([int(mid) for mid in test_movie_ids])
    movies_df = movies_df.copy()
    movies_df['movieId'] = movies_df['movieId'].astype(int)
    
    test_movies = movies_df[movies_df['movieId'].isin(test_movie_ids)].copy()
    
    missing_movies = test_movie_ids - set(test_movies['movieId'])
    if missing_movies:
        for movie_id in missing_movies:
            new_row = pd.DataFrame([{
                'movieId': movie_id,
                'title': f'Unknown Movie {movie_id}',
                'genres': '(no genres listed)'
            }])
            test_movies = pd.concat([test_movies, new_row], ignore_index=True)
    
    genre_cols = [col.replace('genre_', '') for col in feature_cols]
    
    for genre in genre_cols:
        test_movies[f'genre_{genre}'] = smoothing_factor
    
    for idx, row in test_movies.iterrows():
        genres_str = str(row['genres'])
        if genres_str != '(no genres listed)' and genres_str != 'nan':
            movie_genres = genres_str.split('|')
            n_genres = len(movie_genres)
            
            genre_value = 1.0 - (smoothing_factor * (n_genres - 1))
            
            for genre in movie_genres:
                col_name = f'genre_{genre}'
                if col_name in test_movies.columns:
                    test_movies.at[idx, col_name] = genre_value
    
    X_test = test_movies[feature_cols].values
    
    probs = gmm_model.predict_proba(X_test)
    
    test_gmm_map = {str(int(mid)): probs[i] for i, mid in enumerate(test_movies['movieId'])}
    
    return test_gmm_map

# ====================== FUNGSI KONVERSI KE SURPRISE DENGAN SAVE OUTPUT ======================

def convert_to_surprise_format_with_save(train_ratings, test_ratings, output_key, rating_scale=(0.5, 5), show_tables=True):
    """
    Konversi DataFrame ke format Surprise Dataset dengan menyimpan output.
    
    Parameter:
        train_ratings (DataFrame): Data rating training
        test_ratings (DataFrame): Data rating test
        output_key (str): Key untuk menyimpan output
        rating_scale (tuple): Skala rating (min, max)
        show_tables (bool): Tampilkan output
        
    Returns:
        tuple: (trainset, testset)
    """
    output_container = st.empty()
    
    save_output(output_container, output_key, "LANGKAH 5: KONVERSI KE FORMAT SURPRISE", "info")
    start_time = time.time()
    
    reader = Reader(rating_scale=rating_scale)
    
    train_df = train_ratings.copy()
    train_df['userId'] = train_df['userId'].astype(str)
    train_df['movieId'] = train_df['movieId'].astype(str)
    
    data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    testset = []
    if not test_ratings.empty:
        test_df = test_ratings.copy()
        test_df['userId'] = test_df['userId'].astype(str)
        test_df['movieId'] = test_df['movieId'].astype(str)
        testset = [(str(row['userId']), str(row['movieId']), float(row['rating'])) 
                   for _, row in test_df.iterrows()]
    
    duration = time.time() - start_time
    save_output(output_container, output_key, 
               f"Konversi selesai! Trainset: {trainset.n_users} users, {trainset.n_items} items ({duration:.2f} detik)", "success")
    
    if show_tables:
        save_output(output_container, output_key, f"TABEL: SEMUA User dan Ratingnya (Total: {trainset.n_users} User)", "markdown")
        user_samples = []
        for u in range(trainset.n_users):
            raw_uid = trainset.to_raw_uid(u)
            n_ratings = len(trainset.ur[u])
            ratings_list = [r for (_, r) in trainset.ur[u]]
            avg_rating = np.mean(ratings_list) if ratings_list else 0
            user_samples.append({
                'User ID': raw_uid,
                'Jumlah Rating': n_ratings,
                'Rata-rata Rating': f"{avg_rating:.3f}",
                'Min Rating': min(ratings_list) if ratings_list else 0,
                'Max Rating': max(ratings_list) if ratings_list else 0
            })
        save_output(output_container, output_key, pd.DataFrame(user_samples), "dataframe_full")
    
    return trainset, testset

# ====================== FUNGSI TRAINING SVD DENGAN SAVE OUTPUT ======================

def train_svd_model_with_save(trainset, output_key, n_factors=100, n_epochs=30, lr_all=0.005, reg_all=0.02, 
                   random_state=42, show_tables=True):
    """
    Training model SVD dengan menyimpan output.
    
    Parameter:
        trainset: Surprise trainset
        output_key (str): Key untuk menyimpan output
        n_factors (int): Jumlah faktor latent
        n_epochs (int): Jumlah epoch training
        lr_all (float): Learning rate
        reg_all (float): Regularization parameter
        random_state (int): Random seed
        show_tables (bool): Tampilkan output
        
    Returns:
        SVD: Trained SVD model
    """
    output_container = st.empty()
    
    save_output(output_container, output_key, "LANGKAH 6: TRAINING MODEL SVD", "info")
    start_time = time.time()
    
    svd = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state,
        verbose=False
    )
    
    svd.fit(trainset)
    
    duration = time.time() - start_time
    save_output(output_container, output_key, f"Training SVD selesai! Waktu: {duration:.2f} detik", "success")
    
    if show_tables:
        save_output(output_container, output_key, "TABEL: Parameter Model SVD", "markdown")
        svd_params = pd.DataFrame({
            'Parameter': ['n_factors', 'n_epochs', 'learning_rate', 'regularization'],
            'Nilai': [n_factors, n_epochs, lr_all, reg_all]
        })
        save_output(output_container, output_key, svd_params, "dataframe")
    
    return svd

# ====================== FUNGSI TRAINING HYBRID DENGAN SAVE OUTPUT ======================
def train_hybrid_model_with_save(svd_model, gmm_map, trainset, testset, output_key, 
                                  movies_df=None, genre_cols=None, ratings_df=None,
                                  show_tables=True):
    """
    Training Hybrid GMM+SVD Model menggunakan calculate_adaptive_weights().
    """
    output_container = st.empty()
    
    save_output(output_container, output_key, "LANGKAH 7: TRAINING MODEL HYBRID GMM+SVD", "info")
    start_time = time.time()
    
    hybrid_model = HybridGMM_SVD(
        svd_model, 
        gmm_map,
        movies_df=movies_df,
        genre_cols=genre_cols,
        ratings_df=ratings_df
    )
    hybrid_model.fit(trainset, show_tables=show_tables)
    
    duration = time.time() - start_time
    save_output(output_container, output_key, f"Training Hybrid selesai! Waktu: {duration:.2f} detik", "success")
    
    return hybrid_model

# ====================== HYBRID GMM + SVD MODEL ======================
class HybridGMM_SVD:
    """
    Hybrid Model dengan adaptive weighting menggunakan calculate_adaptive_weights().
    Menggabungkan SVD (Collaborative Filtering) dan GMM (Content-based).
    """
    
    def __init__(self, svd_model, gmm_map, movies_df=None, genre_cols=None, ratings_df=None):
        self.svd = svd_model
        self.gmm_map = gmm_map if gmm_map is not None else {}
        self.trainset = None
        self.user_pref = None  # Akan diinisialisasi di fit()
        self.item_vectors = {}
        self.movies_df = movies_df
        self.genre_cols = genre_cols
        self.ratings_df = ratings_df
        
    def fit(self, trainset, show_tables=True):
        self.trainset = trainset
        
        # Inisialisasi user_pref dengan zeros jika belum ada
        if self.gmm_map and len(self.gmm_map) > 0:
            vector_dim = len(next(iter(self.gmm_map.values())))
            self.user_pref = np.zeros((trainset.n_users, vector_dim))
            rating_counts = np.zeros(trainset.n_users)
            
            for u, i, r in trainset.all_ratings():
                raw_iid = trainset.to_raw_iid(i)
                raw_iid_str = str(raw_iid)
                
                if raw_iid_str in self.gmm_map:
                    normalized_rating = (r - 3.0) / 2.0
                    self.user_pref[u] += normalized_rating * self.gmm_map[raw_iid_str]
                    rating_counts[u] += 1
                    self.item_vectors[str(i)] = self.gmm_map[raw_iid_str]
            
            for u in range(trainset.n_users):
                if rating_counts[u] > 0:
                    self.user_pref[u] /= rating_counts[u]
                    norm = np.linalg.norm(self.user_pref[u])
                    if norm > 0:
                        self.user_pref[u] = self.user_pref[u] / norm
        else:
            # Jika gmm_map kosong, buat user_pref dengan dimensi 1
            self.user_pref = np.zeros((trainset.n_users, 1))
        
        return self
    
    def cosine_similarity(self, a, b):
        """
        Menghitung cosine similarity antara dua vektor.
        
        Parameters:
            a (numpy.ndarray): Vektor pertama
            b (numpy.ndarray): Vektor kedua
            
        Returns:
            float: Nilai cosine similarity antara -1 dan 1
        """
        # Handle jika salah satu vektor adalah None
        if a is None or b is None:
            return 0.0
        
        # Konversi ke numpy array jika perlu
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        if not isinstance(b, np.ndarray):
            b = np.array(b)
        
        # Hitung norm (panjang) dari masing-masing vektor
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Jika salah satu vektor adalah nol (panjang 0)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        # Hitung dot product
        dot_product = np.dot(a, b)
        
        # Hitung cosine similarity
        similarity = dot_product / (norm_a * norm_b)
        
        # Clip nilai ke range [-1, 1] untuk menghindari floating point error
        similarity = np.clip(similarity, -1.0, 1.0)
        
        # Konversi dari numpy float ke Python float
        return float(similarity)

    def _get_adaptive_alpha(self, uid):
        """Menghitung alpha menggunakan calculate_adaptive_weights()"""
        weights = calculate_adaptive_weights(
            user_data=uid,
            trainset=self.trainset,
            movies_df=self.movies_df,
            genre_cols=self.genre_cols,
            ratings_df=self.ratings_df,
            svd_model=self.svd,
            hybrid_model=self
        )
        return weights['cf_weight']

    def predict(self, uid, iid, alpha=None):
        """
        Prediksi rating.
        
        Parameters:
            uid: User ID
            iid: Item ID
            alpha: Bobot CF (0-1). 
                - Jika None: dihitung otomatis dengan calculate_adaptive_weights()
                - Jika diisi: digunakan langsung (untuk keperluan CV)
        
        Returns:
            tuple: (predicted_rating, gmm_used)
        """
        try:
            svd_pred = self.svd.predict(uid, iid).est
        except:
            svd_pred = self.trainset.global_mean if self.trainset else 3.0
        
        iid_str = str(iid)
        
        try:
            u = self.trainset.to_inner_uid(str(uid))
            user_found = True
        except:
            user_found = False
        
        item_vector = None
        if iid_str in self.item_vectors:
            item_vector = self.item_vectors[iid_str]
        elif iid_str in self.gmm_map:
            item_vector = self.gmm_map[iid_str]
            self.item_vectors[iid_str] = item_vector
        
        # Jika user tidak ditemukan atau item tidak punya vector GMM
        if not user_found or item_vector is None:
            return svd_pred, False
        
        # CEK: Pastikan user_pref tidak None dan index valid
        if self.user_pref is None:
            # Jika user_pref belum diinisialisasi, return SVD prediction
            return svd_pred, False
        
        if u >= len(self.user_pref):
            # Index out of bounds, return SVD prediction
            return svd_pred, False
        
        # Gunakan alpha yang diberikan atau hitung otomatis
        if alpha is not None:
            cf_weight = alpha
        else:
            cf_weight = self._get_adaptive_alpha(uid)
        
        # Hitung cosine similarity dengan safe handling
        user_vec = self.user_pref[u]
        if user_vec is None:
            return svd_pred, False
        
        sim = self.cosine_similarity(user_vec, item_vector)
        if np.isnan(sim):
            sim = 0.0
        
        sim = (sim + 1.0) / 2.0
        
        if u < len(self.trainset.ur):
            user_mean = np.mean([r for (_, r) in self.trainset.ur[u]])
        else:
            user_mean = self.trainset.global_mean
        
        content_score = user_mean + (sim - 0.5) * 2.0
        content_score = np.clip(content_score, 0.5, 5.0)
        
        hybrid_pred = cf_weight * svd_pred + (1 - cf_weight) * content_score
        hybrid_pred = np.clip(hybrid_pred, 0.5, 5.0)
        
        return hybrid_pred, True


# ====================== MODIFIED PROCESS NEW USER ======================
def process_new_user_flow_with_adaptive_weights(initial_ratings, movies_df, hybrid_model, trainset, 
                                               svd_model, ratings_df, genre_cols, 
                                               cluster_genre_mapping=None,
                                               show_tables=True):
    """
    Proses flow untuk user baru dengan integrasi langsung ke model.
    """
    if show_tables:
        st.info("MEMPROSES USER BARU DENGAN INTEGRASI LANGSUNG KE MODEL")
    start_time = time.time()
    
    if initial_ratings is None:
        initial_ratings = {}
    
    n_initial_ratings = len(initial_ratings)
    
    # Ambil genre dari key yang benar
    selected_fav_genres = st.session_state.get('new_user_genres', [])
    if not selected_fav_genres:
        selected_fav_genres = st.session_state.get('selected_fav_genres', [])
    if not selected_fav_genres:
        selected_fav_genres = st.session_state.get('temp_genres', []) 
    if show_tables:
        st.info(f"Menerima {n_initial_ratings} rating awal")
        if n_initial_ratings > 0:
            sample_ratings = list(initial_ratings.items())[:5]
            for movie_id, rating in sample_ratings:
                movie_info = movies_df[movies_df['movieId'] == int(movie_id)]
                title = movie_info['title'].values[0] if not movie_info.empty else f"Movie {movie_id}"
                st.write(f"  - {title}: {rating}")
    
    if n_initial_ratings == 0:
        if len(selected_fav_genres) < 3:
            st.error(f"Untuk user tanpa rating, pilih minimal 3 genre!")
            raise ValueError("Minimal 3 genre favorit diperlukan")
        else:
            if show_tables:
                st.success(f"Cold-start user dengan {len(selected_fav_genres)} genre favorit")
    
    all_user_ids = []
    for uid in trainset.all_users():
        raw_uid = trainset.to_raw_uid(uid)
        try:
            all_user_ids.append(int(raw_uid))
        except:
            continue
    
    for uid in st.session_state.registered_users.keys():
        try:
            all_user_ids.append(int(uid))
        except:
            continue
    
    new_user_id = max(all_user_ids) + 1 if all_user_ids else 100000
    
    temp_user_data = {
        'user_id': new_user_id,
        'n_ratings': n_initial_ratings,
        'initial_ratings': initial_ratings,
        'fav_genres': selected_fav_genres
    }
    
    genre_diversity = calculate_genre_diversity(new_user_id, trainset, movies_df, genre_cols)
    rating_entropy = calculate_rating_entropy(new_user_id, trainset)
    
    if n_initial_ratings == 0 and len(selected_fav_genres) == 0:
        if show_tables:
            st.error("User baru dengan 0 rating WAJIB memilih minimal 3 genre favorit!")
        raise ValueError("Minimal 3 genre favorit diperlukan untuk user tanpa rating")

    if n_initial_ratings < 5:
        cf_weight = 0.20
        method_used = "rule_cold_start_rating_lt_5"
    else:
        cf_weight = 0.5
        method_used = "fallback"
    
    adaptive_weights = {
        'cf_weight': cf_weight,
        'cb_weight': 1.0 - cf_weight,
        'n_ratings': n_initial_ratings,
        'factors': {
            'quantity': np.log1p(n_initial_ratings) / np.log1p(100),
            'quality': 0.5,
            'embedding': 0.3,
            'gmm': 0.5,
            'diversity': genre_diversity,
            'entropy': rating_entropy,
            'temporal': 0.5,
            'error': 0.5
        },
        'confidence_level': min(0.1 + n_initial_ratings/100, 0.9),
        'exploration_factor': max(0.3, 1.0 - cf_weight),
        'diversity_factor': 0.5 + genre_diversity * 0.5,
        'popularity_bias': 1.0 - (np.log1p(n_initial_ratings) / np.log1p(100)),
        'method': method_used,
        'is_cold_start': True
    }
    
    if show_tables:
        st.info(f"Bobot adaptif (CF={cf_weight:.4f}, CB={1-cf_weight:.4f})")
        st.info(f"Confidence: {adaptive_weights['confidence_level']:.2f}, Diversity: {adaptive_weights['diversity_factor']:.2f}")
    
    all_ratings = []
    
    for uid, iid, rating in trainset.all_ratings():
        raw_uid = trainset.to_raw_uid(uid)
        raw_iid = trainset.to_raw_iid(iid)
        all_ratings.append((str(raw_uid), str(raw_iid), rating))
    
    for movie_id, rating in initial_ratings.items():
        all_ratings.append((str(new_user_id), str(movie_id), float(rating)))
    
    ratings_df_new = pd.DataFrame(all_ratings, columns=['userId', 'movieId', 'rating'])
    
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df_new[['userId', 'movieId', 'rating']], reader)
    new_trainset = data.build_full_trainset()
    
    if 'svd_params' in st.session_state:
        params = st.session_state.svd_params
        n_factors = params.get('n_factors', 100)
        n_epochs = params.get('n_epochs', 30)
        lr_all = params.get('lr_all', 0.005)
        reg_all = params.get('reg_all', 0.02)
        random_state = params.get('random_state', 42)
    else:
        n_factors = 100
        n_epochs = 30
        lr_all = 0.005
        reg_all = 0.02
        random_state = 42
    
    new_svd = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state,
        verbose=False
    )
    new_svd.fit(new_trainset)
    
    new_hybrid = HybridGMM_SVD(new_svd, hybrid_model.gmm_map)
    new_hybrid.fit(new_trainset, show_tables=False)

    # ========== TAMBAHKAN 3 BARIS INI ==========
    if initial_ratings:
        update_user_pref_with_negative_ratings(new_hybrid, new_user_id, initial_ratings, movies_df)
        if show_tables and any(r <= 2.0 for r in initial_ratings.values()):
            st.info("Rating negatif terdeteksi! Film dengan genre serupa akan dipenalti.")
    # ===========================================

    st.session_state.trainset = new_trainset
    st.session_state.svd = new_svd
    st.session_state.hybrid = new_hybrid
    
    user_data = {
        'user_id': new_user_id,
        'initial_ratings': initial_ratings.copy() if initial_ratings else {},
        'fav_genres': selected_fav_genres.copy() if selected_fav_genres else [],
        'n_ratings': n_initial_ratings,
        'integrated': True,
        'created_at': time.time(),
        'adaptive_weights': adaptive_weights
    }
    
    st.session_state.registered_users[str(new_user_id)] = user_data
    
    if show_tables:
        duration = time.time() - start_time
        st.success(f"User baru BERHASIL DIINTEGRASIKAN ke model! ID: {new_user_id} ({duration:.2f} detik)")
        
        # ========== PERBAIKAN: Reset state dengan benar ==========
        # Hapus key yang tidak perlu lagi
        keys_to_delete = ['new_user_genres', 'temp_genres', 'selected_fav_genres']
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        # ========================================================
        
        st.info(f"Jumlah rating: {n_initial_ratings}")
        st.info(f"Alpha (CF Weight): {cf_weight:.4f}")
        
        with st.expander("Detail Bobot Adaptif"):
            st.json(adaptive_weights)
        
        if n_initial_ratings > 0:
            with st.expander("Detail Rating User Baru"):
                rating_list = []
                for movie_id, rating in initial_ratings.items():
                    movie_info = movies_df[movies_df['movieId'] == int(movie_id)]
                    title = movie_info['title'].values[0] if not movie_info.empty else f"Movie {movie_id}"
                    rating_list.append({'Film': title, 'Rating': rating})
                st.dataframe(pd.DataFrame(rating_list))
    
    return {
        'user_id': new_user_id,
        'user_type': 'integrated',
        'adaptive_weights': adaptive_weights,
        'n_ratings': n_initial_ratings,
        'fav_genres': selected_fav_genres,
        'integrated': True,
        'initial_ratings': initial_ratings
    }

def get_recommendations_final(user_id, user_type, hybrid_model, svd_model, 
                                trainset, movies_df, ratings_df, gmm_map, 
                                genre_cols=None, cluster_genre_mapping=None,
                                selected_genres=None, top_n=None, 
                                fav_genres=None, max_candidates=20000,
                                force_cold_start=False, user_adaptive_weights=None):
    """
    Mendapatkan rekomendasi dengan RULE-BASED Adaptive Weight.
    """
    user_id_str = str(user_id)
    registered_users = st.session_state.get('registered_users', {})
    negatif_movies = []
    
    if user_id_str in registered_users:
        initial_ratings = registered_users[user_id_str].get('initial_ratings', {})
        for movie_id, rating in initial_ratings.items():
            if rating <= 2.0:
                negatif_movies.append(movie_id)
    
    st.session_state[f'user_negatif_{user_id}'] = negatif_movies
    # ==================================================
    

    
    import numpy as np
    from tqdm import tqdm
    
    # Inisialisasi
    global_mean = ratings_df['rating'].mean() if ratings_df is not None else 3.0
    
    # Cek apakah user baru atau existing
    is_new_user = False
    n_user_ratings = 0
    try:
        u_inner = trainset.to_inner_uid(str(user_id))
        n_user_ratings = len(trainset.ur[u_inner])
        is_new_user = False
        print(f"User {user_id} ditemukan di trainset dengan {n_user_ratings} rating")
    except:
        is_new_user = True
        print(f"User {user_id} adalah user baru (tidak ditemukan di trainset)")
    
    if is_new_user or n_user_ratings < 5:
        global_mean = max(global_mean, 3.2)
        print(f"User {user_id} adalah cold-start, menggunakan global_mean={global_mean:.2f}")
    
    # Validasi gmm_map
    if gmm_map is None or len(gmm_map) == 0:
        print(f"WARNING: gmm_map is empty or None! Size: {len(gmm_map) if gmm_map is not None else 'None'}")
    else:
        print(f"gmm_map size: {len(gmm_map)}")
    
    # Validasi hybrid_model
    if hybrid_model is not None:
        if hasattr(hybrid_model, 'user_pref') and hybrid_model.user_pref is not None:
            print(f"hybrid_model.user_pref shape: {hybrid_model.user_pref.shape}")
        else:
            print("WARNING: hybrid_model.user_pref is None!")
    
    # ====================== KUMPULKAN SEMUA FILM ======================
    all_movies = []
    for mid in movies_df['movieId'].unique():
        all_movies.append(str(mid))
    
    print(f"Total film awal: {len(all_movies)}")
    
    # Filter berdasarkan genre jika ada
    if selected_genres and len(selected_genres) > 0:
        filtered_movies = []
        for mid in tqdm(all_movies, desc="Filtering by genre"):
            try:
                mid_int = int(mid)
                movie_info = movies_df[movies_df['movieId'] == mid_int]
                if not movie_info.empty:
                    movie_genres = str(movie_info['genres'].values[0]).split('|')
                    if any(g in movie_genres for g in selected_genres):
                        filtered_movies.append(mid)
            except:
                continue
        
        if len(filtered_movies) > 0:
            all_movies = filtered_movies
            print(f"Setelah filter genre: {len(all_movies)} film")
    
    # Batasi jumlah kandidat jika perlu
    if len(all_movies) > max_candidates:
        print(f"Membatasi kandidat dari {len(all_movies)} ke {max_candidates}")
        np.random.seed(42)
        all_movies = np.random.choice(all_movies, max_candidates, replace=False).tolist()
    
    # ====================== PISAHKAN FILM BERDASARKAN GENRE FAVORIT ======================
    movies_with_fav_genre = []
    movies_without_fav_genre = []
    
    if fav_genres and len(fav_genres) > 0:
        print(f"Memproses film dengan genre favorit: {fav_genres}")
        for mid in tqdm(all_movies, desc="Processing movies by favorite genre"):
            try:
                mid_int = int(mid)
                movie_info = movies_df[movies_df['movieId'] == mid_int]
                if not movie_info.empty:
                    movie_genres = str(movie_info['genres'].values[0]).split('|')
                    matches = sum(1 for g in movie_genres if g in fav_genres)
                    
                    if matches > 0:
                        movies_with_fav_genre.append((mid, matches, movie_genres))
                    else:
                        movies_without_fav_genre.append((mid, movie_genres))
            except:
                movies_without_fav_genre.append((mid, []))
        
        movies_with_fav_genre.sort(key=lambda x: -x[1])
        print(f"Film dengan genre favorit: {len(movies_with_fav_genre)}")
        print(f"Film tanpa genre favorit: {len(movies_without_fav_genre)}")
    else:
        print("Tidak ada genre favorit, memproses semua film")
        for mid in all_movies:
            movies_without_fav_genre.append((mid, []))
    
    # ====================== HITUNG PREDIKSI BASELINE (SVD) UNTUK SEMUA FILM ======================
    print(f"\n=== Menghitung prediksi SVD untuk {len(all_movies)} film ===")
    
    baseline_recommendations = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if movies_with_fav_genre:
        status_text.text("Menghitung prediksi SVD untuk film favorit...")
        for idx, (mid, matches, _) in enumerate(movies_with_fav_genre):
            try:
                svd_pred = svd_model.predict(str(user_id), str(mid)).est
            except Exception as e:
                print(f"Error predicting SVD for movie {mid}: {e}")
                svd_pred = global_mean
            baseline_recommendations.append((mid, svd_pred))
            if idx % 100 == 0:
                progress_bar.progress(min(idx / len(movies_with_fav_genre), 0.5))
    
    if movies_without_fav_genre:
        status_text.text("Menghitung prediksi SVD untuk film lainnya...")
        for idx, (mid, _) in enumerate(movies_without_fav_genre):
            try:
                svd_pred = svd_model.predict(str(user_id), str(mid)).est
            except Exception as e:
                print(f"Error predicting SVD for movie {mid}: {e}")
                svd_pred = global_mean
            baseline_recommendations.append((mid, svd_pred))
            if idx % 100 == 0:
                progress_bar.progress(0.5 + min(idx / len(movies_without_fav_genre), 0.5))
    
    progress_bar.progress(1.0)
    status_text.empty()
    
    # Urutkan baseline recommendations berdasarkan rating
    baseline_recommendations.sort(key=lambda x: (-float(x[1]), str(x[0])))
    
    # ========== BUAT DICTIONARY UNTUK LOOKUP CEPAT ==========
    baseline_dict = {str(mid): rating for mid, rating in baseline_recommendations}
    baseline_rank_dict = {str(mid): i+1 for i, (mid, _) in enumerate(baseline_recommendations)}
    
    print(f"Selesai menghitung {len(baseline_recommendations)} prediksi SVD")
    print(f"Top 5 baseline ratings:")
    for i, (mid, rating) in enumerate(baseline_recommendations[:5]):
        print(f"  Rank {i+1}: Movie {mid}, Rating {rating:.3f}")
    
    # Simpan dictionary ke session_state untuk digunakan di display function
    st.session_state[f'baseline_dict_{user_id}'] = baseline_dict
    st.session_state[f'baseline_rank_dict_{user_id}'] = baseline_rank_dict
    
    # ====================== HITUNG BOBOT ADAPTIF ======================
    if user_adaptive_weights is None:
        print("\n=== Menghitung bobot adaptif untuk user ===")
        try:
            u_inner = trainset.to_inner_uid(str(user_id))
            n_ratings = len(trainset.ur[u_inner])
            ratings_values = [r for (_, r) in trainset.ur[u_inner]]
            user_mean = np.mean(ratings_values) if ratings_values else None
            user_std = np.std(ratings_values) if ratings_values else None
            print(f"User ratings: n={n_ratings}, mean={user_mean:.3f}, std={user_std:.3f}")
        except:
            n_ratings = 0
            user_mean = None
            user_std = None
            print("User tidak ditemukan di trainset (cold start)")
        
        genre_diversity = calculate_genre_diversity(user_id, trainset, movies_df, genre_cols) if not is_new_user else 0.5
        rating_entropy = calculate_rating_entropy(user_id, trainset) if not is_new_user else 0.5
        print(f"Genre diversity: {genre_diversity:.3f}, Rating entropy: {rating_entropy:.3f}")
        
        # Rule-based adaptive weight
        if n_ratings < 5:
            cf_weight = 0.20
            method_used = "rule_cold_start_rating_lt_5"
        elif n_ratings < 10:
            cf_weight = 0.30
            method_used = "rule_based"
        elif n_ratings < 20:
            cf_weight = 0.35
            method_used = "rule_based"
        elif n_ratings < 50:
            cf_weight = 0.40
            method_used = "rule_based"
        elif n_ratings < 100:
            cf_weight = 0.45
            method_used = "rule_based"
        else:
            cf_weight = 0.50
            method_used = "rule_based"
        
        if user_std is not None and user_std > 1.2:
            cf_weight = max(0.35, cf_weight * 0.95)
            print(f"Adjusting weight due to high rating std: {cf_weight:.3f}")
        
        if genre_diversity is not None and genre_diversity > 0.7:
            cf_weight = min(0.75, cf_weight * 1.05)
            print(f"Adjusting weight due to high genre diversity: {cf_weight:.3f}")
        
        cf_weight = np.clip(cf_weight, 0.20, 0.75)
        print(f"[Rule-based] User {user_id}: n_ratings={n_ratings}, alpha={cf_weight:.4f}")
        
        user_embedding_norm = 0.5
        if not is_new_user:
            try:
                u_inner = trainset.to_inner_uid(str(user_id))
                if hasattr(svd_model, 'pu') and u_inner < len(svd_model.pu):
                    user_vec = svd_model.pu[u_inner]
                    if user_vec is not None:
                        user_embedding_norm = min(1.0, np.linalg.norm(user_vec) / 5.0)
            except:
                pass
        
        gmm_confidence = 0.5
        if not is_new_user and hybrid_model is not None:
            try:
                u_inner = trainset.to_inner_uid(str(user_id))
                if hasattr(hybrid_model, 'user_pref') and u_inner < len(hybrid_model.user_pref):
                    user_vec = hybrid_model.user_pref[u_inner]
                    if user_vec is not None:
                        p = np.abs(user_vec)
                        p_sum = np.sum(p)
                        if p_sum > 0:
                            p = p / p_sum
                        p_safe = p + 1e-10
                        entropy = -np.sum(p_safe * np.log(p_safe))
                        max_entropy = np.log(len(p))
                        gmm_confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
            except:
                pass
        
        user_adaptive_weights = {
            'cf_weight': cf_weight,
            'cb_weight': 1.0 - cf_weight,
            'n_ratings': n_ratings,
            'factors': {
                'quantity': np.log1p(n_ratings) / np.log1p(100) if n_ratings > 0 else 0,
                'quality': user_std / 2.25 if user_std else 0.5,
                'embedding': user_embedding_norm,
                'gmm': gmm_confidence,
                'diversity': genre_diversity if genre_diversity else 0.5,
                'entropy': rating_entropy if rating_entropy else 0.5,
            },
            'method': method_used,
            'is_cold_start': is_new_user or (n_ratings < 5)
        }
    
    cf_weight = user_adaptive_weights['cf_weight']
    print(f"\n[FINAL] User {user_id}: cf_weight={cf_weight:.4f}, cb_weight={1-cf_weight:.4f}")
    
    # ====================== HITUNG PREDIKSI HYBRID ======================
    print("\n=== Menghitung prediksi hybrid ===")
    
    hybrid_recommendations = []
    progress_bar = st.progress(0)
    status_text = st.text("Menghitung prediksi hybrid...")
    
    total_items = len(movies_with_fav_genre) + len(movies_without_fav_genre)
    processed = 0
    
    # Hybrid prediction untuk film dengan genre favorit
    for mid, matches, movie_genres in movies_with_fav_genre:
        # Gunakan dictionary untuk lookup SVD prediction
        svd_pred = baseline_dict.get(str(mid))
        if svd_pred is None:
            try:
                svd_pred = svd_model.predict(str(user_id), str(mid)).est
            except:
                svd_pred = global_mean
        
        # Content-based prediction
        content_pred = global_mean
        try:
            iid_str = str(mid)
            if iid_str in gmm_map:
                item_vector = gmm_map[iid_str]
                try:
                    u_inner = trainset.to_inner_uid(str(user_id))
                    if hasattr(hybrid_model, 'user_pref') and u_inner < len(hybrid_model.user_pref):
                        user_vector = hybrid_model.user_pref[u_inner]
                        if user_vector is not None and item_vector is not None:
                            sim = hybrid_model.cosine_similarity(user_vector, item_vector)
                            if not np.isnan(sim):
                                sim = (sim + 1.0) / 2.0
                                try:
                                    user_ratings = [r for (_, r) in trainset.ur[u_inner]]
                                    user_mean = np.mean(user_ratings) if user_ratings else global_mean
                                except:
                                    user_mean = global_mean
                                content_pred = user_mean + (sim - 0.5) * 2.0
                                content_pred = np.clip(content_pred, 0.5, 5.0)
                except:
                    pass
        except:
            pass
        
        
        if is_new_user or n_user_ratings < 5:
            # Cek apakah user punya rating negatif
            user_ratings_negatif = st.session_state.get(f'user_negatif_{user_id}', [])
            
            if user_ratings_negatif:
                # User punya rating negatif, gunakan content-based dari hybrid model
                try:
                    iid_str = str(mid)
                    if iid_str in gmm_map:
                        item_vector = gmm_map[iid_str]
                        u_inner = trainset.to_inner_uid(str(user_id))
                        if hasattr(hybrid_model, 'user_pref') and u_inner < len(hybrid_model.user_pref):
                            user_vector = hybrid_model.user_pref[u_inner]
                            if user_vector is not None and item_vector is not None:
                                sim = hybrid_model.cosine_similarity(user_vector, item_vector)
                                if not np.isnan(sim):
                                    sim = (sim + 1.0) / 2.0
                                    try:
                                        user_ratings = [r for (_, r) in trainset.ur[u_inner]]
                                        user_mean = np.mean(user_ratings) if user_ratings else global_mean
                                    except:
                                        user_mean = global_mean
                                    content_pred = user_mean + (sim - 0.5) * 2.0
                                    content_pred = np.clip(content_pred, 0.5, 5.0)
                                    
                                    # TAMBAHKAN PENALTI EKSTRA untuk film dengan genre yang dibenci
                                    try:
                                        mid_int = int(mid)
                                        movie_info = movies_df[movies_df['movieId'] == mid_int]
                                        if not movie_info.empty:
                                            movie_genres_list = str(movie_info['genres'].values[0]).split('|')
                                            for hated_mid in user_ratings_negatif:
                                                hated_info = movies_df[movies_df['movieId'] == int(hated_mid)]
                                                if not hated_info.empty:
                                                    hated_genres = str(hated_info['genres'].values[0]).split('|')
                                                    if any(g in movie_genres_list for g in hated_genres):
                                                        content_pred = max(1.0, content_pred - 1.0)
                                                        break
                                    except:
                                        pass
                except:
                    content_pred = global_mean
            else:
                # Tidak ada rating negatif, pakai cold start prediction biasa
                content_pred = predict_cold_start_rating(mid, movies_df, ratings_df, fav_genres, global_mean)

        hybrid_pred = cf_weight * svd_pred + (1 - cf_weight) * content_pred
        
        if fav_genres and len(fav_genres) > 0 and matches > 0:
            boost_factor = 1.0 + (matches * 0.03)
            boost_factor = min(boost_factor, 1.15)
            hybrid_pred = hybrid_pred * boost_factor

        # ========== TAMBAHKAN INI UNTUK PENALTI RATING NEGATIF ==========
        # Cek apakah user punya rating negatif
        user_ratings_negatif = st.session_state.get(f'user_negatif_{user_id}', [])
        if user_ratings_negatif:
            try:
                mid_int = int(mid)
                movie_info = movies_df[movies_df['movieId'] == mid_int]
                if not movie_info.empty:
                    movie_genres = str(movie_info['genres'].values[0]).split('|')
                    # Cek apakah film ini genre-nya sama dengan yang dibenci
                    for hated_mid in user_ratings_negatif:
                        hated_info = movies_df[movies_df['movieId'] == int(hated_mid)]
                        if not hated_info.empty:
                            hated_genres = str(hated_info['genres'].values[0]).split('|')
                            if any(g in movie_genres for g in hated_genres):
                                hybrid_pred = max(1.0, hybrid_pred - 0.8)
                                break
            except:
                pass
        # ================================================================
        
        hybrid_pred = np.clip(hybrid_pred, 0.5, 5.0)
        hybrid_recommendations.append((mid, hybrid_pred))
        
        processed += 1
        if processed % 100 == 0:
            progress_bar.progress(processed / total_items)

        if is_new_user or n_user_ratings < 5:
            if fav_genres and len(fav_genres) > 0:
                # Hitung ulang matches untuk memastikan
                try:
                    mid_int = int(mid)
                    movie_info = movies_df[movies_df['movieId'] == mid_int]
                    if not movie_info.empty:
                        movie_genres = str(movie_info['genres'].values[0]).split('|')
                        matches = sum(1 for g in movie_genres if g in fav_genres)
                        
                        if matches == 0:
                            # PENALTI EKSTRA untuk film tanpa genre favorit
                            penalty = 1
                            original_pred = hybrid_pred
                            hybrid_pred = hybrid_pred - penalty
                            hybrid_pred = max(1.0, hybrid_pred)  # Minimal rating 1.0
                            print(f"[PENALTY] Movie {mid}: {original_pred:.3f} -> {hybrid_pred:.3f}")
                except:
                    pass
    
    # Hybrid prediction untuk film tanpa genre favorit
    for mid, movie_genres in movies_without_fav_genre:
        svd_pred = baseline_dict.get(str(mid))
        if svd_pred is None:
            try:
                svd_pred = svd_model.predict(str(user_id), str(mid)).est
            except:
                svd_pred = global_mean
        
        content_pred = global_mean
        try:
            iid_str = str(mid)
            if iid_str in gmm_map:
                item_vector = gmm_map[iid_str]
                try:
                    u_inner = trainset.to_inner_uid(str(user_id))
                    if hasattr(hybrid_model, 'user_pref') and u_inner < len(hybrid_model.user_pref):
                        user_vector = hybrid_model.user_pref[u_inner]
                        if user_vector is not None and item_vector is not None:
                            sim = hybrid_model.cosine_similarity(user_vector, item_vector)
                            if not np.isnan(sim):
                                sim = (sim + 1.0) / 2.0
                                try:
                                    user_ratings = [r for (_, r) in trainset.ur[u_inner]]
                                    user_mean = np.mean(user_ratings) if user_ratings else global_mean
                                except:
                                    user_mean = global_mean
                                content_pred = user_mean + (sim - 0.5) * 2.0
                                content_pred = np.clip(content_pred, 0.5, 5.0)
                except:
                    pass
        except:
            pass
        
        if is_new_user or n_user_ratings < 5:
            content_pred = predict_cold_start_rating(mid, movies_df, ratings_df, fav_genres, global_mean)
        
        hybrid_pred = cf_weight * svd_pred + (1 - cf_weight) * content_pred
        hybrid_pred = np.clip(hybrid_pred, 0.5, 5.0)
        hybrid_recommendations.append((mid, hybrid_pred))
        
        processed += 1
        if processed % 100 == 0:
            progress_bar.progress(processed / total_items)
    
    progress_bar.progress(1.0)
    status_text.empty()
    
    hybrid_recommendations.sort(key=lambda x: (-float(x[1]), str(x[0])))
    
    hated_genres_key = f'hated_genres_{user_id}'
    hated_genres = st.session_state.get(hated_genres_key, [])
    
    if hated_genres:
        print(f"\n[PENALTY] Memberikan penalti untuk genre yang dibenci: {hated_genres}")
        penalized_count = 0
        
        # Buat list baru dengan rating yang sudah dipenalti
        penalized_recs = []
        for mid, score in hybrid_recommendations:
            try:
                mid_int = int(mid)
                movie_info = movies_df[movies_df['movieId'] == mid_int]
                if not movie_info.empty:
                    movie_genres = str(movie_info['genres'].values[0]).split('|')
                    
                    # Hitung berapa genre yang dibenci ada di film ini
                    hated_count = sum(1 for g in movie_genres if g in hated_genres)
                    
                    if hated_count > 0:
                        # PENALTI: turunkan rating berdasarkan jumlah genre yang dibenci
                        # Rating 1 = sangat dibenci (penalti besar)
                        # Rating 2 = agak dibenci (penalti sedang)
                        
                        # Cek rating asli user untuk genre ini
                        original_score = score
                        
                        # Penalti: kurangi 1.5 poin per genre yang dibenci (maksimal turun ke 1.0)
                        penalty = min(1.5 * hated_count, 3.5)
                        new_score = max(1.0, original_score - penalty)
                        
                        # Tambahkan penalti tambahan jika rating asli user untuk genre ini adalah 1
                        # (ini sudah ditangani oleh hated_genres)
                        
                        penalized_recs.append((mid, new_score))
                        penalized_count += 1
                        print(f"  [PENALTY] Movie {mid}: {original_score:.3f} -> {new_score:.3f} (hated genres: {hated_count})")
                    else:
                        penalized_recs.append((mid, score))
                else:
                    penalized_recs.append((mid, score))
            except:
                penalized_recs.append((mid, score))
        
        # Urutkan ulang setelah penalti
        penalized_recs.sort(key=lambda x: (-float(x[1]), str(x[0])))
        hybrid_recommendations = penalized_recs
        
        print(f"[PENALTY] Selesai: {penalized_count} film terkena penalti")
    # =========================================================================
    
    # Return results
    if top_n is not None:
        return hybrid_recommendations[:top_n], baseline_recommendations[:top_n], user_adaptive_weights
    else:
        return hybrid_recommendations, baseline_recommendations, user_adaptive_weights



    print(f"Selesai menghitung {len(hybrid_recommendations)} prediksi hybrid")
    print(f"Top 5 hybrid ratings:")
    for i, (mid, rating) in enumerate(hybrid_recommendations[:5]):
        print(f"  Rank {i+1}: Movie {mid}, Rating {rating:.3f}")
    
    # Return results
    if top_n is not None:
        print(f"\nReturning top {top_n} recommendations")
        return hybrid_recommendations[:top_n], baseline_recommendations[:top_n], user_adaptive_weights
    else:
        print(f"\nReturning all {len(hybrid_recommendations)} recommendations")
        return hybrid_recommendations, baseline_recommendations, user_adaptive_weights
    
def display_user_info(user_id, trainset, registered_data, movies_df, ratings_df):
    """
    Menampilkan informasi lengkap user dalam bentuk yang rapi.
    """
    st.markdown("### Informasi User")
    
    # Cek apakah user di trainset atau registered
    is_registered = registered_data is not None
    is_in_trainset = False
    
    if trainset is not None:
        try:
            u_inner = trainset.to_inner_uid(str(user_id))
            is_in_trainset = True
        except:
            pass
    
    # Container utama
    info_container = st.container()
    
    with info_container:
        # Row 1: Basic Info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("User ID", str(user_id))
        
        with col2:
            if is_in_trainset:
                n_ratings = len(trainset.ur[u_inner])
                st.metric("Jumlah Rating", f"{n_ratings:,}")
            elif is_registered:
                n_ratings = registered_data.get('n_ratings', 0)
                st.metric("Jumlah Rating", f"{n_ratings:,}")
            else:
                st.metric("Jumlah Rating", "0")
        
        with col3:
            if is_in_trainset and trainset is not None:
                ratings_values = [r for (_, r) in trainset.ur[u_inner]]
                if ratings_values:
                    avg_rating = np.mean(ratings_values)
                    st.metric("Rata-rata Rating", f"{avg_rating:.2f}")
                else:
                    st.metric("Rata-rata Rating", "N/A")
            elif is_registered and registered_data.get('initial_ratings'):
                ratings_values = list(registered_data['initial_ratings'].values())
                if ratings_values:
                    avg_rating = np.mean(ratings_values)
                    st.metric("Rata-rata Rating", f"{avg_rating:.2f}")
                else:
                    st.metric("Rata-rata Rating", "N/A")
            else:
                st.metric("Rata-rata Rating", "N/A")
        
        with col4:
            if is_in_trainset and trainset is not None:
                ratings_values = [r for (_, r) in trainset.ur[u_inner]]
                if len(ratings_values) > 1:
                    std_rating = np.std(ratings_values)
                    st.metric("Std Dev Rating", f"{std_rating:.3f}")
                else:
                    st.metric("Std Dev Rating", "N/A")
            elif is_registered and registered_data.get('initial_ratings'):
                ratings_values = list(registered_data['initial_ratings'].values())
                if len(ratings_values) > 1:
                    std_rating = np.std(ratings_values)
                    st.metric("Std Dev Rating", f"{std_rating:.3f}")
                else:
                    st.metric("Std Dev Rating", "N/A")
            else:
                st.metric("Std Dev Rating", "N/A")
        
        st.markdown("---")
        
        # Row 2: Genre Favorit (jika ada)
        fav_genres = []
        
        # Ambil dari registered_data
        if is_registered:
            fav_genres = registered_data.get('fav_genres', [])
        
        # Jika tidak ada, coba dari session_state
        if not fav_genres and st.session_state.get('current_fav_genres'):
            fav_genres = st.session_state.get('current_fav_genres', [])
        
        if fav_genres:
            st.markdown("#### Genre Favorit")
            cols = st.columns(min(len(fav_genres), 5))
            for idx, genre in enumerate(fav_genres[:10]):
                with cols[idx % 5]:
                    st.markdown(f"- **{genre}**")
            
            if len(fav_genres) > 10:
                st.caption(f"... dan {len(fav_genres) - 10} genre lainnya")
        else:
            st.info("Belum ada genre favorit yang dipilih")
        
        # Row 3: Adaptive Weight (jika ada)
        adaptive_weights = None
        
        # Ambil dari registered_data
        if is_registered and 'adaptive_weights' in registered_data:
            adaptive_weights = registered_data['adaptive_weights']
        elif st.session_state.get('current_adaptive_weights'):
            adaptive_weights = st.session_state.get('current_adaptive_weights')
        
        if adaptive_weights:
            st.markdown("#### Bobot Adaptif")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cf_weight = adaptive_weights.get('cf_weight', 0.5)
                st.metric("Bobot CF (SVD)", f"{cf_weight:.3f}")
            
            with col2:
                cb_weight = adaptive_weights.get('cb_weight', 0.5)
                st.metric("Bobot CB (GMM)", f"{cb_weight:.3f}")
            
            with col3:
                method = adaptive_weights.get('method', 'unknown')
                method_display = {
                    'rule_cold_start_rating_lt_5': 'Cold Start (rating < 5)',
                    'rule_based': 'Rule-based',
                    'fallback': 'Fallback'
                }.get(method, method)
                st.metric("Metode", method_display)
            
            if adaptive_weights.get('is_cold_start', False):
                st.warning("Cold Start Mode - User memiliki rating yang sedikit")
    
    return is_in_trainset, is_registered

def display_user_ratings_table(user_id, trainset, registered_data, movies_df, ratings_df, top_n=20):
    """
    Menampilkan tabel rating user dengan rapi.
    """
    st.markdown("### Daftar Rating User")
    
    ratings_list = []
    
    # Ambil rating dari trainset
    if trainset is not None:
        try:
            u_inner = trainset.to_inner_uid(str(user_id))
            for iid_inner, rating in trainset.ur[u_inner]:
                raw_iid = trainset.to_raw_iid(iid_inner)
                
                # Ambil informasi film
                try:
                    movie_id_int = int(raw_iid)
                    movie_info = movies_df[movies_df['movieId'] == movie_id_int]
                    if not movie_info.empty:
                        title = movie_info['title'].values[0]
                        genres = movie_info['genres'].values[0]
                    else:
                        title = f"Movie {raw_iid}"
                        genres = "Unknown"
                except:
                    title = f"Movie {raw_iid}"
                    genres = "Unknown"
                
                ratings_list.append({
                    'No': len(ratings_list) + 1,
                    'Movie ID': raw_iid,
                    'Judul Film': title[:60] + '...' if len(title) > 60 else title,
                    'Genre': genres[:40] + '...' if len(genres) > 40 else genres,
                    'Rating': rating
                })
        except:
            pass
    
    # Ambil rating dari registered users
    if registered_data and 'initial_ratings' in registered_data:
        for movie_id, rating in registered_data['initial_ratings'].items():
            # Cek apakah sudah ada di list
            if not any(r['Movie ID'] == str(movie_id) for r in ratings_list):
                try:
                    movie_info = movies_df[movies_df['movieId'] == int(movie_id)]
                    if not movie_info.empty:
                        title = movie_info['title'].values[0]
                        genres = movie_info['genres'].values[0]
                    else:
                        title = f"Movie {movie_id}"
                        genres = "Unknown"
                except:
                    title = f"Movie {movie_id}"
                    genres = "Unknown"
                
                ratings_list.append({
                    'No': len(ratings_list) + 1,
                    'Movie ID': str(movie_id),
                    'Judul Film': title[:60] + '...' if len(title) > 60 else title,
                    'Genre': genres[:40] + '...' if len(genres) > 40 else genres,
                    'Rating': rating
                })
    
    if ratings_list:
        # Urutkan berdasarkan rating tertinggi
        ratings_list.sort(key=lambda x: (-x['Rating'], x['Judul Film']))
        
        # Update nomor urut
        for idx, item in enumerate(ratings_list):
            item['No'] = idx + 1
        
        ratings_df_user = pd.DataFrame(ratings_list)
        
        # Tampilkan statistik ringkas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rating", f"{len(ratings_list)}")
        with col2:
            avg_rating = np.mean([r['Rating'] for r in ratings_list])
            st.metric("Rata-rata Rating", f"{avg_rating:.2f}")
        with col3:
            high_rating = sum(1 for r in ratings_list if r['Rating'] >= 4.0)
            low_rating = sum(1 for r in ratings_list if r['Rating'] <= 2.0)
            st.metric("Rating Tinggi (>=4)", f"{high_rating}", delta=f"Rendah: {low_rating}")
        
        # Tampilkan tabel
        if len(ratings_list) > top_n:
            st.info(f"Menampilkan {top_n} dari {len(ratings_list)} rating (diurutkan berdasarkan rating tertinggi)")
            display_df = ratings_df_user.head(top_n)
        else:
            display_df = ratings_df_user
        
        # Style untuk tabel
        def color_rating(val):
            if isinstance(val, (int, float)):
                if val >= 4.5:
                    return 'background-color: #90EE90; font-weight: bold'
                elif val >= 4.0:
                    return 'background-color: #C8E6C9'
                elif val >= 3.0:
                    return 'background-color: #FFF9C4'
                elif val >= 2.0:
                    return 'background-color: #FFCCBC'
                else:
                    return 'background-color: #FFCDD2; font-weight: bold'
            return ''
        
        styled_df = display_df.style.applymap(color_rating, subset=['Rating'])
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Download button
        csv = ratings_df_user.to_csv(index=False)
        st.download_button(
            label="Download Daftar Rating (CSV)",
            data=csv,
            file_name=f"user_{user_id}_ratings.csv",
            mime="text/csv",
            key=f"download_ratings_{user_id}"
        )
    else:
        st.info("User belum memiliki rating")
        
        # Tampilkan form untuk menambah rating
        with st.expander("Tambah Rating untuk User Ini", expanded=False):
            st.info("Rating yang ditambahkan akan digunakan untuk personalisasi rekomendasi")
            
            search_query = st.text_input("Cari film:", key=f"search_{user_id}", placeholder="Judul atau genre...")
            
            if search_query:
                results = search_movies(movies_df, search_query, top_n=10)
            else:
                results = movies_df.head(10)
            
            for _, row in results.iterrows():
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{row['title']}**")
                    st.caption(f"Genre: {row['genres']}")
                with col2:
                    rating_val = st.slider("Rating", 0.5, 5.0, 3.0, 0.5, 
                                         key=f"rate_{user_id}_{row['movieId']}", 
                                         label_visibility="collapsed")
                with col3:
                    if st.button("Tambah", key=f"add_{user_id}_{row['movieId']}"):
                        # Simpan rating
                        if 'temp_user_ratings' not in st.session_state:
                            st.session_state.temp_user_ratings = {}
                        if str(user_id) not in st.session_state.temp_user_ratings:
                            st.session_state.temp_user_ratings[str(user_id)] = {}
                        st.session_state.temp_user_ratings[str(user_id)][row['movieId']] = rating_val
                        st.success(f"Rating {rating_val} untuk {row['title']} ditambahkan!")
                        st.rerun()
                st.markdown("---")
            
            if st.session_state.get('temp_user_ratings', {}).get(str(user_id), {}):
                st.markdown("#### Rating yang akan ditambahkan:")
                temp_ratings = st.session_state.temp_user_ratings[str(user_id)]
                for mid, rating in temp_ratings.items():
                    movie_info = movies_df[movies_df['movieId'] == int(mid)]
                    title = movie_info['title'].values[0] if not movie_info.empty else f"Movie {mid}"
                    st.write(f"- {title}: {rating}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Simpan ke User", key=f"save_{user_id}"):
                        try:
                            # Proses integrasi rating baru
                            result = process_new_user_flow_with_adaptive_weights(
                                st.session_state.temp_user_ratings[str(user_id)],
                                movies_df,
                                st.session_state.hybrid,
                                st.session_state.trainset,
                                st.session_state.svd,
                                ratings_df,
                                st.session_state.genre_cols,
                                cluster_genre_mapping=st.session_state.cluster_genre_mapping,
                                show_tables=False
                            )
                            st.success(f"Rating berhasil ditambahkan! User ID: {result['user_id']}")
                            del st.session_state.temp_user_ratings[str(user_id)]
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                with col2:
                    if st.button("Batal", key=f"cancel_{user_id}"):
                        del st.session_state.temp_user_ratings[str(user_id)]
                        st.rerun()

def predict_cold_start_rating(movie_id, movies_df, ratings_df, fav_genres, global_mean=3.0):
    """
    Prediksi rating untuk user cold-start berdasarkan genre favorit.
    DIPERBAIKI: Film tanpa match genre favorit mendapat PENALTI.
    """
    try:
        mid_int = int(movie_id)
        movie_info = movies_df[movies_df['movieId'] == mid_int]
        if movie_info.empty:
            return global_mean
        
        movie_genres = str(movie_info['genres'].values[0]).split('|')
        
        if fav_genres and len(fav_genres) > 0:
            matches = sum(1 for g in movie_genres if g in fav_genres)
            
            if matches > 0:
                # Film dengan genre favorit: rating tinggi
                match_ratio = matches / len(fav_genres)
                rating = 4.0 + (match_ratio * 1.0)
                rating = min(5.0, rating)
                print(f"[MATCH] Movie {movie_id}: matches={matches}, rating={rating:.3f}")
                return rating
            else:
                # PERBAIKAN: Film TANPA match genre favorit mendapat PENALTI
                # Gunakan rating rata-rata film tapi dikurangi penalti
                movie_ratings = ratings_df[ratings_df['movieId'] == mid_int]['rating']
                if len(movie_ratings) > 0:
                    base_rating = movie_ratings.mean()
                else:
                    base_rating = global_mean
                
                # Penalti 0.5 - 1.0 poin tergantung seberapa jauh dari genre favorit
                penalty = 0.75  # Penalti tetap
                rating = max(2.0, base_rating - penalty)
                rating = min(3.5, rating)  # Batasi maksimal 3.5 untuk film tanpa match
                
                print(f"[NO MATCH] Movie {movie_id}: base={base_rating:.3f}, penalty={penalty}, rating={rating:.3f}")
                return rating
        
        # Jika tidak ada genre favorit yang dipilih
        movie_ratings = ratings_df[ratings_df['movieId'] == mid_int]['rating']
        if len(movie_ratings) > 0:
            return max(2.5, movie_ratings.mean())
        return max(2.5, global_mean)
        
    except Exception as e:
        print(f"Error predict_cold_start_rating for {movie_id}: {e}")
        return global_mean

# ====================== FUNGSI CROSS-VALIDATION DENGAN SAVE OUTPUT ======================
def run_hybrid_cross_validation_with_save(ratings_df, movies_df, output_key, 
                                          n_folds=5, test_size=0.2, random_state=42,
                                          svd_params=None, gmm_params=None,
                                          show_tables=True):
    
    np.random.seed(random_state)
    
    movies_with_ratings = set(ratings_df['movieId'].unique())
    valid_movies = list(movies_with_ratings)
    np.random.shuffle(valid_movies)
    
    movies_per_fold = len(valid_movies) // n_folds
    movie_folds = []
    
    for i in range(n_folds):
        start_idx = i * movies_per_fold
        end_idx = (i + 1) * movies_per_fold if i < n_folds - 1 else len(valid_movies)
        fold_movies = set(valid_movies[start_idx:end_idx])
        movie_folds.append(fold_movies)
    
    fold_results = []
    
    for fold in range(n_folds):
        test_movie_ids = movie_folds[fold]
        train_movie_ids = set()
        for i in range(len(movie_folds)):
            if i != fold:
                train_movie_ids.update(movie_folds[i])
        
        fold_train_df = ratings_df[ratings_df['movieId'].isin(train_movie_ids)].copy()
        fold_test_df = ratings_df[ratings_df['movieId'].isin(test_movie_ids)].copy()
        
        fold_movies_enc, feature_cols, genre_cols = one_hot_encode_genres_with_save(
            movies_df, train_movie_ids, f"cv_fold_{fold}", 
            smoothing_factor=gmm_params['smoothing'], show_tables=False
        )
        
        gmm_map, gmm_model = gmm_soft_clustering_with_save(
            fold_movies_enc, feature_cols, f"cv_fold_{fold}_gmm",
            max_components=gmm_params['max_components'], 
            random_state=random_state + fold, show_tables=False
        )
        
        test_gmm_map = transform_test_movies_with_gmm(
            movies_df, test_movie_ids, gmm_model, feature_cols, 
            smoothing_factor=gmm_params['smoothing']
        )
        full_gmm_map = {**gmm_map, **test_gmm_map}
        
        reader = Reader(rating_scale=(0.5, 5.0))
        fold_train_df['userId'] = fold_train_df['userId'].astype(str)
        fold_train_df['movieId'] = fold_train_df['movieId'].astype(str)
        data = Dataset.load_from_df(fold_train_df[['userId', 'movieId', 'rating']], reader)
        trainset_cv = data.build_full_trainset()
        
        testset_cv = []
        fold_test_df['userId'] = fold_test_df['userId'].astype(str)
        fold_test_df['movieId'] = fold_test_df['movieId'].astype(str)
        for _, row in fold_test_df.iterrows():
            testset_cv.append((str(row['userId']), str(row['movieId']), float(row['rating'])))
        
        svd_cv = SVD(
            n_factors=svd_params['n_factors'], 
            n_epochs=svd_params['n_epochs'],
            lr_all=svd_params['lr_all'], 
            reg_all=svd_params['reg_all'], 
            random_state=random_state + fold, 
            verbose=False
        )
        svd_cv.fit(trainset_cv)
        
        hybrid_cv = HybridGMM_SVD(svd_cv, full_gmm_map)
        hybrid_cv.fit(trainset_cv, show_tables=False)
        
        svd_errors = []
        hybrid_errors = []
        gmm_used_count = 0
        
        for uid, iid, true_r in testset_cv:
            svd_pred = svd_cv.predict(uid, iid).est
            svd_errors.append(abs(svd_pred - true_r))
            
            try:
                u_inner = trainset_cv.to_inner_uid(str(uid))
                n_ratings = len(trainset_cv.ur[u_inner])
                
                if n_ratings < 5:
                    alpha = 0.20
                elif n_ratings < 10:
                    alpha = 0.30
                elif n_ratings < 20:
                    alpha = 0.35
                elif n_ratings < 50:
                    alpha = 0.40
                elif n_ratings < 100:
                    alpha = 0.45
                else:
                    alpha = 0.50
            except:
                alpha = 0.50
            
            hybrid_pred, used_gmm = hybrid_cv.predict(uid, iid, alpha=alpha)
            hybrid_errors.append(abs(hybrid_pred - true_r))
            if used_gmm:
                gmm_used_count += 1
        
        svd_rmse = np.sqrt(np.mean(np.array(svd_errors)**2))
        svd_mae = np.mean(svd_errors)  
        hybrid_rmse = np.sqrt(np.mean(np.array(hybrid_errors)**2))
        hybrid_mae = np.mean(hybrid_errors) 
        
        fold_results.append({
            'fold': fold + 1,
            'svd_rmse': svd_rmse,
            'svd_mae': svd_mae,  
            'hybrid_rmse': hybrid_rmse,
            'hybrid_mae': hybrid_mae,
            'gmm_used_pct': (gmm_used_count / len(testset_cv)) * 100 if len(testset_cv) > 0 else 0
        })
    
    # Di akhir fungsi run_hybrid_cross_validation_with_save
    results_df = pd.DataFrame(fold_results)
    svd_rmse_mean = results_df['svd_rmse'].mean()
    svd_rmse_std = results_df['svd_rmse'].std()
    hybrid_rmse_mean = results_df['hybrid_rmse'].mean()
    hybrid_rmse_std = results_df['hybrid_rmse'].std()
    rmse_improvement = ((svd_rmse_mean - hybrid_rmse_mean) / svd_rmse_mean * 100) if svd_rmse_mean > 0 else 0

    svd_mae_mean = results_df['svd_mae'].mean() if 'svd_mae' in results_df.columns else 0
    svd_mae_std = results_df['svd_mae'].std() if 'svd_mae' in results_df.columns else 0
    hybrid_mae_mean = results_df['hybrid_mae'].mean() if 'hybrid_mae' in results_df.columns else 0
    hybrid_mae_std = results_df['hybrid_mae'].std() if 'hybrid_mae' in results_df.columns else 0
    mae_improvement = ((svd_mae_mean - hybrid_mae_mean) / svd_mae_mean * 100) if svd_mae_mean > 0 else 0

    from scipy import stats
    t_stat_rmse, p_value_rmse = stats.ttest_rel(results_df['svd_rmse'].tolist(), results_df['hybrid_rmse'].tolist())

    # Hitung p-value untuk MAE jika kolom tersedia
    if 'svd_mae' in results_df.columns and 'hybrid_mae' in results_df.columns:
        t_stat_mae, p_value_mae = stats.ttest_rel(results_df['svd_mae'].tolist(), results_df['hybrid_mae'].tolist())
    else:
        t_stat_mae, p_value_mae = 0, 1.0

    return {
        'fold_results': results_df,
        'summary': {
            'svd_rmse_mean': svd_rmse_mean,
            'svd_rmse_std': svd_rmse_std,
            'svd_mae_mean': svd_mae_mean,
            'svd_mae_std': svd_mae_std,
            'hybrid_rmse_mean': hybrid_rmse_mean,
            'hybrid_rmse_std': hybrid_rmse_std,
            'hybrid_mae_mean': hybrid_mae_mean,
            'hybrid_mae_std': hybrid_mae_std,
            'rmse_improvement': rmse_improvement,
            'mae_improvement': mae_improvement,
            'paired_ttest_rmse': {'t_statistic': t_stat_rmse, 'p_value': p_value_rmse},
            'paired_ttest_mae': {'t_statistic': t_stat_mae, 'p_value': p_value_mae}
        }
    }

# ====================== FUNGSI PLOT HASIL CV ======================
def auto_display_cv_plots():
    """
    Menampilkan grafik CV secara otomatis tanpa tombol.
    """
    if not st.session_state.get('tab3_processed', False):
        return
    
    if st.session_state.tab3_results is None:
        return
    
    results = st.session_state.tab3_results
    
    if 'fold_results' not in results:
        return
    
    fold_results = results['fold_results']
    
    required_cols = ['fold', 'svd_rmse', 'hybrid_rmse']
    missing_cols = [col for col in required_cols if col not in fold_results.columns]
    
    if missing_cols:
        st.warning(f"Data hasil CV tidak lengkap. Kolom yang hilang: {missing_cols}")
        return
    
    st.markdown("---")
    st.subheader("Visualisasi Hasil Cross-Validation")
    
    svd_rmse = fold_results['svd_rmse'].values
    hybrid_rmse = fold_results['hybrid_rmse'].values
    rmse_improvement = ((svd_rmse - hybrid_rmse) / svd_rmse) * 100
    rmse_improvement_mean = np.mean(rmse_improvement)
    
    has_mae = 'svd_mae' in fold_results.columns and 'hybrid_mae' in fold_results.columns
    
    if has_mae:
        svd_mae = fold_results['svd_mae'].values
        hybrid_mae = fold_results['hybrid_mae'].values
        mae_improvement = ((svd_mae - hybrid_mae) / svd_mae) * 100
        mae_improvement_mean = np.mean(mae_improvement)
    else:
        mae_improvement_mean = 0
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cross-Validation Results: SVD vs Hybrid GMM+SVD', fontsize=16, fontweight='bold')
    
    folds = fold_results['fold'].values
    
    ax1 = axes[0]
    ax1.plot(folds, svd_rmse, 'b-o', linewidth=2, markersize=8, label='SVD', alpha=0.8)
    ax1.plot(folds, hybrid_rmse, 'g-s', linewidth=2, markersize=8, label='Hybrid', alpha=0.8)
    ax1.fill_between(folds, svd_rmse, hybrid_rmse, 
                      where=(svd_rmse > hybrid_rmse), color='green', alpha=0.2, 
                      label='Hybrid Better')
    ax1.fill_between(folds, hybrid_rmse, svd_rmse,
                      where=(hybrid_rmse > svd_rmse), color='red', alpha=0.2,
                      label='SVD Better')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title(f'RMSE per Fold (Avg Improvement: {rmse_improvement_mean:+.2f}%)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(folds)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10)
    
    for i, (x, y_svd, y_hyb) in enumerate(zip(folds, svd_rmse, hybrid_rmse)):
        ax1.annotate(f'{y_svd:.4f}', (x, y_svd), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
        ax1.annotate(f'{y_hyb:.4f}', (x, y_hyb), textcoords="offset points", 
                    xytext=(0,-12), ha='center', fontsize=8)
    
    ax2 = axes[1]
    if has_mae:
        ax2.plot(folds, svd_mae, 'b-o', linewidth=2, markersize=8, label='SVD', alpha=0.8)
        ax2.plot(folds, hybrid_mae, 'g-s', linewidth=2, markersize=8, label='Hybrid', alpha=0.8)
        ax2.fill_between(folds, svd_mae, hybrid_mae, 
                          where=(svd_mae > hybrid_mae), color='green', alpha=0.2,
                          label='Hybrid Better')
        ax2.fill_between(folds, hybrid_mae, svd_mae,
                          where=(hybrid_mae > svd_mae), color='red', alpha=0.2,
                          label='SVD Better')
        ax2.set_title(f'MAE per Fold (Avg Improvement: {mae_improvement_mean:+.2f}%)', 
                      fontsize=12, fontweight='bold')
        
        for i, (x, y_svd, y_hyb) in enumerate(zip(folds, svd_mae, hybrid_mae)):
            ax2.annotate(f'{y_svd:.4f}', (x, y_svd), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
            ax2.annotate(f'{y_hyb:.4f}', (x, y_hyb), textcoords="offset points", 
                        xytext=(0,-12), ha='center', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'MAE Data Not Available\n(only RMSE calculated)', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title('MAE per Fold (Data Not Available)', fontsize=12, fontweight='bold')
    
    ax2.set_xlabel('Fold', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_xticks(folds)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=10)
    
    ax3 = axes[2]
    x = np.arange(len(folds))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, rmse_improvement, width, label='RMSE Improvement', 
                    color='steelblue', edgecolor='black', alpha=0.8)
    
    if has_mae:
        bars2 = ax3.bar(x + width/2, mae_improvement, width, label='MAE Improvement',
                        color='coral', edgecolor='black', alpha=0.8)
        ax3.legend(loc='best', fontsize=10)
    else:
        ax3.legend(loc='best', fontsize=10)
    
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:+.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    if has_mae:
        for bar in bars2:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:+.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Fold', fontsize=12)
    ax3.set_ylabel('Improvement (%)', fontsize=12)
    ax3.set_title('Improvement per Fold (Positive = Hybrid Better)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(folds)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    st.pyplot(fig)
    
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # GUNAKAN COUNTER UNTUK MEMBUAT KEY UNIK
        if 'cv_download_counter' not in st.session_state:
            st.session_state.cv_download_counter = 0
        st.session_state.cv_download_counter += 1
        
        st.download_button(
            label=" Download Grafik CV (PNG)",
            data=buf,
            file_name="cv_results.png",
            mime="image/png",
            key=f"auto_download_cv_plot_{st.session_state.cv_download_counter}"
        )
    
    plt.close(fig)
    
    return fig

# ====================== HYBRID GMM + SVD MODEL KHUSUS CV ======================

class HybridGMM_SVD_CV(HybridGMM_SVD):
    """
    Hybrid Model untuk CV dengan alpha eksplisit.
    """
    
    def __init__(self, svd_model, gmm_map, trainset):
        super().__init__(svd_model, gmm_map)
        self.trainset = trainset
    
    def predict_with_alpha(self, uid, iid, alpha):
        """
        Prediksi rating dengan alpha yang diberikan (untuk CV).
        
        Parameter:
            uid: User ID
            iid: Item ID
            alpha: Bobot CF (0-1)
            
        Returns:
            tuple: (predicted_rating, gmm_used)
        """
        try:
            svd_pred = self.svd.predict(uid, iid).est
        except:
            svd_pred = self.trainset.global_mean if self.trainset else 3.0
        
        iid_str = str(iid)
        
        try:
            u = self.trainset.to_inner_uid(str(uid))
            user_found = True
        except:
            user_found = False
        
        item_vector = None
        if iid_str in self.item_vectors:
            item_vector = self.item_vectors[iid_str]
        elif iid_str in self.gmm_map:
            item_vector = self.gmm_map[iid_str]
            self.item_vectors[iid_str] = item_vector
        
        if not user_found or item_vector is None:
            return svd_pred, False
        
        sim = self.cosine_similarity(self.user_pref[u], item_vector)
        if np.isnan(sim):
            sim = 0.0
        
        sim = (sim + 1.0) / 2.0
        
        if u < len(self.trainset.ur):
            ratings = [r for (_, r) in self.trainset.ur[u]]
            user_mean = np.mean(ratings)
        else:
            user_mean = self.trainset.global_mean
        
        content_score = user_mean + (sim - 0.5) * 2.0
        content_score = np.clip(content_score, 0.5, 5.0)
        
        hybrid_pred = alpha * svd_pred + (1 - alpha) * content_score
        hybrid_pred = np.clip(hybrid_pred, 0.5, 5.0)
        
        return hybrid_pred, True

# ====================== FUNGSI FIND USER ======================

def find_user_in_trainset(trainset, user_id):
    """
    Mencari user dalam trainset.
    
    Parameter:
        trainset: Surprise trainset
        user_id: ID user yang dicari
        
    Returns:
        str: Raw user ID jika ditemukan, None jika tidak
    """
    user_str = str(user_id)
    all_users = [trainset.to_raw_uid(uid) for uid in trainset.all_users()]
    
    if user_str in all_users:
        return user_str
    
    try:
        user_int = int(user_id)
        for raw_uid in all_users:
            try:
                if int(raw_uid) == user_int:
                    return raw_uid
            except:
                continue
    except:
        pass
    
    return None

def find_user_in_registered(user_id):
    """
    Mencari user dalam registered users.
    
    Parameter:
        user_id: ID user yang dicari
        
    Returns:
        dict: Data user jika ditemukan di registered_users, None jika tidak
    """
    if not safe_has_session_state():
        return None
    
    trainset = safe_session_state_get('trainset')
    registered_users = safe_session_state_get('registered_users', {})
    
    user_str = str(user_id)
    
    if trainset is not None:
        try:
            trainset.to_inner_uid(user_str)
            return None
        except:
            pass
    
    if user_str in registered_users:
        data = registered_users[user_str].copy()
        
        if 'n_ratings' not in data:
            if 'initial_ratings' in data:
                data['n_ratings'] = len(data['initial_ratings'])
            else:
                data['n_ratings'] = 0
        
        if 'initial_ratings' in data:
            actual_ratings = len(data['initial_ratings'])
            if actual_ratings != data.get('n_ratings', 0):
                data['n_ratings'] = actual_ratings
        
        if user_str in registered_users:
            registered_users[user_str].update(data)
            safe_session_state_set('registered_users', registered_users)
        
        return data
    
    try:
        user_int = int(user_id)
        for uid, data in registered_users.items():
            try:
                if int(uid) == user_int:
                    if trainset is not None:
                        try:
                            trainset.to_inner_uid(str(uid))
                            return None
                        except:
                            pass
                    
                    data_copy = data.copy()
                    return data_copy
            except:
                continue
    except:
        pass
    
    return None

# ====================== FUNGSI FILTER GENRE ======================

def get_genre_selection(movies_df, genre_cols):
    """
    Pilihan genre untuk filter.
    
    Parameter:
        movies_df: DataFrame film
        genre_cols: Daftar genre
        
    Returns:
        list: Genre yang dipilih
    """
    st.subheader("Filter Genre")
    
    genre_counts = {}
    for genre in genre_cols:
        count = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)].shape[0]
        genre_counts[genre] = count
    
    sorted_genres = sorted(genre_counts.items(), key=lambda x: (-x[1], x[0]))
    
    selected = []
    cols = st.columns(3)
    
    for idx, (genre, count) in enumerate(sorted_genres):
        with cols[idx % 3]:
            if st.checkbox(f"{genre} ({count})", key=f"filter_{genre}"):
                selected.append(genre)
    
    if selected:
        st.success(f"{len(selected)} genre dipilih")
        movies_with_genres = movies_df[movies_df['genres'].apply(
            lambda x: any(g in str(x) for g in selected)
        )].shape[0]
        st.info(f"{movies_with_genres:,} film tersedia")
    
    return selected

def display_all_recommendations(hybrid_recs, baseline_recs, movies_df, user_id, 
                               items_per_page=20, selected_genres=None, 
                               is_new_user=False, new_user_details=None,
                               ratings_df=None, genre_cols=None,
                               adaptive_weights=None, user_fav_genres=None):
    
    if user_fav_genres is None:
        user_fav_genres = st.session_state.get('current_fav_genres', [])

    # Tampilkan header
    if adaptive_weights is not None:
        cf_weight = adaptive_weights.get('cf_weight', 0.5)
        method = adaptive_weights.get('method', 'unknown')
        if method == 'rule_cold_start_rating_lt_5':
            st.subheader(f"Rekomendasi User {user_id} | α={cf_weight:.4f} (COLD START - rating < 5)")
        else:
            st.subheader(f"Rekomendasi User {user_id} | α={cf_weight:.4f}")
    elif new_user_details and 'adaptive_weights' in new_user_details:
        cf_weight = new_user_details['adaptive_weights'].get('cf_weight', 0.5)
        st.subheader(f"Rekomendasi User Baru {user_id} | α={cf_weight:.4f}")
    else:
        st.subheader(f"Rekomendasi User {user_id}")

    if user_fav_genres:
        st.info(f"**Genre Favorit:** {', '.join(user_fav_genres)}")
    
    # ====================== GLOBAL MEAN & MOVIE AVG ======================
    global_mean = 3.0
    if ratings_df is not None and not ratings_df.empty:
        global_mean = ratings_df['rating'].mean()
    
    movie_avg_ratings = {}
    if ratings_df is not None and not ratings_df.empty:
        movie_avg = ratings_df.groupby('movieId')['rating'].mean().to_dict()
        for mid, avg in movie_avg.items():
            movie_avg_ratings[str(mid)] = avg
    
    all_movies_by_avg = sorted(movie_avg_ratings.items(), key=lambda x: -x[1])
    movie_avg_rank = {mid: i+1 for i, (mid, _) in enumerate(all_movies_by_avg)}
    
    # ====================== BUAT BASELINE RANK UNTUK SEMUA FILM ======================
    baseline_rank_all = {}
    baseline_rating_all = {}
    baseline_sorted_full = []
    
    if baseline_recs and len(baseline_recs) > 0:
        baseline_sorted_full = sorted(baseline_recs, key=lambda x: (-float(x[1]), str(x[0])))
        
        for i, (mid, rating) in enumerate(baseline_sorted_full):
            mid_str = str(mid)
            baseline_rank_all[mid_str] = i + 1
            baseline_rating_all[mid_str] = rating
        
        print(f"DEBUG: baseline_sorted_full length = {len(baseline_sorted_full)}")
    else:
        print("DEBUG: baseline_recs is None or empty, checking session_state...")
        baseline_dict = st.session_state.get(f'baseline_dict_{user_id}', {})
        baseline_rank_dict = st.session_state.get(f'baseline_rank_dict_{user_id}', {})
        
        if baseline_dict:
            print(f"DEBUG: Found baseline_dict in session_state with {len(baseline_dict)} items")
            for mid, rating in baseline_dict.items():
                baseline_rating_all[mid] = rating
                baseline_rank_all[mid] = baseline_rank_dict.get(mid, 999)
            
            sorted_items = sorted(baseline_rank_all.items(), key=lambda x: x[1])
            baseline_sorted_full = [(mid, baseline_rating_all[mid]) for mid, _ in sorted_items]
        else:
            print("DEBUG: No baseline data found, creating from hybrid as fallback")
            hybrid_temp = sorted(hybrid_recs, key=lambda x: (-float(x[1]), str(x[0])))[:200]
            baseline_sorted_full = []
            for i, (mid, _) in enumerate(hybrid_temp):
                mid_str = str(mid)
                b_rating = movie_avg_ratings.get(mid_str, global_mean)
                baseline_rating_all[mid_str] = b_rating
                baseline_rank_all[mid_str] = i + 1
                baseline_sorted_full.append((mid, b_rating))
    
    # Untuk hybrid, ambil Top 100
    hybrid_sorted = sorted(hybrid_recs, key=lambda x: (-float(x[1]), str(x[0])))[:100]
    
    # Baseline Top 100
    baseline_sorted_100 = baseline_sorted_full[:100] if baseline_sorted_full else []
    baseline_ids_100 = [str(r[0]) for r in baseline_sorted_100]
    hybrid_ids = [str(r[0]) for r in hybrid_sorted]
    
    # ====================== TABEL SVD TOP 100 ======================
    st.markdown("---")
    st.subheader(f"Top 100 Rekomendasi SVD untuk User {user_id}")
    
    svd_table_data = []
    for i, (mid, s_score) in enumerate(baseline_sorted_100):
        s_rank = i + 1
        
        try:
            mid_int = int(mid)
            movie_info = movies_df[movies_df['movieId'] == mid_int]
            if not movie_info.empty:
                film_title = movie_info['title'].values[0]
                film_genre = movie_info['genres'].values[0]
            else:
                film_title = f"Movie {mid}"
                film_genre = "Unknown"
        except:
            film_title = f"Movie {mid}"
            film_genre = "Unknown"
        
        is_fav = False
        if user_fav_genres:
            movie_genres = str(film_genre).split('|')
            is_fav = any(g in movie_genres for g in user_fav_genres)
        
        svd_table_data.append({
            'Rank': s_rank,
            'Judul Film': film_title,
            'Genre': film_genre,
            'Favorit': "Ya" if is_fav else "",
            'Prediksi SVD': f"{s_score:.3f}"
        })
    
    # Pagination SVD
    svd_pages = max(1, (len(svd_table_data) + items_per_page - 1) // items_per_page)
    svd_page_key = f'svd_page_{user_id}'
    
    if svd_page_key not in st.session_state:
        st.session_state[svd_page_key] = 1
    
    svd_current_page = st.session_state[svd_page_key]
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if svd_current_page > 1:
            if st.button("◀ Sebelumnya", key=f"svd_prev_{user_id}"):
                st.session_state[svd_page_key] = svd_current_page - 1
                st.rerun()
    with col2:
        st.write(f"Halaman **{svd_current_page}** dari **{svd_pages}** (Top-100 SVD)")
    with col3:
        if svd_current_page < svd_pages:
            if st.button("Berikutnya ▶", key=f"svd_next_{user_id}"):
                st.session_state[svd_page_key] = svd_current_page + 1
                st.rerun()
    
    svd_start = (svd_current_page - 1) * items_per_page
    svd_end = min(svd_start + items_per_page, len(svd_table_data))
    svd_df_display = pd.DataFrame(svd_table_data[svd_start:svd_end])
    
    def highlight_fav_svd(row):
        if row['Favorit'] == "Ya":
            return ['background-color: #90EE90'] * len(row)
        return [''] * len(row)
    
    styled_svd_df = svd_df_display.style.apply(highlight_fav_svd, axis=1)
    st.dataframe(styled_svd_df, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # ====================== TABEL HYBRID ======================
    st.subheader(f"Top 100 Rekomendasi Hybrid GMM+SVD untuk User {user_id}")
    
    table_data = []
    fav_count = 0
    non_fav_count = 0
    
    for i, (mid, h_score) in enumerate(hybrid_sorted):
        h_rank = i + 1
        mid_str = str(mid)
        
        try:
            mid_int = int(mid)
            movie_info = movies_df[movies_df['movieId'] == mid_int]
            if not movie_info.empty:
                film_title = movie_info['title'].values[0]
                film_genre = movie_info['genres'].values[0]
            else:
                film_title = f"Movie {mid}"
                film_genre = "Unknown"
        except:
            film_title = f"Movie {mid}"
            film_genre = "Unknown"
        
        is_fav = False
        if user_fav_genres:
            movie_genres = str(film_genre).split('|')
            is_fav = any(g in movie_genres for g in user_fav_genres)
            if is_fav:
                fav_count += 1
            else:
                non_fav_count += 1
        
        b_rank = baseline_rank_all.get(mid_str)
        b_rating = baseline_rating_all.get(mid_str)
        
        if b_rank is None and baseline_sorted_full:
            for idx, (b_mid, b_rate) in enumerate(baseline_sorted_full):
                if str(b_mid) == mid_str:
                    b_rank = idx + 1
                    b_rating = b_rate
                    break
        
        if b_rank is None:
            b_rating = movie_avg_ratings.get(mid_str, global_mean)
            pseudo_rank = movie_avg_rank.get(mid_str)
            if pseudo_rank is not None:
                if pseudo_rank <= 100:
                    b_rank_display = pseudo_rank
                else:
                    b_rank_display = "100+"
            else:
                b_rank_display = "N/A"
        else:
            if b_rank <= 100:
                b_rank_display = b_rank
            else:
                b_rank_display = "100+"
            
            if b_rating is None:
                b_rating = movie_avg_ratings.get(mid_str, global_mean)
        
        delta_rating = h_score - b_rating
        delta_rating_formatted = f"{delta_rating:+.3f}"
        
        fav_indicator = "" if is_fav else ""
        
        table_data.append({
            'Rank': h_rank,
            'Judul Film': f"{fav_indicator}{film_title}",
            'Genre': film_genre,
            'Favorit': "Ya" if is_fav else "Tidak",
            'Rating Hybrid': f"{h_score:.3f}",
            'Rank Baseline': b_rank_display,
            'Rating Baseline': f"{b_rating:.3f}",
            'Δ Rating': delta_rating_formatted
        })
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Film di Top 100", "100")
    with col2:
        st.metric("Film dengan Genre Favorit", f"{fav_count}")
    with col3:
        st.metric("Film Lainnya", f"{non_fav_count}")
    
    # Pagination Hybrid
    pages = max(1, (len(table_data) + items_per_page - 1) // items_per_page)
    page = get_page_number(user_id, pages)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if page > 1:
            if st.button("◀ Sebelumnya", key=f"prev_{user_id}"):
                set_page_number(user_id, page - 1)
                st.rerun()
    with col2:
        st.write(f"Halaman **{page}** dari **{pages}** (Top-100 dari {len(hybrid_recs):,} total film)")
    with col3:
        if page < pages:
            if st.button("Berikutnya ▶", key=f"next_{user_id}"):
                set_page_number(user_id, page + 1)
                st.rerun()
    
    start = (page - 1) * items_per_page
    end = min(start + items_per_page, len(table_data))
    
    df_display = pd.DataFrame(table_data[start:end])
    
    def highlight_fav(row):
        if row['Favorit'] == "Ya":
            return ['background-color: #90EE90'] * len(row)
        return [''] * len(row)
    
    def highlight_delta_rating(val):
        if isinstance(val, str) and val.startswith('+'):
            return 'color: green; font-weight: bold'
        elif isinstance(val, str) and val.startswith('-'):
            return 'color: red; font-weight: bold'
        return ''
    
    def highlight_rank_baseline(val):
        if val == "100+" or val == "N/A":
            return 'color: gray; font-style: italic'
        return ''
    
    styled_df = df_display.style.apply(highlight_fav, axis=1)
    styled_df = styled_df.map(highlight_delta_rating, subset=['Δ Rating'])
    styled_df = styled_df.map(highlight_rank_baseline, subset=['Rank Baseline'])
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
    # Top-K Overlap
    st.subheader("Top-K Overlap")
    
    k_values = [5, 10, 20, 50, 100]
    k_values = [k for k in k_values if k <= len(hybrid_ids)]
    
    overlap_data = []
    for k in k_values:
        hybrid_topk = set(hybrid_ids[:k])
        baseline_topk = set(baseline_ids_100[:k]) if len(baseline_ids_100) >= k else set(baseline_ids_100)
        overlap = len(hybrid_topk & baseline_topk)
        overlap_pct = (overlap / k) * 100 if k > 0 else 0
        
        fav_in_topk = 0
        for mid in hybrid_ids[:k]:
            try:
                mid_int = int(mid)
                movie_info = movies_df[movies_df['movieId'] == mid_int]
                if not movie_info.empty:
                    movie_genres = str(movie_info['genres'].values[0]).split('|')
                    if any(g in movie_genres for g in user_fav_genres):
                        fav_in_topk += 1
            except:
                continue
        
        overlap_data.append({
            'K': k,
            'Film Sama': f"{overlap}/{k}",
            'Overlap %': f"{overlap_pct:.1f}%",
            'Film Favorit di Hybrid': f"{fav_in_topk}/{k}",
            '% Favorit': f"{fav_in_topk/k*100:.1f}%" if k > 0 else "0%"
        })
    
    if overlap_data:
        st.dataframe(pd.DataFrame(overlap_data), use_container_width=True)
    
    return df_display, {
        'total_hybrid': len(hybrid_ids),
        'total_baseline': len(baseline_ids_100),
        'overlap_count': len(set(hybrid_ids) & set(baseline_ids_100)),
        'fav_count': fav_count
    }

# ====================== FUNGSI RANK CORRELATION ======================

def calculate_rank_correlation(hybrid_recs, baseline_recs):
    """
    Menghitung 4 metrics korelasi peringkat.
    
    Parameter:
        hybrid_recs: List rekomendasi hybrid
        baseline_recs: List rekomendasi baseline (SVD)
        
    Returns:
        dict: Dictionary berisi korelasi peringkat
    """
    if not hybrid_recs or not baseline_recs:
        return None
    
    hybrid_sorted = sorted(hybrid_recs, key=lambda x: (-x[1], x[0]))
    baseline_sorted = sorted(baseline_recs, key=lambda x: (-x[1], x[0]))
    
    hybrid_rank = {str(mid): i+1 for i, (mid, _) in enumerate(hybrid_sorted)}
    baseline_rank = {str(mid): i+1 for i, (mid, _) in enumerate(baseline_sorted)}
    
    common_movies = set(hybrid_rank.keys()) & set(baseline_rank.keys())
    
    if len(common_movies) < 2:
        return {
            'kendall_tau': None,
            'spearman_rho': None,
            'avg_rank_diff': None,
            'top_10_overlap': None,
            'common_movies': len(common_movies)
        }
    
    common_list = list(common_movies)
    hybrid_ranks = [hybrid_rank[mid] for mid in common_list]
    baseline_ranks = [baseline_rank[mid] for mid in common_list]
    
    try:
        from scipy.stats import kendalltau, spearmanr
        kendall_tau, kendall_p = kendalltau(hybrid_ranks, baseline_ranks)
        spearman_rho, spearman_p = spearmanr(hybrid_ranks, baseline_ranks)
        
        rank_diffs = [abs(h - b) for h, b in zip(hybrid_ranks, baseline_ranks)]
        avg_rank_diff = np.mean(rank_diffs)
        
        hybrid_top10 = set([str(mid) for mid, _ in hybrid_sorted[:10]])
        baseline_top10 = set([str(mid) for mid, _ in baseline_sorted[:10]])
        top10_overlap = len(hybrid_top10 & baseline_top10) / 10 * 100
        
        return {
            'kendall_tau': kendall_tau,
            'kendall_p_value': kendall_p,
            'spearman_rho': spearman_rho,
            'spearman_p_value': spearman_p,
            'avg_rank_diff': avg_rank_diff,
            'top_10_overlap': top10_overlap,
            'common_movies': len(common_movies)
        }
    except:
        return None

# ====================== NEW USER BUTTON ======================

def new_user_button():
    """
    Tombol untuk membuka interface user baru.
    """
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Daftar sebagai User Baru", type="primary", use_container_width=True):
            st.session_state.show_new_user = not st.session_state.get('show_new_user', False)
            if st.session_state.show_new_user:
                st.session_state.new_user_initialized = False
                st.session_state.new_user_ratings = {}
                st.session_state.new_user_ready = False
                st.session_state.temp_genres = []
            st.rerun()

# ====================== FUNGSI INTERFACE USER BARU ======================
def create_new_user_interface(movies_df, genre_cols):
    """
    Interface untuk user baru.
    """
    st.info("USER BARU")
    
    option = st.radio(
        "Pilih opsi:",
        ["Langsung dapatkan rekomendasi (tanpa rating)", "Berikan rating terlebih dahulu"],
        key="new_user_option"
    )
    
    selected_fav_genres = []
    
    # ========== OPSI 1: TANPA RATING (TAMPILKAN GENRE) ==========
    if option == "Langsung dapatkan rekomendasi (tanpa rating)":
        # TAMPILKAN GENRE UNTUK OPSI TANPA RATING
        st.subheader("Pilih Genre Favorit (WAJIB MINIMAL 3)")
        
        genre_counts = {}
        for genre in genre_cols:
            count = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)].shape[0]
            genre_counts[genre] = count
        
        sorted_genres = sorted(genre_counts.items(), key=lambda x: (-x[1], x[0]))
        
        cols = st.columns(3)
        
        for idx, (genre, count) in enumerate(sorted_genres):
            with cols[idx % 3]:
                default_value = genre in st.session_state.get('temp_genres', [])
                if st.checkbox(f"{genre} ({count})", key=f"fav_new_{genre}", value=default_value):
                    selected_fav_genres.append(genre)
        
        st.session_state.temp_genres = selected_fav_genres
        
        if not selected_fav_genres:
            st.error("Anda belum memilih genre favorit!")
        elif len(selected_fav_genres) < 3:
            st.error(f"Pilih minimal 3 genre! (Saat ini: {len(selected_fav_genres)}/3)")
            st.info(f"Genre dipilih: {', '.join(selected_fav_genres)}")
        else:
            st.success(f"{len(selected_fav_genres)} genre favorit dipilih")
        
        st.success("Langsung dapatkan rekomendasi tanpa rating!")
        st.info("Sistem akan menggunakan genre favorit untuk personalisasi dengan bobot adaptif")
        
        button_disabled = len(selected_fav_genres) < 3
        
        if st.button("Dapatkan Rekomendasi Sekarang!", type="primary", 
                    use_container_width=True, disabled=button_disabled):
            st.session_state.new_user_genres = selected_fav_genres
            st.session_state.new_user_ratings = {}
            st.session_state.new_user_ready = True
            return {'genres': selected_fav_genres, 'ratings': {}}
        
        return {}
    
    # ========== OPSI 2: DENGAN RATING (TANPA GENRE) ==========
    else:
        # TANPA checkbox genre, hanya pesan info
        st.info("💡 **Informasi:** Genre favorit akan diekstrak secara otomatis dari rating yang Anda berikan.")
        st.warning("Berikan rating untuk rekomendasi yang lebih personal dengan bobot adaptif!")
        st.info("**PERHATIAN:** User baru dapat memberikan maksimal 4 rating untuk mencegah bias berlebihan.")
        
        if 'new_user_ratings' not in st.session_state:
            st.session_state.new_user_ratings = {}
        
        current_rating_count = len(st.session_state.new_user_ratings)
        max_ratings = 4
        
        if current_rating_count > 0:
            st.progress(current_rating_count / max_ratings, 
                       text=f"Rating: {current_rating_count}/{max_ratings}")
        
        if current_rating_count >= max_ratings:
            st.warning(f"Anda telah mencapai batas maksimal {max_ratings} rating untuk user baru!")
        
        if st.session_state.new_user_ratings:
            st.subheader("Rating Anda")
            ratings_list = []
            for movie_id, rating in st.session_state.new_user_ratings.items():
                movie_info = movies_df[movies_df['movieId'] == movie_id]
                if not movie_info.empty:
                    ratings_list.append({
                        'Film': movie_info['title'].values[0],
                        'Rating': rating,
                        'Genre': movie_info['genres'].values[0]
                    })
            
            if ratings_list:
                st.dataframe(pd.DataFrame(ratings_list), use_container_width=True)
                
                if st.button("Hapus Semua", type="secondary"):
                    st.session_state.new_user_ratings = {}
                    st.rerun()
        
        if current_rating_count < max_ratings:
            search_query = st.text_input("Cari film:", placeholder="Judul atau genre...")
            
            if search_query:
                results = search_movies(movies_df, search_query, top_n=20)
            else:
                results = movies_df.head(20)
            
            st.write(f"**{len(results)} film ditemukan:**")
            
            cols = st.columns(3)
            new_ratings = False
            
            for idx, (_, row) in enumerate(results.iterrows()):
                with cols[idx % 3]:
                    st.write(f"**{row['title']}**")
                    st.write(f"Genre: {row['genres']}")
                    
                    existing = st.session_state.new_user_ratings.get(row['movieId'])
                    
                    if existing:
                        st.info(f"{existing}")
                        new_val = st.slider("Ubah", 0.5, 5.0, float(existing), 0.5,
                                          key=f"change_{row['movieId']}", label_visibility="collapsed")
                        if new_val != existing:
                            if st.button(f"Update", key=f"upd_{row['movieId']}"):
                                st.session_state.new_user_ratings[row['movieId']] = new_val
                                new_ratings = True
                                st.rerun()
                    else:
                        remaining_slots = max_ratings - current_rating_count
                        
                        if remaining_slots > 0:
                            val = st.slider("Rating", 0.5, 5.0, 3.0, 0.5,
                                          key=f"rate_{row['movieId']}", label_visibility="collapsed")
                            if val > 0:
                                button_label = f"Tambah (sisa {remaining_slots-1} slot)" if remaining_slots > 1 else "Tambah (slot terakhir)"
                                if st.button(button_label, key=f"add_{row['movieId']}"):
                                    st.session_state.new_user_ratings[row['movieId']] = val
                                    new_ratings = True
                                    st.rerun()
                        else:
                            st.caption(" Batas maksimal rating tercapai")
                    
                    st.markdown("---")
            
            if new_ratings:
                st.rerun()
        else:
            st.info(" Anda telah mencapai batas maksimal rating. Klik tombol di bawah untuk mendapatkan rekomendasi.")
        
        # ========== TOMBOL GET REKOMENDASI ==========
        if st.session_state.new_user_ratings:
            st.markdown("---")
            n = len(st.session_state.new_user_ratings)
            st.success(f"{n} dari {max_ratings} rating diberikan!")
            
            if st.button("Dapatkan Rekomendasi dengan Bobot Adaptif!", type="primary", 
                        use_container_width=True):
                # Ekstrak genre dari rating yang diberikan
                extracted_genres = extract_genres_from_ratings(
                    st.session_state.new_user_ratings, movies_df, genre_cols
                )
                st.session_state.new_user_ready = True
                return {'genres': extracted_genres, 'ratings': st.session_state.new_user_ratings}
        
        else:
            st.markdown("---")
            st.info("Belum ada rating yang diberikan. Silakan berikan rating di atas.")
            
            if st.button("Dapatkan Rekomendasi dengan Bobot Adaptif!", type="primary", 
                        use_container_width=True, disabled=True):
                # Tombol disabled karena belum ada rating
                pass
        
        return {}

def extract_genres_from_ratings(ratings_dict, movies_df, genre_cols):
    """
    Ekstrak genre favorit dari rating yang diberikan user.
    Genre yang sering muncul dengan rating tinggi akan dipilih.
    """
    genre_scores = {genre: 0 for genre in genre_cols}
    genre_counts = {genre: 0 for genre in genre_cols}
    
    for movie_id, rating in ratings_dict.items():
        try:
            movie_info = movies_df[movies_df['movieId'] == int(movie_id)]
            if not movie_info.empty:
                movie_genres = str(movie_info['genres'].values[0]).split('|')
                for genre in movie_genres:
                    if genre in genre_scores:
                        # Rating tinggi (>=4) diberi bobot lebih
                        weight = rating / 2.0  # Normalisasi rating 0.5-5 ke 0.25-2.5
                        genre_scores[genre] += weight
                        genre_counts[genre] += 1
        except:
            continue
    
    # Hitung rata-rata skor per genre
    avg_scores = {}
    for genre in genre_cols:
        if genre_counts[genre] > 0:
            avg_scores[genre] = genre_scores[genre] / genre_counts[genre]
        else:
            avg_scores[genre] = 0
    
    # Ambil top genre dengan skor tertinggi (minimal 3, maksimal 5)
    sorted_genres = sorted(avg_scores.items(), key=lambda x: -x[1])
    top_genres = [genre for genre, score in sorted_genres[:5] if score > 0.5]
    
    # Pastikan minimal 3 genre (jika kurang, ambil genre populer)
    if len(top_genres) < 3:
        # Ambil genre paling populer dari dataset
        genre_popularity = {}
        for genre in genre_cols:
            count = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)].shape[0]
            genre_popularity[genre] = count
        
        popular_genres = sorted(genre_popularity.items(), key=lambda x: -x[1])
        for genre, _ in popular_genres:
            if genre not in top_genres and len(top_genres) < 3:
                top_genres.append(genre)
    
    print(f"[DEBUG] Extracted genres from ratings: {top_genres}")
    return top_genres

def show_data_summary():
    """Menampilkan jumlah data sebelum dan sesudah pembersihan"""
    if st.session_state.get('ratings') is not None:
        df = st.session_state.ratings
        movies = st.session_state.movies
        
        # Jumlah duplikat
        dup_rating = df.duplicated(subset=['userId', 'movieId', 'rating']).sum()
        dup_movie = movies.duplicated(subset=['movieId']).sum()
        
        # Data sebelum
        before_rating = len(df) + dup_rating
        before_movie = len(movies) + dup_movie
        
        # Data sesudah
        after_rating = len(df)
        after_movie = len(movies)
        after_user = df['userId'].nunique()
        
        st.markdown("### Hasil Pembersihan Data")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**SEBELUM**")
            st.write(f"Rating: {before_rating:,}")
            st.write(f"Film: {before_movie:,}")
        with col2:
            st.write("**SESUDAH**")
            st.write(f"Rating: {after_rating:,} (hapus {dup_rating:,})")
            st.write(f"Film: {after_movie:,} (hapus {dup_movie:,})")
            st.write(f"User: {after_user:,}")

# ====================== MAIN APPLICATION ======================
def main():

    st.set_page_config(page_title="Hybrid Recommender System - GMM+SVD", page_icon="", layout="wide")
        
    
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = RANDOM_SEED
    
    
    if not st.session_state.get('split_done', False):
        split_info_path = os.path.join(MODEL_DIR, 'split_info.pkl')
        if os.path.exists(split_info_path):
            try:
                split_info = joblib.load(split_info_path)
                st.session_state.split_done = split_info.get('split_done', False)
                if st.session_state.split_done:
                    train_df_path = os.path.join(MODEL_DIR, 'train_df.csv')
                    test_df_path = os.path.join(MODEL_DIR, 'test_df.csv')
                    train_movies_path = os.path.join(MODEL_DIR, 'train_movies.pkl')
                    test_movies_path = os.path.join(MODEL_DIR, 'test_movies.pkl')
                    
                    if os.path.exists(train_df_path):
                        st.session_state.train_df = pd.read_csv(train_df_path)
                    if os.path.exists(test_df_path):
                        st.session_state.test_df = pd.read_csv(test_df_path)
                    if os.path.exists(train_movies_path):
                        st.session_state.train_movies = set(joblib.load(train_movies_path))
                    if os.path.exists(test_movies_path):
                        st.session_state.test_movies = set(joblib.load(test_movies_path))
                    
                    st.info("Data split sebelumnya berhasil dimuat!")
            except Exception as e:
                print(f"Error loading split info: {e}")

    if st.session_state.get('grid_results') is None:
        grid_path = os.path.join(MODEL_DIR, 'grid_results.pkl')
        if os.path.exists(grid_path):
            try:
                st.session_state.grid_results = joblib.load(grid_path)
                st.info(" Hasil Grid Search sebelumnya berhasil dimuat!")
            except Exception as e:
                print(f"Error loading grid results: {e}")

    if st.session_state.get('tab3_results') is None:
        cv_path = os.path.join(MODEL_DIR, 'cv_results.pkl')
        if os.path.exists(cv_path):
            try:
                st.session_state.tab3_results = joblib.load(cv_path)
                st.session_state.tab3_processed = True
                st.info(" Hasil Cross-Validation sebelumnya berhasil dimuat!")
            except Exception as e:
                print(f"Error loading CV results: {e}")

    st.title(" Hybrid Recommender System - GMM + SVD")
    st.markdown("---")
    
    model_dir = MODEL_DIR
    saved_models_status = check_saved_models(model_dir)
    if 'model_loaded_from_file' not in st.session_state:
        st.session_state.model_loaded_from_file = False
    
    with st.sidebar:
        st.header("Manajemen Model")
        
        saved_models_status = check_saved_models(MODEL_DIR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if saved_models_status['exists']:
                if st.button("Load Saved Model", type="primary", use_container_width=True):
                    with st.spinner("Memuat model tersimpan..."):
                        if load_all_models(MODEL_DIR):
                            st.session_state.model_loaded_from_file = True
                            st.success("Model berhasil dimuat!")
                            st.rerun()
                        else:
                            st.error("Gagal memuat model")
            else:
                st.info("Belum ada model tersimpan")
        
        with col2:
            if st.session_state.get('model_trained', False):
                if st.button("Save Current Model", type="secondary", use_container_width=True):
                    with st.spinner("Menyimpan model..."):
                        saved_files = save_all_models(MODEL_DIR)
                        if saved_files:
                            st.success(f"Berhasil menyimpan {len(saved_files)} file model ke folder '{MODEL_DIR}'")
                            
                            with st.expander("Detail file tersimpan"):
                                for name, path in saved_files:
                                    st.write(f"- **{name}**: `{path}`")
                            st.balloons()
                        else:
                            st.error(" Gagal menyimpan model - tidak ada data yang tersimpan")
    
    st.sidebar.markdown("---")

    if st.button("Save All Components", type="primary", use_container_width=True):
        with st.spinner("Menyimpan semua komponen..."):
            saved_count, saved_items = auto_save_all_components()
            if saved_count > 0:
                st.success(f"Berhasil menyimpan {saved_count} komponen ke {MODEL_DIR}")
                with st.expander("Detail file tersimpan"):
                    for item in saved_items:
                        st.write(f"- {item}")
                st.balloons()
            else:
                st.warning("Tidak ada data yang tersimpan. Pastikan Anda sudah menjalankan proses di tab-tab sebelumnya.")
    
    if st.session_state.get('model_trained', False) or st.session_state.model_loaded_from_file:
        if st.session_state.model_loaded_from_file:
            st.sidebar.success("Model dimuat dari file")
        else:
            st.sidebar.success("Model sudah di-train")
    else:
        st.sidebar.warning("Model belum di-train atau dimuat")
        st.sidebar.info("Silakan training model di Tab 3, atau load model tersimpan")
    
    if not st.session_state.get('data_loaded', False):
        with st.spinner("Auto-loading data dari path yang ditentukan..."):
            try:
                if os.path.exists(PATH_RATINGS) and os.path.exists(PATH_MOVIES):
                    ratings = pd.read_csv(PATH_RATINGS)
                    movies = pd.read_csv(PATH_MOVIES)
                    
                    movies['movieId'] = movies['movieId'].astype(int)
                    ratings['movieId'] = ratings['movieId'].astype(int)
                    ratings['userId'] = ratings['userId'].astype(int)
                    
                    st.session_state.ratings = ratings
                    st.session_state.movies = movies
                    st.session_state.stats = {
                        'n_users': ratings['userId'].nunique(),
                        'n_items': ratings['movieId'].nunique(),
                        'n_ratings': len(ratings),
                        'sparsity': 1 - len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()),
                        'rating_mean': ratings['rating'].mean(),
                        'rating_std': ratings['rating'].std()
                    }
                    st.session_state.data_loaded = True
                    
                    st.success(f"Data berhasil di-load")
                    
                    # ========== TAMBAHKAN 6 BARIS INI ==========
                    # Tampilkan summary data setelah auto-load
                    n_users = ratings['userId'].nunique()
                    n_items = ratings['movieId'].nunique()
                    n_ratings = len(ratings)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Jumlah User", f"{n_users:,}")
                    with col2:
                        st.metric("Jumlah Film", f"{n_items:,}")
                    with col3:
                        st.metric("Jumlah Rating", f"{n_ratings:,}")
                    # ==========================================
                else:
                    st.error(f"File tidak ditemukan di:\n{PATH_RATINGS}\n{PATH_MOVIES}")
                    st.info("Pastikan path file sesuai dengan lokasi data Anda")
            except Exception as e:
                st.error(f"Error loading data: {e}")

                
    if st.session_state.model_loaded_from_file and not st.session_state.get('model_trained', False):
        st.info("Model berhasil dimuat! Anda dapat langsung menggunakan fitur rekomendasi di Tab 5.")
        st.session_state.model_trained = True
    
    with st.expander("Parameter Model", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("GMM")
            max_components = st.slider("Maks Komponen GMM:", 20, 100, 
                                      st.session_state.get('max_components', 50), 5)
            smoothing = st.slider("Smoothing Factor:", 0.01, 0.1, 
                                 st.session_state.get('smoothing', 0.1), 0.01)
        
        with col2:
            st.subheader("SVD")
            n_factors = st.slider("Faktor SVD:", 20, 200, 
                                 st.session_state.get('n_factors', 100), 20)
            n_epochs = st.slider("Epoch:", 10, 100, 
                                st.session_state.get('n_epochs', 30), 5)
        
        with col3:
            st.subheader("Training")
            lr_all = st.slider("Learning Rate:", 0.001, 0.01, 
                              st.session_state.get('lr_all', 0.005), 0.001)
            reg_all = st.slider("Regularization:", 0.01, 0.1, 
                               st.session_state.get('reg_all', 0.02), 0.01)
            test_size = st.slider("Test size %:", 10, 40, 20, key="test_size_param")
    
    st.session_state.max_components = max_components
    st.session_state.smoothing = smoothing
    st.session_state.n_factors = n_factors
    st.session_state.n_epochs = n_epochs
    st.session_state.lr_all = lr_all
    st.session_state.reg_all = reg_all
    st.session_state.test_size = test_size / 100

    show_tables = True
    
    if 'saved_outputs' not in st.session_state:
        st.session_state.saved_outputs = {}
    
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'new_user_initialized' not in st.session_state:
        st.session_state.new_user_initialized = False
    if 'new_user_ratings' not in st.session_state:
        st.session_state.new_user_ratings = {}
    if 'new_user_ready' not in st.session_state:
        st.session_state.new_user_ready = False
    if 'genre_cols' not in st.session_state:
        st.session_state.genre_cols = []
    if 'registered_users' not in st.session_state:
        st.session_state.registered_users = {}
    if 'show_new_user' not in st.session_state:
        st.session_state.show_new_user = False
    if 'temp_genres' not in st.session_state:
        st.session_state.temp_genres = []
    if 'cluster_genre_mapping' not in st.session_state:
        st.session_state.cluster_genre_mapping = {}
    
    if 'grid_results' not in st.session_state:
        st.session_state.grid_results = None
    if 'gmm_map' not in st.session_state:
        st.session_state.gmm_map = None 
    
    if 'split_done' not in st.session_state:
        st.session_state.split_done = False
    if 'train_df' not in st.session_state:
        st.session_state.train_df = None
    if 'test_df' not in st.session_state:
        st.session_state.test_df = None
    if 'train_movies' not in st.session_state:
        st.session_state.train_movies = None
    if 'test_movies' not in st.session_state:
        st.session_state.test_movies = None
    
    if 'tab1_processed' not in st.session_state:
        st.session_state.tab1_processed = False
    if 'tab2_processed' not in st.session_state:
        st.session_state.tab2_processed = False
    if 'tab3_processed' not in st.session_state:
        st.session_state.tab3_processed = False
    if 'tab4_processed' not in st.session_state:
        st.session_state.tab4_processed = False
    
    if 'tab1_results' not in st.session_state:
        st.session_state.tab1_results = {}
    if 'tab2_results' not in st.session_state:
        st.session_state.tab2_results = {}
    if 'tab3_results' not in st.session_state:
        st.session_state.tab3_results = {}
    if 'tab4_results' not in st.session_state:
        st.session_state.tab4_results = {}
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Data Analysis & Split", "Grid Search", "Model Training", "Cross-Validation", "Recommendations"]
    )
    
    with tab1:
        st.header("Analisis Data & Warm-Start Split")

        if st.session_state.get('data_loaded', False):
            show_data_summary()
            st.markdown("---")
        

        st.markdown("""
        Pada tab ini, Anda dapat:
        1. Melakukan analisis data awal
        2. Melakukan **Warm-Start Item Split** (pembagian data berdasarkan film)
        3. Hasil split akan disimpan dan digunakan di tab-tab berikutnya
        """)
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Data tidak dapat di-load. Periksa path file.")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                process_button = st.button("Load dan Analisis Data", type="primary")
            
            output_key = "tab1_output"
            
            if output_key in st.session_state.saved_outputs:
                st.markdown("### Output Proses Loading Data")
                display_saved_outputs(output_key)
            
            if st.session_state.split_done:
                st.success("Warm-Start Item Split sudah dilakukan!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Film Training", f"{len(st.session_state.train_movies):,}")
                with col2:
                    st.metric("Film Test", f"{len(st.session_state.test_movies):,}")
                with col3:
                    st.metric("Rating Training", f"{len(st.session_state.train_df):,}")
                with col4:
                    st.metric("Rating Test", f"{len(st.session_state.test_df):,}")
                
                with st.expander("Sample Data Training", expanded=False):
                    st.dataframe(st.session_state.train_df.head(20), use_container_width=True)
                
                with st.expander("Sample Data Test", expanded=False):
                    st.dataframe(st.session_state.test_df.head(20), use_container_width=True)
            
            if st.session_state.tab1_processed and st.session_state.tab1_results:
                st.info("Data sudah diproses sebelumnya")
                
                if 'ratings' in st.session_state.tab1_results:
                    st.subheader("Data Rating")
                    st.dataframe(st.session_state.tab1_results['ratings'], use_container_width=True, height=400)
                
                if 'movies' in st.session_state.tab1_results:
                    st.subheader("Data Film")
                    st.dataframe(st.session_state.tab1_results['movies'], use_container_width=True, height=400)
                
                if 'stats' in st.session_state.tab1_results:
                    stats = st.session_state.tab1_results['stats']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Jumlah User", f"{stats['n_users']:,}")
                    with col2:
                        st.metric("Jumlah Film", f"{stats['n_items']:,}")
                    with col3:
                        st.metric("Jumlah Rating", f"{stats['n_ratings']:,}")
                    with col4:
                        st.metric("Sparsity", f"{stats['sparsity']:.2%}")
                
                st.markdown("---")
                st.header("FILM YANG BELUM MEMILIKI RATING")
                
                all_movie_ids = set(st.session_state.movies['movieId'].unique())
                rated_movie_ids = set(st.session_state.ratings['movieId'].unique())
                unrated_movie_ids = all_movie_ids - rated_movie_ids
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Film", f"{len(all_movie_ids):,}")
                with col2:
                    st.metric("Film dengan Rating", f"{len(rated_movie_ids):,}")
                with col3:
                    st.metric("Film TANPA Rating", f"{len(unrated_movie_ids):,}", 
                            delta=f"{len(unrated_movie_ids)/len(all_movie_ids)*100:.2f}% dari total")
                
                if len(unrated_movie_ids) > 0:
                    st.subheader(f"Daftar {min(50, len(unrated_movie_ids))} Film Tanpa Rating (Sample)")
                    
                    unrated_movies_data = []
                    for i, movie_id in enumerate(list(unrated_movie_ids)[:50]):
                        try:
                            movie_info = st.session_state.movies[st.session_state.movies['movieId'] == movie_id]
                            if not movie_info.empty:
                                title = movie_info['title'].values[0]
                                genres = movie_info['genres'].values[0]
                                genre_count = len(genres.split('|')) if genres != '(no genres listed)' else 0
                                
                                unrated_movies_data.append({
                                    'No': i + 1,
                                    'Movie ID': movie_id,
                                    'Judul Film': title,
                                    'Genre': genres,
                                    'Jumlah Genre': genre_count
                                })
                        except:
                            continue
                    
                    if unrated_movies_data:
                        unrated_df = pd.DataFrame(unrated_movies_data)
                        st.dataframe(unrated_df, use_container_width=True, height=500)
                else:
                    st.success("Semua film sudah memiliki rating!")
            
            if process_button:
                clear_saved_outputs(output_key)
                
                with st.spinner("Memuat data dan melakukan split..."):
                    ratings, movies, stats = load_and_filter_data_with_save(
                        PATH_RATINGS, PATH_MOVIES, output_key, show_tables
                    )
                    
                    if ratings is not None:
                        st.session_state.ratings = ratings
                        st.session_state.movies = movies
                        st.session_state.stats = stats
                        st.session_state.data_loaded = True
                        
                        st.session_state.tab1_results = {
                            'ratings': ratings,
                            'movies': movies,
                            'stats': stats
                        }
                        st.session_state.tab1_processed = True
                        
                        st.markdown("---")
                        st.subheader("MELAKUKAN WARM-START ITEM SPLIT...")
                        
                        test_size_val = st.session_state.get('test_size', 0.2)
                        
                        train_df, test_df, train_movies, test_movies = warm_start_item_split_with_save(
                            ratings,
                            movies,
                            output_key,
                            test_size=test_size_val,
                            random_state=RANDOM_SEED,
                            show_tables=show_tables
                        )
                        
                        if train_df is not None:
                            st.session_state.train_df = train_df
                            st.session_state.test_df = test_df
                            st.session_state.train_movies = train_movies
                            st.session_state.test_movies = test_movies
                            st.session_state.split_done = True

                            saved_count, saved_items = auto_save_all_components()
                            st.success(f"Data split berhasil disimpan! {saved_count} komponen tersimpan.")
                            
                            st.success("""
                             **SPLIT SELESAI!**
                            
                            Hasil split telah disimpan dan akan digunakan di tab-tab berikutnya:
                            - **Tab Grid Search**: Menggunakan data split yang sama
                            - **Tab Model Training**: Menggunakan data split yang sama  
                            - **Tab Cross-Validation**: Menggunakan data split yang sama
                            - **Tab Recommendations**: Menggunakan model yang di-train dengan data split ini
                            """)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Film Training", f"{len(train_movies):,}")
                            with col2:
                                st.metric("Film Test", f"{len(test_movies):,}")
                            with col3:
                                st.metric("Rating Training", f"{len(train_df):,}")
                            with col4:
                                st.metric("Rating Test", f"{len(test_df):,}")
                            
                            st.rerun()
    
    with tab2:
        st.header("Grid Search dengan 5-Fold Cross-Validation")
        st.markdown("""
        Mencari kombinasi parameter optimal untuk model Hybrid GMM+SVD menggunakan 5-fold cross-validation.
        
        **Data yang digunakan:** Hasil split dari Tab 1 (hanya data TRAINING)
        """)
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Load data terlebih dahulu di tab Data Analysis")
        elif not st.session_state.get('split_done', False):
            st.warning("Lakukan Warm-Start Split terlebih dahulu di tab Data Analysis")
        else:
            st.info(f"""
            **Data yang akan digunakan untuk Grid Search:**
            - Data Training: {len(st.session_state.train_df):,} rating dari {len(st.session_state.train_movies):,} film
            - Data Test: {len(st.session_state.test_df):,} rating dari {len(st.session_state.test_movies):,} film
            """)
            
            with st.expander("Konfigurasi Grid Search", expanded=True):
                st.subheader("Parameter Grid")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Parameter SVD**")
                    n_factors_options = st.multiselect(
                        "n_factors:",
                        [50, 100, 150, 200],
                        default=[100, 150],
                        key="grid_n_factors"
                    )
                    
                    n_epochs_options = st.multiselect(
                        "n_epochs:",
                        [20, 30, 40, 50],
                        default=[30, 40],
                        key="grid_n_epochs"
                    )
                    
                    lr_options = st.multiselect(
                        "learning_rate:",
                        [0.001, 0.005, 0.01],
                        default=[0.005],
                        key="grid_lr"
                    )
                    
                    reg_options = st.multiselect(
                        "regularization:",
                        [0.01, 0.02, 0.05],
                        default=[0.02],
                        key="grid_reg"
                    )
                
                with col2:
                    st.markdown("**Parameter GMM**")
                    max_components_options = st.multiselect(
                        "max_components:",
                        [30, 50, 70, 100],
                        default=[50],
                        key="grid_max_components"
                    )
                    
                    smoothing_options = st.multiselect(
                        "smoothing:",
                        [0.01, 0.02, 0.05, 0.1],
                        default=[0.1],
                        key="grid_smoothing"
                    )
                    
                    st.markdown("**Cross-Validation**")
                    n_folds = st.slider("Jumlah Folds:", 3, 10, 5, key="grid_folds")
                    
                    random_state = st.number_input("Random Seed:", 0, 100, 42, key="grid_seed")
            
            # Build parameter grid
            param_grid = {}
            
            if n_factors_options:
                param_grid['n_factors'] = n_factors_options
            if n_epochs_options:
                param_grid['n_epochs'] = n_epochs_options
            if lr_options:
                param_grid['lr_all'] = lr_options
            if reg_options:
                param_grid['reg_all'] = reg_options
            if max_components_options:
                param_grid['max_components'] = max_components_options
            if smoothing_options:
                param_grid['smoothing'] = smoothing_options
            
            if param_grid:
                total_combinations = 1
                for values in param_grid.values():
                    total_combinations *= len(values)
                
                st.info(f"Total kombinasi parameter yang akan diuji: **{total_combinations}**")
                
                if total_combinations > 100:
                    st.warning("Total kombinasi > 100. Proses mungkin memakan waktu lama.")
            
            # BUTTON GRID SEARCH
            col1, col2 = st.columns([1, 3])
            with col1:
                grid_button = st.button("Jalankan Grid Search", type="primary", key="run_grid_search")
            
            # ==================== FUNGSI UNTUK MENAMPILKAN HASIL (SATU KALI) ====================
            def display_grid_results(results, show_apply_button=True):
                """Menampilkan hasil grid search (hanya dipanggil sekali)"""
                if not results or results['results_df'].empty:
                    st.warning("Tidak ada hasil grid search")
                    return
                
                st.subheader("Hasil Grid Search - Parameter Terbaik")
                
                best_params = results['best_params']
                best_score = results['best_score']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success("Parameter Optimal:")
                    for key, value in best_params.items():
                        st.metric(key.replace('_', ' ').title(), value)
                
                with col2:
                    st.success("Performa Terbaik:")
                    st.metric("Average RMSE (5-fold CV)", f"{best_score:.6f}")
                
                st.markdown("---")
                st.subheader("Ranking Semua Kombinasi Parameter")
                
                results_df = results['results_df'].copy()
                
                display_cols = []
                for col in results_df.columns:
                    if col not in ['avg_rmse', 'avg_mae', 'rmse_std', 'improvement']:
                        display_cols.append(col)
                
                display_cols.extend(['avg_rmse', 'avg_mae', 'improvement', 'rmse_std'])
                
                display_df = results_df[display_cols].copy()
                display_df['avg_rmse'] = display_df['avg_rmse'].apply(lambda x: f"{x:.6f}")
                display_df['avg_mae'] = display_df['avg_mae'].apply(lambda x: f"{x:.6f}")
                display_df['improvement'] = display_df['improvement'].apply(lambda x: f"{x:+.2f}%")
                display_df['rmse_std'] = display_df['rmse_std'].apply(lambda x: f"{x:.6f}")
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                st.subheader("Visualisasi Pengaruh Parameter")
                
                numeric_params = []
                for col in best_params.keys():
                    if col in ['n_factors', 'n_epochs', 'max_components', 'lr_all', 'reg_all', 'smoothing']:
                        numeric_params.append(col)
                
                if len(numeric_params) >= 2:
                    param1 = numeric_params[0]
                    param2 = numeric_params[1] if len(numeric_params) > 1 else numeric_params[0]
                    
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    
                    ax1 = axes[0]
                    param1_values = results_df[param1].unique()
                    param1_values.sort()
                    
                    avg_rmse_by_param1 = []
                    for val in param1_values:
                        subset = results_df[results_df[param1] == val]
                        avg_rmse_by_param1.append(subset['avg_rmse'].mean())
                    
                    ax1.plot(param1_values, avg_rmse_by_param1, 'b-o', linewidth=2, markersize=8)
                    ax1.set_xlabel(param1.replace('_', ' ').title(), fontsize=11)
                    ax1.set_ylabel('Average RMSE', fontsize=11)
                    ax1.set_title(f'Pengaruh {param1} terhadap Performa', fontsize=12, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    
                    ax2 = axes[1]
                    param2_values = results_df[param2].unique()
                    param2_values.sort()
                    
                    avg_rmse_by_param2 = []
                    for val in param2_values:
                        subset = results_df[results_df[param2] == val]
                        avg_rmse_by_param2.append(subset['avg_rmse'].mean())
                    
                    ax2.plot(param2_values, avg_rmse_by_param2, 'g-o', linewidth=2, markersize=8)
                    ax2.set_xlabel(param2.replace('_', ' ').title(), fontsize=11)
                    ax2.set_ylabel('Average RMSE', fontsize=11)
                    ax2.set_title(f'Pengaruh {param2} terhadap Performa', fontsize=12, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Hasil Grid Search (CSV)",
                    csv,
                    "grid_search_results.csv",
                    "text/csv",
                    key="grid_search_download"
                )
                
                # Tampilkan parameter terbaik untuk training
                if show_apply_button:
                    st.markdown("---")
                    st.subheader("Parameter Terbaik - Siap Digunakan untuk Training")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.success("**Parameter SVD Terbaik:**")
                        for key in ['n_factors', 'n_epochs', 'lr_all', 'reg_all']:
                            if key in best_params:
                                st.metric(key.replace('_', ' ').title(), best_params[key])
                    
                    with col2:
                        st.success("**Parameter GMM Terbaik:**")
                        for key in ['max_components', 'smoothing']:
                            if key in best_params:
                                st.metric(key.replace('_', ' ').title(), best_params[key])
                    
                    with col3:
                        st.success("**Performa Terbaik:**")
                        st.metric("RMSE (5-fold CV)", f"{best_score:.6f}")
                    
                    if st.button("Terapkan Parameter Terbaik untuk Training", key="apply_best_params"):
                        if 'n_factors' in best_params:
                            st.session_state.n_factors = best_params['n_factors']
                        if 'n_epochs' in best_params:
                            st.session_state.n_epochs = best_params['n_epochs']
                        if 'lr_all' in best_params:
                            st.session_state.lr_all = best_params['lr_all']
                        if 'reg_all' in best_params:
                            st.session_state.reg_all = best_params['reg_all']
                        if 'max_components' in best_params:
                            st.session_state.max_components = best_params['max_components']
                        if 'smoothing' in best_params:
                            st.session_state.smoothing = best_params['smoothing']
                        
                        st.success("""
                        **Parameter terbaik TELAH DITERAPKAN!**
                        
                        Parameter di tab Model Training sudah diupdate dengan nilai optimal dari grid search.
                        Silakan langsung ke tab **Model Training** untuk melakukan training.
                        """)
                        
                        if st.button("Langsung ke Model Training", type="primary", use_container_width=True):
                            st.session_state.active_tab = 1
                            st.rerun()
            
            # ==================== TAMPILKAN HASIL YANG SUDAH TERSIMPAN ====================
            if st.session_state.get('grid_results') is not None and not grid_button:
                # Hanya tampilkan jika tombol grid search TIDAK ditekan
                display_grid_results(st.session_state.grid_results, show_apply_button=True)
            
            # ==================== JALANKAN GRID SEARCH BARU ====================
            if grid_button:
                if not param_grid:
                    st.error("Pilih minimal satu parameter untuk di-grid search!")
                else:
                    with st.spinner(f"Menjalankan grid search dengan {total_combinations} kombinasi parameter..."):
                        try:
                            # Jalankan grid search
                            results = run_grid_search_cv(
                                ratings_df=st.session_state.train_df,
                                movies_df=st.session_state.movies,
                                param_grid=param_grid,
                                n_folds=n_folds,
                                random_state=random_state
                            )
                            
                            if results and results['best_params']:
                                # Simpan ke session state
                                st.session_state.grid_results = results
                                
                                # Tampilkan hasil SEKALI saja (tanpa duplikasi)
                                display_grid_results(results, show_apply_button=True)
                                
                                # Simpan ke file
                                saved_count, saved_items = auto_save_all_components()
                                st.success(f"Hasil grid search disimpan! {saved_count} komponen tersimpan.")
                                
                                st.success(f"Grid search selesai! Parameter terbaik ditemukan dengan RMSE = {results['best_score']:.6f}")
                                
                                # Gunakan st.rerun() untuk refresh, tapi pastikan tidak double
                                st.rerun()
                            else:
                                st.error("Grid search gagal. Coba dengan parameter yang berbeda.")
                                
                        except Exception as e:
                            st.error(f"Error: {e}")
                            import traceback
                            st.error(traceback.format_exc())

    with tab3:
        st.header("Training Model Hybrid GMM+SVD")
        st.markdown("""
        Melatih model Hybrid GMM+SVD menggunakan data yang sudah di-split di Tab 1.
        
        **Data yang digunakan:** Hasil split dari Tab 1 (Training Set)
        """)
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Load data terlebih dahulu di tab Data Analysis")
        elif not st.session_state.get('split_done', False):
            st.warning("Lakukan Warm-Start Split terlebih dahulu di tab Data Analysis")
        else:
            st.info(f"""
            **Data Training:** {len(st.session_state.train_df):,} rating dari {len(st.session_state.train_movies):,} film
            **Data Test:** {len(st.session_state.test_df):,} rating dari {len(st.session_state.test_movies):,} film
            """)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                train_button = st.button("Train Model Hybrid", type="primary")
            
            output_key = "tab2_output"
            
            if output_key in st.session_state.saved_outputs:
                with st.expander("Output Proses Training (One-Hot Encoding, GMM, dll)", expanded=False):
                    display_saved_outputs(output_key)
            
            if st.session_state.get('model_trained', False):
                st.success("Model sudah di-train! Menampilkan hasil tersimpan...")
                
                st.markdown("---")
                st.subheader("METRIK EVALUASI MODEL")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    svd_rmse = st.session_state.get('svd_rmse', 0)
                    st.metric("SVD RMSE", f"{svd_rmse:.6f}")
                
                with col2:
                    svd_mae = st.session_state.get('svd_mae', 0)
                    st.metric("SVD MAE", f"{svd_mae:.6f}")
                
                with col3:
                    hybrid_rmse = st.session_state.get('hybrid_rmse', 0)
                    st.metric("Hybrid RMSE", f"{hybrid_rmse:.6f}")
                
                with col4:
                    hybrid_mae = st.session_state.get('hybrid_mae', 0)
                    st.metric("Hybrid MAE", f"{hybrid_mae:.6f}")
                
                svd_rmse = st.session_state.get('svd_rmse', 0)
                hybrid_rmse = st.session_state.get('hybrid_rmse', 0)
                svd_mae = st.session_state.get('svd_mae', 0)
                hybrid_mae = st.session_state.get('hybrid_mae', 0)
                
                if svd_rmse > 0:
                    rmse_imp = ((svd_rmse - hybrid_rmse) / svd_rmse) * 100
                    mae_imp = ((svd_mae - hybrid_mae) / svd_mae) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("RMSE Improvement", f"{rmse_imp:+.2f}%")
                    with col2:
                        st.metric("MAE Improvement", f"{mae_imp:+.2f}%")
                
                st.markdown("---")
                st.subheader("BOBOT ADAPTIF (RULE-BASED)")
                
                st.info("""
                **Rule-Based Adaptive Weight**
                
                **Aturan yang digunakan:**
                | Jumlah Rating | Alpha (CF) | Bobot CB |
                |---------------|------------|----------|
                | < 5 | 0.20 | 0.80 |
                | 5 - 9 | 0.30 | 0.70 |
                | 10 - 19 | 0.35 | 0.65 |
                | 20 - 49 | 0.40 | 0.60 |
                | 50 - 99 | 0.45 | 0.55 |
                | ≥ 100 | 0.50 | 0.50 |
                """)
                
                st.markdown("---")
                st.subheader("TABEL HASIL TRAINING")
                
                if st.session_state.get('svd_predictions_df') is not None:
                    with st.expander("HASIL PREDIKSI SVD PADA TEST SET", expanded=True):
                        st.dataframe(st.session_state.svd_predictions_df, use_container_width=True, height=400)
                        
                        csv = st.session_state.svd_predictions_df.to_csv(index=False)
                        st.download_button(
                            "Download SVD Results",
                            csv,
                            "svd_predictions.csv",
                            "text/csv",
                            key="download_svd_main"
                        )
                
                if st.session_state.get('hybrid_predictions_df') is not None:
                    with st.expander("HASIL PREDIKSI HYBRID GMM+SVD PADA TEST SET", expanded=True):
                        st.dataframe(st.session_state.hybrid_predictions_df, use_container_width=True, height=400)
                        
                        csv = st.session_state.hybrid_predictions_df.to_csv(index=False)
                        st.download_button(
                            "Download Hybrid Results",
                            csv,
                            "hybrid_predictions.csv",
                            "text/csv",
                            key="download_hybrid_main"
                        )
                
                if st.session_state.get('comparison_df') is not None:
                    with st.expander("PERBANDINGAN SVD vs HYBRID PADA TEST SET", expanded=True):
                        st.dataframe(st.session_state.comparison_df, use_container_width=True, height=400)
                        
                        csv = st.session_state.comparison_df.to_csv(index=False)
                        st.download_button(
                            "Download Comparison Results",
                            csv,
                            "comparison_results.csv",
                            "text/csv",
                            key="download_comparison_main"
                        )
                
                st.markdown("---")
                st.subheader("DOWNLOAD SEMUA HASIL")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.session_state.get('svd_predictions_df') is not None:
                        csv = st.session_state.svd_predictions_df.to_csv(index=False)
                        st.download_button(
                            "SVD Results",
                            csv,
                            "svd_predictions.csv",
                            "text/csv",
                            key="download_svd_all"
                        )
                
                with col2:
                    if st.session_state.get('hybrid_predictions_df') is not None:
                        csv = st.session_state.hybrid_predictions_df.to_csv(index=False)
                        st.download_button(
                            "Hybrid Results",
                            csv,
                            "hybrid_predictions.csv",
                            "text/csv",
                            key="download_hybrid_all"
                        )
                
                with col3:
                    if st.session_state.get('comparison_df') is not None:
                        csv = st.session_state.comparison_df.to_csv(index=False)
                        st.download_button(
                            "Comparison Results",
                            csv,
                            "comparison_results.csv",
                            "text/csv",
                            key="download_comparison_all"
                        )
            
            if train_button:
                clear_saved_outputs(output_key)
                
                with st.spinner("Training model... (Proses ini akan memakan waktu)"):
                    try:
                        train_df = st.session_state.train_df
                        test_df = st.session_state.test_df
                        train_movies = st.session_state.train_movies
                        test_movies = st.session_state.test_movies
                        
                        trainset, testset = convert_to_surprise_format_with_save(
                            train_df, test_df, output_key, show_tables=show_tables
                        )
                        st.session_state.trainset = trainset
                        st.session_state.testset = testset
                        
                        movies_enc, feature_cols, genre_cols = one_hot_encode_genres_with_save(
                            st.session_state.movies,
                            train_movies,
                            output_key,
                            smoothing_factor=smoothing,
                            show_tables=show_tables
                        )
                        st.session_state.genre_cols = genre_cols
                        st.session_state.movies_enc = movies_enc
                        
                        gmm_map, gmm_model = gmm_soft_clustering_with_save(
                            movies_enc,
                            feature_cols,
                            output_key,
                            max_components=max_components,
                            random_state=RANDOM_SEED,
                            show_tables=show_tables
                        )
                        
                        cluster_genre_mapping = build_gmm_to_genre_mapping(
                            gmm_model, movies_enc, feature_cols, genre_cols
                        )
                        st.session_state.cluster_genre_mapping = cluster_genre_mapping
                        
                        test_gmm_map = transform_test_movies_with_gmm(
                            st.session_state.movies,
                            test_movies,
                            gmm_model,
                            feature_cols,
                            smoothing_factor=smoothing
                        )
                        
                        full_gmm_map = {**gmm_map, **test_gmm_map}
                        st.session_state.gmm_map = full_gmm_map
                        
                        svd = train_svd_model_with_save(
                            trainset,
                            output_key,
                            n_factors=n_factors,
                            n_epochs=n_epochs,
                            lr_all=lr_all,
                            reg_all=reg_all,
                            random_state=RANDOM_SEED,
                            show_tables=show_tables
                        )
                        st.session_state.svd = svd
                        
                        hybrid = train_hybrid_model_with_save(
                            svd, full_gmm_map, trainset, testset, 
                            output_key,
                            show_tables=show_tables
                        )
                        st.session_state.hybrid = hybrid
                        
                        if len(testset) > 0:
                            st.markdown("---")
                            st.header("EVALUASI PADA TEST SET")
                            
                            svd_errors = []
                            hybrid_errors = []
                            
                            progress_bar = st.progress(0)
                            total_test = len(testset)
                            
                            svd_data = []
                            hybrid_data = []
                            comparison_data = []
                            
                            for idx, (uid, iid, true_r) in enumerate(testset):
                                if idx % 100 == 0:
                                    progress_bar.progress(min(idx / total_test, 1.0))
                                
                                svd_pred = svd.predict(uid, iid).est
                                svd_error = abs(svd_pred - true_r)
                                svd_errors.append(svd_error)
                                
                                hybrid_pred, gmm_used = hybrid.predict(uid, iid)
                                hybrid_error = abs(hybrid_pred - true_r)
                                hybrid_errors.append(hybrid_error)
                                
                                try:
                                    u_inner = trainset.to_inner_uid(str(uid))
                                    n_user_ratings = len(trainset.ur[u_inner])
                                except:
                                    n_user_ratings = 0
                                
                                try:
                                    movie_id_int = int(iid)
                                    movie_info = st.session_state.movies[st.session_state.movies['movieId'] == movie_id_int]
                                    title = movie_info['title'].values[0] if not movie_info.empty else f"Movie {iid}"
                                    genres = movie_info['genres'].values[0] if not movie_info.empty else "Unknown"
                                except:
                                    title = f"Movie {iid}"
                                    genres = "Unknown"
                                
                                try:
                                    u_inner = trainset.to_inner_uid(str(uid))
                                    user_ratings_list = trainset.ur[u_inner]
                                    user_item_ids = [trainset.to_raw_iid(i) for (i, _) in user_ratings_list]
                                    user_ratings_values = [r for (_, r) in user_ratings_list]
                                    user_mean = np.mean(user_ratings_values) if user_ratings_values else 3.0
                                    user_std = np.std(user_ratings_values) if len(user_ratings_values) > 1 else 0.5
                                    timestamps = [i for i in range(len(user_ratings_list))]
                                    
                                    user_embedding_norm = None
                                    if hasattr(svd, 'pu') and u_inner < len(svd.pu):
                                        user_vec = svd.pu[u_inner]
                                        if user_vec is not None:
                                            user_embedding_norm = np.linalg.norm(user_vec)
                                    
                                    genre_diversity = calculate_genre_diversity(uid, trainset, 
                                                                                st.session_state.movies, 
                                                                                genre_cols)
                                    rating_entropy = calculate_rating_entropy(uid, trainset)
                                    
                                    gmm_confidence = 0.5
                                    if hasattr(hybrid, 'user_pref') and u_inner < len(hybrid.user_pref):
                                        user_pref_vec = hybrid.user_pref[u_inner]
                                        if user_pref_vec is not None:
                                            p = np.abs(user_pref_vec)
                                            p_sum = np.sum(p)
                                            if p_sum > 0:
                                                p = p / p_sum
                                            p_safe = p + 1e-10
                                            entropy = -np.sum(p_safe * np.log(p_safe))
                                            max_entropy = np.log(len(p))
                                            gmm_confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
                                    
                                    is_cold_start = n_user_ratings < 5
                                    
                                    weights = calculate_adaptive_weights(
                                        user_data=uid,
                                        trainset=trainset,
                                        n_ratings=n_user_ratings,
                                        user_rating_mean=user_mean,
                                        user_rating_std=user_std,
                                        user_embedding_norm=user_embedding_norm,
                                        gmm_confidence=gmm_confidence,
                                        genre_diversity=genre_diversity,
                                        rating_entropy=rating_entropy,
                                        movies_df=st.session_state.movies,
                                        genre_cols=genre_cols,
                                        ratings_df=st.session_state.ratings,
                                        svd_model=svd,
                                        hybrid_model=hybrid,
                                        is_cold_start=is_cold_start
                                    )
                                    alpha = weights['cf_weight']
                                    
                                except Exception as e:
                                    print(f"Error calculating adaptive weights for user {uid}: {e}")
                                    weights = calculate_adaptive_weights(
                                        user_data=uid,
                                        trainset=trainset,
                                        n_ratings=n_user_ratings
                                    )
                                    alpha = weights['cf_weight']
                                
                                svd_data.append({
                                    'No': idx + 1,
                                    'User ID': uid,
                                    'Film': title[:50] + '...' if len(title) > 50 else title,
                                    'Genre': genres[:30] + '...' if len(genres) > 30 else genres,
                                    'Jml Rating': n_user_ratings,
                                    'Actual': f"{true_r:.1f}",
                                    'Prediksi SVD': f"{svd_pred:.3f}",
                                    'Error': f"{svd_error:.3f}"
                                })
                                
                                gmm_cluster = None
                                gmm_prob = None
                                iid_str = str(iid)
                                if iid_str in full_gmm_map:
                                    gmm_vector = full_gmm_map[iid_str]
                                    gmm_cluster = np.argmax(gmm_vector)
                                    gmm_prob = gmm_vector[gmm_cluster]
                                
                                hybrid_data.append({
                                    'No': idx + 1,
                                    'User ID': uid,
                                    'Film': title[:50] + '...' if len(title) > 50 else title,
                                    'Genre': genres[:30] + '...' if len(genres) > 30 else genres,
                                    'Jml Rating': n_user_ratings,
                                    'α (Rule-based)': f"{alpha:.4f}",
                                    'GMM Cluster': gmm_cluster if gmm_cluster is not None else '-',
                                    'GMM Prob': f"{gmm_prob:.4f}" if gmm_prob is not None else '-',
                                    'Actual': f"{true_r:.1f}",
                                    'Prediksi Hybrid': f"{hybrid_pred:.3f}",
                                    'Error': f"{hybrid_error:.3f}"
                                })
                                
                                error_diff = svd_error - hybrid_error
                                improvement = (error_diff / svd_error) * 100 if svd_error > 0 else 0
                                
                                if svd_error < hybrid_error:
                                    winner = "SVD"
                                elif hybrid_error < svd_error:
                                    winner = "Hybrid"
                                else:
                                    winner = "Seri"
                                
                                comparison_data.append({
                                    'No': idx + 1,
                                    'User ID': uid,
                                    'Jml Rating': n_user_ratings,
                                    'α (Rule-based)': f"{alpha:.4f}",
                                    'Actual': f"{true_r:.1f}",
                                    'SVD Pred': f"{svd_pred:.3f}",
                                    'SVD Error': f"{svd_error:.3f}",
                                    'Hybrid Pred': f"{hybrid_pred:.3f}",
                                    'Hybrid Error': f"{hybrid_error:.3f}",
                                    'Δ Error': f"{error_diff:+.3f}",
                                    'Improvement': f"{improvement:+.1f}%",
                                    'Winner': winner
                                })
                            
                            progress_bar.progress(1.0)
                            
                            svd_rmse = np.sqrt(np.mean(np.array(svd_errors)**2))
                            svd_mae = np.mean(svd_errors)
                            hybrid_rmse = np.sqrt(np.mean(np.array(hybrid_errors)**2))
                            hybrid_mae = np.mean(hybrid_errors)
                            
                            st.session_state.svd_rmse = svd_rmse
                            st.session_state.svd_mae = svd_mae
                            st.session_state.hybrid_rmse = hybrid_rmse
                            st.session_state.hybrid_mae = hybrid_mae
                            
                            st.session_state.svd_predictions_df = pd.DataFrame(svd_data)
                            st.session_state.hybrid_predictions_df = pd.DataFrame(hybrid_data)
                            st.session_state.comparison_df = pd.DataFrame(comparison_data)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("SVD RMSE", f"{svd_rmse:.6f}")
                            with col2:
                                st.metric("SVD MAE", f"{svd_mae:.6f}")
                            with col3:
                                st.metric("Hybrid RMSE", f"{hybrid_rmse:.6f}")
                            with col4:
                                st.metric("Hybrid MAE", f"{hybrid_mae:.6f}")
                            
                            rmse_imp = ((svd_rmse - hybrid_rmse) / svd_rmse) * 100
                            mae_imp = ((svd_mae - hybrid_mae) / svd_mae) * 100
                            
                            st.success(f"""
                            **Ringkasan Improvement:**
                            - RMSE: {svd_rmse:.6f} → {hybrid_rmse:.6f} ({rmse_imp:+.2f}%)
                            - MAE: {svd_mae:.6f} → {hybrid_mae:.6f} ({mae_imp:+.2f}%)
                            """)
                            
                            st.markdown("---")
                            st.subheader("BOBOT ADAPTIF (RULE-BASED)")
                            st.info("""
                            **Rule-Based Adaptive Weight**
                            
                            Aturan yang digunakan:
                            | Jumlah Rating | Alpha (CF) | Bobot CB |
                            |---------------|------------|----------|
                            | < 5 | 0.20 | 0.80 |
                            | 5 - 9 | 0.30 | 0.70 |
                            | 10 - 19 | 0.35 | 0.65 |
                            | 20 - 49 | 0.40 | 0.60 |
                            | 50 - 99 | 0.45 | 0.55 |
                            | ≥ 100 | 0.50 | 0.50 |
                            """)
                            
                            st.subheader("Hasil Prediksi SVD pada Test Set")
                            st.dataframe(st.session_state.svd_predictions_df, use_container_width=True, height=400)
                            
                            st.subheader("Hasil Prediksi Hybrid pada Test Set")
                            st.dataframe(st.session_state.hybrid_predictions_df, use_container_width=True, height=400)
                            
                            st.subheader("Perbandingan SVD vs Hybrid")
                            st.dataframe(st.session_state.comparison_df, use_container_width=True, height=400)
                        
                        st.session_state.model_trained = True
                        st.session_state.tab2_processed = True

                        saved_count, saved_items = auto_save_all_components()
                        st.success(f"Model berhasil disimpan! {saved_count} komponen tersimpan di {MODEL_DIR}")

                        with st.expander("Detail file yang tersimpan"):
                            for item in saved_items:
                                st.write(f"- {item}")

                        if st.session_state.get('model_loaded_from_file', False):
                            st.info("Model dimuat dari file. Klik tombol di bawah untuk menampilkan hasil evaluasi.")
                            
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                if st.button("Tampilkan Hasil Evaluasi Model", type="primary", use_container_width=True):
                                    if reevaluate_and_display_models():
                                        st.rerun()

                        st.markdown("---")
                        st.subheader("SIMPAN MODEL")
                        st.info("Simpan model yang telah di-train ke dalam folder untuk digunakan di sesi berikutnya")
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if st.button("Simpan Model ke File", type="primary", use_container_width=True):
                                with st.spinner("Menyimpan model..."):
                                    saved_files = save_all_models(MODEL_DIR)
                                    if saved_files:
                                        st.success(f"Berhasil menyimpan {len(saved_files)} file model ke folder '{MODEL_DIR}'")
                                        
                                        with st.expander("Detail file yang tersimpan"):
                                            for name, path in saved_files:
                                                st.write(f"- **{name}**: `{path}`")
                                        
                                        st.balloons()
                                    else:
                                        st.error("Gagal menyimpan model")
                        
                        st.success("""
                        **TRAINING SELESAI!** 
                        
                        Semua hasil telah tersimpan permanen dan akan selalu tampil.
                        """)
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.error(traceback.format_exc())

    with tab4:
        st.header("Cross-Validation: SVD vs Hybrid GMM+SVD")
        st.markdown("""
        Melakukan cross-validation untuk membandingkan performa SVD dan Hybrid GMM+SVD.
        
        **Data yang digunakan:** Hasil split dari Tab 1 (hanya data TRAINING)
        """)
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Load data terlebih dahulu di tab Data Analysis")
        elif not st.session_state.get('split_done', False):
            st.warning("Lakukan Warm-Start Split terlebih dahulu di tab Data Analysis")
        else:
            st.info(f"""
            **Data Training yang akan digunakan untuk CV:** {len(st.session_state.train_df):,} rating dari {len(st.session_state.train_movies):,} film
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                n_folds = st.slider("Jumlah Folds:", 3, 10, 5, key="cv_folds")
            with col2:
                cv_seed = st.number_input("Random Seed:", 0, 100, 42, key="cv_seed")
            
            svd_params = {
                'n_factors': n_factors,
                'n_epochs': n_epochs,
                'lr_all': lr_all,
                'reg_all': reg_all
            }
            
            gmm_params = {
                'max_components': max_components,
                'smoothing': smoothing
            }
            
            st.info(f"""
            **Parameter Cross-Validation:**
            - Folds: {n_folds}
            - SVD: n_factors={n_factors}, n_epochs={n_epochs}
            - GMM: max_components={max_components}, smoothing={smoothing}
            """)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                cv_button = st.button("Jalankan Cross-Validation", type="primary")
            
            output_key = "tab3_output"
            if output_key in st.session_state.saved_outputs:
                st.markdown("### Output Proses Cross-Validation")
                display_saved_outputs(output_key)
            
            
            
            if st.session_state.tab3_processed and st.session_state.tab3_results:
                st.success("Cross-validation sudah dijalankan sebelumnya")
                
                results = st.session_state.tab3_results
                
                if 'fold_results' in results:
                    st.markdown("### Hasil per Fold")
                    display_df = results['fold_results'].copy()
                    st.dataframe(display_df, use_container_width=True)
                
                if 'summary' in results:
                    # ========== TAMBAHKAN BARIS INI ==========
                    summary = results['summary']  # <-- AMBIL SUMMARY DARI RESULTS
                    # =======================================
                    
                    svd_rmse_mean = summary.get('svd_rmse_mean', 0)
                    svd_rmse_std = summary.get('svd_rmse_std', 0)
                    svd_mae_mean = summary.get('svd_mae_mean', 0)
                    svd_mae_std = summary.get('svd_mae_std', 0)
                    
                    hybrid_rmse_mean = summary.get('hybrid_rmse_mean', 0)
                    hybrid_rmse_std = summary.get('hybrid_rmse_std', 0)
                    hybrid_mae_mean = summary.get('hybrid_mae_mean', 0)
                    hybrid_mae_std = summary.get('hybrid_mae_std', 0)
                    
                    rmse_improvement = summary.get('rmse_improvement', 0)
                    mae_improvement = summary.get('mae_improvement', 0)
                    
                    # Ambil p-value dari summary (sekarang summary sudah terdefinisi)
                    if 'paired_ttest_rmse' in summary:
                        p_value_rmse = summary['paired_ttest_rmse'].get('p_value', 1.0)
                    else:
                        p_value_rmse = 1.0
                    
                    if 'paired_ttest_mae' in summary:
                        p_value_mae = summary['paired_ttest_mae'].get('p_value', 1.0)  # <-- SEKARANG AMAN
                    else:
                        # Fallback: hitung dari fold_results
                        if 'fold_results' in results and 'svd_mae' in results['fold_results'].columns:
                            from scipy import stats
                            svd_mae_list = results['fold_results']['svd_mae'].tolist()
                            hybrid_mae_list = results['fold_results']['hybrid_mae'].tolist()
                            _, p_value_mae = stats.ttest_rel(svd_mae_list, hybrid_mae_list)
                        else:
                            p_value_mae = 1.0
                    
                    if 'paired_ttest_mae' in summary:
                        p_value_mae = summary['paired_ttest_mae'].get('p_value', 1.0)
                    else:
                        # Jika tidak ada, hitung dari fold_results jika tersedia
                        if 'fold_results' in results and 'svd_mae' in results['fold_results'].columns:
                            from scipy import stats
                            svd_mae_list = results['fold_results']['svd_mae'].tolist()
                            hybrid_mae_list = results['fold_results']['hybrid_mae'].tolist()
                            _, p_value_mae = stats.ttest_rel(svd_mae_list, hybrid_mae_list)
                        else:
                            p_value_mae = 1.0
                    
                    st.markdown("### Ringkasan Hasil Cross-Validation")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("SVD RMSE", f"{svd_rmse_mean:.6f} ± {svd_rmse_std:.6f}")
                        st.metric("SVD MAE", f"{svd_mae_mean:.6f} ± {svd_mae_std:.6f}")
                    with col2:
                        st.metric("Hybrid RMSE", f"{hybrid_rmse_mean:.6f} ± {hybrid_rmse_std:.6f}",
                                delta=f"{rmse_improvement:+.2f}%")
                        st.metric("Hybrid MAE", f"{hybrid_mae_mean:.6f} ± {hybrid_mae_std:.6f}",
                                delta=f"{mae_improvement:+.2f}%")
                    with col3:
                        # Tampilkan p-value MAE
                        st.metric("p-value (MAE)", format_p_value(p_value_mae))
                        if p_value_mae < 0.05:
                            st.success("Signifikan (p < 0.05)")
                        else:
                            st.warning("Tidak signifikan (p ≥ 0.05)")
                        
                        
                        # Tampilkan p-value RMSE
                        st.metric("p-value (RMSE)", format_p_value(p_value_rmse))
                        if p_value_rmse < 0.05:
                            st.success("Signifikan (p < 0.05)")
                        else:
                            st.warning("Tidak signifikan (p ≥ 0.05)")
                auto_display_cv_plots()
            
            if cv_button:
                st.session_state.tab3_processed = False
                if 'tab3_results' in st.session_state:
                    del st.session_state.tab3_results

                output_key = "tab3_output"
                if output_key in st.session_state.saved_outputs:
                    del st.session_state.saved_outputs[output_key]
                
                with st.spinner(f"Running {n_folds}-fold CV dengan calculate_adaptive_weights()..."):
                    clear_saved_outputs(output_key)
                    
                    try:
                        train_df = st.session_state.train_df
                        movies_df = st.session_state.movies
                        
                        st.info(f"Menjalankan {n_folds}-fold CV pada {len(train_df):,} rating...")
                        
                        movies_with_ratings = set(train_df['movieId'].unique())
                        valid_movies = list(movies_with_ratings)
                        np.random.seed(cv_seed)
                        np.random.shuffle(valid_movies)
                        
                        movies_per_fold = len(valid_movies) // n_folds
                        movie_folds = []
                        
                        for i in range(n_folds):
                            start_idx = i * movies_per_fold
                            end_idx = (i + 1) * movies_per_fold if i < n_folds - 1 else len(valid_movies)
                            fold_movies = set(valid_movies[start_idx:end_idx])
                            movie_folds.append(fold_movies)
                        
                        fold_results = []
                        progress_bar = st.progress(0)
                        
                        for fold in range(n_folds):
                            progress_bar.progress((fold) / n_folds)
                            
                            st.write(f"**Fold {fold+1}/{n_folds}**")
                            
                            test_movie_ids = movie_folds[fold]
                            train_movie_ids = set()
                            for i in range(len(movie_folds)):
                                if i != fold:
                                    train_movie_ids.update(movie_folds[i])
                            
                            fold_train_df = train_df[train_df['movieId'].isin(train_movie_ids)].copy()
                            fold_test_df = train_df[train_df['movieId'].isin(test_movie_ids)].copy()
                            
                            if len(fold_train_df) == 0 or len(fold_test_df) == 0:
                                st.warning(f"Fold {fold+1} skipped - insufficient data")
                                continue
                            
                            fold_movies_enc, feature_cols, genre_cols = one_hot_encode_genres_with_save(
                                movies_df, train_movie_ids, f"cv_fold_{fold}", 
                                smoothing_factor=smoothing, show_tables=False
                            )
                            
                            gmm_map, gmm_model = gmm_soft_clustering_with_save(
                                fold_movies_enc, feature_cols, f"cv_fold_{fold}_gmm",
                                max_components=max_components, random_state=cv_seed + fold, show_tables=False
                            )
                            
                            test_gmm_map = transform_test_movies_with_gmm(
                                movies_df, test_movie_ids, gmm_model, feature_cols, smoothing_factor=smoothing
                            )
                            full_gmm_map = {**gmm_map, **test_gmm_map}
                            
                            reader = Reader(rating_scale=(0.5, 5.0))
                            fold_train_df['userId'] = fold_train_df['userId'].astype(str)
                            fold_train_df['movieId'] = fold_train_df['movieId'].astype(str)
                            data = Dataset.load_from_df(fold_train_df[['userId', 'movieId', 'rating']], reader)
                            trainset_cv = data.build_full_trainset()
                            
                            testset_cv = []
                            fold_test_df['userId'] = fold_test_df['userId'].astype(str)
                            fold_test_df['movieId'] = fold_test_df['movieId'].astype(str)
                            for _, row in fold_test_df.iterrows():
                                testset_cv.append((str(row['userId']), str(row['movieId']), float(row['rating'])))
                            
                            svd_cv = SVD(
                                n_factors=n_factors, n_epochs=n_epochs,
                                lr_all=lr_all, reg_all=reg_all, random_state=cv_seed + fold, verbose=False
                            )
                            svd_cv.fit(trainset_cv)
                            
                            hybrid_cv = HybridGMM_SVD(svd_cv, full_gmm_map)
                            hybrid_cv.fit(trainset_cv, show_tables=False)
                            
                            svd_errors = []
                            hybrid_errors = []
                            gmm_used_count = 0
                            
                            for uid, iid, true_r in testset_cv:
                                svd_pred = svd_cv.predict(uid, iid).est
                                svd_errors.append(abs(svd_pred - true_r))
                                
                                try:
                                    u_inner = trainset_cv.to_inner_uid(str(uid))
                                    n_ratings_user = len(trainset_cv.ur[u_inner])
                                    
                                    if n_ratings_user < 5:
                                        alpha = 0.20
                                    elif n_ratings_user < 10:
                                        alpha = 0.30
                                    elif n_ratings_user < 20:
                                        alpha = 0.35
                                    elif n_ratings_user < 50:
                                        alpha = 0.40
                                    elif n_ratings_user < 100:
                                        alpha = 0.45
                                    else:
                                        alpha = 0.50
                                except:
                                    alpha = 0.50
                                
                                hybrid_pred, used_gmm = hybrid_cv.predict(uid, iid, alpha=alpha)  
                                hybrid_errors.append(abs(hybrid_pred - true_r))
                                if used_gmm:
                                    gmm_used_count += 1
                            
                            svd_rmse = np.sqrt(np.mean(np.array(svd_errors)**2))
                            svd_mae = np.mean(svd_errors)
                            hybrid_rmse = np.sqrt(np.mean(np.array(hybrid_errors)**2))
                            hybrid_mae = np.mean(hybrid_errors)
                            
                            fold_results.append({
                                'fold': fold + 1,
                                'svd_rmse': svd_rmse,
                                'svd_mae': svd_mae,
                                'hybrid_rmse': hybrid_rmse,
                                'hybrid_mae': hybrid_mae,
                                'gmm_used_pct': (gmm_used_count / len(testset_cv)) * 100 if len(testset_cv) > 0 else 0,
                                'train_ratings': len(fold_train_df),
                                'test_ratings': len(fold_test_df)
                            })
                            
                            st.write(f" SVD RMSE: {svd_rmse:.6f} | Hybrid RMSE: {hybrid_rmse:.6f}")
                        
                        progress_bar.progress(1.0)
                        
                        if len(fold_results) == 0:
                            st.error("Tidak ada hasil fold yang berhasil!")
                        else:
                            results_df = pd.DataFrame(fold_results)
            
                            svd_rmse_mean = results_df['svd_rmse'].mean()
                            svd_rmse_std = results_df['svd_rmse'].std()
                            hybrid_rmse_mean = results_df['hybrid_rmse'].mean()
                            hybrid_rmse_std = results_df['hybrid_rmse'].std()
                            rmse_improvement = ((svd_rmse_mean - hybrid_rmse_mean) / svd_rmse_mean * 100) if svd_rmse_mean > 0 else 0
                            
                            svd_mae_mean = results_df['svd_mae'].mean()
                            hybrid_mae_mean = results_df['hybrid_mae'].mean()
                            mae_improvement = ((svd_mae_mean - hybrid_mae_mean) / svd_mae_mean * 100) if svd_mae_mean > 0 else 0
                            
                            from scipy import stats
                            t_stat_rmse, p_value_rmse = stats.ttest_rel(results_df['svd_rmse'].tolist(), results_df['hybrid_rmse'].tolist())
                            t_stat_mae, p_value_mae = stats.ttest_rel(results_df['svd_mae'].tolist(), results_df['hybrid_mae'].tolist())
                            
                            # ========== BUAT VARIABLE summary ==========
                            summary = {
                                'svd_rmse_mean': svd_rmse_mean,
                                'svd_rmse_std': svd_rmse_std,
                                'svd_mae_mean': svd_mae_mean,
                                'svd_mae_std': results_df['svd_mae'].std(),
                                'hybrid_rmse_mean': hybrid_rmse_mean,
                                'hybrid_rmse_std': hybrid_rmse_std,
                                'hybrid_mae_mean': hybrid_mae_mean,
                                'hybrid_mae_std': results_df['hybrid_mae'].std(),
                                'rmse_improvement': rmse_improvement,
                                'mae_improvement': mae_improvement,
                                'n_folds': len(fold_results),
                                'paired_ttest_rmse': {'t_statistic': t_stat_rmse, 'p_value': p_value_rmse},
                                'paired_ttest_mae': {'t_statistic': t_stat_mae, 'p_value': p_value_mae}
                            }
                            # =========================================
                            
                            st.markdown("### Ringkasan Hasil Cross-Validation")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("SVD RMSE", f"{svd_rmse_mean:.6f} ± {svd_rmse_std:.6f}")
                                st.metric("SVD MAE", f"{svd_mae_mean:.6f}")
                            with col2:
                                st.metric("Hybrid RMSE", f"{hybrid_rmse_mean:.6f} ± {hybrid_rmse_std:.6f}",
                                        delta=f"{rmse_improvement:+.2f}%")
                                st.metric("Hybrid MAE", f"{hybrid_mae_mean:.6f}",
                                        delta=f"{mae_improvement:+.2f}%")
                            with col3:
                                # ========== SEKARANG summary sudah terdefinisi ==========
                                p_value_mae_display = summary.get('paired_ttest_mae', {}).get('p_value', None)
                                if p_value_mae_display is not None:
                                    st.metric("p-value (MAE)", format_p_value(p_value_mae_display))
                                else:
                                    st.metric("p-value (MAE)", "N/A")
                                # ========================================================
                                
                                if p_value_rmse < 0.05:
                                    st.success("Signifikan (p < 0.05)")
                                else:
                                    st.warning("Tidak signifikan (p ≥ 0.05)")
                                
                                st.metric("p-value (RMSE)", format_p_value(p_value_rmse))
                            
                            st.markdown("### Hasil per Fold")
                            st.dataframe(results_df, use_container_width=True)

                            st.markdown("---")
                            st.subheader("Visualisasi Hasil Cross-Validation")
                            
                            st.session_state.tab3_results = {
                                'fold_results': results_df,
                                'summary': summary
                            }
                            st.session_state.tab3_processed = True

                            saved_count, saved_items = auto_save_all_components()
                            st.success(f"Hasil Cross-Validation disimpan! {saved_count} komponen tersimpan.")

                            auto_display_cv_plots()

                            st.success(f"Cross-validation selesai! Improvement RMSE: {rmse_improvement:+.2f}%, MAE: {mae_improvement:+.2f}%")

                    
                    
                    except Exception as e:
                        st.error(f"Error saat menjalankan CV: {e}")
                        import traceback
                        st.error(traceback.format_exc())
    
    with tab5:
        st.header("Rekomendasi Personal dengan Bobot Adaptif")
        st.markdown("""
        Mendapatkan rekomendasi film untuk user tertentu menggunakan model yang sudah di-train.
        
        **Model yang digunakan:** Hasil training dari Tab 3
        """)
        
        if not st.session_state.get('model_trained', False):
            st.warning("Train model terlebih dahulu di tab Model Training")
        else:
            max_candidates = 2000
            items_per_page = 20
            
            current_user_id = None
            user_fav_genres = []
            user_adaptive_weights = None
            registered_data = None
            found_in_trainset = None

            if st.session_state.model_loaded_from_file:
                if st.session_state.get('gmm_map') is None:
                    st.error("GMM Map tidak ditemukan! Model tidak lengkap. Silakan training ulang model.")
                    st.stop()
                
                if st.session_state.hybrid is not None:
                    if not hasattr(st.session_state.hybrid, 'user_pref') or st.session_state.hybrid.user_pref is None:
                        st.warning("Hybrid model belum memiliki user_pref. Melakukan fitting ulang...")
                        if st.session_state.trainset is not None:
                            st.session_state.hybrid.fit(st.session_state.trainset, show_tables=False)
                            st.success("Hybrid model berhasil di-rebuild!")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("Input User")
                
                new_user_button()
                
                st.markdown("---")
                
                if st.session_state.show_new_user:
                    st.info("Pendaftaran User Baru dengan Bobot Adaptif")
                    create_new_user_interface(
                        st.session_state.movies,
                        st.session_state.genre_cols
                    )
                    
                    if st.session_state.new_user_ready:
                        try:
                            selected_fav_genres = st.session_state.get('selected_fav_genres', [])
                            
                            result = process_new_user_flow_with_adaptive_weights(
                                st.session_state.new_user_ratings,
                                st.session_state.movies,
                                st.session_state.hybrid,
                                st.session_state.trainset,
                                st.session_state.svd,
                                st.session_state.ratings,
                                st.session_state.genre_cols,
                                cluster_genre_mapping=st.session_state.cluster_genre_mapping,
                                show_tables=True
                            )
                            st.session_state.new_user_result = result
                            
                            if selected_fav_genres:
                                st.session_state.current_fav_genres = selected_fav_genres
                            elif result.get('fav_genres'):
                                st.session_state.current_fav_genres = result['fav_genres']
                            
                            st.session_state.new_user_ready = False
                            st.session_state.new_user_initialized = False
                            st.session_state.show_new_user = False
                            st.session_state.temp_genres = []
                            
                            st.success(f"User baru {result['user_id']} BERHASIL DIINTEGRASIKAN ke model!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                st.markdown("---")
                
                st.subheader("Pencarian User")
                
                if 'trainset' in st.session_state:
                    rating_counts = {
                        '0': 0, '1-2': 0, '3-4': 0, '5-9': 0, 
                        '10-19': 0, '20-49': 0, '50+': 0
                    }
                    
                    rating_examples = {
                        '0': [], '1-2': [], '3-4': [], 
                        '5-9': [], '10-19': [], '20-49': [], '50+': []
                    }
                    
                    for uid in st.session_state.trainset.all_users():
                        raw_uid = st.session_state.trainset.to_raw_uid(uid)
                        n_rat = len(st.session_state.trainset.ur[uid])
                        
                        if n_rat == 0:
                            rating_counts['0'] += 1
                            if len(rating_examples['0']) < 2:
                                rating_examples['0'].append((raw_uid, n_rat))
                        elif n_rat < 3:
                            rating_counts['1-2'] += 1
                            if len(rating_examples['1-2']) < 2:
                                rating_examples['1-2'].append((raw_uid, n_rat))
                        elif n_rat < 5:
                            rating_counts['3-4'] += 1
                            if len(rating_examples['3-4']) < 2:
                                rating_examples['3-4'].append((raw_uid, n_rat))
                        elif n_rat < 10:
                            rating_counts['5-9'] += 1
                            if len(rating_examples['5-9']) < 2:
                                rating_examples['5-9'].append((raw_uid, n_rat))
                        elif n_rat < 20:
                            rating_counts['10-19'] += 1
                            if len(rating_examples['10-19']) < 2:
                                rating_examples['10-19'].append((raw_uid, n_rat))
                        elif n_rat < 50:
                            rating_counts['20-49'] += 1
                            if len(rating_examples['20-49']) < 2:
                                rating_examples['20-49'].append((raw_uid, n_rat))
                        else:
                            rating_counts['50+'] += 1
                            if len(rating_examples['50+']) < 2:
                                rating_examples['50+'].append((raw_uid, n_rat))
                    
                    registered_counts = {'0': 0, '1-2': 0, '3-4': 0, '5-9': 0, '10-19': 0, '20-49': 0, '50+': 0}
                    
                    for uid, data in st.session_state.registered_users.items():
                        n_rat = data.get('n_ratings', 0)
                        
                        if n_rat == 0:
                            registered_counts['0'] += 1
                        elif n_rat < 3:
                            registered_counts['1-2'] += 1
                        elif n_rat < 5:
                            registered_counts['3-4'] += 1
                        elif n_rat < 10:
                            registered_counts['5-9'] += 1
                        elif n_rat < 20:
                            registered_counts['10-19'] += 1
                        elif n_rat < 50:
                            registered_counts['20-49'] += 1
                        else:
                            registered_counts['50+'] += 1
                    
                    total_counts = {k: rating_counts[k] + registered_counts[k] for k in rating_counts.keys()}
                    
                    with st.expander("Statistik User", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("0 rating", f"{total_counts['0']} user")
                            if rating_examples['0']:
                                examples = rating_examples['0'][:2]
                                examples_str = ', '.join([f'{u}({n}r)' for u,n in examples])
                                st.caption(f"Contoh: {examples_str}")
                            
                            st.metric("1-2 rating", f"{total_counts['1-2']} user")
                            if rating_examples['1-2']:
                                examples = rating_examples['1-2'][:2]
                                examples_str = ', '.join([f'{u}({n}r)' for u,n in examples])
                                st.caption(f"Contoh: {examples_str}")
                            
                            st.metric("3-4 rating", f"{total_counts['3-4']} user")
                            if rating_examples['3-4']:
                                examples = rating_examples['3-4'][:2]
                                examples_str = ', '.join([f'{u}({n}r)' for u,n in examples])
                                st.caption(f"Contoh: {examples_str}")
                        
                        with col2:
                            st.metric("5-9 rating", f"{total_counts['5-9']} user")
                            if rating_examples['5-9']:
                                examples = rating_examples['5-9'][:2]
                                examples_str = ', '.join([f'{u}({n}r)' for u,n in examples])
                                st.caption(f"Contoh: {examples_str}")
                            
                            st.metric("10-19 rating", f"{total_counts['10-19']} user")
                            if rating_examples['10-19']:
                                examples = rating_examples['10-19'][:2]
                                examples_str = ', '.join([f'{u}({n}r)' for u,n in examples])
                                st.caption(f"Contoh: {examples_str}")
                        
                        with col3:
                            st.metric("20-49 rating", f"{total_counts['20-49']} user")
                            if rating_examples['20-49']:
                                examples = rating_examples['20-49'][:2]
                                examples_str = ', '.join([f'{u}({n}r)' for u,n in examples])
                                st.caption(f"Contoh: {examples_str}")
                            
                            st.metric(">=50 rating", f"{total_counts['50+']} user")
                            if rating_examples['50+']:
                                examples = rating_examples['50+'][:2]
                                examples_str = ', '.join([f'{u}({n}r)' for u,n in examples])
                                st.caption(f"Contoh: {examples_str}")
                        
                        if st.session_state.registered_users:
                            integrated_count = sum(1 for d in st.session_state.registered_users.values() if d.get('integrated', False))
                            st.info(f"{len(st.session_state.registered_users)} user baru (terintegrasi: {integrated_count})")
                    
                    input_user_id = st.number_input("User ID:", min_value=1, value=611, key="rec_user")
                    
                    found_in_trainset = find_user_in_trainset(st.session_state.trainset, input_user_id)
                    registered_data = find_user_in_registered(input_user_id)
                    
                    if found_in_trainset:
                        st.success(f"User {input_user_id} ditemukan di DATASET ASLI")
                        current_user_id = found_in_trainset
                        
                        try:
                            u_inner = st.session_state.trainset.to_inner_uid(str(current_user_id))
                            n_rat = len(st.session_state.trainset.ur[u_inner])
                            user_ratings_values = [r for (_, r) in st.session_state.trainset.ur[u_inner]]
                            user_mean = np.mean(user_ratings_values) if user_ratings_values else 3.0
                            user_std = np.std(user_ratings_values) if len(user_ratings_values) > 1 else 0.5
                            
                            is_cold_start = n_rat < 5
                            if is_cold_start:
                                st.warning(f"User {current_user_id} memiliki rating sangat sedikit ({n_rat} rating) - akan diproses sebagai COLD START")
                            
                            user_embedding_norm = None
                            if hasattr(st.session_state.svd, 'pu') and u_inner < len(st.session_state.svd.pu):
                                user_vec = st.session_state.svd.pu[u_inner]
                                if user_vec is not None:
                                    user_embedding_norm = np.linalg.norm(user_vec)
                            
                            genre_diversity = calculate_genre_diversity(current_user_id, st.session_state.trainset, 
                                                                    st.session_state.movies, st.session_state.genre_cols)
                            rating_entropy = calculate_rating_entropy(current_user_id, st.session_state.trainset)
                            
                            gmm_confidence = 0.5
                            if hasattr(st.session_state.hybrid, 'user_pref') and u_inner < len(st.session_state.hybrid.user_pref):
                                user_pref_vec = st.session_state.hybrid.user_pref[u_inner]
                                if user_pref_vec is not None:
                                    p = np.abs(user_pref_vec)
                                    p_sum = np.sum(p)
                                    if p_sum > 0:
                                        p = p / p_sum
                                    p_safe = p + 1e-10
                                    entropy = -np.sum(p_safe * np.log(p_safe))
                                    max_entropy = np.log(len(p))
                                    gmm_confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
                            
                            user_adaptive_weights = calculate_adaptive_weights(
                                user_data=current_user_id,
                                trainset=st.session_state.trainset,
                                n_ratings=n_rat,
                                user_rating_mean=user_mean,
                                user_rating_std=user_std,
                                user_embedding_norm=user_embedding_norm,
                                gmm_confidence=gmm_confidence,
                                genre_diversity=genre_diversity,
                                rating_entropy=rating_entropy,
                                movies_df=st.session_state.movies,
                                genre_cols=st.session_state.genre_cols,
                                ratings_df=st.session_state.ratings,
                                svd_model=st.session_state.svd,
                                hybrid_model=st.session_state.hybrid,
                                is_cold_start=is_cold_start
                            )
                        except Exception as e:
                            st.error(f"Error menghitung bobot adaptif: {e}")
                            user_adaptive_weights = calculate_adaptive_weights(
                                user_data=current_user_id,
                                trainset=st.session_state.trainset
                            )
                            st.info(f"User {current_user_id} | alpha={user_adaptive_weights['cf_weight']:.4f} (fallback)")
                        
                        with st.expander("Detail Bobot Adaptif untuk User Ini"):
                            st.json(user_adaptive_weights)
                        
                    elif registered_data:
                        st.success(f"User {input_user_id} ditemukan di REGISTERED USERS")
                        current_user_id = str(input_user_id)
                        user_data_details = registered_data
                        
                        n_rat = user_data_details.get('n_ratings', 0)
                        if n_rat < 5:
                            st.warning(f"User registered memiliki {n_rat} rating - diproses sebagai COLD START")
                        
                        st.info(f"Jumlah rating: {n_rat}")
                        
                        with st.expander("Detail User Registered"):
                            st.json(user_data_details)
                        
                        user_adaptive_weights = user_data_details.get('adaptive_weights')
                        user_fav_genres = user_data_details.get('fav_genres', [])
                        
                        if user_fav_genres:
                            st.session_state.current_fav_genres = user_fav_genres    
                    else:
                        st.warning(f"User {input_user_id} tidak ditemukan dalam sistem")
                        current_user_id = None
                        user_fav_genres = []

                    if found_in_trainset or registered_data:
                        # TAMPILKAN INFORMASI USER
                        is_in_trainset, is_registered = display_user_info(
                            current_user_id, 
                            st.session_state.trainset, 
                            registered_data,
                            st.session_state.movies,
                            st.session_state.ratings
                        )
                        
                        # TAMPILKAN TABEL RATING USER
                        display_user_ratings_table(
                            current_user_id,
                            st.session_state.trainset,
                            registered_data,
                            st.session_state.movies,
                            st.session_state.ratings,
                            top_n=20
                        )
                        
                        st.markdown("---")
                        
                        # Filter Genre
                        selected_genres = get_genre_selection(
                            st.session_state.movies,
                            st.session_state.genre_cols
                        )
                        
                        # Tombol Generate Rekomendasi
                        if st.button("Generate Rekomendasi dengan Bobot Adaptif", type="primary", key="gen_rec"):
                            set_page_number(current_user_id, 1)
                            
                            st.session_state.current_fav_genres = user_fav_genres
                            
                            if not user_fav_genres and selected_genres:
                                user_fav_genres = selected_genres
                                st.info(f"Menggunakan genre dari filter: {', '.join(user_fav_genres)}")
                                st.session_state.current_fav_genres = user_fav_genres
                            
                            with st.spinner("Menghasilkan rekomendasi dengan Bobot Adaptif..."):
                                hybrid_recs, baseline_recs, _ = get_recommendations_final(
                                    user_id=current_user_id,
                                    user_type='existing',
                                    hybrid_model=st.session_state.hybrid,
                                    svd_model=st.session_state.svd,
                                    trainset=st.session_state.trainset,
                                    movies_df=st.session_state.movies,
                                    ratings_df=st.session_state.ratings,
                                    gmm_map=st.session_state.get('gmm_map', {}),
                                    genre_cols=st.session_state.genre_cols,
                                    cluster_genre_mapping=st.session_state.cluster_genre_mapping,
                                    selected_genres=selected_genres,
                                    top_n=None,
                                    fav_genres=user_fav_genres,
                                    max_candidates=2000,
                                    force_cold_start=True,
                                    user_adaptive_weights=user_adaptive_weights
                                )
                                
                                st.session_state[f'hybrid_recs_{current_user_id}'] = hybrid_recs
                                st.session_state[f'baseline_recs_{current_user_id}'] = baseline_recs
                                st.session_state[f'recs_generated_{current_user_id}'] = True
                                
                                st.success(f"Berhasil menghasilkan {len(hybrid_recs)} rekomendasi dengan Bobot Adaptif!")
                                st.rerun()
                    else:
                        st.info("Masukkan User ID dan klik di luar input box untuk mencari user")
            
            with col_right:
                st.subheader("Hasil")
                
                if current_user_id is not None:
                    if st.session_state.get(f'recs_generated_{current_user_id}', False):
                        hybrid_recs = st.session_state.get(f'hybrid_recs_{current_user_id}', [])
                        baseline_recs = st.session_state.get(f'baseline_recs_{current_user_id}', [])
                        
                        if hybrid_recs and baseline_recs:
                            if user_fav_genres and len(user_fav_genres) > 0:
                                valid_count = 0
                                invalid_movies = []
                                
                                for movie_id, rating in hybrid_recs[:20]:
                                    try:
                                        mid_int = int(movie_id)
                                        movie_info = st.session_state.movies[st.session_state.movies['movieId'] == mid_int]
                                        if not movie_info.empty:
                                            movie_genres = str(movie_info['genres'].values[0]).split('|')
                                            if any(g in movie_genres for g in user_fav_genres):
                                                valid_count += 1
                                            else:
                                                movie_title = movie_info['title'].values[0]
                                                invalid_movies.append(f"{movie_title} ({', '.join(movie_genres)})")
                                    except:
                                        continue
                                
                                if valid_count < 10:
                                    st.error(f"Hanya {valid_count}/20 film teratas yang sesuai genre favorit!")
                                    if invalid_movies:
                                        with st.expander("Film yang tidak sesuai genre:"):
                                            for movie in invalid_movies[:10]:
                                                st.write(f"- {movie}")
                                else:
                                    st.success(f"{valid_count}/20 film teratas sesuai genre favorit!")
                            
                            if user_adaptive_weights:
                                cold_start_tag = " (COLD START MODE)" if user_adaptive_weights.get('is_cold_start', False) else ""
                                st.info(f"User {current_user_id}{cold_start_tag} | alpha={user_adaptive_weights.get('cf_weight', 0.5):.4f}")

                            if user_adaptive_weights is None:
                                try:
                                    u_inner = st.session_state.trainset.to_inner_uid(str(current_user_id))
                                    n_rat = len(st.session_state.trainset.ur[u_inner])
                                    user_ratings_values = [r for (_, r) in st.session_state.trainset.ur[u_inner]]
                                    user_std = np.std(user_ratings_values) if len(user_ratings_values) > 1 else 0.5
                                    user_embedding_norm = None
                                    if hasattr(st.session_state.svd, 'pu') and u_inner < len(st.session_state.svd.pu):
                                        user_vec = st.session_state.svd.pu[u_inner]
                                        if user_vec is not None:
                                            user_embedding_norm = np.linalg.norm(user_vec)
                                    genre_diversity = calculate_genre_diversity(current_user_id, st.session_state.trainset, 
                                                                            st.session_state.movies, st.session_state.genre_cols)
                                    rating_entropy = calculate_rating_entropy(current_user_id, st.session_state.trainset)
                                    gmm_confidence = 0.5
                                    if hasattr(st.session_state.hybrid, 'user_pref') and u_inner < len(st.session_state.hybrid.user_pref):
                                        user_pref_vec = st.session_state.hybrid.user_pref[u_inner]
                                        if user_pref_vec is not None:
                                            p = np.abs(user_pref_vec)
                                            p_sum = np.sum(p)
                                            if p_sum > 0:
                                                p = p / p_sum
                                            p_safe = p + 1e-10
                                            entropy = -np.sum(p_safe * np.log(p_safe))
                                            max_entropy = np.log(len(p))
                                            gmm_confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
                                    is_cold_start = n_rat < 5
                                    
                                    user_adaptive_weights = calculate_adaptive_weights(
                                        user_data=current_user_id,
                                        trainset=st.session_state.trainset,
                                        n_ratings=n_rat,
                                        user_rating_std=user_std,
                                        user_embedding_norm=user_embedding_norm,
                                        gmm_confidence=gmm_confidence,
                                        genre_diversity=genre_diversity,
                                        rating_entropy=rating_entropy,
                                        movies_df=st.session_state.movies,
                                        genre_cols=st.session_state.genre_cols,
                                        ratings_df=st.session_state.ratings,
                                        svd_model=st.session_state.svd,
                                        hybrid_model=st.session_state.hybrid,
                                        is_cold_start=is_cold_start
                                    )
                                except:
                                    pass

                            st.session_state.current_adaptive_weights = user_adaptive_weights

                            if user_fav_genres is None or len(user_fav_genres) == 0:
                                if registered_data and registered_data.get('fav_genres'):
                                    user_fav_genres = registered_data['fav_genres']
                                elif selected_genres:
                                    user_fav_genres = selected_genres
                                elif st.session_state.get('current_fav_genres'):
                                    user_fav_genres = st.session_state.get('current_fav_genres')

                            st.session_state.current_fav_genres = user_fav_genres

                            df, analysis = display_all_recommendations(
                                hybrid_recs=hybrid_recs,
                                baseline_recs=baseline_recs,
                                movies_df=st.session_state.movies,
                                user_id=current_user_id,
                                items_per_page=items_per_page,
                                selected_genres=selected_genres,
                                is_new_user=False,
                                new_user_details=None,
                                ratings_df=st.session_state.ratings,
                                genre_cols=st.session_state.genre_cols,
                                adaptive_weights=user_adaptive_weights,
                                user_fav_genres=user_fav_genres
                            )
                            
                            st.markdown("---")
                            
                        else:
                            st.warning("Tidak ada rekomendasi yang tersedia")
                    else:
                        st.info("Klik tombol 'Generate Rekomendasi dengan Bobot Adaptif' di sebelah kiri")
                        
                                                
# ====================== FUNGSI UPDATE USER PREF DENGAN RATING NEGATIF ======================
def update_user_pref_with_negative_ratings(hybrid_model, user_id, ratings_dict, movies_df):
    """
    Update user preference dengan mempertimbangkan rating negatif.
    """
    try:
        # Cek apakah user ada di trainset
        try:
            u_inner = hybrid_model.trainset.to_inner_uid(str(user_id))
        except:
            print(f"User {user_id} tidak ditemukan")
            return False
        
        # Cek user_pref
        if hybrid_model.user_pref is None:
            return False
        
        if u_inner >= len(hybrid_model.user_pref):
            return False
        
        # ========== PERBAIKAN: Jangan gunakan factor negatif ==========
        # Simpan daftar genre yang dibenci secara terpisah
        hated_genres = set()
        
        for movie_id, rating in ratings_dict.items():
            iid_str = str(movie_id)
            
            if iid_str in hybrid_model.gmm_map:
                if rating <= 2.0:  # Rating buruk
                    # JANGAN update user_pref dengan negatif!
                    # Simpan genre yang dibenci saja
                    try:
                        mid_int = int(movie_id)
                        movie_info = movies_df[movies_df['movieId'] == mid_int]
                        if not movie_info.empty:
                            movie_genres = str(movie_info['genres'].values[0]).split('|')
                            for g in movie_genres:
                                hated_genres.add(g)
                    except:
                        pass
                elif rating >= 4.0:  # Rating bagus
                    factor = (rating - 3.0) / 2.0  # 4→0.5, 5→1.0
                    hybrid_model.user_pref[u_inner] += factor * hybrid_model.gmm_map[iid_str]
                # Rating 3: abaikan
        
        # Simpan hated_genres ke session_state untuk digunakan di rekomendasi
        st.session_state[f'hated_genres_{user_id}'] = list(hated_genres)
        
        # Normalisasi hanya jika ada perubahan positif
        norm = np.linalg.norm(hybrid_model.user_pref[u_inner])
        if norm > 0:
            hybrid_model.user_pref[u_inner] = hybrid_model.user_pref[u_inner] / norm
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    main()