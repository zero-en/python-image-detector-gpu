import numpy as np
import cupy as cp
from . import hog_histogram_gpu


def hog_gpu(image_cp: cp.ndarray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(3, 3),
            transform_sqrt=False,
            feature_vector=True):
    """
    GPUを用いて画像のHOG特徴配列を計算する

    Args:
        image_cp (cp.array): 画像のcupy配列
        orientations (int, optional): Defaults to 9. 勾配方向の数
        pixels_per_cell (tuple, optional): Defaults to (8, 8). Cell毎の画素数
        cells_per_block (tuple, optional): Defaults to (3, 3). BlockごとのCell数
        transform_sqrt (bool, optional): Defaults to False.
        feature_vector (bool, optional): Defaults to True. HOG特徴を一次元化して返すかどうか

    Returns:
        cp.array: HOG特徴配列
    """

    if not isinstance(image_cp, cp.ndarray):
        raise ValueError("image_cp isn't cp.array.")

    if transform_sqrt:
        image_cp = cp.sqrt(image_cp)

    if image_cp.dtype.kind == 'u':
        image_cp = image_cp.astype('float')

    # x軸、y軸への勾配量を計算する
    g_row_cp, g_col_cp = _hog_channel_gradient_gpu(image_cp)

    # Cell毎の勾配方向と勾配量を計算する
    s_row, s_col = image_cp.shape[:2]
    c_row, c_col = pixels_per_cell
    b_row, b_col = cells_per_block
    n_cells_row = int(s_row // c_row)
    n_cells_col = int(s_col // c_col)
    orientation_histogram_cp = cp.zeros((n_cells_row, n_cells_col, orientations),
                                        dtype=np.double)

    hog_histogram_gpu.hog_histograms_gpu(g_col_cp, g_row_cp, c_col, c_row, s_col, s_row,
                                         n_cells_col, n_cells_row,
                                         orientations, orientation_histogram_cp)

    # Block毎の勾配量を正規化する
    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    normalized_blocks = cp.zeros((n_blocks_row, n_blocks_col,
                                  b_row, b_col, orientations),
                                 dtype=np.double)

    _hog_normalize_block_gpu_l2_hys(orientation_histogram_cp,
                                    n_cells_col, n_cells_row,
                                    n_blocks_col, n_blocks_row,
                                    b_col, b_row,
                                    orientations, normalized_blocks)

    # HOG特徴リストの1次元化
    if feature_vector:
        normalized_blocks = normalized_blocks.ravel()

    return normalized_blocks


def _hog_normalize_block_gpu_l2_hys(hist,
                                    number_of_cells_columns, number_of_cells_rows,
                                    number_of_blocks_columns, number_of_blocks_rows,
                                    cells_per_block_columns, cells_per_block_rows,
                                    number_of_orientations, blocks, eps=1e-5):
    """
    GPUを用いて画像のブロックごとのHOG特徴を正則化する
    正則化にはL2_hysを用いる(GPU使用)

    Args:
        hist (cp.array): 画像のヒストグラム
        number_of_cells_columns (int): 縦方向のCellの数
        number_of_cells_rows (int): 横方向のCellの数
        number_of_blocks_columns (int): 縦方向のBlockの数
        number_of_blocks_rows (int): 横方向のBlockの数
        cells_per_block_columns (int): ブロック毎の縦方向のCellの数
        cells_per_block_rows (int): ブロック毎の横方向のCellの数
        number_of_orientations (int): 勾配方向数
        blocks (cp.array): output. この配列に計算値が出力される
    """

    nocr = cp.ones((number_of_blocks_rows, number_of_blocks_columns))
    c = cp.arange(number_of_blocks_columns).reshape((1, number_of_blocks_columns))
    r = cp.arange(number_of_blocks_rows).reshape((number_of_blocks_rows, 1))

    cp.ElementwiseKernel(
        '''
        T c, T r, 
        T nocc, T nocr, T nobc, T nobr, T cpbc, T cpbr, T noo, T eps,
        raw D hist
        ''',
        'raw E blocks',
        '''
        double result = 0.f;
        for (int c_i = 0; c_i < cpbc; c_i++) {
            for (int r_i = 0; r_i < cpbr; r_i++) {
                for (int o_i = 0; o_i < noo; o_i++) {
                    int ind = {
                        (r + r_i) * nocc * noo
                    +          (c + c_i) * noo
                    +                      o_i
                    };
                    result += powf(hist[ind], 2);
                }
            }
        }
        for (int c_i = 0; c_i < cpbc; c_i++) {
            for (int r_i = 0; r_i < cpbr; r_i++) {
                for (int o_i = 0; o_i < noo; o_i++) {
                    int ind = {
                        (r + r_i) * nocc * noo
                    +          (c + c_i) * noo
                    +                      o_i
                    };
                    int ind_b = {
                        r * nobc * cpbr * cpbc * noo
                    +          c * cpbr * cpbc * noo
                    +               r_i * cpbc * noo
                    +                      c_i * noo
                    +                            o_i
                    };
                    blocks[ind_b] = min(hist[ind] / sqrt(result + powf(eps, 2)), 0.2f);
                }
            }
        }

        result = 0.f;
        for (int c_i = 0; c_i < cpbc; c_i++) {
            for (int r_i = 0; r_i < cpbr; r_i++) {
                for (int o_i = 0; o_i < noo; o_i++) {
                    int ind_b = {
                        r * nobc * cpbr * cpbc * noo
                    +          c * cpbr * cpbc * noo
                    +               r_i * cpbc * noo
                    +                      c_i * noo
                    +                            o_i
                    };
                    result += powf(blocks[ind_b], 2);
                }
            }
        }
        for (int c_i = 0; c_i < cpbc; c_i++) {
            for (int r_i = 0; r_i < cpbr; r_i++) {
                for (int o_i = 0; o_i < noo; o_i++) {
                    int ind_b = {
                        r * nobc * cpbr * cpbc * noo
                    +          c * cpbr * cpbc * noo
                    +               r_i * cpbc * noo
                    +                      c_i * noo
                    +                            o_i
                    };
                    blocks[ind_b] = blocks[ind_b] / sqrt(result + powf(eps, 2));
                }
            }
        }
        ''',
        'hog_normalize_block_l2_hys',
    )(nocr * c, nocr * r,
      number_of_cells_columns, number_of_cells_rows,
      number_of_blocks_columns, number_of_blocks_rows,
      cells_per_block_columns, cells_per_block_rows,
      number_of_orientations, eps,
      hist, blocks)


def _hog_channel_gradient_gpu(channel: cp.ndarray):
    """各座標における勾配量を求める(GPU使用)"""
    g_row = cp.empty(channel.shape, dtype=np.double)
    g_row[0, :] = 0
    g_row[-1, :] = 0
    g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
    g_col = cp.empty(channel.shape, dtype=np.double)
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]

    return g_row, g_col
