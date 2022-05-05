import cupy as cp


def hog_histograms_gpu(gradient_columns,
                       gradient_rows,
                       cell_columns, cell_rows,
                       size_columns, size_rows,
                       number_of_cells_columns, number_of_cells_rows,
                       number_of_orientations,
                       orientation_histogram):
    """
    HOGヒストグラムを作成する

    Args:
        gradient_columns (cp.array): 各画素毎の縦方向の勾配配列
        gradient_rows (cp.array): 各画素ごとの横方向の勾配配列
        cell_columns (int): Cell内の縦方向の画素数
        cell_rows (int): Cell内の横方向の画素数
        size_columns (int): 縦方向の画素数
        size_rows (int): 横方向の画素数
        number_of_cells_columns (int): 縦方向のCellの総数
        number_of_cells_rows (int): 横方向のCellの総数
        number_of_orientations (int): 勾配方向数
        orientation_histogram (cp.array): output. 勾配配列を出力する
    """

    # 勾配量を計算する
    magnitude = cp.hypot(gradient_columns, gradient_rows)

    # 勾配方向を計算する
    orientation = cp.rad2deg(cp.arctan2(gradient_rows, gradient_columns)) % 180

    # 変数準備
    cr = cell_rows * number_of_cells_rows
    cc = cell_columns * number_of_cells_columns
    number_of_orientations_per_180 = 180. / number_of_orientations
    noxy = cp.ones((cr, cc))
    x = cp.arange(cc).reshape((1, cc))
    y = cp.arange(cr).reshape((cr, 1))

    # Cell毎の勾配量と勾配方向を計算する
    cp.ElementwiseKernel(
        '''
        T x, T y, F o,
        T cc, T cr,
        T cell_columns, T cell_rows,
        T nocc, T nocr, T noo,
        T size_columns, T size_rows, T noop_180,
        raw D magnitude
        ''',
        'raw E hist',
        '''
        int c_i = x / cell_columns;
        int r_i = y / cell_rows;
        int o_i = o / noop_180;
        int ind = {
            r_i * nocc * noo
        +          c_i * noo
        +                o_i
        };
        atomicAdd(&hist[ind], magnitude[y*size_columns+x]/(cell_columns * cell_rows));
        ''',
        'hog_histograms',
    )(noxy * x, noxy * y, orientation[:cr, :cc],
      cc, cr,
      cell_columns, cell_rows,
      number_of_cells_columns, number_of_cells_rows, number_of_orientations,
      size_columns, size_rows, number_of_orientations_per_180, magnitude,
      orientation_histogram)
