from src.hog_gpu import hog_gpu

import cupy as cp
import numpy as np
import skimage.transform as transform
from skimage.feature import hog
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from typing import Tuple, List


def detect(
        detector,
        image: np.ndarray,
        reduction: float,
        resize_repeat: int,
        min_ratio: float,
        target_size: Tuple[int, int],
        orientation: int,
        pixels_per_cell: Tuple[int, int],
        cells_per_block: Tuple[int, int],
        threshold: float,
        use_gpu: bool,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """
    HOG特徴量とSVMを使って画像検出を行う

    :param detector: 検知器(GridSearchSVのestimator)
    :param image: 画素配列. グレースケールであること
    :param reduction: ループごとの縮小率
    :param resize_repeat: ループ回数
    :param min_ratio: 最低縮小率
    :param target_size: 探索窓の基礎サイズ
    :param orientation: 勾配方向の数
    :param pixels_per_cell: セルごとのピクセル数
    :param cells_per_block: ブロックごとのセル数
    :param threshold: (required 0 <= threshold <= 1)しきい値. 判定に用いる
    :param use_gpu: gpuを使用するかどうか
    :return: tuple(tuple(検出矩形), 学習データに似ているかどうかのスコア(0～1))のリスト
    """

    scales = _get_scales(
        image=image,
        target_size=target_size,
        reduction=reduction,
        repeat=resize_repeat,
        min_ratio=min_ratio,
    )
    image_resizes = Parallel(n_jobs=-1, backend="threading")(
        [delayed(_resize)(image, s, s) for s in scales]
    )

    if use_gpu:
        image_resizes_cp = [cp.asarray(l) for l in image_resizes]
        detects = [
            _detect_gpu(
                detector=detector,
                img_parse=i,
                img_size=image.shape,
                target_size=target_size,
                orientation=orientation,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                threshold=threshold,
            )
            for i in image_resizes_cp
        ]
    else:
        detects = [
            _detect_cpu(
                detector=detector,
                img_parse=i,
                img_size=image.shape,
                target_size=target_size,
                orientation=orientation,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                threshold=threshold,
            )
            for i in image_resizes
        ]

    likelihoods = [rect for rects in detects for rect in rects if rects != []]
    return likelihoods


def _get_scales(
        image: np.ndarray,
        target_size: Tuple[int, int],
        reduction: float,
        repeat: int,
        min_ratio: float,
) -> List[float]:
    """ループ数と1ループごとの縮小率から各ステップにおける縮小率のリストを返す"""

    scale = 1.0
    scales = [scale]
    height, width = image.shape
    min_height, min_width = (height * min_ratio, width * min_ratio)
    target_height, target_width = target_size
    for i in range(repeat - 1):
        scale, height, width = scale * reduction, int(height / reduction), int(width / reduction)
        if (
                min_height > height
                or min_width > width
                or target_height > height
                or target_width > width
        ):
            break
        scales.append(scale)
    return scales


def _resize(image: np.ndarray, hscale: float, wscale: float) -> np.ndarray:
    """縮小した画像を返却する"""

    height, width = image.shape
    reduction = transform.resize(
        image,
        (int(height / hscale), int(width / wscale)),
        mode="constant",
        anti_aliasing=True,
    )
    return reduction


def _detect_cpu(
        detector: GridSearchCV,
        img_parse: np.ndarray,
        img_size: Tuple[int, int],
        target_size: Tuple[int, int],
        orientation: int,
        pixels_per_cell: Tuple[int, int],
        cells_per_block: Tuple[int, int],
        threshold: float,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """スライディングウィンドウ法を用いて画像の中から閾値以上のスコアを持つ矩形を導き出す"""
    original_height, original_width = img_size
    target_height, target_width = target_size[:2]
    height, width = img_parse.shape[:2]
    scale_height = height / original_height
    scale_width = width / original_width
    n_cellsx = int(np.floor(width // pixels_per_cell[0]))
    n_cellsy = int(np.floor(height // pixels_per_cell[1]))
    n_blocksx = (n_cellsx - cells_per_block[0]) + 1
    n_blocksy = (n_cellsy - cells_per_block[1]) + 1
    n_celltx = int(np.floor(target_width // pixels_per_cell[0]))
    n_cellty = int(np.floor(target_height // pixels_per_cell[1]))
    n_blocktx = (n_celltx - cells_per_block[0]) + 1
    n_blockty = (n_cellty - cells_per_block[1]) + 1
    likelihoods = []

    fd = hog(
        img_parse,
        orientations=orientation,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=False,
        block_norm="L2-Hys",
    )
    fd = fd.reshape(
        (n_blocksy, n_blocksx, cells_per_block[0], cells_per_block[1], orientation)
    )

    for x in range(n_blocksx - n_blocktx + 1):
        for y in range(n_blocksy - n_blockty + 1):
            # fmt: off
            # SVMの出力値をシグモイド関数で0~1に正規化
            estimated = 1 / (1 + (np.exp(-1 * detector.decision_function(
                fd[y:y + n_blockty, x:x + n_blocktx].reshape(1, -1)))))[0]
            # fmt: on
            if estimated >= threshold:
                x_parse = int(original_width * x * pixels_per_cell[0] / width)
                y_parse = int(original_height * y * pixels_per_cell[1] / height)
                target_height_parse = int(target_height / scale_height)
                target_width_parse = int(target_width / scale_width)

                likelihoods.append(
                    (
                        x_parse,
                        y_parse,
                        x_parse + target_width_parse,
                        y_parse + target_height_parse,
                        estimated,
                    )
                )

    return [((l[0], l[1], l[2], l[3]), l[4]) for l in likelihoods]


def _detect_gpu(
        detector: GridSearchCV,
        img_parse: cp.ndarray,
        img_size: Tuple[int, int],
        target_size: Tuple[int, int],
        orientation: int,
        pixels_per_cell: Tuple[int, int],
        cells_per_block: Tuple[int, int],
        threshold: float,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """スライディングウィンドウ法を用いて画像の中から閾値以上のスコアを持つ矩形を導き出す(GPU使用)"""

    original_height, original_width = img_size
    target_height, target_width = target_size[:2]
    height, width = img_parse.shape[:2]
    scale_height = height / original_height
    scale_width = width / original_width
    n_cellsx = int(np.floor(width // pixels_per_cell[0]))
    n_cellsy = int(np.floor(height // pixels_per_cell[1]))
    n_blocksx = (n_cellsx - cells_per_block[0]) + 1
    n_blocksy = (n_cellsy - cells_per_block[1]) + 1
    n_celltx = int(np.floor(target_width // pixels_per_cell[0]))
    n_cellty = int(np.floor(target_height // pixels_per_cell[1]))
    n_blocktx = (n_celltx - cells_per_block[0]) + 1
    n_blockty = (n_cellty - cells_per_block[1]) + 1

    fd = hog_gpu(
        img_parse,
        orientations=orientation,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        feature_vector=False,
    )

    scores = _decision_function_gpu(
        detector=detector,
        X=fd,
        number_of_blocks_columns=n_blocksx,
        number_of_blocks_rows=n_blocksy,
        cells_per_block_columns=cells_per_block[0],
        cells_per_block_rows=cells_per_block[1],
        number_of_orientations=orientation,
        target_columns=n_blocktx,
        target_rows=n_blockty,
    )

    estimated = cp.asnumpy(_sigmoid_gpu(scores))
    tx, ty = n_blocksx - n_blocktx + 1, n_blocksy - n_blockty + 1
    A = np.ones((ty, tx))
    x = np.arange(tx).reshape((1, tx))
    y = np.arange(ty).reshape((ty, 1))

    pos = estimated >= threshold
    estimated_score = estimated[pos]
    estimated_x = (A * x)[pos] * original_width * pixels_per_cell[0] / width
    estimated_y = (A * y)[pos] * original_height * pixels_per_cell[1] / height
    likelihoods = np.hstack(
        (
            estimated_x.reshape((-1, 1)),
            estimated_y.reshape((-1, 1)),
            estimated_x.reshape((-1, 1)) + int(target_width / scale_width),
            estimated_y.reshape((-1, 1)) + int(target_height / scale_height),
            estimated_score.reshape((-1, 1)),
        )
    )

    return [((l[0], l[1], l[2], l[3]), l[4]) for l in likelihoods]


def _decision_function_gpu(
        detector: GridSearchCV,
        X: cp.ndarray,
        number_of_blocks_columns: int,
        number_of_blocks_rows: int,
        cells_per_block_columns: int,
        cells_per_block_rows: int,
        number_of_orientations: int,
        target_columns: int,
        target_rows: int,
) -> cp.ndarray:
    """Xの各要素において`どのくらい学習データに似ているか`を計算する(GPU使用)"""

    n_features = detector.coef_.shape[1]
    # if X.shape[1] != n_features:
    #     raise ValueError("X has %d features per sample; expecting %d"
    #                      % (X.shape[1], n_features))

    coef = cp.asarray(detector.coef_)
    intercept = cp.asarray(detector.intercept_)

    number_of_stride_columns = number_of_blocks_columns - target_columns + 1
    number_of_stride_rows = number_of_blocks_rows - target_rows + 1
    scores = cp.zeros((number_of_stride_rows, number_of_stride_columns))
    nocrb = cp.ones((number_of_stride_rows, number_of_stride_columns, n_features))

    c = cp.arange(number_of_stride_columns).reshape((1, number_of_stride_columns, 1))
    r = cp.arange(number_of_stride_rows).reshape((number_of_stride_rows, 1, 1))
    b = cp.arange(n_features).reshape((1, 1, n_features))

    bncnrcro = cp.ones(
        (
            target_rows,
            target_columns,
            cells_per_block_rows,
            cells_per_block_columns,
            number_of_orientations,
        )
    )
    bnc = cp.arange(target_columns).reshape((1, target_columns, 1, 1, 1))
    bnr = cp.arange(target_rows).reshape((target_rows, 1, 1, 1, 1))
    bc = cp.arange(cells_per_block_columns).reshape(
        (1, 1, 1, cells_per_block_columns, 1)
    )
    br = cp.arange(cells_per_block_rows).reshape((1, 1, cells_per_block_rows, 1, 1))
    bo = cp.arange(number_of_orientations).reshape((1, 1, 1, 1, number_of_orientations))

    cp.ElementwiseKernel(
        """
        T c, T r, T b,
        T nobc, T nobr, T cpbc, T cpbr, T noo, T tc, T tr,
        raw T bnc, raw T bnr, raw T bc, raw T br, raw T bo,
        raw E X, raw E coef
        """,
        "raw E scores",
        """
        int ind_b = { b };
        unsigned int bnc_i = bnc[ind_b];
        unsigned int bnr_i = bnr[ind_b];
        unsigned int c_i = bc[ind_b];
        unsigned int r_i = br[ind_b];
        unsigned int o_i = bo[ind_b];
        int ind = {
            r * (nobc - tc + 1) + c
        };
        int ind_x = {
            (r+bnr_i) * nobc * cpbr * cpbc * noo
        +          (c+bnc_i) * cpbr * cpbc * noo
        +                       r_i * cpbc * noo
        +                              c_i * noo
        +                                    o_i
        };
        atomicAdd(&scores[ind], X[ind_x]*coef[ind_b]);
        """,
        "decision_function",
    )(
        nocrb * c,
        nocrb * r,
        nocrb * b,
        number_of_blocks_columns,
        number_of_blocks_rows,
        cells_per_block_columns,
        cells_per_block_rows,
        number_of_orientations,
        target_columns,
        target_rows,
        bncnrcro * bnc,
        bncnrcro * bnr,
        bncnrcro * bc,
        bncnrcro * br,
        bncnrcro * bo,
        X,
        coef,
        scores,
    )

    return scores + intercept


def _sigmoid_gpu(scores: cp.ndarray):
    """シグモイド関数(GPU使用)"""
    outputs = cp.zeros(scores.shape)

    cp.ElementwiseKernel(
        """
        T score
        """,
        "raw E output",
        """
        double sigmoid_range = 34.538776394910684;

        if (score <= -sigmoid_range) {
            output[i] = 1e-15;
        } else if (score >= sigmoid_range) {
            output[i] = 1 - 1e-15;
        } else {
            output[i] = 1.0 / (1.0 + exp(-score));
        }

        """,
        "sigmoid",
    )(
        scores, outputs
    )

    return outputs


def _sigmoid(x: float) -> float:
    """シグモイド関数"""
    sigmoid_range = 34.538776394910684
    if x <= -sigmoid_range:
        return 1e-15
    if x >= sigmoid_range:
        return 1.0 - 1e-15

    return 1.0 / (1.0 + np.exp(-x))


def similarity(
        detector: GridSearchCV,
        image: np.ndarray,
        target_size: Tuple[int, int],
        orientation: int,
        pixels_per_cell: Tuple[int, int],
        cells_per_block: Tuple[int, int],
) -> float:
    """対象画像と学習データとの類似度を調べる"""
    # imageをtarget_sizeの大きさに縮小
    hscale, wscale = image.shape[0] / target_size[0], image.shape[1] / target_size[1]
    rimage = _resize(image, hscale, wscale)

    # rimageのhogを取得
    height, width = rimage.shape[:2]
    n_cellsx = int(np.floor(width // pixels_per_cell[0]))
    n_cellsy = int(np.floor(height // pixels_per_cell[1]))
    n_blocksx = (n_cellsx - cells_per_block[0]) + 1
    n_blocksy = (n_cellsy - cells_per_block[1]) + 1

    fd = hog(
        rimage,
        orientations=orientation,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=False,
        block_norm="L2-Hys",
    )

    fd = fd.reshape(
        (n_blocksy, n_blocksx, cells_per_block[0], cells_per_block[1], orientation)
    )

    # fmt: off
    # SVMの出力値をシグモイド関数で0~1に正規化
    estimated = 1 / (1 + (np.exp(-1 * detector.decision_function(fd.reshape(1, -1)))))[0]
    # fmt: on

    return estimated
