import os

import cupy as cp
from skimage.io import imread
from src.hog_gpu import hog_gpu
from src.estimate_model import EstimateModel
from skimage.feature import hog
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import argparse

if __name__ == "__main__":
    # コマンドライン引数の準備
    parser = argparse.ArgumentParser()
    parser.add_argument('use_gpu', type=int, choices=[0, 1], help='GPUを使うかどうか')
    args = parser.parse_args()
    use_gpu = True if args.use_gpu == 1 else False

    if use_gpu:
        print("GPU使用")

    # 各種パラメータ
    train_dir = 'data/train'
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (3, 3)
    tuned_parameters = [{"C": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}]
    cv = 10

    # 変数定義
    X = []
    y = []
    listdir = []
    listdir.extend([(f"{train_dir}/pos/{f}", True) for f in os.listdir(path=f"{train_dir}/pos") if f != '.gitkeep'])
    listdir.extend([(f"{train_dir}/neg/{f}", False) for f in os.listdir(path=f"{train_dir}/neg") if f != '.gitkeep'])

    # 各画像のHOG特徴量を取得
    shape = None
    for file_name, is_positive in listdir:
        image = imread(file_name, as_gray=True)
        if shape is not None and image.shape != shape:
            raise ValueError("image size do not match.")
        shape = image.shape

        if use_gpu:
            X.append(cp.asnumpy(hog_gpu(image_cp=cp.asarray(image),
                                        orientations=orientations,
                                        pixels_per_cell=pixels_per_cell,
                                        cells_per_block=cells_per_block)))
        else:
            X.append(hog(image,
                         orientations=orientations,
                         pixels_per_cell=pixels_per_cell,
                         cells_per_block=cells_per_block,
                         multichannel=False,
                         block_norm='L2-Hys'))

        y.append(1 if is_positive else 0)

    # 教師データから学習
    gscv = GridSearchCV(
        svm.LinearSVC(penalty="l2", max_iter=10000),
        tuned_parameters,
        cv=cv,
    )
    gscv.fit(X, y)
    svm_detector = gscv.best_estimator_
    svm_detector.fit(X, y)

    # pickleファイルに保存
    model = EstimateModel(svm_detector, shape, orientations, pixels_per_cell, cells_per_block)
    with open('data/estimate_model.pkl', "wb") as f:
        pickle.dump(model, f)
