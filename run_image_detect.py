import argparse
import os
import pickle

from src.detector import detect
from src.estimate_model import EstimateModel
from src.non_max_suppression import non_max_suppression
from PIL import Image, ImageDraw
import numpy as np
from src.timer import Timer

if __name__ == "__main__":
    # コマンドライン引数の準備
    parser = argparse.ArgumentParser()
    parser.add_argument('use_gpu', type=int, choices=[0, 1], help='GPUを使うかどうか')
    args = parser.parse_args()
    use_gpu = True if args.use_gpu == 1 else False

    # 各種パラメータ
    reduction = 1.05
    resize_repeat = 32
    min_ratio = 0.3
    threshold = 0.8
    overlap_thresh = 0.6
    listdir = [f"data/test/{f}" for f in os.listdir(path="data/test") if f != '.gitkeep']

    if use_gpu:
        print("GPU使用")

    # 検出器を取得する
    with open("./data/estimate_model.pkl", "rb") as f:
        estimate_model: EstimateModel = pickle.load(f)

    t = Timer()
    for path in listdir:
        target_image = Image.open(path)

        # 計測開始
        t.start()
        likelihoods = detect(detector=estimate_model.svm_estimator,
                             image=np.asarray(target_image.convert("L")),
                             reduction=reduction,
                             resize_repeat=resize_repeat,
                             min_ratio=min_ratio,
                             target_size=estimate_model.shape,
                             orientation=estimate_model.orientation,
                             pixels_per_cell=estimate_model.pixels_per_cell,
                             cells_per_block=estimate_model.cells_per_block,
                             threshold=threshold,
                             use_gpu=use_gpu
                             )
        t.stop()
        # 計測終了

        # 重なっている検出矩形を削減
        likelihoods = non_max_suppression(likelihoods, overlap_thresh=overlap_thresh)

        for likelihood in likelihoods:
            (rect, score) = likelihood
            start = (rect[0], rect[1])
            end = (rect[2], rect[3])
            draw = ImageDraw.Draw(target_image)
            draw.rectangle(rect)
            output_file_name = os.path.splitext(os.path.basename(path))[0]
            target_image.save(f"data/output/{output_file_name}.png", "PNG")

    print("elapsed_time: " + str(t.elapsed_time))
