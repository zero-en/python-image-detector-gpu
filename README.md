## 概要

- GPGPUを使って、スライディングウィンドウ法とHOG特徴量+SVM(サポートベクターマシン)を用いた教師あり画像検出処理を高速化してみた
    - 主にHOG特徴量計算部分を高速化した
    - グレースケール画像のみ対応

- sklearnのhog関数を使った画像検出と今回自作したhog_gpu関数を使った画像検出の処理時間を比較した

## 実行方法

1. データセットを下記ディレクトリに展開する

- data/train/pos
    - positiveデータ
- data/train/neg
    - negativeデータ
- data/test
    - テストデータ(検出画像)

2. HOG特徴量+SVMによる学習を実行

```bash
# CPU使用
python run_compute_image_hog.py 0

# GPU使用
python run_compute_image_hog.py 1
```

3. HOG特徴量+SVMによる画像検出を実行
    1. data/test内の各画像に対して処理を実行し、検出された矩形を描画してdata/output内に吐き出す

```bash
# CPU使用
python run_image_detect.py 0

# CPU使用
python run_image_detect.py 1
```

## 計測

- 内容
    - 下記データセットとパラメータを用いて学習モデルを作成し、付属している全テストデータに対して、検出処理(`detect.py`の`detect`関数)が終わるまでの時間を計測した
        - https://github.com/MenukaIshan/Data-sets-for-opencv-classifier-training/blob/master/Other%20Image%20Datasets%20collected/UIUC%20Image%20Database%20for%20Car%20Detection/CarData.zip
            - 教師データ(positive: 550枚、negative: 500枚)
                - CarData/TrainImages内の画像データ
            - 検出テストデータ(170枚)
                - CarData/TestImages内の画像データ
    - 計測箇所は`run_image_detect.py`の下記部分
        - 内部では、`use_gpu`がTrueの場合自作の`hog_gpu`、Falseの場合はsklearnの`hog`関数が使われるようになっている
    ```python
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
    ```

- 実行環境

| 項目           | 内容                          |
|--------------|-----------------------------|
| OS           | Windows 10 Professional     |
| CPU          | AMD Ryzen 8 3700X           |
| Memory       | 32GB                        |
| GPU          | NVIDIA GeForce GTX 1060 6GB |
| Python       | Python 3.7.0                |
| CUDA toolkit | 11.6                        |

- 結果
    - テストデータ(170枚)への検出処理時間の合計
        - 約30％高速化

| 項目    | 処理時間     |
|-------|----------|
| CPUのみ | 35.14sec |
| GPU使用 | 24.28sec |

## 参考

- データセット(使用させていただいたデータセット)
    - https://github.com/MenukaIshan/Data-sets-for-opencv-classifier-training

- NML(Non Maximum Suppression)について(ソースコード参考)
    - https://pystyle.info/opencv-non-maximum-suppression/
