import numpy as np
import cv2
import sys # エラー表示のためにsysをインポート
# 学籍番号K24004を想定したパスに修正
from my_module.K21999.lecture05_camera_image_capture import MyVideoCapture

def lecture05_01():
    # カメラキャプチャ実行（保存せずに画像を取得）
    app = MyVideoCapture()
    # カメラ映像表示と画像キャプチャ（'q'キーを押すまで待機）を実行
    app.run()  

    capture_img: cv2.Mat = app.get_img()  # ← get_img()で最後にキャプチャされた画像を取得

    if capture_img is None:
        print("エラー: カメラ画像がキャプチャされませんでした。'q'キーを押して画像をキャプチャしましたか？", file=sys.stderr)
        return

    # Google検索画面画像を読み込み
    google_img: cv2.Mat = cv2.imread('images/google.png')
    
    if google_img is None:
        print("エラー: images/google.pngが見つかりませんでした。ファイルパスを確認してください。", file=sys.stderr)
        return

    g_height, g_width, _ = google_img.shape
    c_height, c_width, _ = capture_img.shape
    print("google.png:", google_img.shape)
    print("camera:", capture_img.shape)

    # --- キャプチャ画像をタイル状（グリッド状）に敷き詰める ---
    tiled_img = np.zeros_like(google_img)
    for y in range(0, g_height, c_height):
        for x in range(0, g_width, c_width):
            # 貼り付け範囲を計算
            y_end = min(y + c_height, g_height)
            x_end = min(x + c_width, g_width)

            # 貼り付け範囲に合わせて切り取る
            cropped = capture_img[0:(y_end - y), 0:(x_end - x)]
            tiled_img[y:y_end, x:x_end] = cropped

    # --- 白部分をキャプチャ画像タイルで置換 ---
    result_img = google_img.copy()
    # google.pngの白色部分を特定 (R=255, G=255, B=255)
    white_mask = (google_img == [255, 255, 255]).all(axis=2)
    # マスクされた部分をタイル画像で置換
    result_img[white_mask] = tiled_img[white_mask]

    # --- 合成結果を保存 ---
    cv2.imwrite('images/google_composite.png', result_img)
    print("合成画像を保存しました: images/google_composite.png")

    # 結果表示
    cv2.imshow("Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()