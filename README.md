# VAE MNIST 手寫數字重建

使用變分自編碼器（VAE）重建 MNIST 手寫數字

## 安裝

```bash
pip install -r requirements.txt
```

## 執行

```bash
python VAE_MNIST_Optimized.py
```

## Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/VAE_MNIST/blob/main/VAE_MNIST.ipynb)

## 作業要求

- [x] 使用 MNIST 資料集
- [x] 實作 VAE (Encoder + Decoder)
- [x] Reparameterization trick
- [x] Adam 優化器
- [x] 顯示每個 epoch 平均損失
- [x] 輸出重建圖像

## 結果

- 訓練 epochs: 5
- 損失: ~165 → ~110
- 輸出檔案: reconstruction.png
