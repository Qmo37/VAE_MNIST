# VAE MNIST - 變分自編碼器手寫數字重建

使用變分自編碼器（Variational Autoencoder, VAE）重建 MNIST 手寫數字資料集。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Qmo37/VAE_MNIST/blob/master/VAE_MNIST.ipynb)

*因使用 Local IDE 開發，因此在 Colab 上執行前需執行從 .py 檔同步到 .ipynb 檔方能執行。

## 📋 作業要求

### ✅ 已完成項目
- [x] **使用 MNIST 資料集**作為輸入資料
- [x] **Encoder**：將 28x28 圖像（展平為784維向量）轉換為潛在空間的均值（mu）與對數方差（logvar）
- [x] **使用 reparameterization trick**產生潛在變數 z
- [x] **Decoder**：將 z 解碼回圖像空間，重建輸入圖像
- [x] **使用 Adam 優化器**
- [x] **顯示每個 epoch 平均損失**
- [x] **輸出重建圖像結果**

## 🏗️ 模型架構

```
輸入 (784維) → 編碼器 → 潛在空間 (20維) → 解碼器 → 輸出 (784維)

編碼器:  784 → 400 → (mu=20, logvar=20)
解碼器:  20 → 400 → 784
```

## 📦 安裝

### 需求環境
- Python 3.8+
- PyTorch
- matplotlib

### 安裝依賴
```bash
pip install -r requirements.txt
```

### 或單獨安裝
```bash
pip install torch torchvision matplotlib
```

## 🚀 執行

```bash
python VAE_MNIST.py
```

## 📊 資料集資訊

程式會自動下載所需的 MNIST 檔案：
- `train-images-idx3-ubyte.gz` (60,000 訓練圖像)
- `train-labels-idx1-ubyte.gz` (60,000 訓練標籤)
- `t10k-images-idx3-ubyte.gz` (10,000 測試圖像)
- `t10k-labels-idx1-ubyte.gz` (10,000 測試標籤)

## 📈 預期結果

### 訓練過程
```
Epoch 1 Average loss: 165.2341
Epoch 2 Average loss: 132.4521
Epoch 3 Average loss: 118.7834
Epoch 4 Average loss: 109.2156
Epoch 5 Average loss: 98.4231
```

### 輸出檔案
- `reconstruction.png` - 原始圖像與重建圖像的比較
- `data/` 資料夾 - 下載的 MNIST 資料集

## 🧠 核心概念

### 1. 編碼器 (Encoder)
將輸入圖像轉換為潛在空間的參數（均值和對數方差）

### 2. 重參數化技巧 (Reparameterization Trick)
```python
z = mu + std * epsilon
```
使得反向傳播能夠通過隨機採樣過程

### 3. 解碼器 (Decoder)
將潛在變數重建為圖像

### 4. VAE 損失函數
```python
總損失 = 重建損失 + KL 散度損失
```

## 📁 檔案結構

```
VAE_MNIST/
├── VAE_MNIST.py          # 主要實作檔案
├── requirements.txt      # 依賴套件
├── README.md            # 說明文件
├── reconstruction.png   # 重建結果圖像
└── data/               # MNIST 資料集
```
