# 📈 نظام التنبؤ بالمبيعات (Sales Prediction)

مشروع Machine Learning بسيط يتنبأ بالمبيعات من 6 عوامل باستخدام **Linear Regression**،
مع بيانات في ملف CSV منفصل، وواجهة رسومية لتعديل البيانات وإعادة التدريب.

## المدخلات والمخرجات

| المدخل | المعنى | النطاق |
|---|---|---|
| `advertising` | ميزانية الإعلان | 100 – 5000 |
| `price` | سعر المنتج | 5 – 100 |
| `discount` | نسبة الخصم % | 0 – 40 |
| `customers` | عدد العملاء | 20 – 500 |
| `day_of_week` | اليوم (1–7) | 1 – 7 |
| `stock` | المخزون | 10 – 1000 |

**المخرج:** `sales` (المبيعات المتوقعة)

## هيكل المشروع

```
sales_project/
├── generate_data.py   # يولّد البيانات → sales_data.csv
├── sales_data.csv     # ملف البيانات (50,000 صف)
├── train_model.py     # يقرأ الـ CSV، يدرّب، يقيّم، ويحفظ الموديل
├── predict.py         # توقع من الطرفية (يسألك عن القيم)
├── sales_core.py      # دوال مشتركة (تحميل، تدريب، توقع)
├── ui.py              # واجهة Streamlit
├── model.ipynb        # شرح خطوة بخطوة + رسومات
├── requirements.txt
└── models/            # يتكوّن بعد التدريب (نسخة لكل تدريب)
    └── v1_<تاريخ_ووقت>/
        ├── best_model.pkl
        ├── preprocessor.pkl
        └── features.pkl
```

## التثبيت

```bash
# (اختياري) بيئة افتراضية
python -m venv venv
source venv/bin/activate        # على Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## التشغيل

### الطريقة 1: الواجهة الرسومية (الأسهل)

```bash
python generate_data.py          # فقط لو ما عندك sales_data.csv
streamlit run ui.py
```

افتح `http://localhost:8501` ثم:

1. **تبويب التدريب** ← "ابدأ التدريب" (أول مرة لازم).
2. **تبويب التوقع** ← أدخل القيم واضغط "احسب التوقع".
3. **تبويب تعديل البيانات** ← عدّل/أضف/احذف صفوفاً أو ارفع CSV، ثم أعد التدريب.

### الطريقة 2: الطرفية

```bash
python generate_data.py
python train_model.py
python predict.py
```

### الطريقة 3: النوتبوك

```bash
jupyter notebook model.ipynb
```

## استخدام الموديل داخل كودك

```python
import sales_core as core

model_dir, model, scaler = core.load_latest()
result = core.predict(model, scaler, {
    "advertising": 2500, "price": 30, "discount": 15,
    "customers": 200, "day_of_week": 5, "stock": 600,
})
print(result)   # ≈ 4029.73
```

## استخدام بياناتك الحقيقية

1. جهّز ملف CSV بنفس الأعمدة السبعة بالضبط (6 مدخلات + `sales`).
2. استبدل `sales_data.csv` به (أو ارفعه من الواجهة).
3. أعد التدريب.

لتغيير الأعمدة، عدّل `FEATURES` و`TARGET` في `sales_core.py` (وفي `train_model.py` و`predict.py` لو بتستخدمهما).

## النتائج على البيانات المولّدة

| المقياس | القيمة |
|---|---|
| R² | ≈ 0.999 |
| MAE | ≈ 39.8 |
| MSE | ≈ 2481 |

> النتيجة العالية متوقعة لأن البيانات مولّدة من معادلة خطية مع ضوضاء بسيطة.
> مع بيانات حقيقية توقّع أرقاماً أقل، وقد تحتاج موديلات غير خطية (Random Forest / Gradient Boosting).

## ملاحظات مهمة

- تعديل البيانات **لا يحدّث الموديل تلقائياً**، لازم تعيد التدريب.
- الموديل خطي، فلا تثق بتوقعاته **خارج نطاق** القيم اللي اتدرّب عليها.
- كل تدريب يحفظ نسخة جديدة في `models/`، وتقدر تمسح القديمة يدوياً.
- الـ `StandardScaler` محفوظ مع الموديل ولازم يُستخدم دائماً قبل التوقع.



v## Project Structure

| File | Purpose |
|------|---------|
| `generate_data.py` | Generates the sales dataset |
| `train_model.py` | Trains the regression model and saves it |
| `predict.py` | Loads the saved model and predicts sales values |
| `ui.py` | Simple user interface for making predictions |
| `model.ipynb` | Notebook with exploration and model experiments |

## Requirements

- Python 3.9+
- <numpy, pandas, scikit-learn, ...>

Install them with:

```bash
pip install numpy pandas scikit-learn
```

## How to Run

1. Generate the data:
```bash
   python generate_data.py
```
2. Train the model:
```bash
   python train_model.py
```
3. Make a prediction:
```bash
   python predict.py
```
4. Launch the interface:
```bash
   python ui.py
```

## Results

- Model: <Linear Regression / ...>
- Evaluation: <R² = ..., MAE = ...>
