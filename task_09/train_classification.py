import time
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    print("=" * 60)
    print("TASK 8 & 9: TIEN XU LY VA SO SANH 3 MO HINH PHAN LOAI")
    print("=" * 60)

    # ---------------------------------------------------------
    # PHAN 1: TASK 8 - TIEN XU LY DU LIEU
    # ---------------------------------------------------------
    print("\n[1] Dang tai va tien xu ly du lieu...")
    X, y = load_breast_cancer(return_X_y=True)

    # Chia tap du lieu: 70% Train, 15% Validation, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    # Chuan hoa StandardScaler (Chi fit tren Train de tranh Data Leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f" - So luong mau Train: {X_train_scaled.shape[0]}")
    print(f" - So luong mau Val:   {X_val_scaled.shape[0]}")
    print(f" - So luong mau Test:  {X_test_scaled.shape[0]}")

    # ---------------------------------------------------------
    # PHAN 2: TASK 9 - HUAN LUYEN VA DANH GIA 3 THUAT TOAN
    # ---------------------------------------------------------
    print("\n[2] Bat dau huan luyen 3 thuat toan...")

    # Khoi tao 3 mo hinh
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    results = []

    for model_name, model in models.items():
        # Đo thoi gian Train
        start_train = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_train

        # Đo thoi gian Predict tren tap Test
        start_infer = time.time()
        y_pred = model.predict(X_test_scaled)
        infer_time = time.time() - start_infer

        # Tinh toan cac chi so danh gia
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Luu ket qua
        results.append({
            "Model": model_name,
            "Train Time (s)": round(train_time, 4),
            "Infer Time (s)": round(infer_time, 4),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
        })

    # ---------------------------------------------------------
    # PHAN 3: TONG HOP VA IN BANG KET QUA SO SANH
    # ---------------------------------------------------------
    print("\n[3] KET QUA SO SANH TREN TAP TEST:")
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
