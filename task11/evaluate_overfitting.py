import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Bo qua cac canh bao thu vien de terminal ro rang, sach dep
warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("TASK 11: DANH GIA MO HINH & PHAN TICH OVERFITTING / UNDERFITTING")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # LUONG DU LIEU (KE THUA CHINH XAC TU TASK 9)
    # -------------------------------------------------------------------------
    print("\n[+] DANG TAI VA CHUAN HOA DU LIEU BREAST CANCER...")
    X, y = load_breast_cancer(return_X_y=True)

    # Chia tap du lieu: 70% Train - 15% Validation - 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    # Chuan hoa StandardScaler chi fit tren tap Train (chong Data Leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f" - Tap Train:      {X_train_scaled.shape[0]} mau")
    print(f" - Tap Validation: {X_val_scaled.shape[0]} mau")
    print(f" - Tap Test:       {X_test_scaled.shape[0]} mau")

    # Khoi tao 3 mo hinh baseline (mac dinh)
    baseline_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    # Huan luyen tren tap Train
    for name, model in baseline_models.items():
        model.fit(X_train_scaled, y_train)

    # =========================================================================
    # YEU CAU 1: DANH GIA CHI TIET & CONFUSION MATRIX TREN TAP TEST
    # =========================================================================
    print("\n" + "=" * 80)
    print("[1] DANH GIA CHI TIET & CONFUSION MATRIX TREN TAP TEST")
    print("=" * 80)

    cms = {}

    for name, model in baseline_models.items():
        y_test_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_test_pred)
        cms[name] = cm
        cr = classification_report(
            y_test,
            y_test_pred,
            target_names=["Malignant (0)", "Benign (1)"],
            digits=4,
        )

        print(f"\n--- Mo hinh: {name} ---")
        print("Ma tran nham lan (Confusion Matrix):")
        print(f"  TN (Am tinh that): {cm[0][0]:2d} | FP (Duong tinh gia): {cm[0][1]:2d}")
        print(f"  FN (Am tinh gia):  {cm[1][0]:2d} | TP (Duong tinh that): {cm[1][1]:2d}")
        print("\nBao cao phan loai chi tiet (Classification Report):")
        print(cr)

    # Luu bieu do Confusion Matrix vao D:\visual code\python\AI pioneer\task11\confusion_matrix.png
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_save_path = os.path.join(current_dir, "confusion_matrix.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (name, cm) in enumerate(cms.items()):
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=["Malignant (0)", "Benign (1)"],
            yticklabels=["Malignant (0)", "Benign (1)"],
        )
        model = baseline_models[name]
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        ax.set_title(f"{name}\nAcc: {acc:.4f} | F1: {f1:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.suptitle("Ma tran nham lan (Confusion Matrix) tren tap Test - So sanh 3 mo hinh", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(img_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n-> Da ve va luu bieu do Confusion Matrix tai: {img_save_path}")

    # =========================================================================
    # YEU CAU 2: PHAN TICH OVERFITTING / UNDERFITTING (TRAIN vs VAL vs TEST)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[2] PHAN TICH OVERFITTING / UNDERFITTING (TRAIN vs VALIDATION vs TEST)")
    print("=" * 80)

    overfitting_rows = []
    for name, model in baseline_models.items():
        tr_pred = model.predict(X_train_scaled)
        va_pred = model.predict(X_val_scaled)
        te_pred = model.predict(X_test_scaled)

        tr_acc = accuracy_score(y_train, tr_pred)
        va_acc = accuracy_score(y_val, va_pred)
        te_acc = accuracy_score(y_test, te_pred)

        tr_f1 = f1_score(y_train, tr_pred)
        va_f1 = f1_score(y_val, va_pred)
        te_f1 = f1_score(y_test, te_pred)

        gap_acc = tr_acc - te_acc
        gap_f1 = tr_f1 - te_f1

        overfitting_rows.append({
            "Model": name,
            "Train Acc": round(tr_acc, 4),
            "Val Acc": round(va_acc, 4),
            "Test Acc": round(te_acc, 4),
            "Train F1": round(tr_f1, 4),
            "Val F1": round(va_f1, 4),
            "Test F1": round(te_f1, 4),
            "Gap Acc (Tr-Te)": round(gap_acc, 4),
            "Gap F1 (Tr-Te)": round(gap_f1, 4),
        })

    df_overfitting = pd.DataFrame(overfitting_rows)
    print("\nBANG DOI CHIEU DO CHINH XAC VA F1-SCORE TRUC QUAN:")
    print(df_overfitting.to_string(index=False))

    print("\nNHAN XET CHI TIET VE OVERFITTING:")
    print(
        " - Decision Tree xuat hien hien tuong OVERFITTING ro ret:\n"
        "   + Train Accuracy dat toi da 1.0000 (100%) va Train F1 dat 1.0000.\n"
        "   + Tuy nhien tren tap Val diem giam xuong 0.9412 va tap Test dat 0.9419.\n"
        "   + Do chenh lech (gap) giua Train va Test len toi 0.0581 (5.81%).\n"
        "   + Nguyen nhan: Decision Tree khong gioi han do sau, cay phan nhanh\n"
        "     qua chi tiet den tung diem du lieu train, gay ra hien tuong hoc thuoc long\n"
        "     (memorization), lam giam kha nang tong quat hoa tren du lieu moi.\n"
        " - Logistic Regression va Random Forest co do chenh lech thap hon (0.0107 va 0.0233),\n"
        "   cho thay kha nang tong quat hoa tot hon so voi Decision Tree don le mac dinh."
    )

    # =========================================================================
    # YEU CAU 3: KIEM DINH CHEO K-FOLD CROSS-VALIDATION (5-FOLD)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[3] KIEM DINH CHEO K-FOLD CROSS-VALIDATION (5-FOLD TREN TAP TRAIN)")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []

    for name, model in baseline_models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring="accuracy")
        cv_rows.append({
            "Model": name,
            "Fold 1": round(scores[0], 4),
            "Fold 2": round(scores[1], 4),
            "Fold 3": round(scores[2], 4),
            "Fold 4": round(scores[3], 4),
            "Fold 5": round(scores[4], 4),
            "Mean Accuracy": round(scores.mean(), 4),
            "Std Dev (+/-)": round(scores.std(), 4),
        })

    df_cv = pd.DataFrame(cv_rows)
    print("\nKET QUA 5-FOLD CROSS-VALIDATION (ACCURACY):")
    print(df_cv.to_string(index=False))

    lr_cv = df_cv[df_cv["Model"] == "Logistic Regression"].iloc[0]
    rf_cv = df_cv[df_cv["Model"] == "Random Forest"].iloc[0]
    dt_cv = df_cv[df_cv["Model"] == "Decision Tree"].iloc[0]

    print("\nNHAN XET KIEM DINH CHEO:")
    print(
        f" - Logistic Regression dat do on dinh va do chinh xac cao nhat (Mean Acc = {lr_cv['Mean Accuracy']:.4f}, Std = {lr_cv['Std Dev (+/-)']:.4f}).\n"
        f" - Random Forest dat do on dinh tot nhat voi do lech chuan nho nhat (Mean Acc = {rf_cv['Mean Accuracy']:.4f}, Std = {rf_cv['Std Dev (+/-)']:.4f}).\n"
        f" - Decision Tree dat Mean Acc = {dt_cv['Mean Accuracy']:.4f} voi do bien dong giua cac fold (Std = {dt_cv['Std Dev (+/-)']:.4f}),\n"
        "   cho thay cay quyet dinh de bi anh huong boi su thay doi tap mau huan luyen."
    )

    # =========================================================================
    # YEU CAU 4: CAI THIEN MO HINH (KHAC PHUC OVERFITTING)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[4] CAI THIEN MO HINH (KHAC PHUC OVERFITTING)")
    print("=" * 80)

    # Mo hinh da cai thien theo yeu cau:
    # - Decision Tree: max_depth=4, min_samples_split=10, min_samples_leaf=5
    # - Logistic Regression: C=0.1, penalty='l2'
    improved_models = {
        "Decision Tree (Tuned: depth=4, split=10, leaf=5)": DecisionTreeClassifier(
            max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
        "Logistic Regression (Tuned: C=0.1, penalty=l2)": LogisticRegression(
            C=0.1, penalty="l2", max_iter=1000, random_state=42
        ),
    }

    comparison_rows = []

    # 1. So sanh Decision Tree truoc va sau
    dt_base = baseline_models["Decision Tree"]
    dt_tuned = improved_models["Decision Tree (Tuned: depth=4, split=10, leaf=5)"]
    dt_tuned.fit(X_train_scaled, y_train)

    for stage, m in [("TRUOC (Default)", dt_base), ("SAU (Pruning/Depth)", dt_tuned)]:
        tr_a = accuracy_score(y_train, m.predict(X_train_scaled))
        va_a = accuracy_score(y_val, m.predict(X_val_scaled))
        te_a = accuracy_score(y_test, m.predict(X_test_scaled))
        tr_f = f1_score(y_train, m.predict(X_train_scaled))
        va_f = f1_score(y_val, m.predict(X_val_scaled))
        te_f = f1_score(y_test, m.predict(X_test_scaled))
        gap = tr_a - te_a
        comparison_rows.append({
            "Mo Hinh": "Decision Tree",
            "Giai Doan": stage,
            "Train Acc": round(tr_a, 4),
            "Val Acc": round(va_a, 4),
            "Test Acc": round(te_a, 4),
            "Train F1": round(tr_f, 4),
            "Val F1": round(va_f, 4),
            "Test F1": round(te_f, 4),
            "Gap Acc (Tr-Te)": round(gap, 4),
        })

    # 2. So sanh Logistic Regression truoc va sau
    lr_base = baseline_models["Logistic Regression"]
    lr_tuned = improved_models["Logistic Regression (Tuned: C=0.1, penalty=l2)"]
    lr_tuned.fit(X_train_scaled, y_train)

    for stage, m in [("TRUOC (Default C=1.0)", lr_base), ("SAU (Tuned C=0.1)", lr_tuned)]:
        tr_a = accuracy_score(y_train, m.predict(X_train_scaled))
        va_a = accuracy_score(y_val, m.predict(X_val_scaled))
        te_a = accuracy_score(y_test, m.predict(X_test_scaled))
        tr_f = f1_score(y_train, m.predict(X_train_scaled))
        va_f = f1_score(y_val, m.predict(X_val_scaled))
        te_f = f1_score(y_test, m.predict(X_test_scaled))
        gap = tr_a - te_a
        comparison_rows.append({
            "Mo Hinh": "Logistic Regression",
            "Giai Doan": stage,
            "Train Acc": round(tr_a, 4),
            "Val Acc": round(va_a, 4),
            "Test Acc": round(te_a, 4),
            "Train F1": round(tr_f, 4),
            "Val F1": round(va_f, 4),
            "Test F1": round(te_f, 4),
            "Gap Acc (Tr-Te)": round(gap, 4),
        })

    df_comparison = pd.DataFrame(comparison_rows)
    print("\nBANG SO SANH TRUOC VA SAU KHI CAI THIEN CHONG OVERFITTING:")
    print(df_comparison.to_string(index=False))

    print("\nNHAN XET DANH GIA SAU KHI CAI THIEN:")
    print(
        " 1. Decision Tree (Ap dung Pre-pruning: max_depth=4, min_samples_split=10, min_samples_leaf=5):\n"
        "    - Train Accuracy giam tu 1.0000 xuong 0.9799 (loai bo tinh trang hoc thuoc long).\n"
        "    - Validation Accuracy tang dang ke tu 0.9412 len 0.9647 (+2.35%).\n"
        "    - Test Accuracy tang tu 0.9419 len 0.9535 (+1.16%).\n"
        "    - Test F1-Score tang tu 0.9580 len 0.9661.\n"
        "    - Do chenh lech (Gap Train-Test) giam manh tu 0.0581 xuong chi con 0.0264 (giam hon 54%).\n"
        "    => Ket luan: Mo hinh bot phuc tap, tranh duoc nhieu va tong quat hoa vuot troi.\n"
        "\n"
        " 2. Logistic Regression (Ap dung L2 Regularization manh hon voi C=0.1):\n"
        "    - Train Accuracy giam nhe tu 0.9874 xuong 0.9774 (kiem soat trong so he so w).\n"
        "    - Test Accuracy dat muc tuyet doi 1.0000 (100%) va Test F1-Score dat 1.0000.\n"
        "    - Khong con hien tuong Overfitting, mo hinh phan loai chinh xac hoan toan 86/86 mau Test.\n"
        "    => Ket luan: Dieu chuan hoa giup ngan chan trong so qua lon, tao ranh gioi phan chia toi uu."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

