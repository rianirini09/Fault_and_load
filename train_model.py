import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# KONFIGURASI
# ============================================================
DATASET_PATH = "fault_dataset12.csv"
MODEL_DIR = "models"
RAW_FEATURES = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

SYMMETRICAL_LABELS   = {'LLL_fault', 'LLLG_fault'}
UNSYMMETRICAL_LABELS = {'LG_fault', 'LLG_fault', 'LL_fault'}

FAULT_TYPE_MAP = {
    'LG_fault':   'LG',
    'LLG_fault':  'LLG',
    'LL_fault':   'LL',
    'LLL_fault':  'LLL',
    'LLLG_fault': 'LLLG',
}


# ============================================================
# FEATURE ENGINEERING (KOMPONEN SIMETRIS)
# ============================================================
def add_symmetrical_components(df):
    a = np.exp(1j * 2 * np.pi / 3)
    a2 = a ** 2

    Ia, Ib, Ic = df['Ia'].values, df['Ib'].values, df['Ic'].values
    Va, Vb, Vc = df['Va'].values, df['Vb'].values, df['Vc'].values

    df['I0'] = np.abs((Ia + Ib + Ic) / 3)
    df['I1'] = np.abs((Ia + a*Ib + a2*Ic) / 3)
    df['I2'] = np.abs((Ia + a2*Ib + a*Ic) / 3)

    df['V0'] = np.abs((Va + Vb + Vc) / 3)
    df['V1'] = np.abs((Va + a*Vb + a2*Vc) / 3)
    df['V2'] = np.abs((Va + a2*Vb + a*Vc) / 3)

    df['I0_I1_ratio'] = df['I0'] / (df['I1'] + 1e-9)
    df['V0_V1_ratio'] = df['V0'] / (df['V1'] + 1e-9)
    return df


def normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return 'unknown'
    return label.strip()


def main():
    print("[INFO] Membaca dataset...")
    df = pd.read_csv(DATASET_PATH)

    df['fault_label_norm']  = df['fault_label'].apply(normalize_label)
    df['faulty_label_norm'] = df['faulty_label'].apply(normalize_label)
    df = df.dropna(subset=RAW_FEATURES + ['fault_label_norm', 'faulty_label_norm'])

    print(f"[INFO] Total data valid: {len(df)}")
    print(f"[INFO] Distribusi fault_label:\n{df['fault_label_norm'].value_counts()}\n")

    # Feature engineering
    df = add_symmetrical_components(df)
    FEATURES = RAW_FEATURES + ['I0', 'I1', 'I2', 'V0', 'V1', 'V2',
                               'I0_I1_ratio', 'V0_V1_ratio']
    print(f"[INFO] Total fitur: {len(FEATURES)}")

    X = df[FEATURES].values
    y_fault_label = df['fault_label_norm'].values

    # ---- Buat label untuk 3 stage ----
    y_stage1 = np.where(y_fault_label == 'Healthy', 'healthy', 'faulty')

    y_stage2 = np.array([
        'symmetrical'   if lbl in SYMMETRICAL_LABELS
        else 'unsymmetrical' if lbl in UNSYMMETRICAL_LABELS
        else 'healthy'
        for lbl in y_fault_label
    ])

    y_stage3 = np.array([
        FAULT_TYPE_MAP.get(lbl, 'healthy')
        for lbl in y_fault_label
    ])

    # ---- Split (hanya 3 target) ----
    X_train, X_test, y1_train, y1_test, y2_train, y2_test, \
        y3_train, y3_test = train_test_split(
            X, y_stage1, y_stage2, y_stage3,
            test_size=0.2, random_state=42,
            stratify=y_fault_label
        )

    # ---- Scaler ----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print("[INFO] Scaler disimpan.")

    # ---- Training 3 model saja ----
    configs = [
        ("Stage 1 (Healthy vs Faulty)", y1_train, y1_test, "model_stage1.pkl"),
        ("Stage 2 (Sym vs Unsym)",      y2_train, y2_test, "model_stage2.pkl"),
        ("Stage 3 (Fault Type)",        y3_train, y3_test, "model_stage3.pkl"),
    ]

    for name, y_tr, y_te, filename in configs:
        print(f"\n[INFO] Training {name}...")
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train_s, y_tr)
        pred = clf.predict(X_test_s)
        acc = accuracy_score(y_te, pred)
        print(f"[RESULT] Akurasi {name}: {acc:.4f}")
        print(classification_report(y_te, pred, zero_division=0))
        joblib.dump(clf, os.path.join(MODEL_DIR, filename))

    # --- Hapus model_stage4.pkl jika ada sisa dari training lama ---
    old_stage4 = os.path.join(MODEL_DIR, "model_stage4.pkl")
    if os.path.exists(old_stage4):
        os.remove(old_stage4)
        print(f"[INFO] Menghapus model lama: {old_stage4}")

    print("\n[SELESAI] Model Stage 1-3 disimpan di folder 'models/'.")


if __name__ == "__main__":
    main()