import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Medical Q&A Dataset
# -----------------------------
print("Processing Medical Q&A dataset...")
qna_df = pd.read_csv(RAW_DIR / "medical_qna_train.csv")

with open(OUT_DIR / "medical_qna.txt", "w", encoding="utf-8") as f:
    for _, row in qna_df.iterrows():
        q = str(row.get("Question", "")).strip()
        a = str(row.get("Answer", "")).strip()
        if q and a:
            f.write(
            f"QUESTION: {q}\n"
            f"ANSWER: {a}\n\n"
            "----\n\n"
        )
                 
# -----------------------------
# Medical Device Manuals Dataset
# -----------------------------
print("Processing Medical Device Manuals...")
device_df = pd.read_csv(RAW_DIR / "medical_device_manuals.csv")

with open(OUT_DIR / "medical_devices.txt", "w", encoding="utf-8") as f:
    for _, row in device_df.iterrows():
        device_name = str(row.get("Device_Name", "")).strip()
        model = str(row.get("Model_Number", "")).strip()
        manufacturer = str(row.get("Manufacturer", "")).strip()
        device_class = str(row.get("Device_Class", "")).strip()
        indications = str(row.get("Indications_for_Use", "")).strip()
        contraindications = str(row.get("Contraindications", "")).strip()
        sterilization = str(row.get("Sterilization_Method", "")).strip()

        if not device_name:
            continue

        f.write(
            f"DEVICE NAME: {device_name}\n"
            f"MODEL NUMBER: {model}\n"
            f"MANUFACTURER: {manufacturer}\n"
            f"DEVICE CLASS: {device_class}\n\n"
            f"INDICATIONS FOR USE:\n{indications}\n\n"
            f"CONTRAINDICATIONS:\n{contraindications}\n\n"
            f"STERILIZATION METHOD:\n{sterilization}\n\n"
            "----\n\n"
        )

print("Data preparation complete.")
