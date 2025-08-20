import streamlit as st
import pandas as pd
from io import BytesIO
import os
import requests
import numpy as np
import tempfile
# import sys
from app.models.match_descriptions import create_match_product

EMBEDDINGS_URL = "https://github.com/naamaLuxenburg/sarel_catalog_plus/releases/download/v1/db_embeddings_BioBERT-mnli-snli-scinli-scitail-mednli-stsb.npy"

# def load_embeddings():
#     """Download once if missing, then load into memory."""
#     if not os.path.exists(LOCAL_PATH):
#         print("Downloading embeddings file for the first time...")  # Only in logs
#         response = requests.get(EMBEDDINGS_URL)
#         response.raise_for_status()
#         os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
#         with open(LOCAL_PATH, "wb") as f:
#             f.write(response.content)
#     return np.load(LOCAL_PATH, allow_pickle=True)

@st.cache_resource
def load_embeddings():
    """Download embeddings once per session, load into memory, and delete temp file."""
    print("Downloading embeddings for this session...")
    r = requests.get(EMBEDDINGS_URL)
    r.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    embeddings = np.load(tmp_path, allow_pickle=True)
    os.remove(tmp_path)
    print("Done loading embeddings")
    return embeddings



st.set_page_config(page_title="Sarel Catalog Enrichment", layout="centered")

# ✅ load silently (no user notification)
db_embeddings = load_embeddings()


st.title("העשרת קטלוג שראל")
st.write("שלב ראשון - העלאת קובץ ובדיקת נתונים")
uploaded_file = st.file_uploader("בחר קובץ Excel", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("קובץ נטען בהצלחה!", icon="✅")
    st.write("תצוגה מקדימה של הקובץ:")
    st.dataframe(df)

    st.markdown("יש למלא את שם העמודה בקובץ של **התיאור** ובמידה ויש גם את **מקט יצרן**")
    
    col_input_desc = st.text_input("שם עמודת התיאור (חובה)")
    col_input_manu = st.text_input("שם עמודת מקט יצרן (לא חובה)")

    if st.button("הרץ תהליך התאמה"):
        if not col_input_desc:
            st.error("אנא מלא את שם עמודת התיאור לפני הרצה ❗")
        elif col_input_desc not in df.columns:
            st.error(f"עמודה '{col_input_desc}' לא נמצאה בקובץ ❗")
        elif col_input_manu and col_input_manu not in df.columns:
            st.error(f"עמודה '{col_input_manu}' לא נמצאה בקובץ ❗")
        else:
            st.info("מריץ התאמה...", icon="🔄")
       
            df_result = create_match_product(df,db_embeddings, col_input_desc, col_input_manu)

            st.success("ההתאמה הסתיימה! ✅")
            st.dataframe(df_result)

            
            # Simulate backend result (just reuse original DataFrame for now)
            output = BytesIO()
            df_result.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)

            clicked=st.download_button(
                label="📥 הורד קובץ תוצאה",
                data=output,
                file_name="matched_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            # Only print when user clicks download
            if clicked:
                print("User clicked download button. File sent!")
           
