import streamlit as st
import pandas as pd
from io import BytesIO
# import os
# import sys
from app.models.match_descriptions import create_match_product

st.set_page_config(page_title="Sarel Catalog Enrichment", layout="centered")

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
       
            df_result = create_match_product(df,col_input_desc, col_input_manu)

            st.success("ההתאמה הסתיימה! ✅")
            st.dataframe(df_result)

            
            # Simulate backend result (just reuse original DataFrame for now)
            output = BytesIO()
            df_result.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)

            st.download_button(
                label="📥 הורד קובץ תוצאה",
                data=output,
                file_name="matched_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
