import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Sarel Catalog Enrichment", layout="centered")

st.title("העשרת קטלוג שראל")
st.write("שלב ראשון - העלאת קובץ ובדיקת נתונים")

uploaded_file = st.file_uploader("בחר קובץ Excel", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("קובץ נטען בהצלחה!", icon="✅")
    st.write("תצוגה מקדימה של הקובץ:")
    st.dataframe(df)

    if st.button("הרץ תהליך (כרגע דמה)"):
        # Simulated processing
        st.info("מעבד... (לא באמת עושה כלום כרגע)", icon="🔄")
        
        # Simulate backend result (just reuse original DataFrame for now)
        output = BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)

        st.success("עיבוד הסתיים! לחץ להורדה.")
        st.download_button(
            label="📥 הורד קובץ מעובד",
            data=output,
            file_name="processed_file.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
