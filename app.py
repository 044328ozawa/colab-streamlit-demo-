
import streamlit as st

st.title("Colab & Streamlit Cloud デモ")
st.write("これはGoogle Colabで作成したStreamlitアプリです。")

name = st.text_input("お名前を入力してください", "")
if name:
    st.write(f"{name}さん、こんにちは！")
