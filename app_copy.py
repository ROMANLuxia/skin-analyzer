import streamlit as st
import google.generativeai as genai

# ページ設定
st.set_page_config(page_title="Safety Copywriter | Luxia", layout="wide")

st.title("🛡️ セーフティ・コピーライター")
st.markdown("薬機法管理者の知見を学習したAIが、あなたのキャッチコピーをリーガルチェックします。")

# --- サイドバー ---
st.sidebar.header("1. API設定")
api_key = st.sidebar.text_input("Gemini APIキーを入力", type="password")

st.sidebar.divider()
st.sidebar.header("2. カテゴリ選択")
category = st.sidebar.selectbox("商品区分", ["化粧品（スキンケア・ヘアケア）", "美容機器・雑貨", "健康食品"])

# --- メインロジック ---
input_text = st.text_area("チェックしたい文章を入力してください", height=200, placeholder="例：この美容液でシミが消えて、ニキビも治ります！")

if st.button("リーガルチェック実行"):
    if not api_key:
        st.error("APIキーを入力してください")
    elif not input_text:
        st.warning("文章を入力してください")
    else:
        with st.spinner("薬機法・景表法に照らして解析中..."):
            try:
                genai.configure(api_key=api_key)
                
                # 専門知識をシステムプロンプトに凝縮
                system_instruction = f"""
                あなたは日本の「薬機法」および「景品表示法」に精通した、美容業界専門のリーガルチェッカーです。
                対象カテゴリ：{category}
                
                以下の指示に従って、厳格かつ魅力的な校正を行ってください：
                1. NG表現の指摘：医学的効能（治る、消える、若返る等）や最大級表現、根拠のない期間保証を特定してください。
                2. 理由の解説：なぜその表現が法的にリスクがあるのか、簡潔に論理的に説明してください。
                3. 代替案の提示：元の訴求力を維持しつつ、合法的な「言い換え」を3パターン提案してください（例：「シミが消える」→「乾燥によるくすみを防ぎ、明るい印象へ」）。
                """
                
                # 【修正】判明した最新の有効モデルを正確に指定
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash", 
                    system_instruction=system_instruction
                )
                
                response = model.generate_content(input_text)
                
                st.divider()
                st.subheader("📊 診断結果・修正案")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

st.sidebar.info("提供：株式会社Luxia (2026)")