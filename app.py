import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps

# ページ設定
st.set_page_config(page_title="Comprehensive Skin Analyzer Pro", layout="wide")

st.title("高機能肌診断：ハイブリッド解析 & 総合カウンセリング")

# --- サイドバー：キャリブレーション（位置合わせ） ---
st.sidebar.header("1. オーダーメイド位置合わせ")
st.sidebar.markdown("※お客様の骨格に合わせてAIの解析枠を調整します")

show_mask = st.sidebar.checkbox("👁️ 調整モード（除外マスクと検知枠を表示）", value=True)

with st.sidebar.expander("📍 顔全体の調整", expanded=True):
    face_x = st.slider("左右移動 (X軸)", -200, 200, 0)
    face_y = st.slider("上下移動 (Y軸)", -200, 200, 0)
    face_scale = st.slider("顔のサイズ (倍率)", 0.5, 2.0, 1.0)

with st.sidebar.expander("👁️ 目の調整 (誤検知防止)", expanded=False):
    eye_y = st.slider("目の上下移動", -100, 100, 0)
    eye_spread = st.slider("目の間隔 (左右)", -100, 100, 0)
    eye_size = st.slider("目のマスクサイズ", 0.5, 2.0, 1.0)

with st.sidebar.expander("👃 鼻・頬の調整 (毛穴検知エリア)", expanded=False):
    nose_y = st.slider("鼻の上下移動", -150, 150, 0)
    nose_size = st.slider("鼻の検知エリア広さ", 0.5, 2.0, 1.0)

with st.sidebar.expander("👄 口・ほうれい線の調整", expanded=False):
    mouth_y = st.slider("口の上下移動", -100, 100, 0)
    mouth_size = st.slider("口のマスクサイズ", 0.5, 3.0, 1.5)

st.sidebar.divider()

# --- サイドバー：感度調整スライダー ---
st.sidebar.header("2. 解析感度の調整")
acne_redness = st.sidebar.slider("🔴 炎症の強さ (大きいほど濃い赤のみ)", 50, 200, 130)
acne_min_size = st.sidebar.slider("🔴 炎症ニキビ検知サイズ", 10, 200, 80)
pore_sens = st.sidebar.slider("🔵 鼻の黒ずみ感度", 2, 20, 8)
cheek_pore_sens = st.sidebar.slider("🌸 頬の毛穴検知感度", 2, 20, 8)
spot_sens = st.sidebar.slider("🟦 シミ検知感度", 20, 150, 40)
nasolabial_sens = st.sidebar.slider("〽️ ほうれい線検知感度", 10, 100, 30)
sagging_sens = st.sidebar.slider("🟪 フェイスラインたるみ感度", 10, 100, 40)

st.sidebar.divider()
st.sidebar.header("3. 画像アップロード")
uploaded_file = st.sidebar.file_uploader("顧客の肌写真をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    img_array = np.array(image)
    h, w, c = img_array.shape
    
    scale = 640 / w
    img_res = cv2.resize(img_array, (int(w * scale), int(h * scale)))
    img_bgr = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
    img_draw = img_bgr.copy()
    h_res, w_res, _ = img_res.shape

    with st.spinner('AIが全顔のトラブルをマッピングしています...'):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # ==========================================
        # 座標とマスクの計算
        # ==========================================
        mask_roi = np.zeros((h_res, w_res), dtype=np.uint8)
        
        # 1. 顔全体
        center_face_x = (w_res // 2) + face_x
        center_face_y = int(h_res * 0.55) + face_y
        axes_face = (int(w_res * 0.40 * face_scale), int(h_res * 0.45 * face_scale))
        cv2.ellipse(mask_roi, (center_face_x, center_face_y), axes_face, 0, 0, 360, 255, -1)

        # 2. 目 (除外)
        eye_w, eye_h = int(w_res * 0.15 * eye_size), int(h_res * 0.08 * eye_size)
        left_eye_x = int(w_res * 0.35) + face_x - eye_spread
        right_eye_x = int(w_res * 0.65) + face_x + eye_spread
        eye_y_pos = int(h_res * 0.45) + face_y + eye_y
        cv2.ellipse(mask_roi, (left_eye_x, eye_y_pos), (eye_w, eye_h), 0, 0, 360, 0, -1)   
        cv2.ellipse(mask_roi, (right_eye_x, eye_y_pos), (eye_w, eye_h), 0, 0, 360, 0, -1)  

        # 3. 口 (除外)
        mouth_x = int(w_res * 0.50) + face_x
        mouth_y_pos = int(h_res * 0.75) + face_y + mouth_y
        mouth_w, mouth_h = int(w_res * 0.15 * mouth_size), int(h_res * 0.08 * mouth_size)
        cv2.ellipse(mask_roi, (mouth_x, mouth_y_pos), (mouth_w, mouth_h), 0, 0, 360, 0, -1) 

        # 4. 鼻 (青丸エリア)
        nose_center_x = (w_res // 2) + face_x
        nose_center_y = int(h_res * 0.55) + face_y + nose_y
        nose_w = int(w_res * 0.25 * nose_size)
        nose_h = int(h_res * 0.25 * nose_size)
        nose_x1, nose_x2 = nose_center_x - nose_w // 2, nose_center_x + nose_w // 2
        nose_y1, nose_y2 = nose_center_y - nose_h // 2, nose_center_y + nose_h // 2

        # 5. 頬エリア (ピンク点エリア) / ほうれい線エリア (緑線エリア) の計算
        cheek_w = int(w_res * 0.22 * face_scale)
        left_cheek_x1, left_cheek_x2 = nose_x1 - cheek_w, nose_x1
        right_cheek_x1, right_cheek_x2 = nose_x2, nose_x2 + cheek_w
        cheek_y1, cheek_y2 = eye_y_pos + eye_h, mouth_y_pos # 目の下から口の高さまで

        # ==========================================
        # 解析ロジック
        # ==========================================
        # ニキビ（赤丸）
        lower_red1 = np.array([0, acne_redness, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, acne_redness, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_acne_color = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        mask_acne = cv2.bitwise_and(mask_acne_color, mask_acne_color, mask=mask_roi)
        contours_acne, _ = cv2.findContours(mask_acne, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        acne_count = 0
        for cnt in contours_acne:
            area = cv2.contourArea(cnt)
            if area > acne_min_size:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity > 0.25: 
                        (x, y), r = cv2.minEnclosingCircle(cnt)
                        cv2.circle(img_draw, (int(x), int(y)), int(r), (0, 0, 255), 2)
                        acne_count += 1

        # 毛穴検知ベース
        thresh_pore = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, pore_sens)
        thresh_pore = cv2.bitwise_and(thresh_pore, thresh_pore, mask=mask_roi)
        contours_pore, _ = cv2.findContours(thresh_pore, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pore_count = 0
        cheek_pore_count = 0
        for cnt in contours_pore:
            if 2 < cv2.contourArea(cnt) < 30:
                x, y, w_s, h_s = cv2.boundingRect(cnt)
                # 鼻の黒ずみ（青丸）
                if nose_x1 < x < nose_x2 and nose_y1 < y < nose_y2:
                    (cx, cy), r = cv2.minEnclosingCircle(cnt)
                    cv2.circle(img_draw, (int(cx), int(cy)), int(r), (255, 0, 0), 1)
                    pore_count += 1
                # 頬の毛穴（ピンク丸）
                elif (left_cheek_x1 < x < left_cheek_x2 or right_cheek_x1 < x < right_cheek_x2) and (cheek_y1 < y < cheek_y2):
                    (cx, cy), r = cv2.minEnclosingCircle(cnt)
                    cv2.circle(img_draw, (int(cx), int(cy)), int(r), (255, 105, 180), 1) # Hot Pink
                    cheek_pore_count += 1

        # シミ（水色）
        lower_spot = np.array([5, spot_sens, 50])
        upper_spot = np.array([20, 150, 150])
        mask_spot = cv2.bitwise_and(cv2.inRange(hsv, lower_spot, upper_spot), mask_roi)
        contours_spot, _ = cv2.findContours(mask_spot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spot_count = 0
        for cnt in contours_spot:
            if 20 < cv2.contourArea(cnt) < 200:
                x, y, w_s, h_s = cv2.boundingRect(cnt)
                cv2.rectangle(img_draw, (x, y), (x + w_s, y + h_s), (255, 255, 0), 1)
                spot_count += 1

        # 線（エッジ）の抽出ベース
        edges = cv2.Canny(blur, 30, 100)
        edges = cv2.bitwise_and(edges, edges, mask=mask_roi)
        
        # ほうれい線（緑の線）- 頬の内側半分を下に向かう線を抽出
        mask_naso = np.zeros((h_res, w_res), dtype=np.uint8)
        cv2.rectangle(mask_naso, (nose_x1 - cheek_w//2, nose_y1), (nose_x1, mouth_y_pos + mouth_h), 255, -1) # 左ほうれい線エリア
        cv2.rectangle(mask_naso, (nose_x2, nose_y1), (nose_x2 + cheek_w//2, mouth_y_pos + mouth_h), 255, -1) # 右ほうれい線エリア
        edges_naso = cv2.bitwise_and(edges, mask_naso)
        lines_naso = cv2.HoughLinesP(edges_naso, 1, np.pi/180, threshold=nasolabial_sens, minLineLength=15, maxLineGap=5)
        
        nasolabial_count = 0
        if lines_naso is not None:
            for line in lines_naso:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 20 < angle < 80: # 斜めの線のみ抽出
                    cv2.line(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2) # 緑線
                    nasolabial_count += 1

        # フェイスラインたるみ（紫の線）- 顎周りのみ
        mask_sag = np.zeros((h_res, w_res), dtype=np.uint8)
        cv2.rectangle(mask_sag, (0, mouth_y_pos), (w_res, h_res), 255, -1) # 口より下
        edges_sag = cv2.bitwise_and(edges, mask_sag)
        lines_sag = cv2.HoughLinesP(edges_sag, 1, np.pi/180, threshold=sagging_sens, minLineLength=20, maxLineGap=10)
        
        sagging_count = 0
        if lines_sag is not None:
            for line in lines_sag:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 20 < angle < 80: 
                    cv2.line(img_draw, (x1, y1), (x2, y2), (255, 0, 255), 2) # 紫線
                    sagging_count += 1

        # くすみ（オレンジ枠）
        grid_h, grid_w = h_res // 2, w_res // 2
        min_bright = 255
        dull_zone = None
        for r in range(2):
            for c in range(2):
                y1, y2 = r * grid_h, (r + 1) * grid_h
                x1, x2 = c * grid_w, (c + 1) * grid_w
                zone_mask = mask_roi[y1:y2, x1:x2]
                zone_gray = gray[y1:y2, x1:x2]
                if cv2.countNonZero(zone_mask) > 0:
                    zone_mean = cv2.mean(zone_gray, mask=zone_mask)[0]
                    if zone_mean < min_bright:
                        min_bright = zone_mean
                        dull_zone = (x1, y1, x2, y2)
        if dull_zone:
            dx1, dy1, dx2, dy2 = dull_zone
            cv2.rectangle(img_draw, (dx1+10, dy1+10), (dx2-10, dy2-10), (0, 165, 255), 3)
            cv2.putText(img_draw, "DULLNESS", (dx1 + 15, dy1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        transparency_score = int(cv2.mean(gray, mask=mask_roi)[0] / 2.55)
        
        # AIスコア計算（減点ロジック更新）
        ai_base_score = 50 
        ai_penalty = (acne_count * 1.5) + (spot_count * 0.5) + (sagging_count * 1.0) + (nasolabial_count * 1.5) + (pore_count * 0.2) + (cheek_pore_count * 0.1)
        ai_score = max(15, int(ai_base_score - ai_penalty))

        # ==========================================
        # 調整モード（マスクと検知枠の可視化）
        # ==========================================
        if show_mask:
            overlay = np.zeros_like(img_draw)
            overlay[mask_roi == 0] = (0, 0, 0) 
            overlay[mask_roi == 255] = (200, 200, 200) 
            img_draw = cv2.addWeighted(img_draw, 0.6, overlay, 0.4, 0)
            
            # 鼻枠（黄）
            cv2.rectangle(img_draw, (nose_x1, nose_y1), (nose_x2, nose_y2), (0, 255, 255), 1)
            # 頬枠（ピンク）
            cv2.rectangle(img_draw, (left_cheek_x1, cheek_y1), (left_cheek_x2, cheek_y2), (255, 105, 180), 1)
            cv2.rectangle(img_draw, (right_cheek_x1, cheek_y1), (right_cheek_x2, cheek_y2), (255, 105, 180), 1)
            # ほうれい線枠（緑）
            cv2.rectangle(img_draw, (nose_x1 - cheek_w//2, nose_y1), (nose_x1, mouth_y_pos + mouth_h), (0, 255, 0), 1)
            cv2.rectangle(img_draw, (nose_x2, nose_y1), (nose_x2 + cheek_w//2, mouth_y_pos + mouth_h), (0, 255, 0), 1)
            
            cv2.putText(img_draw, "ADJUSTMENT MODE ON", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- UI表示 ---
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("全顔マッピング & トラブル可視化")
        st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("🔴ニキビ / 🔵鼻黒ずみ / 🌸頬毛穴 / 🟦シミ / 〽️緑:ほうれい線 / 🟪紫:たるみ / 🟧くすみ")

    with col2:
        st.subheader("AI検出データ")
        st.write(f"- 炎症ニキビ: {acne_count}箇所")
        st.write(f"- ほうれい線の影: {nasolabial_count}箇所")
        st.write(f"- 頬のたるみ毛穴: {cheek_pore_count}箇所")
        st.write(f"- 鼻の黒ずみ: {pore_count}箇所")
        st.write(f"- フェイスラインたるみ: {sagging_count}箇所")
        st.write(f"- シミ候補: {spot_count}箇所")

    st.divider()

    st.subheader("詳細カウンセリング（全10問）")
    
    with st.form("skin_form"):
        c1, c2 = st.columns(2)
        
        with c1:
            q1 = st.selectbox("1. 洗顔後、何もつけないと肌はどのように感じますか？", ["強くつっぱる・カサつく", "部分的に乾燥する", "特に変化はない", "すぐに皮脂が出る"])
            q2 = st.selectbox("2. 夕方になると、Tゾーン（額・鼻）のテカリはどうですか？", ["かなりテカる・メイクが崩れる", "少しテカる", "ほとんど気にならない", "逆に乾燥する"])
            q3 = st.selectbox("3. 化粧品を変えた時や季節の変わり目に赤み・ピリピリ感が出ますか？", ["よくある", "たまにある", "ほとんどない"])
            q4 = st.selectbox("4. ニキビや吹き出物ができる頻度と場所は？", ["常にTゾーンにある", "Uゾーン（顎・フェイスライン）にできやすい", "生理前などたまにできる", "めったにできない"])
            q5 = st.selectbox("5. 現在の肌の『くすみ感』や『透明感のなさ』をどう感じていますか？", ["夕方になると顔が暗く見える", "常に顔全体がくすんでいる", "あまり気にならない"])
        
        with c2:
            q6 = st.selectbox("6. 毎日の平均睡眠時間はどのくらいですか？", ["5時間未満", "5〜6時間", "7時間以上"])
            q7 = st.selectbox("7. 外出時（曇りや冬を含む）の紫外線（UV）対策は？", ["毎日欠かさず行っている", "夏場や晴れの日だけ行っている", "あまり気にしていない"])
            q8 = st.selectbox("8. 食生活について、油っぽいものや甘いものをよく食べますか？", ["週に4日以上食べる", "週に1〜3日程度", "意識して控えている"])
            q9 = st.selectbox("9. 日常生活で強いストレスを感じる、または疲れが顔に出やすいですか？", ["常に感じる", "たまに感じる", "あまり感じない"])
            q10 = st.selectbox("10. 本日、最も優先して改善したいお悩みは何ですか？", ["たるみ・ほうれい線", "毛穴の開き・黒ずみ", "シミ・くすみ", "ニキビ・赤み", "乾燥・小ジワ"])
            
        submit_button = st.form_submit_button("AI解析×カウンセリング 総合診断結果を出す")

    if submit_button:
        lifestyle_score = 0
        if q1 == "特に変化はない": lifestyle_score += 5
        elif q1 == "部分的に乾燥する": lifestyle_score += 3
        if q2 == "ほとんど気にならない": lifestyle_score += 5
        elif q2 == "少しテカる": lifestyle_score += 3
        if q3 == "ほとんどない": lifestyle_score += 5
        elif q3 == "たまにある": lifestyle_score += 3
        if q4 == "めったにできない": lifestyle_score += 5
        elif q4 == "生理前などたまにできる": lifestyle_score += 3
        if q5 == "あまり気にならない": lifestyle_score += 5
        if q6 == "7時間以上": lifestyle_score += 5
        elif q6 == "5〜6時間": lifestyle_score += 3
        if q7 == "毎日欠かさず行っている": lifestyle_score += 5
        elif q7 == "夏場や晴れの日だけ行っている": lifestyle_score += 2
        if q8 == "意識して控えている": lifestyle_score += 5
        elif q8 == "週に1〜3日程度": lifestyle_score += 3
        if q9 == "あまり感じない": lifestyle_score += 5
        elif q9 == "たまに感じる": lifestyle_score += 3
        
        lifestyle_score = min(50, lifestyle_score + 5)
        total_score = ai_score + lifestyle_score

        if total_score >= 80:
            score_color = "normal" 
            score_eval = "素晴らしい状態です！維持しましょう。"
        elif total_score >= 60:
            score_color = "off" 
            score_eval = "改善の余地あり。今のケアを見直す時期です。"
        else:
            score_color = "inverse" 
            score_eval = "【警告】今すぐ根本的な集中ケアが必要です！"

        st.divider()
        st.header("📊 総合診断結果")
        
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("AI肌解析スコア (50点満点)", f"{ai_score} 点")
        sc2.metric("生活習慣スコア (50点満点)", f"{lifestyle_score} 点")
        sc3.metric("🏆 総合肌スコア (100点満点)", f"{total_score} 点", delta=score_eval, delta_color=score_color)

        skin_type = ""
        if q1 == "強くつっぱる・カサつく" and q2 in ["ほとんど気にならない", "逆に乾燥する"]: skin_type = "乾燥肌"
        elif q1 == "すぐに皮脂が出る" and q2 == "かなりテカる・メイクが崩れる": skin_type = "脂性肌"
        elif q1 == "部分的に乾燥する" and q2 in ["かなりテカる・メイクが崩れる", "少しテカる"]: skin_type = "混合肌"
        elif q1 == "特に変化はない": skin_type = "普通肌"
        else: skin_type = "混合肌"

        if q3 in ["よくある", "たまにある"]: skin_type += "（敏感傾向）"

        st.subheader(f"🔍 あなたの肌質：【{skin_type}】")
        
        col_adv1, col_adv2 = st.columns(2)

        with col_adv1:
            st.subheader("💡 根本改善のための生活習慣アドバイス")
            if q7 != "毎日欠かさず行っている":
                st.warning("【光老化リスク】曇りの日や室内でもUVケアを徹底しないと、画像に現れている潜在シミが表面化するリスクが極めて高い状態です。")
            if q6 == "5時間未満" or q9 == "常に感じる":
                st.info("【ターンオーバー遅延】睡眠不足とストレスにより肌の生まれ変わりが滞っています。これが『くすみ（オレンジ枠）』の最大の原因です。")
            if q8 == "週に4日以上食べる" or "Uゾーン" in q4:
                st.error("【食生活と大人ニキビ】糖分・脂質の過剰摂取は『糖化』や皮脂分泌を招きます。ビタミンB群・Cを積極的に摂取してください。")
            if "乾燥" in skin_type:
                st.success("【バリア機能の低下】洗顔後のつっぱりは水分保持力低下の証拠です。洗いすぎを避け、セラミド等でバリア機能を補ってください。")

        with col_adv2:
            st.subheader("🧴 スコアを上げるための推奨プログラム")
            if q10 == "たるみ・ほうれい線" or nasolabial_count > 3 or sagging_count > 5:
                st.success("**【推奨】NIVORA エイジングケアライン（新ブランド開発中）**\n緑の線（ほうれい線）と紫の線（フェイスライン）にアプローチし、肌の土台からハリを再構築します。")
            elif q10 == "シミ・くすみ" or spot_count > 5 or transparency_score < 60:
                st.info("**【推奨】ROMAN スキンケアローション EX**\nナイアシンアミドがくすみを晴らし、全顔の透明感を底上げします。オレンジ枠のエリアに重ね付けを行ってください。")
            elif q10 in ["ニキビ・赤み", "毛穴の開き・黒ずみ"] or acne_count > 3 or cheek_pore_count > 10:
                st.error("**【推奨】ROMAN アクネケアエッセンス ＆ ローション**\n赤みのある炎症を鎮めつつ、ピンクの点（頬の乾燥毛穴）にはローションでしっかりと水分を補給し引き締めるケアが急務です。")
            else:
                st.success("**【推奨】ROMAN スキンケアローション EX（基本保湿）**\n現状のスコアを維持・向上させるため、バリア機能を高い状態で維持してください。")

st.sidebar.info("このツールはサロン専用の物販支援システムです。スコアはお客様のモチベーション管理にご活用ください。")