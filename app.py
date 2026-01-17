import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
import plotly.express as px
from transformers import pipeline

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="CintaMeter AI",
    page_icon="💖",
    layout="centered"  # Mengubah layout jadi centered agar fokus di tengah (seperti aplikasi HP)
)

# --- CUSTOM CSS & FONTS (UI MEWAH) ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;600;800&family=Pacifico&display=swap');

    /* Background Utama dengan Gradasi Lembut */
    .stApp {
        background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E1 100%);
        font-family: 'Nunito', sans-serif;
    }

    /* Judul Utama */
    h1 {
        font-family: 'Pacifico', cursive;
        color: #FF4B4B;
        text-align: center;
        font-size: 3.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0px;
    }
    
    /* Subjudul */
    h3 {
        font-family: 'Nunito', sans-serif;
        font-weight: 800;
        color: #555;
    }

    /* Card Style (Glassmorphism) */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
        text-align: center;
    }

    /* Styling Tombol Upload */
    .stFileUploader {
        padding: 20px;
        border: 2px dashed #FF4B4B;
        border-radius: 15px;
        background-color: rgba(255, 255, 255, 0.5);
    }

    /* Styling Tombol Analisis */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        border: none;
        border-radius: 50px;
        height: 60px;
        font-size: 20px;
        font-weight: 800;
        font-family: 'Nunito', sans-serif;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.6);
    }

    /* Angka Skor Besar */
    .big-score {
        font-family: 'Fredoka One', cursive;
        font-size: 80px;
        background: -webkit-linear-gradient(#FF4B4B, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1;
    }
    
    /* Teks User Profile */
    .user-name {
        font-weight: 800;
        font-size: 1.2rem;
        color: #333;
        margin-bottom: 5px;
    }
    .user-stat {
        font-size: 0.9rem;
        color: #666;
    }

</style>
""", unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL (CACHE) ---
@st.cache_resource
def load_model():
    model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

# --- FUNGSI PARSING & CLEANING ---
def parse_whatsapp_txt(file_content):
    pattern = r'^\[(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}\.\d{2}\.\d{2}\s?(?:AM|PM)?)\] ([^:]+): (.*)$'
    data = []
    lines = file_content.split('\n')
    date, sender = None, None
    message_buffer = []

    for line in lines:
        line = line.strip()
        match = re.match(pattern, line)
        if match:
            if date and sender and message_buffer:
                data.append([date, sender, " ".join(message_buffer)])
            date_time_str, sender, msg = match.groups()
            msg = msg.lstrip('\u200e')
            date = date_time_str
            message_buffer = [msg]
        else:
            if message_buffer:
                message_buffer.append(line)
    
    if date and sender and message_buffer:
        data.append([date, sender, " ".join(message_buffer)])

    df = pd.DataFrame(data, columns=['Waktu', 'Pengirim', 'Pesan'])
    return df

def bersihkan_teks(teks):
    teks = str(teks).lower()
    if "<media omitted>" in teks or "message deleted" in teks: return ""
    teks = re.sub(r'http\S+', '', teks)
    return teks.strip()

# --- LOGIKA SENTIMEN ---
def analyze_sentiment(df, nlp_model):
    results = []
    my_bar = st.progress(0)
    total = len(df)
    
    for i, text in enumerate(df['Pesan_Bersih']):
        try:
            output = nlp_model(text[:512])[0]
            score = output['score']
            label = output['label']
            val = 1 * score if label == 'positive' else -1 * score if label == 'negative' else 0
            results.append(val)
        except:
            results.append(0)
        
        if i % 10 == 0:
            my_bar.progress(min((i + 1) / total, 1.0))
            
    my_bar.empty()
    return results

# --- MAIN APP UI ---
def main():
    # Header
    st.markdown("<h1>CintaMeter AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; margin-bottom: 30px;'>Seberapa cocok kalian berdasarkan chat WhatsApp? Biarkan AI yang menilai! 💘</p>", unsafe_allow_html=True)

    # Container Utama
    with st.container():
        st.markdown("<div class='glass-card'><h4>📂 Langkah 1: Upload Chat</h4><p style='font-size:0.9rem; color:#888;'>Ekspor chat WA ke .txt (tanpa media) lalu upload di sini.</p></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type="txt", label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode("utf-8")
            df = parse_whatsapp_txt(content)
            df['Pesan_Bersih'] = df['Pesan'].apply(bersihkan_teks)
            df = df[df['Pesan_Bersih'] != ""]
            
            st.info(f"✅ Berhasil membaca {len(df)} pesan. Siap dianalisis!")
            
            # Tombol Analisis Besar
            st.write("")
            if st.button("✨ Cek Kecocokan Kami! ✨"):
                with st.spinner('🔍 Sedang membaca perasaan kalian...'):
                    nlp_model = load_model()
                    df['Skor_Sentimen'] = analyze_sentiment(df, nlp_model)
                
                # --- HASIL ANALISIS ---
                senders = df['Pengirim'].value_counts().head(2).index.tolist()
                if len(senders) < 2:
                    st.error("⚠️ Butuh minimal 2 orang dalam chat.")
                    return
                
                p1, p2 = senders[0], senders[1]
                stats = df.groupby('Pengirim')['Skor_Sentimen'].agg(['mean', 'count'])
                
                # Kalkulasi Skor
                sentimen_p1 = stats.loc[p1, 'mean']
                sentimen_p2 = stats.loc[p2, 'mean']
                count_p1 = stats.loc[p1, 'count']
                count_p2 = stats.loc[p2, 'count']
                
                avg_score = (sentimen_p1 + sentimen_p2) / 2
                score_positivity = max(0, min(100, (avg_score + 0.2) * 200))
                balance_ratio = 1 - (abs(count_p1 - count_p2) / (count_p1 + count_p2))
                score_balance = balance_ratio * 100
                final_match_score = (score_positivity * 0.6) + (score_balance * 0.4)
                
                # --- TAMPILAN DASHBOARD HASIL ---
                st.write("---")
                
                # Skor Utama
                st.markdown(f"""
                <div class='glass-card'>
                    <h2 style='margin:0; color:#555;'>Tingkat Kecocokan</h2>
                    <div class='big-score'>{final_match_score:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # Kesimpulan
                if final_match_score > 80:
                    st.balloons()
                    msg = "😍 MATCH MADE IN HEAVEN! Kalian sangat cocok!"
                    alert_type = st.success
                elif final_match_score > 60:
                    st.balloons()
                    msg = "🥰 COCOK! Hubungan yang sehat dan positif."
                    alert_type = st.info
                else:
                    msg = "🤔 PERLU USAHA. Komunikasi bisa ditingkatkan lagi."
                    alert_type = st.warning
                alert_type(msg)

                # Card Statistik Kiri Kanan
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class='glass-card'>
                        <div class='user-name'>{p1}</div>
                        <div class='user-stat'>🗣️ {count_p1} Chat</div>
                        <div class='user-stat'>😊 Vibe: {sentimen_p1:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class='glass-card'>
                        <div class='user-name'>{p2}</div>
                        <div class='user-stat'>🗣️ {count_p2} Chat</div>
                        <div class='user-stat'>😊 Vibe: {sentimen_p2:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Grafik Timeline
                st.markdown("### 📈 Grafik Perasaan (Timeline)")
                df['Index'] = range(len(df))
                window_size = max(5, int(len(df)/20))
                df_viz = df.copy()
                df_viz['MA_Sentiment'] = df_viz.groupby('Pengirim')['Skor_Sentimen'].transform(lambda x: x.rolling(window_size).mean())
                
                fig = px.line(df_viz, x='Index', y='MA_Sentiment', color='Pengirim', 
                              labels={'Index': 'Waktu', 'MA_Sentiment': 'Mood'},
                              color_discrete_sequence=['#FF4B4B', '#4B7BFF'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Gagal memproses file. Pastikan format txt WhatsApp benar. Error: {e}")

if __name__ == "__main__":
    main()