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
    layout="centered"
)

# --- CUSTOM CSS (HIGH CONTRAST & READABLE) ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700;900&family=Pacifico&display=swap');

    /* Background Aplikasi: Abu-abu sangat muda agar tidak silau */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Judul Utama */
    h1 {
        font-family: 'Pacifico', cursive;
        color: #ff2b2b; /* Merah yang lebih gelap agar terbaca */
        text-align: center;
        font-size: 3rem !important;
        margin-bottom: 10px;
    }
    
    /* Subjudul & Teks Biasa */
    p, .stMarkdown {
        font-family: 'Nunito', sans-serif;
        color: #2c3e50 !important; /* Biru gelap hampir hitam */
        font-size: 1.1rem;
    }

    /* Card Style: PUTIH SOLID (Bukan transparan) agar tulisan jelas */
    .clean-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Bayangan lembut */
        border: 1px solid #e1e4e8;
        margin-bottom: 20px;
        text-align: center;
    }

    /* Tombol Upload */
    .stFileUploader {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #ff2b2b;
    }

    /* Tombol Analisis */
    .stButton>button {
        width: 100%;
        background-color: #ff2b2b; /* Merah solid */
        color: white !important;
        border: none;
        border-radius: 10px;
        height: 55px;
        font-size: 18px;
        font-weight: 900;
        font-family: 'Nunito', sans-serif;
        box-shadow: 0 4px 0 #c92222; /* Efek tombol timbul */
        transition: all 0.1s;
    }
    .stButton>button:active {
        transform: translateY(4px);
        box-shadow: none;
    }

    /* Angka Skor Besar */
    .big-score {
        font-family: 'Nunito', sans-serif;
        font-weight: 900;
        font-size: 70px;
        color: #ff2b2b;
        margin: 0;
        line-height: 1;
    }
    
    /* Label User */
    .user-name {
        font-weight: 900;
        font-size: 1.4rem;
        color: #1a1a1a; /* Hitam pekat */
        margin-bottom: 5px;
    }
    .user-stat {
        font-size: 1rem;
        color: #4a4a4a; /* Abu tua */
        font-weight: 600;
    }
    
    /* Garis Pemisah */
    hr {
        border-top: 2px solid #eee;
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
    # --- TAMBAHAN: BATASI PESAN AGAR CEPAT ---
    # Jika chat lebih dari 1500, ambil 1500 terakhir saja
    limit = 5000
    if len(df) > limit:
        # Memberi info ke user kalau data dipotong
        st.warning(f"File terlalu besar ({len(df)} pesan). Menganalisis {limit} pesan terakhir saja agar tidak timeout.")
        df = df.tail(limit).reset_index(drop=True)
    # -----------------------------------------

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
    st.markdown("<h1>CintaMeter AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 30px;'><b>Analisis Kecocokan Pasangan</b><br>Unggah riwayat chat WhatsApp (.txt) untuk melihat hasilnya.</p>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='clean-card'><h4>📂 Upload File Chat</h4><p style='font-size:0.9rem;'>Pastikan format ekspor WhatsApp benar (tanpa media).</p></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type="txt", label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode("utf-8")
            df = parse_whatsapp_txt(content)
            df['Pesan_Bersih'] = df['Pesan'].apply(bersihkan_teks)
            df = df[df['Pesan_Bersih'] != ""]
            
            # --- PERBAIKAN DI SINI: LIMIT DATA SEBELUM ANALISIS ---
            total_chat = len(df)
            limit = 5000 # Batas maksimal chat yang dianalisis agar tidak error
            
            if total_chat > limit:
                st.warning(f"⚠️ Chat kamu sangat banyak ({total_chat} pesan). Demi kecepatan, AI hanya akan menganalisis {limit} pesan terakhir.")
                # Kita potong DataFrame-nya DI SINI
                df = df.tail(limit).reset_index(drop=True)
            
            st.success(f"✅ Siap menganalisis {len(df)} pesan percakapan.")
            
            # Tombol Analisis Besar
            st.write("")
            if st.button("Mulai Analisis Sekarang!"):
                with st.spinner('⏳ Sedang menganalisis kata-kata kalian...'):
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
                
                # --- TAMPILAN HASIL ---
                st.write("---")
                
                # Kesimpulan Warna Warni tapi Jelas
                if final_match_score > 80:
                    st.balloons()
                    match_color = "#28a745" # Hijau
                    msg = "SANGAT COCOK! Chemistry Kuat."
                elif final_match_score > 60:
                    st.balloons()
                    match_color = "#17a2b8" # Biru
                    msg = "COCOK. Komunikasi Positif."
                else:
                    match_color = "#ffc107" # Kuning
                    msg = "NETRAL / PERLU USAHA."

                # Skor Utama
                st.markdown(f"""
                <div class='clean-card' style='border: 2px solid {match_color};'>
                    <h3 style='margin:0; color:#555;'>SKOR KECOCOKAN</h3>
                    <div class='big-score' style='color:{match_color}'>{final_match_score:.1f}%</div>
                    <h4 style='color:{match_color}; margin-top:10px;'>{msg}</h4>
                </div>
                """, unsafe_allow_html=True)

                # Card Statistik Kiri Kanan
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class='clean-card'>
                        <div class='user-name'>{p1}</div>
                        <div class='user-stat'>💬 {count_p1} Chat</div>
                        <div class='user-stat'>❤️ Vibe: {sentimen_p1:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class='clean-card'>
                        <div class='user-name'>{p2}</div>
                        <div class='user-stat'>💬 {count_p2} Chat</div>
                        <div class='user-stat'>❤️ Vibe: {sentimen_p2:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Grafik Timeline
                st.markdown("### 📈 Grafik Mood Percakapan")
                df['Index'] = range(len(df))
                window_size = max(5, int(len(df)/20))
                df_viz = df.copy()
                df_viz['MA_Sentiment'] = df_viz.groupby('Pengirim')['Skor_Sentimen'].transform(lambda x: x.rolling(window_size).mean())
                
               # --- UPDATE GRAFIK: TITLE, WARNA & LABEL ---
                # Mapping warna manual: P1 = Merah (#d62728), P2 = Biru (#1f77b4)
                warna_pasti = {p1: '#d62728', p2: '#1f77b4'}
                
                fig = px.line(df_viz, x='Index', y='MA_Sentiment', color='Pengirim', 
                              title='Grafik Mood', # Judul Grafik ditambahkan
                              labels={'Index': 'Urutan Pesan', 'MA_Sentiment': 'Mood (Positif/Negatif)'},
                              color_discrete_map=warna_pasti, # Menggunakan mapping warna manual
                              template="plotly_white") # Memaksa tema terang
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='#f8f9fa',
                    # Memaksa seluruh font menjadi Hitam Pekat (#000000)
                    font=dict(color='#000000', family='Nunito, sans-serif', size=14),
                    title_font_color="#000000",
                    legend_title_font_color="#000000"
                )
                
                # Memaksa warna sumbu X dan Y menjadi hitam
                fig.update_xaxes(title_font=dict(color='#000000'), tickfont=dict(color='#000000'), showgrid=True, gridcolor='#e1e4e8')
                fig.update_yaxes(title_font=dict(color='#000000'), tickfont=dict(color='#000000'), showgrid=True, gridcolor='#e1e4e8')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Keterangan Legenda Manual
                st.markdown(f"""
                <div style='text-align:center; margin-top: -10px; margin-bottom: 30px;'>
                    <span style='color:#d62728; font-weight:bold; font-size:1.1rem;'>🔴 {p1}</span> 
                    &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; 
                    <span style='color:#1f77b4; font-weight:bold; font-size:1.1rem;'>🔵 {p2}</span>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Gagal memproses file. Error: {e}")

if __name__ == "__main__":
    main()








