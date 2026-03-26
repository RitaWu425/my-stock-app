import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from FinMind.data import DataLoader
import google.generativeai as genai
import warnings
import os
import urllib.request
from matplotlib import font_manager
from datetime import datetime, timedelta, time

# 基礎設定
warnings.filterwarnings('ignore')
st.set_page_config(page_title="股票籌碼智慧診斷系統", layout="wide")

# --- 1. 中文字體與初始化 ---
@st.cache_resource
def init_all():
    font_path = 'font.ttf'
    if not os.path.exists(font_path):
        try:
            url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
            urllib.request.urlretrieve(url, font_path)
        except: pass
    if os.path.exists(font_path):
        try:
            fe = font_manager.FontEntry(fname=font_path, name='NotoSansCJKtc')
            font_manager.fontManager.ttflist.insert(0, fe)
            plt.rcParams['font.family'] = fe.name
        except: pass
    plt.rcParams['axes.unicode_minus'] = False
    return DataLoader()

dl = init_all()

# --- 樣式自定義 (標題與數據字體) ---
st.markdown("""
    <style>
    .report-title { font-size: 28px !important; font-weight: bold; margin-bottom: 10px; }
    .data-text { font-size: 18px !important; font-weight: 500; }
    .stMetric label { font-size: 16px !important; }
    .stMetric div { font-size: 24px !important; }
    .report-section { font-size: 20px !important; font-weight: bold; color: #E0E0E0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 結束日期自動判定 ---
now = datetime.now()
default_end_date = now.date() - timedelta(days=1)

# --- 3. 側邊欄 ---
st.sidebar.header("📊 診斷參數設定")
股票代號 = st.sidebar.text_input("輸入股票代號", value="3481")
開始日期 = st.sidebar.date_input("開始日期", value=pd.to_datetime("2026-02-01"))
結束日期 = st.sidebar.date_input("結束日期", value=default_end_date)
執行診斷 = st.sidebar.button("開始執行診斷")

if not 執行診斷:
    st.markdown('<p class="report-title">🚀 台股籌碼智慧診斷系統</p>', unsafe_allow_html=True)
    st.info("👈 請在左側輸入設定，並按下「開始執行診斷」。")
    st.markdown(f"💡 **目前系統判定日期**：資料擷取至 `{結束日期}`")
    st.markdown("""
    ### 系統深度分析包含：
    * **籌碼面**：法人動向、融資券、借券回補。
    * **基本面**：財報營收與毛利趨勢。
    * **技術面**：均線結構與指標診斷。
    """)
else:
    try:
        with st.spinner('正在彙整深度報告數據...'):
            # A. 資料抓取
            個股資訊 = dl.taiwan_stock_info()
            股名 = 個股資訊[個股資訊['stock_id'] == 股票代號]['stock_name'].values[0]
            
            股價資料 = dl.taiwan_stock_daily(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            法人資料 = dl.taiwan_stock_institutional_investors(stock_id=股票代號, start_date=str(開始日期))
            融資券資料 = dl.taiwan_stock_margin_purchase_short_sale(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            借券資料 = dl.get_data(dataset="TaiwanDailyShortSaleBalances", data_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            
            # 財報抓取
            財報年 = str(datetime.now().year - 1)
            營收資料 = dl.taiwan_stock_month_revenue(stock_id=股票代號, start_date=f"{財報年}-01-01")
            損益表 = dl.taiwan_stock_financial_statement(stock_id=股票代號, start_date=f"{財報年}-01-01")

            if 股價資料.empty:
                st.error("此區間無交易資料。")
                st.stop()

            # B. 核心數據處理
            股價資料['date'] = pd.to_datetime(股價資料['date'])
            股價資料['5MA'] = 股價資料['close'].rolling(5).mean()
            vol_col = 'Trading_Volume' if 'Trading_Volume' in 股價資料.columns else 'volume'
            股價資料['Vol_Lots'] = 股價資料[vol_col] // 1000
            
            最新 = 股價資料.iloc[-1]
            昨日 = 股價資料.iloc[-2]
            最新5MA = 最新['5MA']
            五日均量 = 股價資料['Vol_Lots'].tail(5).mean()

            # RSI
            delta = 股價資料['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
            最新RSI = (100 - (100 / (1 + (gain / loss.replace(0, 0.001))))).iloc[-1]

            # 法人運算
            def get_net_buy(df, name):
                if df.empty: return 0
                target = df[df['name'] == name]
                return (target['buy'].sum() - target['sell'].sum()) // 1000

            外資 = get_net_buy(法人資料, 'Foreign_Investor')
            投信 = get_net_buy(法人資料, 'Investment_Trust')
            自營 = get_net_buy(法人資料, 'Dealer_self')
            權證 = get_net_buy(法人資料, 'Dealer_Hedging')
            集中度 = ((外資 + 投信) * 1000 / 股價資料[vol_col].sum()) * 100 if not 法人資料.empty else 0

        # --- 顯示介面 ---
        st.markdown(f'<p class="report-title">📊 {股票代號} {股名} 終極診斷報告</p>', unsafe_allow_html=True)
        
        # 數據面板
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最新股價", f"{最新['close']}", f"{最新['close']-昨日['close']:.2f}")
        m2.metric("5MA 均線", f"{最新5MA:.1f}")
        m3.metric("RSI 指標", f"{最新RSI:.1f}")
        m4.metric("籌碼集中度", f"{集中度:.2f}%")

        # 法人與權證動向 (加大字體)
        st.markdown('<p class="report-section">🍇 法人與權證動向</p>', unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)
        f1.markdown(f'<p class="data-text">外資：<span style="color:{"#ff4b4b" if 外資<0 else "#00c853"}">{外資:+,d}</span> 張</p>', unsafe_allow_html=True)
        f2.markdown(f'<p class="data-text">投信：<span style="color:{"#ff4b4b" if 投信<0 else "#00c853"}">{投信:+,d}</span> 張</p>', unsafe_allow_html=True)
        f3.markdown(f'<p class="data-text">自營商：<span style="color:{"#ff4b4b" if 自營<0 else "#00c853"}">{自營:+,d}</span> 張</p>', unsafe_allow_html=True)
        f4.markdown(f'<p class="data-text">權證避險：<span style="color:{"#ff4b4b" if 權證<0 else "#00c853"}">{權證:+,d}</span> 張</p>', unsafe_allow_html=True)

        # 信用與借券 (加大字體)
        st.markdown('<p class="report-section">📉 信用與借券數據詳解</p>', unsafe_allow_html=True)
        今日融資 = 0; 融券餘額 = 0
        if not 融資券資料.empty:
            今日融資 = (融資券資料.iloc[-1]['MarginPurchaseTodayBalance'] - 融資券資料.iloc[-2]['MarginPurchaseTodayBalance']) // 1000
            融券餘額 = 融資券資料.iloc[-1]['ShortSaleTodayBalance'] // 1000
        
        s1, s2 = st.columns(2)
        s1.markdown(f'<p class="data-text">融資變動：`{今日融資:+,d}` 張 | 融券餘額：`{融券餘額:,.0f}` 張</p>', unsafe_allow_html=True)
        
        今日還券 = 0; 借券餘額 = 0; 還券比 = 0
        if not 借券資料.empty:
            sbl_最新 = 借券資料.iloc[-1]
            借券餘額 = sbl_最新['SBLShortSalesPreviousDayBalance'] // 1000
            今日還券 = sbl_最新['SBLShortSalesReturns'] // 1000
            還券比 = (今日還券 / 最新['Vol_Lots']) * 100 if 最新['Vol_Lots'] > 0 else 0
            s2.markdown(f'<p class="data-text">借券餘額：`{借券餘額:,.0f}` 張 | 今日還券：`{今日還券:,.0f}` ({還券比:.1f}%)</p>', unsafe_allow_html=True)

        # 借券動向分析區塊
        st.markdown("#### ● [借券動向分析]")
        借券賣出 = sbl_最新['SBLShortSalesShortSales'] // 1000 if not 借券資料.empty else 0
        if 今日還券 > 借券賣出:
            st.success(f"今日『還券』大於『賣出』，淨回補 `{abs(借券賣出-今日還券):,.0f}` 張，空頭力量消退中。")
        else:
            st.error(f"今日『賣出』大於『還券』，法人空方力道仍存。")

        # --- 🧠 深度拆解說明 (比照附圖豐富化) ---
        st.markdown("---")
        st.markdown('<p class="report-section">📝 數據深度拆解與診斷</p>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📋 區間觀察總結")
            st.write(f"1. **集中度分析**：{'🔵 偏多' if 集中度>0.5 else '🟡 盤整'} ({集中度:.2f}%)")
            st.write(f"2. **量能觀察**：今日成交量 ({最新['Vol_Lots']:,.0f}張) {'低於' if 最新['Vol_Lots'] < 五日均量 else '高於'} 五日均量 ({五日均量:,.0f}張)。")
            st.write(f"3. **借券觀察**：{'空頭回補中' if 今日還券 > 借券賣出 else '空方動能持續'}。")
            
        with c2:
            st.markdown("#### 🔍 各維度詳解")
            st.markdown(f"● **[技術面]**: {'股價站上 5MA，短線轉強。' if 最新['close'] > 最新5MA else '股價低於 5MA，趨勢轉弱。'}")
            st.markdown(f"● **[法人面]**: {'法人合買，動能強勁。' if 外資>0 and 投信>0 else '法人意見分歧，拉扯盤整。'}")
            st.markdown(f"● **[指標面]**: RSI目前為 `{最新RSI:.1f}`，{'進入超賣區，醞釀反彈。' if 最新RSI < 30 else '處於強勢區。' if 最新RSI > 70 else '中性偏弱。'}")
            st.markdown(f"● **[籌碼面]**: {'籌碼趨於集中。' if 集中度 > 0 else '典型大戶丟、散戶撿，結構凌亂。'}")

        st.markdown("---")
        # 最終判定
        if 最新['close'] < 最新5MA and 集中度 < 0:
            st.warning("💡 **最終進出場判斷：【盤整觀望】** - 訊號不一且動能不足，建議觀察 5MA 支撐強度。")
        else:
            st.success("💡 **最終進出場判斷：【偏多看待】** - 籌碼或技術面有撐，可待量能噴發。")

        # --- 📊 戰情圖 ---
        tab1, tab2 = st.tabs(["價格趨勢", "財報基本面"])
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(股價資料['date'], 股價資料['close'], color='#1f77b4', label='收盤價')
            ax1.plot(股價資料['date'], 股價資料['5MA'], color='#ff7f0e', linestyle='--', label='5MA')
            ax1.legend(); st.pyplot(fig1)
        with tab2:
            if not 營收資料.empty:
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                ax2.bar(pd.to_datetime(營收資料['date']), 營收資料['revenue']//1000000, color='#add8e6', alpha=0.6, label='營收(百萬)')
                if not 損益表.empty:
                    ax3 = ax2.twinx()
                    gpm = 損益表[損益表['type'] == 'GrossProfitMargin']
                    ax3.plot(pd.to_datetime(gpm['date']), gpm['value'], color='red', marker='o', label='毛利%')
                st.pyplot(fig2)

        # --- AI 顧問 (優化報錯與提示) ---
        st.markdown("---")
        st.subheader("🤖 AI 投資顧問分析")
        if "GEMINI_API_KEY" in st.secrets:
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"分析台股{股票代號}{股名}: 價格{最新['close']}, 外資{外資:+,d}, RSI{最新RSI:.1f}。請給予專業投資建議。"
                response = model.generate_content(prompt)
                st.info(response.text)
            except: st.warning("🕒 AI 引擎目前忙碌中，請參考上方內建之深度拆解說明。")
    
    except Exception as e:
        st.error(f"❌ 診斷發生錯誤：{e}")
