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

# --- 樣式設定 (標題縮小、數據放大) ---
st.markdown("""
    <style>
    .small-title { font-size: 24px !important; font-weight: bold; margin-bottom: 15px; }
    .big-data-text { font-size: 20px !important; font-weight: 500; line-height: 1.6; }
    .stMetric label { font-size: 16px !important; }
    .stMetric div { font-size: 26px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 日期判定 ---
now = datetime.now()
default_end_date = now.date() - timedelta(days=1)

# --- 3. 側邊欄 ---
st.sidebar.header("📊 診斷參數設定")
股票代號 = st.sidebar.text_input("輸入股票代號", value="3481")
開始日期 = st.sidebar.date_input("開始日期", value=pd.to_datetime("2026-02-01"))
結束日期 = st.sidebar.date_input("結束日期", value=default_end_date)
執行診斷 = st.sidebar.button("開始執行診斷")

if not 執行診斷:
    st.markdown('<p class="small-title">🚀 台股籌碼智慧診斷系統</p>', unsafe_allow_html=True)
    st.info("👈 請在左側輸入設定，並按下「開始執行診斷」。")
    st.markdown(f"💡 **目前系統判定日期**：資料擷取至 `{結束日期}`")
    st.markdown("""
    本系統整合以下深度分析：
    - **籌碼面**：三大法人動向、融資券變動、借券回補天數。
    - **技術面**：5MA 趨勢、RSI 強弱指標。
    - **AI 顧問**：基於數據的投資亮點與風險分析。
    """)
else:
    try:
        with st.spinner('正在分析大數據...'):
            # A. 資料抓取
            個股資訊 = dl.taiwan_stock_info()
            股名 = 個股資訊[個股資訊['stock_id'] == 股票代號]['stock_name'].values[0]
            
            # 1. 交易與籌碼數據
            股價資料 = dl.taiwan_stock_daily(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            法人資料 = dl.taiwan_stock_institutional_investors(stock_id=股票代號, start_date=str(開始日期))
            融資券資料 = dl.taiwan_stock_margin_purchase_short_sale(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            借券資料 = dl.get_data(dataset="TaiwanDailyShortSaleBalances", data_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            
            # 2. 基本面數據 (抓取最近一期往前推一年)
            # 抓取最近 500 天資料以確保包含四季財報與 12 個月營收
            base_date = (now - timedelta(days=500)).strftime('%Y-%m-%d')
            營收資料 = dl.taiwan_stock_month_revenue(stock_id=股票代號, start_date=base_date).tail(12)
            損益表 = dl.taiwan_stock_financial_statement(stock_id=股票代號, start_date=base_date)
            毛利資料 = 損益表[損益表['type'] == 'GrossProfitMargin'].tail(4) # 最近四季

            # B. 核心運算
            股價資料['date'] = pd.to_datetime(股價資料['date'])
            股價資料['5MA'] = 股價資料['close'].rolling(5).mean()
            vol_col = 'Trading_Volume' if 'Trading_Volume' in 股價資料.columns else 'volume'
            股價資料['Vol_Lots'] = 股價資料[vol_col] // 1000
            
            最新 = 股價資料.iloc[-1]
            昨日 = 股價資料.iloc[-2]
            最新股價 = 最新['close']
            最新5MA = 最新['5MA']
            今日張數 = 最新['Vol_Lots']
            今日5MA量 = 股價資料['Vol_Lots'].tail(5).mean()

            # RSI
            delta = 股價資料['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
            最新RSI = (100 - (100 / (1 + (gain / loss.replace(0, 0.001))))).iloc[-1]

            # 法人數據處理
            def get_net_buy(df, name):
                if df.empty: return 0
                target = df[df['name'] == name]
                return (target['buy'].sum() - target['sell'].sum()) // 1000

            區間外資 = get_net_buy(法人資料, 'Foreign_Investor')
            區間投信 = get_net_buy(法人資料, 'Investment_Trust')
            區間自營 = get_net_buy(法人資料, 'Dealer_self')
            區間權證 = get_net_buy(法人資料, 'Dealer_Hedging')
            籌碼集中度 = ((區間外資 + 區間投信) * 1000 / 股價資料[vol_col].sum()) * 100 if not 法人資料.empty else 0

        # --- 顯示介面 ---
        st.markdown(f'<p class="small-title">📊 {股票代號} {股名} 終極診斷報告</p>', unsafe_allow_html=True)
        
        # 儀表板
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最新股價", f"{最新股價}", f"{最新股價-昨日['close']:.2f}")
        m2.metric("5MA 均線", f"{最新5MA:.1f}")
        m3.metric("RSI 指標", f"{最新RSI:.1f}")
        m4.metric("籌碼集中度", f"{籌碼集中度:.2f}%")

        # 法人動向 (放大字體)
        st.markdown("### 👥 法人與權證動向")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<p class="big-data-text">外資：{區間外資:+,d} 張</p>', unsafe_allow_html=True)
        c2.markdown(f'<p class="big-data-text">投信：{區間投信:+,d} 張</p>', unsafe_allow_html=True)
        c3.markdown(f'<p class="big-data-text">自營商：{區間自營:+,d} 張</p>', unsafe_allow_html=True)
        c4.markdown(f'<p class="big-data-text">權證避險：{區間權證:+,d} 張</p>', unsafe_allow_html=True)

        # 信用與借券 (放大字體)
        st.markdown("### 📉 信用與借券數據詳解")
        d1, d2 = st.columns(2)
        今日融資變動 = 0; 融券總餘額 = 0
        if not 融資券資料.empty:
            今日融資變動 = (融資券資料.iloc[-1]['MarginPurchaseTodayBalance'] - 融資券資料.iloc[-2]['MarginPurchaseTodayBalance']) // 1000
            融券總餘額 = 融資券資料.iloc[-1]['ShortSaleTodayBalance'] // 1000
        d1.markdown(f'<p class="big-data-text">融資變動：{今日融資變動:+,d} 張 | 融券餘額：{融券總餘額:,.0f} 張</p>', unsafe_allow_html=True)

        最新借券餘額 = 0; 今日還券 = 0
        if not 借券資料.empty:
            sbl = 借券資料.iloc[-1]
            最新借券餘額 = sbl['SBLShortSalesPreviousDayBalance'] // 1000
            今日還券 = sbl['SBLShortSalesReturns'] // 1000
            還券比 = (今日還券 / 今日張數) * 100 if 今日張數 > 0 else 0
            d2.markdown(f'<p class="big-data-text">借券餘額：{最新借券餘額:,.0f} 張 | 今日還券：{今日還券:,.0f} ({還券比:.1f}%)</p>', unsafe_allow_html=True)

        # --- 法人行為深度分析 (補回) ---
        st.markdown("---")
        st.write("📝 **法人籌碼分析：**")
        if not 法人資料.empty:
            df_inst = 法人資料.sort_values('date')
            latest_day = df_inst.iloc[-1]
            外資買賣 = latest_day[latest_day['name'] == 'Foreign_Investor']['buy'].sum() - latest_day[latest_day['name'] == 'Foreign_Investor']['sell'].sum()
            投信買賣 = latest_day[latest_day['name'] == 'Investment_Trust']['buy'].sum() - latest_day[latest_day['name'] == 'Investment_Trust']['sell'].sum()
            
            if 外資買賣 > 0 and 投信買賣 > 0:
                st.write("✅ **[英雄所見略同]**：外資與投信今日同步買超，通常是強力的止跌或攻擊訊號。")
            elif 外資買賣 < 0 and 投信買賣 > 0:
                st.write("⚠️ **[土洋對作]**：投信力挺但外資在倒貨，需觀察 5MA 支撐是否能守住。")
            elif 外資買賣 > 0 and 投信買賣 < 0:
                st.write("⚠️ **[外熱內冷]**：外資回頭補貨，但內資投信先行獲利了結，股價易陷入震盪。")
            else:
                st.write("❌ **[法人棄守]**：主要法人同步站回賣方，短線建議保守看待。")

            sitc_5d = 區間投信
            if sitc_5d > 500:
                st.write(f"🔥 **[投信鎖碼]**：投信近期累計買超 `{sitc_5d:,.0f}` 張，有作帳行情跡象。")
        else:
            st.write("暫無足夠的法人進出資料。")

        # --- ８. 完整智慧診斷輸出 ---
        st.markdown("---")
        st.success("🧠 **圖表智慧診斷總結**")
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.write(f"● **[趨勢判讀]**: {'⚠️ 均線壓制：目前股價低於 5MA。' if 最新股價 < 最新5MA else '✅ 短線轉強：股價已站在 5MA 之上。'}")
            if 最新借券餘額 > 0:
                st.write(f"● **[籌碼鎖定]**: 借券餘額 `{最新借券餘額:,.0f}` 張。{'餘額偏高，需防範空頭打壓。' if 最新借券餘額 > 5000 else '籌碼相對穩定。'}")
            else:
                st.write("● **[籌碼鎖定]**: 該個股暫無借券數據紀錄。")
        
        with diag_col2:
            力道 = "🔥 買盤積極：帶量且高於均量。" if 今日張數 > 今日5MA量 else "🧊 追價乏力：量能萎縮中。"
            st.write(f"● **[買盤力道]**: {力道}")
            st.write(f"● **[指標訊號]**: RSI(`{最新RSI:.1f}`) 處於 {'超賣區' if 最新RSI < 30 else '中性區'}。")
        
        st.info(f"🔍 **操作核心建議**：目前 {股票代號} 的籌碼集中度為 `{籌碼集中度:.2f}%`。若集中度轉正且 RSI 站回 30 以上，則具備更強的『軋空』底氣。")

        # --- 📊 綜合戰情圖 (包含營收毛利二合一) ---
        st.markdown("---")
        tab1, tab2 = st.tabs(["價格均線圖", "基本面：營收與毛利趨勢"])
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(股價資料['date'], 股價資料['close'], label='收盤價', color='#1f77b4')
            ax1.plot(股價資料['date'], 股價資料['5MA'], label='5MA', color='#ff7f0e', linestyle='--')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax1.legend(); st.pyplot(fig1)
            
        with tab2:
            if not 營收資料.empty:
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                # 營收長條圖
                ax2.bar(pd.to_datetime(營收資料['date']), 營收資料['revenue']//1000000, color='#add8e6', alpha=0.7, label='月營收(百萬)')
                ax2.set_ylabel("營收 (百萬)")
                # 毛利折線圖 (共用 X 軸)
                if not 毛利資料.empty:
                    ax3 = ax2.twinx()
                    ax3.plot(pd.to_datetime(毛利資料['date']), 毛利資料['value'], color='red', marker='o', linewidth=2, label='毛利率%')
                    ax3.set_ylabel("毛利率 (%)")
                st.pyplot(fig2)
            else:
                st.write("暫無基本面資料可供顯示。")

        # --- AI 顧問 ---
        st.markdown("---")
        if "GEMINI_API_KEY" in st.secrets:
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"分析台股{股票代號}{股名}: 價{最新股價}, 外資{區間外資:+,d}, RSI{最新RSI:.1f}。請給予專業投資建議。"
                response = model.generate_content(prompt)
                st.info(response.text)
            except: st.warning("🕒 AI 引擎目前忙碌中。")

    except Exception as e:
        st.error(f"❌ 診斷發生錯誤：{e}")
