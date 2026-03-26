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
from datetime import datetime, timedelta

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

# --- 樣式設定 (精確對調字體大小與正負色) ---
st.markdown("""
    <style>
    /* 儀表板標籤與數值對調感 */
    .stMetric label { font-size: 14px !important; color: #BBBBBB !important; }
    .stMetric div[data-testid="stMetricValue"] { font-size: 32px !important; font-weight: bold !important; }
    
    /* 法人數據區塊 */
    .data-label { font-size: 18px; font-weight: bold; color: #FFFFFF; }
    .val-pos { font-size: 20px; font-weight: bold; color: #ff4b4b; } /* 正值為紅 */
    .val-neg { font-size: 20px; font-weight: bold; color: #00c853; } /* 負值為綠 */
    .val-neu { font-size: 20px; font-weight: bold; color: #FFFFFF; }
    </style>
    """, unsafe_allow_html=True)

def color_val(val, text_func=lambda x: f"{x:+,d}"):
    """自動判斷正負值並套用顏色標籤"""
    if val > 0: return f'<span class="val-pos">{text_func(val)}</span>'
    elif val < 0: return f'<span class="val-neg">{text_func(val)}</span>'
    return f'<span class="val-neu">{text_func(val)}</span>'

# --- 2. 日期與側邊欄 ---
now = datetime.now()
default_end_date = now.date() - timedelta(days=1)

st.sidebar.header("📊 診斷參數設定")
股票代號 = st.sidebar.text_input("輸入股票代號", value="3481")
開始日期 = st.sidebar.date_input("開始日期", value=pd.to_datetime("2026-02-01"))
結束日期 = st.sidebar.date_input("結束日期", value=default_end_date)
執行診斷 = st.sidebar.button("開始執行診斷")

if not 執行診斷:
    st.title("🚀 台股籌碼智慧診斷系統")
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
        with st.spinner('正在分析數據中...'):
            # A. 資料抓取
            個股資訊 = dl.taiwan_stock_info()
            股名 = 個股資訊[個股資訊['stock_id'] == 股票代號]['stock_name'].values[0]
            
            股價資料 = dl.taiwan_stock_daily(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            法人資料 = dl.taiwan_stock_institutional_investors(stock_id=股票代號, start_date=str(開始日期))
            融資券資料 = dl.taiwan_stock_margin_purchase_short_sale(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            借券資料 = dl.get_data(dataset="TaiwanDailyShortSaleBalances", data_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            
            # 財報 (往前抓 500 天確保跨年)
            base_date = (now - timedelta(days=500)).strftime('%Y-%m-%d')
            營收資料 = dl.taiwan_stock_month_revenue(stock_id=股票代號, start_date=base_date).tail(12)
            損益表 = dl.taiwan_stock_financial_statement(stock_id=股票代號, start_date=base_date)

            if 股價資料.empty: st.stop()

            # B. 運算
            股價資料['date'] = pd.to_datetime(股價資料['date'])
            股價資料['5MA'] = 股價資料['close'].rolling(5).mean()
            vol_col = 'Trading_Volume' if 'Trading_Volume' in 股價資料.columns else 'volume'
            
            最新 = 股價資料.iloc[-1]
            昨日 = 股價資料.iloc[-2]
            最新股價 = 最新['close']
            最新5MA = 最新['5MA']
            今日張數 = 最新[vol_col] // 1000
            今日5MA量 = 股價資料[vol_col].tail(5).mean() // 1000

            # RSI
            delta = 股價資料['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
            最新RSI = (100 - (100 / (1 + (gain / loss.replace(0, 0.001))))).iloc[-1]

            # 法人數據匯總
            def net_buy(df, name):
                if df.empty: return 0
                return (df[df['name'] == name]['buy'].sum() - df[df['name'] == name]['sell'].sum()) // 1000

            外資 = net_buy(法人資料, 'Foreign_Investor')
            投信 = net_buy(法人資料, 'Investment_Trust')
            自營 = net_buy(法人資料, 'Dealer_self')
            權證 = net_buy(法人資料, 'Dealer_Hedging')
            籌碼集中度 = ((外資 + 投信) * 1000 / 股價資料[vol_col].sum()) * 100 if not 法人資料.empty else 0

        # --- 顯示介面 ---
        st.subheader(f"📈 {股票代號} {股名} 終極診斷報告")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最新股價", f"{最新股價}", f"{最新股價-昨日['close']:.2f}")
        m2.metric("5MA 均線", f"{最新5MA:.1f}")
        m3.metric("RSI 指標", f"{最新RSI:.1f}")
        m4.metric("籌碼集中度", f"{籌碼集中度:.2f}%")

        # 1. 法人動向
        st.markdown("### 👥 法人與權證動向")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<p class="data-label">外資：{color_val(外資)} 張</p>', unsafe_allow_html=True)
        c2.markdown(f'<p class="data-label">投信：{color_val(投信)} 張</p>', unsafe_allow_html=True)
        c3.markdown(f'<p class="data-label">自營商：{color_val(自營)} 張</p>', unsafe_allow_html=True)
        c4.markdown(f'<p class="data-label">權證避險：{color_val(權證)} 張</p>', unsafe_allow_html=True)

        # 2. 信用與借券
        st.markdown("### 📉 信用與借券數據詳解")
        d1, d2 = st.columns(2)
        融資變動 = 0; 融券餘額 = 0
        if not 融資券資料.empty:
            融資變動 = (融資券資料.iloc[-1]['MarginPurchaseTodayBalance'] - 融資券資料.iloc[-2]['MarginPurchaseTodayBalance']) // 1000
            融券餘額 = 融資券資料.iloc[-1]['ShortSaleTodayBalance'] // 1000
        d1.markdown(f'<p class="data-label">融資變動：{color_val(融資變動)} 張 | 融券餘額：{融券餘額:,.0f} 張</p>', unsafe_allow_html=True)

        最新借券餘額 = 0; 今日還券 = 0
        if not 借券資料.empty:
            sbl = 借券資料.iloc[-1]
            最新借券餘額 = sbl['SBLShortSalesPreviousDayBalance'] // 1000
            今日還券 = sbl['SBLShortSalesReturns'] // 1000
            還券比 = (今日還券 / 今日張數) * 100 if 今日張數 > 0 else 0
            d2.markdown(f'<p class="data-label">借券餘額：{最新借券餘額:,.0f} 張 | 今日還券：{color_val(今日還券, lambda x: f"{x:,.0f}")} ({還券比:.1f}%)</p>', unsafe_allow_html=True)

        # --- 3. 法人籌碼分析 (徹底修復崩潰問題) ---
        st.markdown("---")
        st.write("📝 **法人籌碼分析：**")
        if not 法人資料.empty:
            # 取得最後一個交易日的明細
            last_date = 法人資料['date'].max()
            last_inst = 法人資料[法人資料['date'] == last_date]
            
            def get_single_day(df, name):
                target = df[df['name'] == name]
                return (target['buy'].sum() - target['sell'].sum()) if not target.empty else 0
            
            當日外資 = get_single_day(last_inst, 'Foreign_Investor')
            當日投信 = get_single_day(last_inst, 'Investment_Trust')
            
            if 當日外資 > 0 and 當日投信 > 0:
                st.write("✅ **[英雄所見略同]**：外資與投信今日同步買超，通常是強力的止跌或攻擊訊號。")
            elif 當日外資 < 0 and 當日投信 > 0:
                st.write("⚠️ **[土洋對作]**：投信力挺但外資在倒貨，需觀察 5MA 支撐是否能守住。")
            elif 當日外資 > 0 and 當日投信 < 0:
                st.write("⚠️ **[外熱內冷]**：外資回頭補貨，但內資投信先行獲利了結。")
            else:
                st.write("❌ **[法人棄守]**：主要法人同步站回賣方，短線建議保守看待。")

            if 投信 > 500:
                st.write(f"🔥 **[投信鎖碼]**：投信近期累計買超 `{投信:,.0f}` 張，有作帳行情跡象。")
        
        # --- 4. 智慧診斷 ---
        st.success("🧠 **圖表智慧診斷總結**")
        colA, colB = st.columns(2)
        with colA:
            st.write(f"● **[趨勢判讀]**: {'⚠️ 均線壓制' if 最新股價 < 最新5MA else '✅ 短線轉強'} (5MA: {最新5MA:.1f})")
            st.write(f"● **[籌碼鎖定]**: 借券餘額 `{最新借券餘額:,.0f}` 張，{'需防範壓力' if 最新借券餘額 > 5000 else '籌碼穩定'}。")
        with colB:
            力道 = "🔥 買盤積極" if 今日張數 > 今日5MA量 else "🧊 追價乏力"
            st.write(f"● **[買盤力道]**: {力道} (今日 {今日張數:,.0f} / 均量 {今日5MA量:,.0f})")
            st.write(f"● **[指標訊號]**: RSI (`{最新RSI:.1f}`) {'超賣醞釀反彈' if 最新RSI < 30 else '中性盤整'}。")

        # --- 5. 財報圖表 ---
        st.markdown("---")
        t1, t2 = st.tabs(["價格趨勢", "基本面營收毛利"])
        with t1:
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(股價資料['date'], 股價資料['close'], label='收盤價')
            ax1.plot(股價資料['date'], 股價資料['5MA'], label='5MA', linestyle='--')
            ax1.legend(); st.pyplot(fig1)
        with t2:
            if not 營收資料.empty:
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                ax2.bar(pd.to_datetime(營收資料['date']), 營收資料['revenue']//1000000, color='#add8e6', label='營收(百萬)')
                if not 損益表.empty:
                    ax3 = ax2.twinx()
                    gpm = 損益表[損益表['type'] == 'GrossProfitMargin'].tail(4)
                    ax3.plot(pd.to_datetime(gpm['date']), gpm['value'], color='red', marker='o', label='毛利率%')
                st.pyplot(fig2)

        # AI 顧問
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-1.5-flash')
            res = model.generate_content(f"分析台股{股票代號}{股名}: 價{最新股價}, 外資{外資:+,d}, 借券回補。請繁體中文簡短分析。")
            st.info(res.text)

    except Exception as e:
        st.error(f"❌ 診斷發生錯誤：{e}")
