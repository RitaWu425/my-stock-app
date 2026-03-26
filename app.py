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
st.set_page_config(page_title="台股籌碼智慧診斷系統", layout="wide")

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

# --- CSS 樣式修正 (字體大小對調與正負色) ---
st.markdown("""
    <style>
    .stMetric label { font-size: 14px !important; color: #BBBBBB !important; }
    .stMetric div[data-testid="stMetricValue"] { font-size: 32px !important; font-weight: bold !important; }
    .data-label { font-size: 18px; font-weight: bold; color: #FFFFFF; margin-bottom: 5px; }
    .val-pos { font-size: 22px; font-weight: bold; color: #ff4b4b; } /* 紅色 */
    .val-neg { font-size: 22px; font-weight: bold; color: #00c853; } /* 綠色 */
    .val-neu { font-size: 22px; font-weight: bold; color: #FFFFFF; }
    </style>
    """, unsafe_allow_html=True)

def get_color_html(val, is_pct=False):
    fmt = f"{val:+.2f}%" if is_pct else f"{val:+,d}"
    if val > 0: return f'<span class="val-pos">{fmt}</span>'
    elif val < 0: return f'<span class="val-neg">{fmt}</span>'
    return f'<span class="val-neu">{fmt}</span>'

# --- 2. 日期判定與側邊欄 ---
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
            
            # 財報 (抓取最近一年份)
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
            最新股價, 最新5MA = 最新['close'], 最新['5MA']
            今日張數 = 最新[vol_col] // 1000
            今日5MA量 = 股價資料[vol_col].tail(5).mean() // 1000

            # RSI
            delta = 股價資料['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
            最新RSI = (100 - (100 / (1 + (gain / loss.replace(0, 0.001))))).iloc[-1]

            # 法人運算
            def net_sum(df, name):
                if df.empty: return 0
                return (df[df['name'] == name]['buy'].sum() - df[df['name'] == name]['sell'].sum()) // 1000

            外資, 投信 = net_sum(法人資料, 'Foreign_Investor'), net_sum(法人資料, 'Investment_Trust')
            自營, 權證 = net_sum(法人資料, 'Dealer_self'), net_sum(法人資料, 'Dealer_Hedging')
            籌碼集中度 = ((外資 + 投信) * 1000 / 股價資料[vol_col].sum()) * 100 if not 法人資料.empty else 0

        # --- 顯示介面 ---
        st.subheader(f"📊 {股票代號} {股名} 終極診斷報告")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最新股價", f"{最新股價}", f"{最新股價-昨日['close']:.2f}")
        m2.metric("5MA 均線", f"{最新5MA:.1f}")
        m3.metric("RSI 指標", f"{最新RSI:.1f}")
        m4.metric("籌碼集中度", f"{籌碼集中度:.2f}%")

        # 法人動向
        st.markdown("### 👥 法人與權證動向 (區間累計)")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<p class="data-label">外資：{get_color_html(外資)} 張</p>', unsafe_allow_html=True)
        c2.markdown(f'<p class="data-label">投信：{get_color_html(投信)} 張</p>', unsafe_allow_html=True)
        c3.markdown(f'<p class="data-label">自營商：{get_color_html(自營)} 張</p>', unsafe_allow_html=True)
        c4.markdown(f'<p class="data-label">權證避險：{get_color_html(權證)} 張</p>', unsafe_allow_html=True)

        # 信用與借券
        st.markdown("### 📉 信用交易與借券 (SBL)")
        d1, d2 = st.columns(2)
        融資變動 = (融資券資料.iloc[-1]['MarginPurchaseTodayBalance'] - 融資券資料.iloc[-2]['MarginPurchaseTodayBalance']) // 1000 if not 融資券資料.empty else 0
        融券餘額 = 融資券資料.iloc[-1]['ShortSaleTodayBalance'] // 1000 if not 融資券資料.empty else 0
        d1.markdown(f'<p class="data-label">融資變動：{get_color_html(融資變動)} 張 | 融券餘額：{融券餘額:,.0f} 張</p>', unsafe_allow_html=True)

        最新借券餘額 = 0; 今日還券 = 0; 借券賣出 = 0
        if not 借券資料.empty:
            sbl = 借券資料.iloc[-1]
            最新借券餘額 = sbl['SBLShortSalesPreviousDayBalance'] // 1000
            今日還券 = sbl['SBLShortSalesReturns'] // 1000
            借券賣出 = sbl['SBLShortSalesShortSales'] // 1000
            還券比 = (今日還券 / 今日張數) * 100 if 今日張數 > 0 else 0
            d2.markdown(f'<p class="data-label">借券餘額：{最新借券餘額:,.0f} 張 | 今日還券：{get_color_html(今日還券)} ({還券比:.1f}%)</p>', unsafe_allow_html=True)
        
        # 借券分析文字
        if 今日還券 > 借券賣出:
            st.success(f"💥 今日「借券賣出」：{借券賣出} 張 | 今日『還券』大於『賣出』，淨回補 {今日還券-借券賣出:,.0f} 張，空頭力量消退中。")
        else:
            st.error(f"💥 今日「借券賣出」：{借券賣出} 張 | 賣出大於還券，法人空方力道仍存。")

        # --- 法人與智慧診斷文字區 ---
        st.markdown("---")
        st.write("📝 **法人籌碼分析：**")
        if not 法人資料.empty:
            last_inst = 法人資料[法人資料['date'] == 法人資料['date'].max()]
            f_buy = last_inst[last_inst['name'] == 'Foreign_Investor']['buy'].sum() - last_inst[last_inst['name'] == 'Foreign_Investor']['sell'].sum()
            t_buy = last_inst[last_inst['name'] == 'Investment_Trust']['buy'].sum() - last_inst[last_inst['name'] == 'Investment_Trust']['sell'].sum()
            if f_buy > 0 and t_buy > 0: st.write("✅ **[英雄所見略同]**：外資與投信今日同步買超，通常是強力的止跌或攻擊訊號。")
            elif f_buy < 0 and t_buy > 0: st.write("⚠️ **[土洋對作]**：投信力挺但外資在倒貨，需觀察 5MA 支撐是否能守住。")
            elif f_buy > 0 and t_buy < 0: st.write("⚠️ **[外熱內冷]**：外資回頭補貨，但內資投信先行獲利了結，股價易陷入震盪。")
            else: st.write("❌ **[法人棄守]**：主要法人同步站回賣方，短線建議保守看待。")

        st.success("🧠 **圖表智慧診斷總結**")
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.write(f"● **[技術面]**: {'⚠️ 均線壓制' if 最新股價 < 最新5MA else '✅ 短線轉強'}")
            st.write(f"● **[籌碼鎖定]**: 借券餘額 `{最新借券餘額:,.0f}` 張。{'餘額偏高，需防範打壓。' if 最新借券餘額 > 5000 else '籌碼穩定。'}")
        with diag_col2:
            st.write(f"● **[買盤力道]**: {'🔥 買盤積極' if 今日張數 > 今日5MA量 else '🧊 追價乏力'}")
            st.write(f"● **[指標訊號]**: RSI(`{最新RSI:.1f}`) 處於 {'超賣區' if 最新RSI < 30 else '中性區'}。")

        # --- 圖表區 (補回法人五日趨勢圖與財報二合一圖) ---
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["價格趨勢", "法人五日進出趨勢", "基本面：營收與毛利趨勢"])
        
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(股價資料['date'], 股價資料['close'], label='收盤價')
            ax1.plot(股價資料['date'], 股價資料['5MA'], label='5MA', linestyle='--')
            ax1.legend(); st.pyplot(fig1)

        with tab2:
            if not 法人資料.empty:
                inst_pivot = 法人資料.pivot(index='date', columns='name', values=['buy', 'sell'])
                inst_net = (inst_pivot['buy'] - inst_pivot['sell']) // 1000
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                inst_net[['Foreign_Investor', 'Investment_Trust']].tail(5).plot(kind='bar', ax=ax2)
                ax2.set_title("近五日法人買賣超 (張)")
                st.pyplot(fig2)

        with tab3:
            if not 營收資料.empty:
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                ax3.bar(pd.to_datetime(營收資料['date']), 營收資料['revenue']//1000000, color='#add8e6', label='營收(百萬)')
                if not 損益表.empty:
                    ax4 = ax3.twinx()
                    gpm = 損益表[損益表['type'] == 'GrossProfitMargin'].tail(4)
                    ax4.plot(pd.to_datetime(gpm['date']), gpm['value'], color='red', marker='o', label='毛利率%')
                st.pyplot(fig3)
            else: st.warning("暫無基本面資料可供顯示。")

    except Exception as e:
        st.error(f"❌ 診斷發生錯誤：{e}")
