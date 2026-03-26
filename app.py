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
from datetime import datetime, time

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

# --- 2. 結束日期自動判定邏輯 ---
now = datetime.now()
if now.time() >= time(14, 0):
    default_end_date = now.date()
else:
    default_end_date = now.date() - pd.Timedelta(days=1)

# --- 3. 側邊欄參數設定 ---
st.sidebar.header("📊 診斷參數設定")
股票代號 = st.sidebar.text_input("輸入股票代號", value="3481")
開始日期 = st.sidebar.date_input("開始日期", value=pd.to_datetime("2026-02-01"))
結束日期 = st.sidebar.date_input("結束日期", value=default_end_date)

執行診斷 = st.sidebar.button("開始執行診斷")

# --- 4. 畫面分支邏輯 ---
if not 執行診斷:
    st.title("🚀 台股籌碼智慧診斷系統")
    st.info("👈 請在左側輸入股票代號及日期，並按下「開始執行診斷」。")
    st.markdown(f"💡 **目前系統判定日期**：資料擷取至 `{結束日期}`")
    st.markdown("""
    本系統整合以下深度分析：
    - **籌碼面**：三大法人動向、融資券變動、借券回補天數。
    - **技術面**：5MA 趨勢、RSI 強弱指標。
    - **AI 顧問**：基於數據的投資亮點與風險分析。
    """)
else:
    try:
        with st.spinner('正在從 FinMind 抓取並分析資料...'):
            # A. 資料抓取
            個股資訊 = dl.taiwan_stock_info()
            股名 = 個股資訊[個股資訊['stock_id'] == 股票代號]['stock_name'].values[0]
            
            股價資料 = dl.taiwan_stock_daily(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            法人資料 = dl.taiwan_stock_institutional_investors(stock_id=股票代號, start_date=str(開始日期))
            融資券資料 = dl.taiwan_stock_margin_purchase_short_sale(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            借券資料 = dl.get_data(dataset="TaiwanDailyShortSaleBalances", data_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))

            if 股價資料.empty:
                st.error("此日期區間無股價資料。")
                st.stop()

            # B. 核心數據計算
            股價資料['date'] = pd.to_datetime(股價資料['date'])
            股價資料['5MA'] = 股價資料['close'].rolling(5).mean()
            vol_col = 'Trading_Volume' if 'Trading_Volume' in 股價資料.columns else 'volume'
            股價資料['Vol_Lots'] = 股價資料[vol_col] // 1000

            最新 = 股價資料.iloc[-1]
            昨日 = 股價資料.iloc[-2]
            最新股價 = 最新['close']
            最新5MA = 最新['5MA']
            
            delta = 股價資料['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
            最新RSI = (100 - (100 / (1 + (gain / loss.replace(0, 0.001))))).iloc[-1]

            籌碼集中度 = 0.0
            if not 法人資料.empty:
                法人總計 = (法人資料['buy'].sum() - 法人資料['sell'].sum()) // 1000
                籌碼集中度 = (法人總計 / 股價資料['Vol_Lots'].sum()) * 100

        # --- 5. 網頁視覺化輸出：儀表板 ---
        st.title(f"📈 {股票代號} {股名} 終極診斷報告")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最新股價", f"{最新股價}", f"{最新股價-昨日['close']:.2f}")
        col2.metric("5MA 均線", f"{最新5MA:.1f}")
        col3.metric("RSI 指標", f"{最新RSI:.1f}")
        col4.metric("籌碼集中度", f"{籌碼集中度:.2f}%")

        # --- 6. 法人與信用 ---
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        def get_net_buy(df, name):
            if df.empty: return 0
            target = df[df['name'] == name]
            return (target['buy'].sum() - target['sell'].sum()) // 1000

        區間外資 = get_net_buy(法人資料, 'Foreign_Investor')
        區間投信 = get_net_buy(法人資料, 'Investment_Trust')
        區間自營 = get_net_buy(法人資料, 'Dealer_self')
        區間權證 = get_net_buy(法人資料, 'Dealer_Hedging')
        
        今日融資變動 = 0
        融券總餘額 = 0
        if not 融資券資料.empty:
            今日融資變動 = (融資券資料.iloc[-1]['MarginPurchaseTodayBalance'] - 融資券資料.iloc[-2]['MarginPurchaseTodayBalance']) // 1000
            融券總餘額 = 融資券資料.iloc[-1]['ShortSaleTodayBalance'] // 1000

        with c1:
            st.subheader("👥 法人與權證動向")
            st.write(f"外資：`{區間外資:+,d}` | 投信：`{區間投信:+,d}`")
            st.write(f"自營商：`{區間自營:+,d}` | 權證避險：`{區間權證:+,d}`")
        with c2:
            st.subheader("📉 信用與借券數據")
            st.write(f"融資變動：`{今日融資變動:+,d}` 張")
            st.write(f"融券總餘額：`{融券總餘額:,.0f}` 張")

        # --- 7. 借券解析邏輯 ---
        連續回補 = 0
        最新借券餘額 = 0
        if not 借券資料.empty:
            sbl_最新 = 借券資料.iloc[-1]
            最新借券餘額 = sbl_最新['SBLShortSalesPreviousDayBalance'] // 1000
            今日還券 = sbl_最新['SBLShortSalesReturns'] // 1000
            今日張數 = 最新['Vol_Lots']
            還券比 = (今日還券 / 今日張數) * 100 if 今日張數 > 0 else 0
            for i in range(len(借券資料)-1, -1, -1):
                if 借券資料.iloc[i]['SBLShortSalesReturns'] > 借券資料.iloc[i]['SBLShortSalesShortSales']: 
                    連續回補 += 1
                else: break
            st.info(f"💡 **借券摘要**：目前連續回補 `{連續回補}` 天。最新餘額 `{最新借券餘額:,.0f}` 張，還券力道 `{還券比:.2f}%`。")

        # --- 8. 🔍 數據深度拆解說明 ---
        st.markdown("---")
        st.subheader("🔍 數據深度拆解說明")
        col_left, col_right = st.columns(2)
        with col_left:
            st.write(f"● **[技術面]**: {'✅ 股價站在 5MA 之上' if 最新股價 > 最新5MA else '⚠️ 股價低於 5MA'}")
            st.write(f"● **[法人面]**: {'外資、投信站回買方' if 區間外資 > 0 and 區間投信 > 0 else '法人買賣力道交錯'}")
        with col_right:
            st.write(f"● **[指標面]**: RSI `{最新RSI:.1f}` ({'超賣' if 最新RSI < 30 else '中性'})")
            均量 = 股價資料['Vol_Lots'].rolling(5).mean().iloc[-1]
            st.write(f"● **[量價面]**: {'🔥 買盤積極' if 最新['Vol_Lots'] > 均量 else '🧊 追價乏力'}")

        # --- 9. 圖表顯示 ---
        st.subheader("📊 籌碼與技術戰情圖")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(股價資料['date'], 股價資料['close'], label='收盤價', color='blue')
        ax.plot(股價資料['date'], 股價資料['5MA'], label='5MA', color='orange', linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.legend()
        st.pyplot(fig)

        # --- 10. AI 投資顧問分析 ---
        st.markdown("---")
        st.subheader("🤖 AI 投資顧問「白話」分析")
        if "GEMINI_API_KEY" in st.secrets:
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model_names = ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-pro']
                ai_content = ""
                for m_name in model_names:
                    try:
                        model = genai.GenerativeModel(m_name)
                        ai_prompt = f"股票:{股票代號} {股名}, 價格:{最新股價}, RSI:{最新RSI:.1f}, 外資:{區間外資:+,d}, 投信:{區間投信:+,d}, 借券回補:{連續回補}天。請以台股專家身份給出300字內繁體中文建議。"
                        response = model.generate_content(ai_prompt)
                        ai_content = response.text
                        if ai_content: break
                    except: continue
                if ai_content: st.info(f"💡 **AI 診斷結果**：\n\n{ai_content}")
                else: st.warning("🕒 AI 引擎暫時忙碌中。")
            except Exception as e: st.error(f"AI 啟動失敗：{e}")
        else:
            st.warning("⚠️ 找不到 GEMINI_API_KEY。")

    except Exception as e:
        st.error(f"❌ 診斷失敗：{e}")
