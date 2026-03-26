import streamlit as st
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from FinMind.data import DataLoader

# --- 0. 網頁基本設定 ---
st.set_page_config(page_title="股票籌碼診斷系統", layout="wide")

# 阻斷警告
warnings.filterwarnings('ignore')

# --- 1. 中文字體處理 (針對 Streamlit Cloud 環境優化) ---
@st.cache_resource
def install_font():
    if not os.path.exists('font.ttf'):
        os.system('wget -O font.ttf https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf')
    import matplotlib.font_manager as fm
    fm.fontManager.addfont('font.ttf')
    plt.rc('font', family='Noto Sans CJK TC')
    plt.rcParams['axes.unicode_minus'] = False

install_font()

# --- 2. 側邊欄：互動參數輸入 ---
st.sidebar.header("📊 診斷參數設定")
股票代號 = st.sidebar.text_input("輸入股票代號", value="3481")
開始日期 = st.sidebar.date_input("開始日期", value=pd.to_datetime("2026-02-01"))
結束日期 = st.sidebar.date_input("結束日期", value=pd.to_datetime("2026-03-25"))

if st.sidebar.button("開始執行診斷"):
    dl = DataLoader()
    
    try:
        # 資料抓取
        with st.spinner('正在從 FinMind 抓取資料...'):
            個股資訊 = dl.taiwan_stock_info()
            股名 = 個股資訊[個股資訊['stock_id'] == 股票代號]['stock_name'].values[0]
            
            法人資料 = dl.taiwan_stock_institutional_investors(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            股價資料 = dl.taiwan_stock_daily(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            融資券資料 = dl.taiwan_stock_margin_purchase_short_sale(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            借券資料 = dl.get_data(dataset="TaiwanDailyShortSaleBalances", data_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))

        # --- 3. 核心數據計算 ---
        股價資料['date'] = pd.to_datetime(股價資料['date'])
        股價資料['5MA'] = 股價資料['close'].rolling(5).mean()
        股價資料['Trading_Volume_Lots'] = 股價資料['Trading_Volume'] // 1000
        股價資料['5MA_Vol'] = 股價資料['Trading_Volume_Lots'].rolling(5).mean()

        delta = 股價資料['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
        rs = gain / loss
        股價資料['RSI'] = 100 - (100 / (1 + rs))

        最新, 昨日 = 股價資料.iloc[-1], 股價資料.iloc[-2]
        最新股價, 最新5MA, 最新RSI = 最新['close'], 最新['5MA'], 最新['RSI']
        今日張數, 今日5MA量 = 最新['Trading_Volume_Lots'], 最新['5MA_Vol']

        def get_net_buy(df, name):
            target = df[df['name'] == name]
            return (target['buy'].sum() - target['sell'].sum()) // 1000

        區間外資 = get_net_buy(法人資料, 'Foreign_Investor')
        區間投信 = get_net_buy(法人資料, 'Investment_Trust')
        區間自營 = get_net_buy(法人資料, 'Dealer_self')
        區間權證 = get_net_buy(法人資料, 'Dealer_Hedging')
        法人總計 = 區間外資 + 區間投信 + 區間自營 + 區間權證
        籌碼集中度 = (法人總計 / 股價資料['Trading_Volume_Lots'].sum()) * 100

        今日融資變動 = (融資券資料.iloc[-1]['MarginPurchaseTodayBalance'] - 融資券資料.iloc[-2]['MarginPurchaseTodayBalance']) // 1000
        融券總餘額 = 融資券資料.iloc[-1]['ShortSaleTodayBalance'] // 1000

        # --- 4. 網頁視覺化輸出 ---
        st.title(f"📈 {股票代號} {股名} 終極診斷報告")
        
        # 頂部儀表板
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最新股價", f"{最新股價} 元", f"{最新股價-昨日['close']:.2f}")
        col2.metric("5MA 均線", f"{最新5MA:.1f}")
        col3.metric("RSI 指標", f"{最新RSI:.1f}")
        col4.metric("籌碼集中度", f"{籌碼集中度:.2f}%")

        # 法人與信用
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("👥 法人與權證動向")
            st.write(f"外資：`{區間外資:+,d}` | 投信：`{區間投信:+,d}`")
            st.write(f"自營商：`{區間自營:+,d}` | 權證避險：`{區間權證:+,d}`")
        with c2:
            st.subheader("📉 信用與借券數據")
            st.write(f"融資變動：`{今日融資變動:+,d}` 張")
            st.write(f"融券總餘額：`{融券總餘額:,.0f}` 張")

        # 借券解析邏輯
        if not 借券資料.empty:
            sbl_最新 = 借券資料.iloc[-1]
            最新借券餘額 = sbl_最新['SBLShortSalesPreviousDayBalance'] // 1000
            今日還券 = sbl_最新['SBLShortSalesReturns'] // 1000
            今日借券賣出 = sbl_最新['SBLShortSalesShortSales'] // 1000
            還券比 = (今日還券 / 今日張數) * 100 if 今日張數 > 0 else 0
            
            連續回補 = 0
            for i in range(len(借券資料)-1, -1, -1):
                if 借券資料.iloc[i]['SBLShortSalesReturns'] > 借券資料.iloc[i]['SBLShortSalesShortSales']: 連續回補 += 1
                else: break

            st.info(f"💡 **借券賣出解析**：最新餘額 `{最新借券餘額:,.0f}` 張 | 連續回補 `{連續回補}` 天 | 還券比 `{還券比:.2f}%`")

        # 深度拆解說明
        st.subheader("🔍 數據深度拆解說明")
        st.write(f"● **[技術面]**: {'股價站上 5MA，短線轉強。' if 最新股價 > 最新5MA else '股價低於 5MA，均線壓制明顯。'}")
        st.write(f"● **[法人面]**: {'外資、投信同步站回買方，法人底氣足。' if 區間外資 > 0 and 區間投信 > 0 else '法人買賣力道交錯，尚無一致共識。'}")
        st.write(f"● **[量價面]**: {'『帶量攻擊』且法人買進！【真買盤在追】。' if 今日張數 > 今日5MA量 and 最新股價 > 昨日['close'] and 區間外資 > 0 else '量能表現平平或買盤追價意願不足。'}")

        # 繪圖區
        st.subheader("📊 籌碼與技術戰情圖")
        plot_df = 借券資料.copy()
        plot_df['Net_回補'] = (plot_df['SBLShortSalesReturns'] - plot_df['SBLShortSalesShortSales']) // 1000
        plot_df['Balance'] = plot_df['SBLShortSalesPreviousDayBalance'] // 1000

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(股價資料['date'], 股價資料['close'], label='收盤價', color='blue')
        ax1.plot(股價資料['date'], 股價資料['5MA'], label='5MA', color='orange', linestyle='--')
        ax1.legend()
        ax2.bar(plot_df['date'], plot_df['Net_回補'], color='green', alpha=0.5, label='淨回補')
        ax2_r = ax2.twinx()
        ax2_r.plot(plot_df['date'], plot_df['Balance'], color='red', label='借券餘額')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        st.pyplot(fig)

        # 智慧診斷
        st.success("🧠 **圖表智慧診斷**")
        st.write(f"● **[趨勢]**: {'目前股價回測 5MA 之下' if 最新股價 < 最新5MA else '股價已站穩 5MA'}。")
        st.write(f"● **[籌碼]**: 借券餘額 `{最新借券餘額:,.0f}` 張，空頭鎖單中。")
        st.write(f"● **[診斷]**: RSI(`{最新RSI:.1f}`) 配合連續 `{連續回補}` 天回補，具備轉折特徵。")

    except Exception as e:
        st.error(f"❌ 診斷失敗：{e}")
else:
    st.write("👈 請在左側輸入代號並點擊「開始執行診斷」")