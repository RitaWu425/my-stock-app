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
        個股資訊 = dl.taiwan_stock_info()
    股名 = 個股資訊[個股資訊['stock_id'] == 股票代號]['stock_name'].values[0]

    print(f"\n✅ 正在讀取 {股票代號} {股名} 從 {開始日期} 到 {結束日期} 的資料...")

    法人資料 = dl.taiwan_stock_institutional_investors(stock_id=股票代號, start_date=開始日期, end_date=結束日期)
    股價資料 = dl.taiwan_stock_daily(stock_id=股票代號, start_date=開始日期, end_date=結束日期)
    融資券資料 = dl.taiwan_stock_margin_purchase_short_sale(stock_id=股票代號, start_date=開始日期, end_date=結束日期)
    借券資料 = dl.get_data(dataset="TaiwanDailyShortSaleBalances", data_id=股票代號, start_date=開始日期, end_date=結束日期)

    # --- 4. 核心數據計算 (接續您原本嚴密的邏輯) ---
    股價資料['date'] = pd.to_datetime(股價資料['date'])
    股價資料['5MA'] = 股價資料['close'].rolling(5).mean()
    股價資料['Trading_Volume_Lots'] = 股價資料['Trading_Volume'] // 1000
    股價資料['5MA_Vol'] = 股價資料['Trading_Volume_Lots'].rolling(5).mean()

    # RSI 計算
    delta = 股價資料['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    rs = gain / loss
    股價資料['RSI'] = 100 - (100 / (1 + rs))

    最新, 昨日 = 股價資料.iloc[-1], 股價資料.iloc[-2]
    最新股價, 最新5MA, 最新RSI = 最新['close'], 最新['5MA'], 最新['RSI']
    今日張數, 今日5MA量 = 最新['Trading_Volume_Lots'], 最新['5MA_Vol']

    # 法人分項計算 (張)
    def get_net_buy(df, name):
        target = df[df['name'] == name]
        return (target['buy'].sum() - target['sell'].sum()) // 1000

    區間外資 = get_net_buy(法人資料, 'Foreign_Investor')
    區間投信 = get_net_buy(法人資料, 'Investment_Trust')
    區間自營 = get_net_buy(法人資料, 'Dealer_self')
    區間權證 = get_net_buy(法人資料, 'Dealer_Hedging')
    法人總計 = 區間外資 + 區間投信 + 區間自營 + 區間權證
    籌碼集中度 = (法人總計 / 股價資料['Trading_Volume_Lots'].sum()) * 100

    # 信用與借券
    今日融資變動 = (融資券資料.iloc[-1]['MarginPurchaseTodayBalance'] - 融資券資料.iloc[-2]['MarginPurchaseTodayBalance']) // 1000
    融券總餘額 = 融資券資料.iloc[-1]['ShortSaleTodayBalance'] // 1000

    # 借券變數初始化
    最新借券餘額, 今日還券, 今日借券賣出, 借券淨變動, 還券比, 連續回補 = 0, 0, 0, 0, 0.0, 0

    if not 借券資料.empty:
        借券資料['date'] = pd.to_datetime(借券資料['date'])
        sbl_最新 = 借券資料.iloc[-1]
        最新借券餘額 = sbl_最新['SBLShortSalesPreviousDayBalance'] // 1000
        今日還券 = sbl_最新['SBLShortSalesReturns'] // 1000
        今日借券賣出 = sbl_最新['SBLShortSalesShortSales'] // 1000
        借券淨變動 = 今日借券賣出 - 今日還券
        還券比 = (今日還券 / 今日張數) * 100 if 今日張數 > 0 else 0
        for i in range(len(借券資料)-1, -1, -1):
            if 借券資料.iloc[i]['SBLShortSalesReturns'] > 借券資料.iloc[i]['SBLShortSalesShortSales']: 連續回補 += 1
            else: break

    # --- 5. 輸出報表 (您的原邏輯) ---
    print(f"\n--- 📊 {股票代號} {股名} 終極診斷總表 ---")
    print(f"💰 股價/技術：{最新股價} 元 (5MA: {最新5MA:.1f} | RSI: {最新RSI:.1f})")
    print(f"📊 量能表現：今日 {今日張數:,.0f} 張 (5MA量: {今日5MA量:.1f} 張)")
    print(f"🎯 區間籌碼集中度：{籌碼集中度:.2f}%")

    print(f"\n--- 👥 法人與權證動向 (區間累計張數) ---")
    print(f"外資：{區間外資:+,d} | 投信：{區間投信:+,d}")
    print(f"自營商：{區間自營:+,d} | 權證避險：{區間權證:+,d}")

    print(f"\n--- 📉 信用交易與借券 (SBL) ---")
    print(f"融資變動：{今日融資變動:+,d} 張 | 融券總餘額：{融券總餘額:,.0f} 張")

    if not 借券資料.empty: # Only print borrow-specific details if data exists
        print(f"最新借券餘額：{最新借券餘額:,.0f} 張")
        print(f"💥 今日「借券賣出」：{今日借券賣出:,.0f} 張")
        print(f"今日還券：{今日還券:,.0f} 張 (還券比: {還券比:.2f}%) | 連續回補：{連續回補} 天")

        # 通用分析邏輯 (Correctly placed within the if block)
        if 今日還券 > 今日借券賣出:
            print(f"● [動向]: 今日『還券』大於『賣出』，淨回補 {abs(借券淨變動):,.0f} 張，空頭力量消退中。")
        elif 今日借券賣出 > 今日還券:
            print(f"● [動向]: 今日『賣出』大於『還券』，淨增加 {借券淨變動:,.0f} 張空單，法人持續加壓。")
        else:
            print(f"● [動向]: 今日賣出與還券持平，空頭力道進入觀望期。")

        if 最新借券餘額 > 今日5MA量 * 10:
            print(f"● [潛力]: 借券餘額規模巨大，若配合 RSI 低檔，軋空動能極強.")

    print("\n--- 🔍 數據深度拆解說明 ---")
    # [技術面]
    print(f"● [技術面]: {'股價站上 5MA，短線轉強。' if 最新股價 > 最新5MA else '股價低於 5MA，均線壓制明顯。'}")
    # [法人面]
    分析法人 = "外資、投信同步站回買方，法人底氣足。" if 區間外資 > 0 and 區間投信 > 0 else "法人買賣力道交錯，尚無一致共識。"
    print(f"● [法人面]: {分析法人}")
    # [籌碼面]
    分析籌碼 = "籌碼集中度偏低，且融資若續增將不利洗盤。" if 籌碼集中度 < 1 else "籌碼集中度尚可，法人控盤穩定。"
    print(f"● [籌碼面]: {分析籌碼}")
    # [指標面]
    print(f"● [指標面]: {'RSI 超賣 ({:.1f})，注意反彈訊號。'.format(最新RSI) if 最新RSI < 30 else '指標中性偏弱。'}")

    # --- NEW: 量價面與買盤力道總結 ---
    # B. 量價面與「買盤追價」分析
    if 今日張數 > 今日5MA量 and 最新股價 > 昨日['close']:
        if 籌碼集中度 > 0 or 區間外資 > 0:
            print(f"● [量價面]: 『帶量攻擊』且法人同步買進！這是【真買盤在追】，上攻動能紮實。")
        else:
            print(f"● [量價面]: 帶量上漲但法人未買，可能是散戶進場或單純空頭回補（虛漲）。")
    elif 今日張數 < 今日5MA量:
        print(f"● [量價面]: 價漲量縮或量能低於均值，代表追價意願不足，需防高位震盪。")
    else:
        print(f"● [量價面]: 量能表現平平，目前動能尚未爆發。")

    # F. 買盤力道總結 (買盤在追分析)
    if 還券比 > 15 and 最新股價 > 昨日['close']:
        print(f"⚠️ [預警]: 今日漲幅中，有 {還券比:.1f}% 的交易來自還券，『軋空回補』力道大於『實體買盤』。")
    elif 籌碼集中度 > 5 and 今日張數 > 今日5MA量:
        print(f"✅ [確認]: 籌碼集中度極高且量能噴發，確認為『實體大戶買盤』強力進駐。")
    # --- END NEW SECTIONS ---

    # --- 5. 繪圖 ---
    if not 借券資料.empty:
        plot_df = 借券資料.copy()
        plot_df['Net_回補'] = (plot_df['SBLShortSalesReturns'] - plot_df['SBLShortSalesShortSales']) // 1000
        plot_df['Balance'] = plot_df['SBLShortSalesPreviousDayBalance'] // 1000

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'{股票代號} {股名} - 籌碼與技術戰情圖', fontsize=18, fontweight='bold')
        ax1.plot(股價資料['date'], 股價資料['close'], label='收盤價', color='blue', linewidth=2)
        ax1.plot(股價資料['date'], 股價資料['5MA'], label='5MA', color='orange', linestyle='--')
        ax1.set_ylabel('價格 (元)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.bar(plot_df['date'], plot_df['Net_回補'], label='淨回補量(張)', color='green', alpha=0.5)
        ax2.set_ylabel('當日淨回補張數', color='green')
        ax2.axhline(0, color='black', linewidth=1)
        ax2_r = ax2.twinx()
        ax2_r.plot(plot_df['date'], plot_df['Balance'], label='借券餘額', color='red', linewidth=2)
        ax2_r.set_ylabel('總借券餘額 (張)', color='red')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # --- 6. 結構化圖表智慧診斷 ---
        print("\n--- 🧠 圖表智慧診斷 ---")
        print(f"● [趨勢]: {'目前股價回測 5MA 之下，待帶量站回方可確認反轉。' if 最新股價 < 最新5MA else '股價已站穩 5MA，開啟短期反彈波。'}")
        print(f"● [籌碼]: 借券餘額維持在 {最新借券餘額:,.0f} 張高檔，紅線平移顯示空頭「鎖單」並未大量撤退。")
        print(f"● [追價]: {'量縮整理中，買盤追價力道尚未釋放。' if 今日張數 < 今日5MA量 else '量能放大且還券比上升，空頭被迫回補引發追價。'}")
        print(f"● [診斷]: RSI({最新RSI:.1f}) 於超賣區鈍化後出現連續 {連續回補} 天回補，初步具備底部轉折特徵。")


except Exception as e:
    print(f"❌ 發生錯誤：請確認代號是否正確或日期格式是否為 YYYY-MM-DD")
    print(f"錯誤訊息: {e}")
