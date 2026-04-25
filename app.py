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
    .stMetric label { font-size: 20px !important; color: #BBBBBB !important; }
    .stMetric div[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: normal !important; }
    .data-label { font-size: 18px; font-weight: normal; color: #FFFFFF; margin-bottom: 5px; }
    .val-pos { font-size: 22px; font-weight: bold; color: #ff4b4b; } /* 紅色 */
    .val-neg { font-size: 22px; font-weight: bold; color: #00c853; } /* 綠色 */
    .val-neu { font-size: 22px; font-weight: normal; color: #FFFFFF; }
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
股票代號 = st.sidebar.text_input("輸入股票代號", value="2330")
開始日期 = st.sidebar.date_input("開始日期", value=default_end_date - timedelta(days=365))
結束日期 = st.sidebar.date_input("結束日期", value=default_end_date)
執行診斷 = st.sidebar.button("開始執行診斷")

if not 執行診斷:
    st.title("🚀 台股籌碼智慧診斷系統")
    st.info("👈 請在左側輸入股票代號及日期，並按下「開始執行診斷」。")
    st.markdown(f"💡 **目前系統判定日期**：資料擷取至 :green[{結束日期}]")
    st.markdown("""
    本系統整合以下深度分析：
    - **籌碼面**：三大法人動向、融資券變動、借券回補天數。
    - **技術面**：5MA 趨勢、RSI 強弱指標。
    - **AI 顧問**：基於數據的投資亮點與風險分析。
    """
)
else: # 執行診斷 = True
    DEBUG = True # Set to True to show debug info, False to hide

    # --- 【除錯補強 1】：在 try 開始前強制初始化所有顯示變數，防止 NameError ---
    # 這樣即使下方抓資料失敗，變數依然存在，不會出現 '今日5MA量' is not defined
    外資 = 投信 = 自營 = 權證 = 籌碼集中度 = 0
    區間外資 = 區間投信 = 區間自營 = 區間權證 = 0
    今日融資變動 = 融券總餘額 = 0
    最新借券餘額 = 今日還券 = 借券賣出 = 連續回補 = 還券比 = 0
    最新股價 = 最新5MA = 最新RSI = 今日張數 = 今日5MA量 = 0
    大盤收盤 = 大盤漲跌 = 大盤漲跌幅 = 大盤成交量 = 0
    大盤融資餘額 = 大盤融資增減 = 大盤融券餘額 = 大盤融券增減 = 0
    主力買賣超 = 0 # Initialize new variable
    散戶買賣超 = 0 # Initialize new variable
    昨日 = {"close": 0} # 防止昨日['close']抓不到
    股名 = "載入中..."

    try:
        with st.spinner('正在分析數據中...'):
            # 1. 資料抓取 (保持原樣)
            個股資訊 = dl.taiwan_stock_info()
            try:
                股名 = 個股資訊[個股資訊['stock_id'] == 股票代號]['stock_name'].values[0]
            except: 股名 = "未知"

            法人資料 = dl.taiwan_stock_institutional_investors(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            股價資料 = dl.taiwan_stock_daily(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            融資券資料 = dl.taiwan_stock_margin_purchase_short_sale(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            借券資料 = dl.get_data(dataset="TaiwanDailyShortSaleBalances", data_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            主力散戶資料 = dl.get_data(
                dataset='TaiwanStockTotalInstitutionalInvestors',
                data_id=股票代號,
                start_date=str(開始日期),
                end_date=str(結束日期)
            )
            if DEBUG:
                st.write("--- 主力散戶資料 原始資料 ---")
                st.write("Columns:", 主力散戶資料.columns.tolist())
                st.write("Head:", 主力散戶資料.head())
                st.write("Info:")
                主力散戶資料.info()
                if 'name' in 主力散戶資料.columns:
                    st.write("Unique names:", 主力散戶資料['name'].unique())
                st.write("Tail:", 主力散戶資料.tail())

            # 財報與大盤抓取 (保持原樣)
            財報開始日 = (pd.to_datetime(結束日期) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            基本面資料 = dl.taiwan_stock_financial_statement(stock_id=股票代號, start_date=財報開始日)
            大盤資料 = dl.taiwan_stock_daily(stock_id="TAIEX", start_date=str(開始日期), end_date=str(結束日期))

            # Debug print for 大盤資料 (Moved to st.write for Streamlit visibility)
            if DEBUG:
                st.write("--- 大盤資料 (TAIEX) 原始資料 ---")
                st.write("Columns:", 大盤資料.columns.tolist())
                st.write("Tail:", 大盤資料.tail())

            # 新增：大盤融資融券資料
            融資券總表 = dl.taiwan_stock_margin_purchase_short_sale_total(
            start_date=str(開始日期),
            end_date=str(結束日期)
            )
            if DEBUG:
                st.write("--- 融資券總表 原始資料 ---")
                st.write("Columns:", 融資券總表.columns.tolist())
                st.write("Tail:", 融資券總表.tail())
            # --- 【除錯補強 2】：修正大盤計算，增加 empty 判定 ---
            if not 大盤資料.empty and len(大盤資料) >= 2:
                大盤最新 = 大盤資料.iloc[-1]
                大盤收盤 = float(大盤最新["close"])
                大盤漲跌 = float(大盤最新["spread"])
                前日收盤 = float(大盤資料.iloc[-2]["close"])
                大盤漲跌幅 = (大盤漲跌 / 前日收盤) * 100
                大盤成交量 = float(大盤最新.get("Trading_money", 0)) / 1e8 # Corrected column name
            # [B] 大盤資券補回
            if not 融資券總表.empty:
                # Get the latest date available in the DataFrame
                latest_date = 融資券總表['date'].max()
                latest_data = 融資券總表[融資券總表['date'] == latest_date]

                # Initialize variables to avoid NameError if specific names are not found
                mp_money_today = mp_money_yes = ss_today = ss_yes = 0.0

                # Extract data for MarginPurchaseMoney
                mp_money_row = latest_data[latest_data['name'] == 'MarginPurchaseMoney']
                if not mp_money_row.empty:
                    mp_money_today = mp_money_row['TodayBalance'].iloc[0]
                    mp_money_yes = mp_money_row['YesBalance'].iloc[0]

                # Extract data for ShortSale
                ss_row = latest_data[latest_data['name'] == 'ShortSale']
                if not ss_row.empty:
                    ss_today = ss_row['TodayBalance'].iloc[0]
                    ss_yes = ss_row['YesBalance'].iloc[0]

                # Calculate user-requested metrics
                # 融資餘額 (MarginPurchaseMoney TodayBalance) in millions (億元)
                # 1 億元 = 100,000,000 元
                大盤融資餘額 = mp_money_today / 100000000.0

                # 融資增減 (MarginPurchaseMoney TodayBalance – MarginPurchaseMoney YesBalance) in millions (億元)
                大盤融資增減 = (mp_money_today - mp_money_yes) / 100000000.0

                # 融券餘額 (ShortSale TodayBalance) in thousands of shares (張)
                # 更正：FinMind的ShortSale TodayBalance單位就是張，不需再除以1000
                大盤融券餘額 = ss_today # 原始單位即為「張」

                # 融券增減 (ShortSale TodayBalance – ShortSale YesBalance) in thousands of shares (張)
                # 更正：FinMind的ShortSale TodayBalance單位就是張，不需再除以1000
                大盤融券增減 = (ss_today - ss_yes) # 原始單位即為「張」

            # --- 【除錯補強 3】：修正 KeyError: 'data'，確保股價資料不為空才執行 ---
            if not 股價資料.empty and len(股價資料) >= 2:
                # 只有在有資料時才進行日期轉換與指標計算
                if 'date' in 股價資料.columns:
                    股價資料['date'] = pd.to_datetime(股價資料['date'])

                股價資料['5MA'] = 股價資料['close'].rolling(5).mean()
                vol_col = 'Trading_Volume' if 'Trading_Volume' in 股價資料.columns else 'volume'
                股價資料['Trading_Volume_Lots'] = 股價資料[vol_col] // 1000
                股價資料['5MA_Vol'] = 股價資料['Trading_Volume_Lots'].rolling(5).mean()

                # RSI 計算
                delta = 股價資料['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
                股價資料['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))

                最新 = 股價資料.iloc[-1]
                昨日 = 股價資料.iloc[-2]
                最新股價 = 最新['close']
                最新5MA = 最新['5MA']
                最新RSI = 最新['RSI']
                今日張數 = 最新['Trading_Volume_Lots']
                今日5MA量 = 最新['5MA_Vol']

            # 3. 法人籌碼計算 (增加判定)
            if not 法人資料.empty:
                def get_net(df, name):
                    target = df[df['name'] == name]
                    return (target['buy'].sum() - target['sell'].sum()) // 1000
                外資 = get_net(法人資料, 'Foreign_Investor')
                投信 = get_net(法人資料, 'Investment_Trust')
                自營 = get_net(法人資料, 'Dealer_self')
                權證 = get_net(法人資料, 'Dealer_Hedging')
                區間外資, 區間投信, 區間自營, 區間權證 = 外資, 投信, 自營, 權證

                總量 = 股價資料['Trading_Volume_Lots'].sum() if not 股價資料.empty else 1
                籌碼集中度 = ((外資 + 投信 + 自營 + 權證) / 總量 * 100)

            # 新增：主力與散戶買賣超計算
            if not 主力散戶資料.empty:
                if 'date' in 主力散戶資料.columns:
                    主力散戶資料['date'] = pd.to_datetime(主力散戶資料['date'])

                latest_date = 主力散戶資料['date'].max()
                latest_data_subset = 主力散戶資料[主力散戶資料['date'] == latest_date]

                main_investors_df = latest_data_subset[latest_data_subset['name'] == 'institutional_investors']
                if not main_investors_df.empty:
                    主力買賣超 = (main_investors_df['buy'].sum() - main_investors_df['sell'].sum()) // 1000
                else:
                    主力買賣超 = 0

                retail_investors_df = latest_data_subset[latest_data_subset['name'] == 'retail_investors']
                if not retail_investors_df.empty:
                    散戶買賣超 = (retail_investors_df['buy'] - retail_investors_df['sell']) // 1000
                else:
                    散戶買賣超 = 0

            # 4. 借券與信用 (增加判定)
            if not 借券資料.empty:
                sbl_最新 = 借券資料.iloc[-1]
                最新借券餘額 = sbl_最新['SBLShortSalesPreviousDayBalance'] // 1000
                今日還券 = sbl_最新['SBLShortSalesReturns'] // 1000
                借券賣出 = sbl_最新['SBLShortSalesShortSales'] // 1000
                還券比 = (今日還券 / 今日張數) * 100 if 今日張數 > 0 else 0

                # 計算連續回補天數
                連續回補 = 0
                for i in range(len(借券資料)-1, -1, -1):
                    if 借券資料.iloc[i]['SBLShortSalesReturns'] > 借券資料.iloc[i]['SBLShortSalesShortSales']:
                        連續回補 += 1
                    else:
                        break

            if not 融資券資料.empty and len(融資券資料) >= 2:
                今日融資變動 = (融資券資料.iloc[-1]['MarginPurchaseTodayBalance'] - 融資券資料.iloc[-2]['MarginPurchaseTodayBalance'])
                融券總餘額 = 融資券資料.iloc[-1]['ShortSaleTodayBalance']

        # --- 4. 網頁視覺化輸出 ---
        st.title(f"📈 {股票代號} {股名} 分析報告")

        # 在頂部儀表板前插入大盤資訊
        # 第一行：收盤指數、漲跌幅、總成交量
        st.subheader("📊 大盤資訊")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("加權指數", f"{大盤收盤:,.2f}", f"{大盤漲跌:+.2f}")
        mc2.metric("漲跌幅", f"{大盤漲跌幅:+.2f}%")
        mc3.metric("總成交量", f"{大盤成交量:,.2f} 億")

        # 第二行：融資增減、融資餘額、融券增減、融券餘額
        mc4, mc5, mc6, mc7 = st.columns(4)
        mc4.metric("融資增減", f"{大盤融資增減:+.2f} 億")
        mc5.metric("融資餘額", f"{大盤融資餘額:,.2f} 億")
        mc6.metric("融券增減", f"{大盤融券增減:+,d} 張")
        mc7.metric("融券餘額", f"{大盤融券餘額:,.0f} 張")

        # --- 原本的個股頂部儀表板 ---
        st.subheader(f"📈 {股票代號} {股名} 個股資訊")

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
            st.write(f"外資：:green[{區間外資:+,d}] | 投信：:green[{區間投信:+,d}]")
            st.write(f"自營商：:green[{區間自營:+,d}] | 權證避險：:green[{區間權證:+,d}]")
        with c2:
            st.subheader("📉 信用與借券數據")
            st.write(f"融資變動：:green[{今日融資變動:+,d}] 張")
            st.write(f"融券總餘額：:green[{融券總餘額:,.0f}] 張")

        # 新增：主力與散戶動向
        st.markdown("---")
        st.subheader("💸 市場參與者動向")
        mc_main, mc_retail = st.columns(2)
        with mc_main:
            st.metric("主力買賣超", f"{主力買賣超:+,d} 張", delta=f"{主力買賣超:+,d}", delta_color="normal")
        with mc_retail:
            st.metric("散戶買賣超", f"{散戶買賣超:+,d} 張", delta=f"{散戶買賣超:+,d}", delta_color="inverse") # inverse color for retail (usually sell is good)

        # --- 借券解析邏輯 (修正變數與縮排) ---
        if not 借券資料.empty:
            sbl_最新 = 借券資料.iloc[-1]
            最新借券餘額 = sbl_最新['SBLShortSalesPreviousDayBalance'] // 1000
            今日還券 = sbl_最新['SBLShortSalesReturns'] // 1000
            # 統一變數名稱為 借券賣出，避免下方判斷式報錯
            借券賣出 = sbl_最新['SBLShortSalesShortSales'] // 1000
            還券比 = (今日還券 / 今日張數) * 100 if 今日張數 > 0 else 0

            # 計算連續回補天數
            連續回補 = 0
            for i in range(len(借券資料)-1, -1, -1):
                if 借券資料.iloc[i]['SBLShortSalesReturns'] > 借券資料.iloc[i]['SBLShortSalesShortSales']:
                    連續回補 += 1
                else:
                    break

            # 1. 使用 Columns 條列關鍵數據卡片
            st.markdown("#### 📊 今日借券關鍵數據")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("最新借券餘額", f"{最新借券餘額:,.0f} 張")
            # 借券賣出增加對股市是負面，所以 delta_color 用 inverse (紅增綠減)
            c2.metric("今日借券賣出", f"{借券賣出:,.0f} 張", delta=f"{借券賣出:,.0f}", delta_color="inverse")
            c3.metric("今日還券", f"{今日還券:,.0f} 張", delta=f"{今日還券:,.0f}")
            c4.metric("還券比", f"{還券比:.2f}%")

            # 2. 顯示文字解析摘要 (包含連續回補燈號)
            status_color = "🟢" if 連續回補 > 0 else "⚪"
            st.info(f"{status_color} **借券賣出摘要**：目前連續回補 :green[{連續回補}] 天。最新餘額 :green[{最新借券餘額:,.0f}] 張，整體還券力道為 :green[{還券比:.2f}]%。")

            # 3. 淨回補動態判斷 (修正語法結構)
            if 今日還券 > 借券賣出:
                st.success(f"💥 今日「借券賣出」：:green[{借券賣出:,.0f}] 張 | 今日『還券』大於『賣出』，淨回補 :green[{今日還券 - 借券賣出:,.0f}] 張，空頭力量消退中。")
            else:
                st.error(f"💥 今日「借券賣出」：:green[{借券賣出:,.0f}]張 | 賣出大於還券，法人空方力道仍存。")
        else:
            st.warning("⚠️ 暫無借券資料可供分析。")
        # --- 數據深度拆解說明 (全面升級版) ---
        st.subheader("🔍 數據深度拆解說明")

        col_left, col_right = st.columns(2)

        with col_left:
            # [技術面]
            st.write(f"● **[技術面]**： {'股價高於 5MA，短線趨勢偏多。' if 最新股價 > 最新5MA else '股價低於 5MA，均線壓制明顯。'}")

            # [法人面]
            分析法人 = "外資、投信同步站回買方，法人底氣足。" if 區間外資 > 0 and 區間投信 > 0 else "法人買賣力道交錯，尚無一致共識。"
            st.write(f"● **[法人面]**： {分析法人}")

            # [籌碼面] (補齊)
            分析籌碼 = "籌碼集中度偏高，法人控盤穩定。" if 籌碼集中度 > 2 else "籌碼集中度較低，股價易受散戶情緒影響。"
            st.write(f"● **[籌碼面]**： {分析籌碼}")

        with col_right:
            # [指標面] (補齊)
            st.write(f"● **[指標面]**： {'RSI 超賣 ({:.1f})，注意跌深反彈訊號。'.format(最新RSI) if 最新RSI < 30 else 'RSI 指標目前處於中性區間。'}")

            # [量價面]
            if 今日張數 > 今日5MA量 and 最新股價 > 昨日['close']:
                st.write("● **[量價面]**： 『帶量攻擊』，顯示買盤追價意願強烈。")
            else:
                st.write("● **[量價面]**： 量能表現平平，目前動能尚未爆發。")

            # [基本面] (優化版)
            if not 基本面資料.empty:
                try:
                    # 篩選營業收入與營業利益
                    df_rev = 基本面資料[基本面資料['type'] == 'Revenue']
                    df_oi = 基本面資料[基本面資料['type'] == 'OperatingIncome']

                    if not df_rev.empty:
                        最新營收 = df_rev['value'].iloc[-1]
                        最新日期 = df_rev['date'].iloc[-1]

                        # 簡單判斷營收規模 (以億為單位)
                        st.write(f"● **[基本面]**: 最新財報日 :green[{最新日期}]。季度營收約 :green[{最新營收/1e8:.2f}] 億元。")

                        # 如果有營業利益，判斷是否獲利
                        if not df_oi.empty:
                            最新利益 = df_oi['value'].iloc[-1]
                            本業狀態 = "✅ 本業獲利" if 最新利益 > 0 else "⚠️ 本業虧損"
                            st.write(f"  └─ {本業狀態}：營業利益 :green[{最新利益/1e8:.2f}] 億元。")
                    else:
                        st.write("● **[基本面]**： 財報格式不符，建議至公開觀測站確認。")
                except Exception as e:
                    st.write(f"● **[基本面]**： 解析財報時發生小誤差。")
            else:
                st.write("● **[基本面]**： 即使追蹤一年仍查無財報，請確認代號是否為 ETF 或特殊股。")


        # --- 5. 繪圖與分頁顯示 ---
        st.markdown("---")
        st.subheader("📈 深度戰情圖表分析")

        # 建立四個分頁 (新增「市場參與者動向」)
        tab1, tab2, tab3, tab4 = st.tabs(["🛡️ 技術與借券診斷", "🏦 三大法人動向", "📉 基本面獲利分析", "👥 市場參與者動向"])

        # --- 分頁 1: 技術與借券診斷 ---
        with tab1:
            plot_price = 股價資料.copy()
            plot_sbl = 借券資料.copy()
            if not plot_sbl.empty:
                plot_sbl['date'] = pd.to_datetime(plot_sbl['date'])
                plot_sbl['Net_回補'] = (plot_sbl['SBLShortSalesReturns'] - plot_sbl['SBLShortSalesShortSales']) // 1000
                plot_sbl['Balance'] = plot_sbl['SBLShortSalesPreviousDayBalance'] // 1000

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                fig.suptitle(f'{股票代號} {股名} - 戰情診斷', fontsize=20, fontweight='bold')

                # 上圖：股價與均線
                ax1.plot(plot_price['date'], plot_price['close'], label='收盤價', color='blue', linewidth=2)
                ax1.plot(plot_price['date'], plot_price['5MA'], label='5MA', color='orange', linestyle='--')
                ax1.set_ylabel('價格 (元)')
                ax1.legend(loc='upper left')
                ax1.grid(True, alpha=0.3)

                # 下圖：借券淨變動 (長條圖) 與 借券餘額 (折線圖)
                ax2.bar(plot_sbl['date'], plot_sbl['Net_回補'], label='淨回補量(張)', color='green', alpha=0.5)
                ax2.set_ylabel('當日淨回補張數', color='green')
                ax2.axhline(0, color='black', linewidth=1)

                ax2_r = ax2.twinx()
                ax2_r.plot(plot_sbl['date'], plot_sbl['Balance'], label='借券餘額', color='red', linewidth=2)
                ax2_r.set_ylabel('總借券餘額 (張)', color='red')

                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                st.pyplot(fig)
                # --- 戰情深度拆解文字 (新增於圖表下方) ---
                st.markdown("---")
                st.markdown("### 📝 技術與借券戰情拆解")

                t_col1, t_col2 = st.columns(2)

                with t_col1:
                    st.write("**【技術形態診斷】**")
                    if 最新股價 > 最新5MA:
                        if 昨日['close'] < 股價資料.iloc[-2]['5MA']:
                            st.write(f"🔥 **短線轉強**：今日股價「帶量噴出」站上 5MA ({最新5MA:.2f})，扭轉近期頹勢，具備短線攻擊轉機。")
                        else:
                            st.write(f"沿 5MA 走揚：股價持續維持在 5MA 之上，屬於強勢多頭排列，只要不破 5MA 慣性，多單可續抱。")
                    else:
                        st.write(f"⚠️ **均線壓制**：股價目前受制於 5MA ({最新5MA:.2f}) 之下。若無法帶量站回，需防範慣性下跌壓力，短線切勿盲目摸底.")

                    rsi_text = "超賣區鈍化" if 最新RSI < 20 else "低檔轉折" if 最新RSI < 30 else "中性震盪" if 最新RSI < 70 else "高檔過熱"
                    st.write(f"● **指標訊號**：RSI 目前為 :green[{最新RSI:.1f}]，處於 :green[{rsi_text}] 區間，建議參考借券動向觀察底部是否成形.")

                with t_col2:
                    st.write("**【籌碼空方動向】**")
                    if 連續回補 >= 3:
                        st.write(f"✅ **空頭認賠回補**：借券已連續 :green[{連續回補}] 天回補。這種『還券大於賣出』的現象若配合股價止跌，通常是法人空單撤退、底部浮現的訊號.")
                    elif 借券賣出 > 今日還券:
                        st.write(f"❌ **空方持續加壓**：今日借券賣出張數仍大於還券，且總餘額維持在 :green[{最新借券餘額:,.0f}] 張高位。顯示空頭勢力尚未鬆手，對股價形成實質壓力.")
                    else:
                        st.write("● **鎖單觀察**：借券餘額變動不大。目前空頭處於「鎖單」狀態，正在等待市場下一個變盤訊號.")

                    st.write(f"● **追價力道**：今日成交張數為 :green[{今日張數:,.0f}] 張，與 5MA 均量相比為 :green[{(今日張數/今日5MA量 if 今日5MA量>0 else 1):.2f}] 倍。{'量能不足，推升力道受限。' if 今日張數 < 今日5MA量 else '量能升溫，市場參與度提高。'}")

                # 智慧診斷小結
                if 最新股價 > 最新5MA and 連續回補 >= 1:
                    st.success(f"💡 **診斷結論**：股價站回短期均線且借券開始回補，籌碼面與技術面出現共振轉強，短線勝率提高.")
                elif 最新股價 < 最新5MA and 借券賣出 > 今日還券:
                    st.error(f"💡 **診斷結論**：技術面受壓且空方持續借券賣出，目前屬於『價弱籌碼散』，建議靜待籌碼洗清.")
            else:
                st.warning("⚠️ 暫無借券資料可繪圖。")

        # --- 分頁 2: 三大法人近五日動向 ---
        with tab2:
            if not 法人資料.empty:
                # 整理資料：僅取外資、投信、自營商，並取最近 5 個交易日
                df_inst = 法人資料.groupby(['date', 'name'])['buy'].sum() - 法人資料.groupby(['date', 'name'])['sell'].sum()
                df_inst = df_inst.unstack().tail(5) / 1000 # 轉為張數 (取最後5天)

                if not df_inst.empty:
                    fig_inst, ax_inst = plt.subplots(figsize=(10, 5))
                    df_inst.plot(kind='bar', ax=ax_inst, width=0.8, color=['#FF9999', '#66B2FF', '#99FF99'])

                    ax_inst.set_ylabel('買賣超張數 (張)')
                    ax_inst.set_title(f'{股票代號} 近五日法人進出趨勢', fontsize=14)
                    ax_inst.axhline(0, color='black', linewidth=1, alpha=0.5)
                    ax_inst.grid(axis='y', linestyle='--', alpha=0.4)
                    plt.xticks(rotation=0)
                    st.pyplot(fig_inst)

                    # --- 法人行為深度分析 (強化版) ---
                latest_day = df_inst.iloc[-1]
                st.write("📝 **今日法人重點評析：**")

                # 取得外資與投信數值
                f_buy = latest_day.get('Foreign_Investor', 0)
                t_buy = latest_day.get('Investment_Trust', 0)

                # 運算：近五日累計與連續性 (判斷鎖碼)
                sitc_5d_sum = df_inst['Investment_Trust'].sum()
                foreign_5d_sum = df_inst['Foreign_Investor'].sum()

                # 建立分析欄位
                inst_col1, inst_col2 = st.columns(2)

                with inst_col1:
                    st.write("**【外資與主力動態】**")
                    if f_buy > 0 and foreign_5d_sum > 1000:
                        st.write(f"🚀 **外資波段加碼**：外資今日買超 :green[{f_buy:,.0f}] 張，且近五日累計買超 :green[{foreign_5d_sum:,.0f}] 張。屬於典型的波段佈局，對股價中長線走勢有利.")
                    elif f_buy < 0 and foreign_5d_sum < -1000:
                        st.write(f"🚨 **外資套利撤出**：外資近五日大舉調節 :green[{abs(foreign_5d_sum):,.0f}] 張。需留意是否因國際盤勢變動或避險需求而進行「提款」，短線承接需謹慎.")
                    elif f_buy > 0 and 最新股價 < 最新5MA:
                        st.write(f"📉 **外資低位接盤**：股價雖在均線下，但外資已開始逢低試單。這通常是「左側交易」訊號，觀察是否能靠外資買盤止跌.")
                    else:
                        st.write("● 外資目前買賣力道互有勝負，尚無明顯的單邊波段趨勢。")

                with inst_col2:
                    st.write("**【投信鎖碼與作帳】**")
                    # 判斷投信鎖碼邏輯：近五日買超顯著且佔比提升
                    if sitc_5d_sum > 500:
                        st.write(f"🔥 **投信強勢鎖碼**：投信近五日積極掃貨 :green[{sitc_5d_sum:,.0f}] 張。在台股中，投信連買通常代表「認養股」行情，具備強大的跟單效應，是作帳行情的前兆.")
                    elif t_buy < 0 and sitc_5d_sum > 1000:
                        st.write(f"⚠️ **高檔獲利了結**：雖然投信先前重倉，但今日出現調節。需防範「作帳變結帳」，若跌破關鍵支撐需警戒.")
                    elif t_buy > 0:
                        st.write(f"✅ **內資投信進場**：今日投信小幅佈局 :green[{t_buy:,.0f}] 張。若後續能出現連續性買盤，則有望形成新的支撐.")
                    else:
                        st.write("● 投信暫無明顯進出。目前籌碼多由外資或市場大戶主導。")

                # 智慧綜合評估
                st.markdown("---")
                if f_buy > 0 and t_buy > 0:
                    st.success(f"💡 **籌碼總結 [英雄所見略同]**：外資與投信同步站在買方。兩大勢力方向一致時，股價『漲多跌少』，為目前最理想的多頭籌碼結構.")
                elif f_buy < 0 and t_buy > 0:
                    st.warning(f"💡 **籌碼總結 [土洋對作]**：投信力挺但外資在倒貨。這種情況下股價通常會劇烈震盪，建議以『投信成本線』作為防守參考.")
                elif f_buy < 0 and t_buy < 0:
                    st.error(f"💡 **籌碼總結 [雙殺警訊]**：法人同步棄守。當內外資皆在出貨時，應避開殺低風險，嚴格執行止損.")

        # --- 分頁 3: 基本面 (營收與毛利) ---
        with tab3:
            fund_start_date = (pd.Timestamp.now() - pd.Timedelta(days=500)).strftime('%Y-%m-%d')
            try:
                fund_data = dl.taiwan_stock_financial_statement(stock_id=股票代號, start_date=fund_start_date)
                if not fund_data.empty:
                    df_rev = fund_data[fund_data['type'] == 'Revenue'].tail(4).copy()
                    df_gp = fund_data[fund_data['type'] == 'GrossProfit'].tail(4).copy()

                    if not df_rev.empty and not df_gp.empty:
                        df_merge = pd.merge(df_rev[['date', 'value']], df_gp[['date', 'value']], on='date', suffixes=('_rev', '_gp'))
                        df_merge['rev_billion'] = df_merge['value_rev'] / 1e8
                        df_merge['gp_margin'] = (df_merge['value_gp'] / df_merge['value_rev']) * 100

                        fig_f, ax_f1 = plt.subplots(figsize=(10, 5))
                        ax_f1.bar(df_merge['date'], df_merge['rev_billion'], color='#BDD7EE', label='季度營收(億)', width=0.4)
                        ax_f1.set_ylabel('營收 (億元)', color='#2F5597', fontweight='bold')

                        ax_f2 = ax_f1.twinx()
                        ax_f2.plot(df_merge['date'], df_merge['gp_margin'], color='#ED7D31', marker='o', linewidth=3, label='毛利率(%)')
                        ax_f2.set_ylabel('毛利率 (%)', color='#ED7D31', fontweight='bold')

                        # 標註毛利率數值
                        for i, txt in enumerate(df_merge['gp_margin']):
                            ax_f2.annotate(f'{txt:.1f}%', (df_merge['date'].iloc[i], df_merge['gp_margin'].iloc[i]),
                                         textcoords="offset points", xytext=(0,10), ha='center', color='#C65911', fontweight='bold')

                        plt.title(f"{股票代號} {股名} - 獲利能力趨勢", fontsize=14)
                        st.pyplot(fig_f)

                        latest_gp = df_merge['gp_margin'].iloc[-1]
                        prev_gp = df_merge['gp_margin'].iloc[-2]
                        gp_trend = "📈 提升" if latest_gp > prev_gp else "📉 下滑"
                        st.info(f"💡 **最新季報摘要**：毛利率 {latest_gp:.2f}%，較上季表現 {gp_trend}。")
                else:
                    st.warning("⚠️ 查無財報數據，可能該公司尚未公佈最新季度報表。")
            except Exception as e:
                st.info("ℹ️ 基本面資料正在更新中或暫時無法存取。")

        # --- 分頁 4: 市場參與者動向 ---
        with tab4:
            if not 主力散戶資料.empty:
                plot_investors = 主力散戶資料.copy()
                plot_investors['date'] = pd.to_datetime(plot_investors['date'])

                # Re-calculate 主力買賣超_張 and 散戶買賣超_張 using filtered data
                main_investors_plot_df = plot_investors[plot_investors['name'] == 'institutional_investors'].copy()
                retail_investors_plot_df = plot_investors[plot_investors['name'] == 'retail_investors'].copy()

                main_investors_plot_df['主力買賣超_張'] = (main_investors_plot_df['buy'] - main_investors_plot_df['sell']) // 1000
                retail_investors_plot_df['散戶買賣超_張'] = (retail_investors_plot_df['buy'] - retail_investors_plot_df['sell']) // 1000

                fig_inv, ax_inv = plt.subplots(figsize=(12, 6))
                if not main_investors_plot_df.empty:
                    ax_inv.plot(main_investors_plot_df['date'], main_investors_plot_df['主力買賣超_張'], label='主力買賣超(張)', color='purple', linewidth=2)
                if not retail_investors_plot_df.empty:
                    ax_inv.plot(retail_investors_plot_df['date'], retail_investors_plot_df['散戶買賣超_張'], label='散戶買賣超(張)', color='gray', linestyle='--', linewidth=1)

                ax_inv.axhline(0, color='black', linewidth=0.8, linestyle='-')
                ax_inv.set_title(f'{股票代號} {股名} - 主力與散戶買賣超趨勢', fontsize=16)
                ax_inv.set_xlabel('日期')
                ax_inv.set_ylabel('買賣超張數 (張)')
                ax_inv.legend()
                ax_inv.grid(True, alpha=0.3)
                ax_inv.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                fig_inv.tight_layout()
                st.pyplot(fig_inv)

                st.markdown("---")
                st.markdown("### 📝 市場參與者動向解析")
                st.write("此圖表展示了主力與散戶在指定期間內的買賣超變化。" "<ul><li><span style='color:purple;font-weight:bold;'>主力買賣超</span>：通常代表大額投資者（包含法人或大戶）的動向，其連續性買賣超對股價影響較大。</li>" "<li><span style='color:gray;font-weight:bold;'>散戶買賣超</span>：反映一般投資者的情緒，與主力動向常呈反向關係。</li></ul>", unsafe_allow_html=True)
                st.write("透過觀察兩者趨勢，可判斷市場籌碼是集中於特定大戶，還是分散於散戶，有助於評估股價的穩定性與未來潛力.")
            else:
                st.warning("⚠️ 暫無主力與散戶買賣超資料可繪圖。")

        # --- ８. 完整智慧診斷輸出 (策略建議強化版) ---
        st.markdown("---")
        st.success("🧠 **圖表智慧診斷總結**")

        # --- 策略邏輯運算 (先判定建議) ---
        # 1. 定義多頭條件
        多頭趨勢 = 最新股價 > 最新5MA
        量能增溫 = 今日張數 > 今日5MA量
        法人挺進 = (外資 > 0 or 投信 > 0)
        指標轉強 = 最新RSI > 50

        # 2. 定義操作建議邏輯
        if 多頭趨勢 and 法人挺進 and 量能增溫:
            建議 = "🚀 **強勢加碼**"
            理由 = "股價站穩均線且法人放量推升，屬於多頭攻擊形態。"
        elif 多頭趨勢 and (not 量能增溫):
            建議 = "📈 **持股續抱**"
            理由 = "趨勢雖多但量能稍顯不足，建議續抱觀察是否能補量衝關。"
        elif (not 多頭趨勢) and 指標轉強 and 法人挺進:
            建議 = "💎 **分批買進**"
            理由 = "股價雖受壓制但指標與法人先行轉強，具備落後補漲潛力。"
        elif 多頭趨勢 and 最新RSI > 80:
            動作 = "💰 **獲利了結**"
            策略 = "指標進入極度超買區，短線乖離過大，建議先入袋為安。"
        elif (not 多頭趨勢) and 外資 < 0 and 投信 < 0:
            建議 = "⚠️ **觀望/減碼**"
            理由 = "技術面破位且法人同步棄守，籌碼散亂，建議退場觀望。"
        else:
            建議 = "⌛ **中性盤整**"
            理由 = "目前量價與籌碼方向不一，建議靜待變盤訊號出現。"

        # --- 介面輸出 ---
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.write(f"● **[趨勢判讀]**: {'✅ 短線轉強：股價已站在 5MA 之上。' if 多頭趨勢 else '⚠️ 均線壓制：目前股價低於 5MA。'}")

            # 籌碼鎖定防錯
            sbl_val = locals().get('最新借券餘額', 0)
            if sbl_val > 0:
                sbl_desc = "餘額偏高，需防範空頭打壓。" if sbl_val > 5000 else "籌碼相對穩定。"
                st.write(f"● **[籌碼鎖定]**: 借券餘額 :green[{sbl_val:,.0f}] 張。{sbl_desc}")
            else:
                st.write("● **[籌碼鎖定]**: 該個股暫無借券數據紀錄。")

        with diag_col2:
            st.write(f"● **[買盤力道]**: {'🔥 買盤積極：帶量且高於均量。' if 量能增溫 else '🧊 追價乏力：量能萎縮中。'}")

            rsi_status = "🔥 超買過熱" if 最新RSI > 70 else "❄️ 超賣低迷" if 最新RSI < 30 else "⚖️ 中性區間"
            st.write(f"● **[指標訊號]**: RSI(:green[{最新RSI:.1f}]) 處於 {rsi_status}。")
            # --- 最終操作建議 (排除非法字元與對齊修復) ---

            # 1. 基礎條件定義 (確保變數存在)
            is_bull = 最新股價 > 最新5MA
            is_inst_buying = (外資 > 0 or 投信 > 0)
            is_vol_up = (今日張數 > 今日5MA量)

            # 2. 判斷明確動作邏輯
            if is_bull and is_inst_buying and is_vol_up:
                動作 = "🎯 **操作動作：強勢進場 / 加碼買進**"
                策略 = "條件全數達成，建議建立核心倉位，回檔不破 5MA 皆可加碼。"
            elif is_bull and 最新RSI > 80:
                動作 = "💰 **操作動作：分批止盈 / 減碼**"
                策略 = "指標進入極度超買區，短線乖離過大，建議先入袋為安。"
            elif (not is_bull) and 最新RSI > 50 and is_inst_buying:
                動作 = "⏳ **操作動作：試探性買進 (分批入坑)**"
                策略 = "技術面尚待轉強，但籌碼先行，建議先投入 20% 資金試水溫。"
            elif 最新股價 < 最新5MA and (外資 < 0 or 投信 < 0):
                動作 = "🚫 **操作動作：觀望 / 停損撤出**"
                策略 = "法人反手倒貨且破均線，建議嚴守 5MA 停損，不宜攤平。"
            else:
                動作 = "⌛ **操作動作：空手觀望**"
                策略 = "方向不明朗，建議等待股價帶量站回 5MA 再行佈局。"

            # 3. 顯示結果 (使用 st.info 確保樣式統一)
            st.info(f"{動作}\n\n**分析策略**：{策略}")

        # --- 9. AI 投資顧問分析 (DEBUG 開關：開發時 True，正式環境 False) ---
        if "GEMINI_API_KEY" in st.secrets:
            try:
                import google.generativeai as genai
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                models = genai.list_models()
                available_models = [m.name for m in models]

                # 1) 只有在 DEBUG=True 時才顯示完整清單
                if DEBUG:
                    st.write("可用模型：", available_models)
                    for m in models:
                        methods = getattr(m, "supported_generation_methods", None)
                        st.write(m.name, "支援的方法：", methods)

                # 2) 模型選擇邏輯：優先 flash-lite → flash → pro
                model_name = None
                if "models/gemini-2.5-flash-lite" in available_models:
                    model_name = "models/gemini-2.5-flash-lite"
                elif "models/gemini-2.5-flash" in available_models:
                    model_name = "models/gemini-2.5-flash"
                else:
                    model = genai.GenerativeModel('gemini-pro') # Fallback to gemini-pro if other models not available
                    model_name = 'gemini-pro'

                # 3) 若沒模型就提示
                if not model_name:
                    st.warning("⚠️ 找不到可用的 Gemini 模型，請檢查 API Key 或 SDK 版本.")
                else:
                    st.write(f"目前使用的模型：{model_name}")

                    model = genai.GenerativeModel(model_name)
                    with st.spinner("🤖 AI 顧問正在同步研讀所有數據..."):
                        # 4) 建立 prompt（請確保下面變數在此區塊之前已定義）
                        ai_prompt = f"""
                        你是一位精通台股與籌碼分析的專家，請使用「繁體中文」及台灣用語，針對以下數據並配合國際總經、中東戰爭、地緣風險、產業前景及個股基本面，提供 450 字內的專業投資建議：
                        股票：{股票代號} {股名}
                        技術面：收盤價 {最新股價}，5MA {最新5MA:.2f}。
                        籌碼面：外資今日 {'買超' if 外資 > 0 else '賣超'} {abs(外資)} 張，投信 {'買超' if 投信 > 0 else '賣超'} {abs(投信)} 張。
                        主力買賣超：{'買超' if 主力買賣超 > 0 else '賣超'} {abs(主力買賣超)} 張。
                        散戶買賣超：{'買超' if 散戶買賣超 > 0 else '賣超'} {abs(散戶買賣超)} 張。
                        目前借券餘額：{最新借券餘額} 張。

                        請直接告訴我：
                        1. 這檔股票目前的亮點在哪？
                        2. 最大的風險是什麼？
                        3. 進出場建議 (買進、加碼、續抱、攤平、獲利了結)。

                        請特別分析主力與中實戶的買賣超動向及其對股價的影響。
                        """

                        # 5) 嘗試呼叫模型（若 generate_content 不存在，會捕捉並顯示錯誤）
                        try:
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content(ai_prompt)
                            # 取出回傳文字（不同 SDK 版本回傳結構可能不同）
                            text = getattr(response, "text", None)
                            if not text:
                                # 嘗試其他常見欄位
                                text = getattr(response, "output_text", None) or getattr(response, "output", None)
                            if text:
                                st.markdown("---")
                                # Modified: Changed st.info to st.markdown to support unsafe_allow_html
                                st.markdown(f"💡 <span style='color:#b3fcd3;'>**AI 診斷結果**</span>：\n\n{text}", unsafe_allow_html=True)
                            else:
                                st.warning("AI 有回應但無法解析回傳內容，請檢查 SDK 版本與回傳格式.")
                        except AttributeError as ae:
                            st.error(f"呼叫模型的方法不存在（AttributeError）：{ae}\n請檢查 SDK 版本或 supported_generation_methods.")
                        except Exception as e:
                            st.warning(f"🕒 AI 服務暫時無法回應。詳情：{e}")

            except Exception as ai_err:
                st.warning(f"🕒 AI 服務暫時無法回應。詳情：{ai_err}")
        else:
            st.error("🔑 尚未在 Streamlit Secrets 設定 GEMINI_API_KEY。")

    except Exception as e:
        st.error(f"❌ 診斷過程發生重大錯誤：{e}")

# --- 10. 初始狀態與按鈕修復 (必須完全「不縮進」，靠最左邊) ---
if "股名" not in locals():
    st.info("👈 請在左側輸入股票代號及日期，並按下「開始執行診斷」。")

