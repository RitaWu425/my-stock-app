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

# --- 1. 中文字體處理 (防錯強化版) ---
@st.cache_resource
def install_font():
    font_path = 'font.ttf'
    # 如果檔案不存在，才進行下載
    if not os.path.exists(font_path):
        try:
            # 使用更穩定的方式下載
            import urllib.request
            url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
            urllib.request.urlretrieve(url, font_path)
        except Exception as e:
            st.error(f"字體下載失敗: {e}")
            return

    # 再次確認檔案存在後才加入字體管理器
    if os.path.exists(font_path):
        import matplotlib.font_manager as fm
        try:
            fm.fontManager.addfont(font_path)
            plt.rc('font', family='Noto Sans CJK TC')
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            st.warning(f"字體掛載警告: {e}")

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
# 強制抓取過去 365 天的財報，確保一定能抓到最新一季
            財報開始日 = (pd.to_datetime(結束日期) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            基本面資料 = dl.taiwan_stock_financial_statement(stock_id=股票代號, start_date=財報開始日)
        
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

        # --- 數據深度拆解說明 (全面升級版) ---
        st.subheader("🔍 數據深度拆解說明")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            # [技術面]
            st.write(f"● **[技術面]**: {'股價高於 5MA，短線趨勢偏多。' if 最新股價 > 最新5MA else '股價低於 5MA，均線壓制明顯。'}")
            
            # [法人面]
            分析法人 = "外資、投信同步站回買方，法人底氣足。" if 區間外資 > 0 and 區間投信 > 0 else "法人買賣力道交錯，尚無一致共識。"
            st.write(f"● **[法人面]**: {分析法人}")
            
            # [籌碼面] (補齊)
            分析籌碼 = "籌碼集中度偏高，法人控盤穩定。" if 籌碼集中度 > 2 else "籌碼集中度較低，股價易受散戶情緒影響。"
            st.write(f"● **[籌碼面]**: {分析籌碼}")

        with col_right:
            # [指標面] (補齊)
            st.write(f"● **[指標面]**: {'RSI 超賣 ({:.1f})，注意跌深反彈訊號。'.format(最新RSI) if 最新RSI < 30 else 'RSI 指標目前處於中性區間。'}")
            
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
                        st.write(f"● **[基本面]**: 最新財報日 `{最新日期}`。季度營收約 `{最新營收/1e8:.2f}` 億元。")
                        
                        # 如果有營業利益，判斷是否獲利
                        if not df_oi.empty:
                            最新利益 = df_oi['value'].iloc[-1]
                            本業狀態 = "✅ 本業獲利" if 最新利益 > 0 else "⚠️ 本業虧損"
                            st.write(f"  └─ {本業狀態}：營業利益 `{最新利益/1e8:.2f}` 億元。")
                    else:
                        st.write("● **[基本面]**: 財報格式不符，建議至公開觀測站確認。")
                except Exception as e:
                    st.write(f"● **[基本面]**: 解析財報時發生小誤差。")
            else:
                st.write("● **[基本面]**: 即使追蹤一年仍查無財報，請確認代號是否為 ETF 或特殊股。")

            # [量價面]
            if 今日張數 > 今日5MA量 and 最新股價 > 昨日['close']:
                st.write("● **[量價面]**: 『帶量攻擊』，顯示買盤追價意願強烈。")
            else:
                st.write("● **[量價面]**: 量能表現平平，目前動能尚未爆發。")

        # 繪圖區
        # --- 5. 繪圖 (修復日期格式報錯) ---
        st.subheader("📊 籌碼與技術戰情圖")
        
        # 準備繪圖資料，強制轉換日期格式以防報錯
        plot_price = 股價資料.copy()
        plot_sbl = 借券資料.copy()
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
        
        # [關鍵修正]：強制格式化 X 軸日期，避免 tz 報錯
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)

        # --- 6. 完整智慧診斷輸出 (補足資訊) ---
        st.markdown("---")
        st.success("🧠 **圖表智慧診斷總結**")
        
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.write(f"● **[趨勢判讀]**: {'⚠️ 均線壓制：目前股價回測 5MA 之下，空方佔優。' if 最新股價 < 最新5MA else '✅ 短線轉強：股價已站穩 5MA，開啟短期反彈。'}")
            st.write(f"● **[籌碼鎖定]**: 借券餘額維持在 `{最新借券餘額:,.0f}` 張高檔。若餘額未減，代表大戶空頭尚未撤退。")
        
        with diag_col2:
            st.write(f"● **[買盤力道]**: {'🔥 買盤積極：帶量且還券比上升，空頭被迫回補引發追價。' if 今日張數 > 今日5MA量 and 借券淨變動 < 0 else '🧊 追價乏力：量能萎縮，市場觀望氣氛濃厚。'}")
            st.write(f"● **[指標訊號]**: RSI(`{最新RSI:.1f}`) 於 {'超賣區' if 最新RSI < 30 else '中性區'}。目前連續 `{連續回補}` 天出現回補跡象。")

        # 最終操作建議 (僅供參考)
        st.info(f"🔍 **操作核心建議**：目前 {股票代號} 的籌碼集中度為 `{籌碼集中度:.2f}%`。若集中度轉正且 RSI 站回 30 以上，則具備更強的『軋空』底氣。")

    except Exception as e:
        st.error(f"❌ 診斷失敗：{e}")
else:
    st.write("👈 請在左側輸入代號並點擊「開始執行診斷」")
