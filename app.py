import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # <--- 務必補上這行
from FinMind.data import DataLoader
import google.generativeai as genai
import warnings
import os
import urllib.request
from matplotlib import font_manager

# 基礎設定
warnings.filterwarnings('ignore')

# --- 0. 網頁基本設定 ---
st.set_page_config(page_title="股票籌碼診斷系統", layout="wide")

# 初始化抓資料工具
dl = DataLoader()

# --- 1. 中文字體處理 ---
@st.cache_resource
def install_font():
    font_path = 'font.ttf'
    # 這裡確保所有縮排都是 4 個空格
    if not os.path.exists(font_path):
        try:
            url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
            urllib.request.urlretrieve(url, font_path)
        except Exception as e:
            st.error(f"字體下載失敗: {e}")
    return font_path

# --- 1. 中文字體處理 (相容性強化版) ---
font_file = install_font()

if os.path.exists(font_file):
    try:
        # 使用最穩定的 FontEntry 方式註冊字體
        fe = font_manager.FontEntry(
            fname=font_file, 
            name='NotoSansCJKtc'
        )
        # 注意這裡：是 fontManager (小寫 m)，且直接插入到 ttflist
        font_manager.fontManager.ttflist.insert(0, fe) 
        plt.rcParams['font.family'] = fe.name
    except Exception as e:
        st.warning(f"字體設定稍微受阻，但不影響程式執行: {e}")

# 設定負號顯示正常
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 側邊欄：互動參數輸入 ---
st.sidebar.header("📊 診斷參數設定")
股票代號 = st.sidebar.text_input("輸入股票代號", value="3481")
開始日期 = st.sidebar.date_input("開始日期", value=pd.to_datetime("2026-02-01"))
結束日期 = st.sidebar.date_input("結束日期", value=pd.to_datetime("2026-03-25"))

if st.sidebar.button("開始執行診斷"):
    dl = DataLoader()
    
    # --- 預設值初始化 (移到最前面，防止 AI 報錯) ---
    最新營收 = 0
    latest_gp = 0
    最新股價 = 0
    最新5MA = 0
    最新RSI = 0
    借券淨變動 = 0
    最新借券餘額 = 0
    連續回補 = 0
    還券比 = 0  

    try:
        # 1. 資料抓取
        with st.spinner('正在從 FinMind 抓取資料...'):
            # --- 注意：以下這幾行必須比 with 縮排 4 個空格 ---
            個股資訊 = dl.taiwan_stock_info()
            股名 = 個股資訊[個股資訊['stock_id'] == 股票代號]['stock_name'].values[0]
            
            法人資料 = dl.taiwan_stock_institutional_investors(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            股價資料 = dl.taiwan_stock_daily(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            融資券資料 = dl.taiwan_stock_margin_purchase_short_sale(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            借券資料 = dl.get_data(dataset="TaiwanDailyShortSaleBalances", data_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            
            # 強制抓取過去 365 天的財報，確保一定能抓到最新一季
            財報開始日 = (pd.to_datetime(結束日期) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            基本面資料 = dl.taiwan_stock_financial_statement(stock_id=股票代號, start_date=財報開始日)
 # 抓取法人買賣超資料 (取最近 30 天以確保有足夠五日數據)
            法人資料 = dl.taiwan_stock_institutional_investors(stock_id=股票代號, start_date=開始日期)
          # --- 預設值初始化 (防止變數未定義報錯) ---
        借券淨變動 = 0
        最新借券餘額 = 0
        連續回補 = 0
        還券比 = 0  
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

        # --- 借券解析邏輯 (優化條列顯示) ---
        if not 借券資料.empty:
            sbl_最新 = 借券資料.iloc[-1]
            最新借券餘額 = sbl_最新['SBLShortSalesPreviousDayBalance'] // 1000
            今日還券 = sbl_最新['SBLShortSalesReturns'] // 1000
            今日借券賣出 = sbl_最新['SBLShortSalesShortSales'] // 1000
            還券比 = (今日還券 / 今日張數) * 100 if 今日張數 > 0 else 0

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
            c2.metric("今日借券賣出", f"{今日借券賣出:,.0f} 張", delta=f"{-今日借券賣出:,.0f}", delta_color="inverse")
            c3.metric("今日還券", f"{今日還券:,.0f} 張", delta=f"{今日還券:,.0f}")
            c4.metric("還券比", f"{還券比:.2f}%")

            # 2. 保留原本的文字解析摘要 (顏色標註更醒目)
            status_color = "🟢" if 連續回補 > 0 else "⚪"
            st.info(f"{status_color} **借券賣出摘要**：目前連續回補 `{連續回補}` 天。最新餘額 `{最新借券餘額:,.0f}` 張，整體還券力道為 `{還券比:.2f}%`。")

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

            # [量價面]
            if 今日張數 > 今日5MA量 and 最新股價 > 昨日['close']:
                st.write("● **[量價面]**: 『帶量攻擊』，顯示買盤追價意願強烈。")
            else:
                st.write("● **[量價面]**: 量能表現平平，目前動能尚未爆發。")
            
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
# --- ６. 基本面：營收與毛利雙軸圖 (進階版) ---
        st.markdown("---")
        st.subheader("📉 基本面：營收與毛利率趨勢")

        fund_start_date = (pd.Timestamp.now() - pd.Timedelta(days=500)).strftime('%Y-%m-%d')
        
        try:
            fund_data = dl.taiwan_stock_financial_statement(stock_id=股票代號, start_date=fund_start_date)

            if not fund_data.empty:
                # 準備營收資料
                df_rev = fund_data[fund_data['type'] == 'Revenue'].tail(4).copy()
                # 準備毛利率資料 (營業毛利 / 營業收入 * 100)
                df_gp = fund_data[fund_data['type'] == 'GrossProfit'].tail(4).copy()
                
                if not df_rev.empty and not df_gp.empty:
                    # 合併資料確保日期對齊
                    df_merge = pd.merge(df_rev[['date', 'value']], df_gp[['date', 'value']], on='date', suffixes=('_rev', '_gp'))
                    df_merge['rev_billion'] = df_merge['value_rev'] / 1e8
                    df_merge['gp_margin'] = (df_merge['value_gp'] / df_merge['value_rev']) * 100

                    # 開始繪圖
                    fig_f, ax1 = plt.subplots(figsize=(10, 5))
                    
                    # 1. 繪製營收長條圖 (左軸)
                    bars = ax1.bar(df_merge['date'], df_merge['rev_billion'], color='#BDD7EE', label='季度營收(億)', width=0.4)
                    ax1.set_ylabel('營收 (億元)', color='#2F5597', fontweight='bold')
                    ax1.tick_params(axis='y', labelcolor='#2F5597')
                    
                    # 2. 建立右軸繪製毛利率折線
                    ax2 = ax1.twinx()
                    ax2.plot(df_merge['date'], df_merge['gp_margin'], color='#ED7D31', marker='o', linewidth=3, label='毛利率(%)')
                    ax2.set_ylabel('毛利率 (%)', color='#ED7D31', fontweight='bold')
                    ax2.tick_params(axis='y', labelcolor='#ED7D31')
                    ax2.set_ylim(df_merge['gp_margin'].min() - 5, df_merge['gp_margin'].max() + 5) # 自動調整間距

                    # 標註毛利率數值
                    for i, txt in enumerate(df_merge['gp_margin']):
                        ax2.annotate(f'{txt:.1f}%', (df_merge['date'].iloc[i], df_merge['gp_margin'].iloc[i]), 
                                     textcoords="offset points", xytext=(0,10), ha='center', color='#C65911', fontweight='bold')

                    plt.title(f"{股票代號} {股名} - 獲利能力分析", fontsize=14)
                    st.pyplot(fig_f)
                    
                    # 💡 自動分析文字
                    latest_gp = df_merge['gp_margin'].iloc[-1]
                    prev_gp = df_merge['gp_margin'].iloc[-2]
                    gp_trend = "📈 提升" if latest_gp > prev_gp else "📉 下滑"
                    st.info(f"💡 **獲利評測**：最新毛利率為 `{latest_gp:.2f}%`，較上季 {gp_trend}。")
                else:
                    st.warning("⚠️ 財報科目不足，無法計算毛利率。")
            else:
                st.warning("⚠️ 查無財報資料。")

        except Exception as e:
            st.write("ℹ️ 基本面數據加載中或暫無資料。")

        # --- 7. 新增：三大法人近五日買賣超圖 ---
        if not 法人資料.empty:
            st.markdown("---")
            st.subheader("🏦 三大法人近五日動向")
            
            # 整理資料：僅取外資、投信、自營商，並取最近 5 個交易日
            df_inst = 法人資料.groupby(['date', 'name'])['buy'].sum() - 法人資料.groupby(['date', 'name'])['sell'].sum()
            df_inst = df_inst.unstack().tail(5) / 1000 # 轉為張數
            
            if not df_inst.empty:
                # 建立畫布
                fig_inst, ax_inst = plt.subplots(figsize=(10, 5))
                df_inst.plot(kind='bar', ax=ax_inst, width=0.8, color=['#FF9999', '#66B2FF', '#99FF99'])
                
                ax_inst.set_ylabel('買賣超張數 (張)')
                ax_inst.set_title(f'{股票代號} 近五日法人進出', fontsize=14)
                ax_inst.axhline(0, color='black', linewidth=1, alpha=0.5)
                ax_inst.grid(axis='y', linestyle='--', alpha=0.4)
                plt.xticks(rotation=0)
                
                st.pyplot(fig_inst)
                
                # --- 法人行為深度分析 ---
                latest_day = df_inst.iloc[-1]
                st.write("📝 **法人籌碼分析：**")
                
                # 判斷外資與投信
                if latest_day['Foreign_Investor'] > 0 and latest_day['Investment_Trust'] > 0:
                    st.write("✅ **[英雄所見略同]**：外資與投信今日同步買超，通常是強力的止跌或攻擊訊號。")
                elif latest_day['Foreign_Investor'] < 0 and latest_day['Investment_Trust'] > 0:
                    st.write("⚠️ **[土洋對作]**：投信力挺但外資在倒貨，需觀察 5MA 支撐是否能守住。")
                elif latest_day['Foreign_Investor'] > 0 and latest_day['Investment_Trust'] < 0:
                    st.write("⚠️ **[外熱內冷]**：外資回頭補貨，但內資投信先行獲利了結，股價易陷入震盪。")
                else:
                    st.write("❌ **[法人棄守]**：主要法人同步站回賣方，短線建議保守看待，避開殺低風險。")
                    
                # 補充：觀察投信連買
                sitc_five_days = df_inst['Investment_Trust'].sum()
                if sitc_five_days > 500: # 五日累積買超五百張以上
                    st.write(f"🔥 **[投信鎖碼]**：投信近五日累計買超 `{sitc_five_days:.0f}` 張，有作帳行情或長期佈局的跡象。")
            else:
                st.write("暫無足夠的法人進出資料。")
        
        # --- ８. 完整智慧診斷輸出 (防錯強健版) ---
        st.markdown("---")
        st.success("🧠 **圖表智慧診斷總結**")
        
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.write(f"● **[趨勢判讀]**: {'⚠️ 均線壓制：目前股價低於 5MA。' if 最新股價 < 最新5MA else '✅ 短線轉強：股價已站在 5MA 之上。'}")
            
            # 檢查是否有借券資料再輸出
            if '最新借券餘額' in locals() and 最新借券餘額 > 0:
                st.write(f"● **[籌碼鎖定]**: 借券餘額 `{最新借券餘額:,.0f}` 張。{'餘額偏高，需防範空頭打壓。' if 最新借券餘額 > 5000 else '籌碼相對穩定。'}")
            else:
                st.write("● **[籌碼鎖定]**: 該個股暫無借券數據紀錄。")
        
        with diag_col2:
            # 買盤力道增加防錯
            if '今日張數' in locals() and '今日5MA量' in locals():
                力道 = "🔥 買盤積極：帶量且高於均量。" if 今日張數 > 今日5MA量 else "🧊 追價乏力：量能萎縮中。"
                st.write(f"● **[買盤力道]**: {力道}")
            
            st.write(f"● **[指標訊號]**: RSI(`{最新RSI:.1f}`) 處於 {'超賣區' if 最新RSI < 30 else '中性區'}。")

        # 最終操作建議 (僅供參考)
        st.info(f"🔍 **操作核心建議**：目前 {股票代號} 的籌碼集中度為 `{籌碼集中度:.2f}%`。若集中度轉正且 RSI 站回 30 以上，則具備更強的『軋空』底氣。")

# --- (前面是你的圖表程式碼，例如 st.pyplot(fig_inst) ) ---

        # --- 9. AI 投資顧問分析 ---
        if "GEMINI_API_KEY" in st.secrets:
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                
                # 這裡改用最保險的順序：先試 1.5-flash，不行就試 gemini-pro
                model_names = ['models/gemini-1.5-flash', 'gemini-1.5-flash', 'gemini-pro']
                ai_model = None
                
                for name in model_names:
                    try:
                        ai_model = genai.GenerativeModel(name)
                        # 測試是否能用，若不行會跳到 except
                        break 
                    except:
                        continue
                
                if ai_model:
                    with st.spinner("🤖 AI 顧問正在同步研讀所有數據..."):
                        ai_prompt = f"""
                        你是一位精通台股與籌碼分析的專家，請針對以下數據提供 300 字內的「繁體中文」大白話投資建議：
                        股票：{股票代號} {股名}
                        技術面：收盤價 {最新股價}，5MA {最新5MA:.2f}。
                        籌碼面：法人近5日買賣超 {latest_day.to_dict()}，借券餘額 {最新借券餘額} 張。
                        基本面：最新季度營收 {最新營收/1e8:.2f} 億元，毛利率 {latest_gp:.2f}%。
                        請直接告訴我：這檔股票目前的亮點在哪？最大的風險是什麼？並做進出場建議(買進、加碼、續抱、獲利了結)。
                        """
                        response = ai_model.generate_content(ai_prompt)
                        st.info(f"💡 **AI 診斷結果**：\n\n{response.text}")
            except Exception as ai_err:
                st.warning(f"🕒 AI 服務忙碌中，請稍後再試。({ai_err})")
        else:
            st.error("🔑 尚未在 Streamlit Secrets 設定 GEMINI_API_KEY。")

    except Exception as e:
        # 只要不是因為按下按鈕觸發的錯誤，都顯示首頁提示
        st.error(f"❌ 診斷過程發生錯誤：{e}")
        
    # 重點：把這行移出 except 之外，確保它在「沒按按鈕」時一定會出現
    if not st.sidebar.button("開始執行診斷", key="check_run") and "股名" not in locals():
        st.info("👈 請在左側輸入股票代號及日期，並按下「開始執行診斷」。")
