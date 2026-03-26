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

# --- 2. 結束日期自動判定 ---
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
        with st.spinner('正在從大數據庫擷取：技術、籌碼、財報、法人數據...'):
            # A. 資料抓取
            個股資訊 = dl.taiwan_stock_info()
            股名 = 個股資訊[個股資訊['stock_id'] == 股票代號]['stock_name'].values[0]
            
            # 1. 交易數據
            股價資料 = dl.taiwan_stock_daily(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            法人資料 = dl.taiwan_stock_institutional_investors(stock_id=股票代號, start_date=str(開始日期))
            融資券資料 = dl.taiwan_stock_margin_purchase_short_sale(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            借券資料 = dl.get_data(dataset="TaiwanDailyShortSaleBalances", data_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            
            # 2. 基本面數據 (抓一年份)
            財報年 = str(datetime.now().year - 1)
            營收資料 = dl.taiwan_stock_month_revenue(stock_id=股票代號, start_date=f"{財報年}-01-01")
            損益表 = dl.taiwan_stock_financial_statements(stock_id=股票代號, start_date=f"{財報年}-01-01")

            if 股價資料.empty:
                st.error("此區間無交易資料。")
                st.stop()

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
            今日5MA量 = 股價資料['Vol_Lots'].rolling(5).mean().iloc[-1]

            # RSI 計算
            delta = 股價資料['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
            最新RSI = (100 - (100 / (1 + (gain / loss.replace(0, 0.001))))).iloc[-1]

            # 法人運算
            def get_net_buy(df, name):
                if df.empty: return 0
                target = df[df['name'] == name]
                return (target['buy'].sum() - target['sell'].sum()) // 1000

            區間外資 = get_net_buy(法人資料, 'Foreign_Investor')
            區間投信 = get_net_buy(法人資料, 'Investment_Trust')
            區間自營 = get_net_buy(法人資料, 'Dealer_self')
            區間權證 = get_net_buy(法人資料, 'Dealer_Hedging')
            籌碼集中度 = ((區間外資 + 區間投信) * 1000 / 股價資料[vol_col].sum()) * 100 if not 法人資料.empty else 0

        # --- 5. 儀表板 ---
        st.title(f"📈 {股票代號} {股名} 終極診斷報告")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最新股價", f"{最新股價}", f"{最新股價-昨日['close']:.2f}")
        m2.metric("5MA 均線", f"{最新5MA:.1f}")
        m3.metric("RSI 指標", f"{最新RSI:.1f}")
        m4.metric("籌碼集中度", f"{籌碼集中度:.2f}%")

        # --- 6. 法人與權證動向 (補回) ---
        st.markdown("---")
        st.subheader("👥 法人與權證動向")
        c1, c2, c3, c4 = st.columns(4)
        c1.write(f"外資：`{區間外資:+,d}` 張")
        c2.write(f"投信：`{區間投信:+,d}` 張")
        c3.write(f"自營商：`{區間自營:+,d}` 張")
        c4.write(f"權證避險：`{區間權證:+,d}` 張")

        # --- 7. 信用與借券數據詳解 ---
        st.subheader("📉 信用與借券數據詳解")
        d1, d2 = st.columns(2)
        今日融資變動, 融券總餘額 = 0, 0
        if not 融資券資料.empty:
            今日融資變動 = (融資券資料.iloc[-1]['MarginPurchaseTodayBalance'] - 融資券資料.iloc[-2]['MarginPurchaseTodayBalance']) // 1000
            融券總餘額 = 融資券資料.iloc[-1]['ShortSaleTodayBalance'] // 1000
        with d1:
            st.write(f"融資變動：`{今日融資變動:+,d}` 張 | 融券餘額：`{融券總餘額:,.0f}` 張")

        連續回補, 最新借券餘額, 還券比 = 0, 0, 0
        if not 借券資料.empty:
            sbl_最新 = 借券資料.iloc[-1]
            最新借券餘額 = sbl_最新['SBLShortSalesPreviousDayBalance'] // 1000
            今日還券 = sbl_最新['SBLShortSalesReturns'] // 1000
            今日借券賣出 = sbl_最新['SBLShortSalesShortSales'] // 1000
            還券比 = (今日還券 / 今日張數) * 100 if 今日張數 > 0 else 0
            借券淨變動 = 今日借券賣出 - 今日還券
            for i in range(len(借券資料)-1, -1, -1):
                if 借券資料.iloc[i]['SBLShortSalesReturns'] > 借券資料.iloc[i]['SBLShortSalesShortSales']: 連續回補 += 1
                else: break
            with d2:
                st.write(f"借券餘額：`{最新借券餘額:,.0f}` 張 | 今日還券：`{今日還券:,.0f}` ({還券比:.1f}%)")

            st.markdown("#### ● [借券動向分析]")
            if 今日還券 > 今日借券賣出:
                st.success(f"今日『還券』大於『賣出』，淨回補 `{abs(借券淨變動):,.0f}` 張，空頭力量消退中。")
            elif 今日借券賣出 > 今日還券:
                st.error(f"今日『賣出』大於『還券』，淨增加 `{借券淨變動:,.0f}` 張空單，法人持續加壓。")

        # --- 8. 🔍 深度拆解說明 (含基本面) ---
        st.markdown("---")
        st.subheader("🔍 數據深度拆解說明")
        
        # 準備基本面數據
        毛利率說明 = "資料讀取中"
        if not 損益表.empty:
            gpm_df = 損益表[損益表['type'] == 'GrossProfitMargin']
            if not gpm_df.empty:
                最新毛利 = gpm_df.iloc[-1]['value']
                毛利率說明 = f"最新毛利率 `{最新毛利:.2f}%`，{'呈現成長趨勢' if 最新毛利 > gpm_df.iloc[-2]['value'] else '毛利稍有承壓'}。"

        # 四大診斷區塊
        g1, g2 = st.columns(2)
        with g1:
            st.info(f"🚩 **[技術診斷]**: {'股價站上 5MA，短線轉強。' if 最新股價 > 最新5MA else '股價低於 5MA，均線壓制明顯。'}\n\n"
                    f"🚩 **[籌碼診斷]**: {分析法人 if (分析法人 := '法人同步看好買進。' if 區間外資>0 and 區間投信>0 else '法人觀望力道交錯。') else ''}")
        with g2:
            st.info(f"🚩 **[追價診斷]**: {'帶量攻擊，買盤積極。' if 今日張數 > 今日5MA量 else '量能萎縮，追價乏力。'}\n\n"
                    f"🚩 **[基本診斷]**: {毛利率說明}")

        # --- 9. 📊 綜合戰情圖 (加入營收與毛利) ---
        st.markdown("---")
        st.subheader("📊 綜合戰情分析圖表")
        
        tab1, tab2 = st.tabs(["價格與均線", "營收與毛利趨勢"])
        
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(股價資料['date'], 股價資料['close'], label='收盤價', color='#1f77b4', linewidth=2)
            ax1.plot(股價資料['date'], 股價資料['5MA'], label='5MA', color='#ff7f0e', linestyle='--')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax1.legend()
            st.pyplot(fig1)
            
        with tab2:
            if not 營收資料.empty:
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                ax2.bar(營收資料['date'], 營收資料['revenue']//1000000, label='營收(百萬)', color='#add8e6', alpha=0.7)
                ax2.set_ylabel('營收')
                if not 損益表.empty:
                    ax3 = ax2.twinx()
                    gpm_plot = 損益表[損益表['type'] == 'GrossProfitMargin']
                    ax3.plot(gpm_plot['date'], gpm_plot['value'], color='red', marker='o', label='毛利率%')
                    ax3.set_ylabel('毛利率(%)')
                st.pyplot(fig2)
            else:
                st.write("暫無營收圖表資料")

        # --- 10. 🤖 AI 投資顧問 (強化版) ---
        st.markdown("---")
        st.subheader("🤖 AI 投資顧問分析")
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
            try:
                genai.configure(api_key=api_key)
                # 嘗試多個模型以防忙碌
                success = False
                for model_ver in ['gemini-1.5-flash', 'gemini-pro']:
                    try:
                        model = genai.GenerativeModel(model_ver)
                        prompt = f"""請以台股專家身份分析{股票代號}{股名}：
                        1. 技術面：收盤{最新股價}, RSI{最新RSI:.1f}
                        2. 籌碼面：外資{區間外資:+,d}, 投信{區間投信:+,d}, 借券回補{連續回補}天
                        3. 基本面：{毛利率說明}
                        提供300字建議包含：亮點、風險、操作建議(續抱/買進/觀望)。"""
                        response = model.generate_content(prompt)
                        st.write(response.text)
                        success = True
                        break
                    except: continue
                if not success: st.warning("🕒 AI 引擎目前真的太忙了，請參考上方四大診斷。")
            except Exception as e:
                st.error(f"AI 連線失敗：{e}")
        else:
            st.warning("⚠️ 找不到 GEMINI_API_KEY，請確認 Secrets 設定。")

    except Exception as e:
        st.error(f"❌ 診斷發生錯誤：{e}")
