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
        with st.spinner('正在深度解析籌碼數據...'):
            # A. 資料抓取
            個股資訊 = dl.taiwan_stock_info()
            股名 = 個股資訊[個股資訊['stock_id'] == 股票代號]['stock_name'].values[0]
            股價資料 = dl.taiwan_stock_daily(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            法人資料 = dl.taiwan_stock_institutional_investors(stock_id=股票代號, start_date=str(開始日期))
            融資券資料 = dl.taiwan_stock_margin_purchase_short_sale(stock_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))
            借券資料 = dl.get_data(dataset="TaiwanDailyShortSaleBalances", data_id=股票代號, start_date=str(開始日期), end_date=str(結束日期))

            if 股價資料.empty:
                st.error("此區間無資料。")
                st.stop()

            # B. 數據計算
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

            # RSI
            delta = 股價資料['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
            最新RSI = (100 - (100 / (1 + (gain / loss.replace(0, 0.001))))).iloc[-1]

            # 法人
            def get_net_buy(df, name):
                if df.empty: return 0
                target = df[df['name'] == name]
                return (target['buy'].sum() - target['sell'].sum()) // 1000

            區間外資 = get_net_buy(法人資料, 'Foreign_Investor')
            區間投信 = get_net_buy(法人資料, 'Investment_Trust')
            籌碼集中度 = ((區間外資 + 區間投信) * 1000 / 股價資料[vol_col].sum()) * 100 if not 法人資料.empty else 0

        # --- 5. 儀表板 ---
        st.title(f"📈 {股票代號} {股名} 終極診斷報告")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最新股價", f"{最新股價}", f"{最新股價-昨日['close']:.2f}")
        col2.metric("5MA 均線", f"{最新5MA:.1f}")
        col3.metric("RSI 指標", f"{最新RSI:.1f}")
        col4.metric("籌碼集中度", f"{籌碼集中度:.2f}%")

        # --- 6. 信用與借券數據 (含詳細長版解析) ---
        st.markdown("---")
        st.subheader("📉 信用與借券數據詳解")
        
        c1, c2 = st.columns(2)
        今日融資變動, 融券總餘額 = 0, 0
        if not 融資券資料.empty:
            今日融資變動 = (融資券資料.iloc[-1]['MarginPurchaseTodayBalance'] - 融資券資料.iloc[-2]['MarginPurchaseTodayBalance']) // 1000
            融券總餘額 = 融資券資料.iloc[-1]['ShortSaleTodayBalance'] // 1000

        with c1:
            st.write(f"融資變動：`{今日融資變動:+,d}` 張")
            st.write(f"融券總餘額：`{融券總餘額:,.0f}` 張")

        連續回補, 最新借券餘額 = 0, 0
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

            with c2:
                st.write(f"最新借券餘額：`{最新借券餘額:,.0f}` 張")
                st.write(f"💥 今日「借券賣出」：`{今日借券賣出:,.0f}` 張")
                st.write(f"今日還券：`{今日還券:,.0f}` 張 (還券比: `{還券比:.2f}%`) | 連續回補：`{連續回補}` 天")

            # --- 深度動向邏輯 ---
            st.markdown("#### ● [借券動向分析]")
            if 今日還券 > 今日借券賣出:
                st.success(f"今日『還券』大於『賣出』，淨回補 `{abs(借券淨變動):,.0f}` 張，空頭力量消退中。")
            elif 今日借券賣出 > 今日還券:
                st.error(f"今日『賣出』大於『還券』，淨增加 `{借券淨變動:,.0f}` 張空單，法人持續加壓。")
            else:
                st.warning("今日賣出與還券持平，空頭力道進入觀望期。")

            if 最新借券餘額 > 今日5MA量 * 10:
                st.info("💡 [潛力]: 借券餘額規模巨大，若配合 RSI 低檔，軋空動能極強。")

        # --- 8. 🔍 數據深度拆解說明 (長版回歸) ---
        st.markdown("---")
        st.subheader("🔍 數據深度拆解說明")
        
        # [技術面]
        st.write(f"● **[技術面]**: {'股價站上 5MA，短線轉強。' if 最新股價 > 最新5MA else '股價低於 5MA，均線壓制明顯。'}")
        # [法人面]
        分析法人 = "外資、投信同步站回買方，法人底氣足。" if 區間外資 > 0 and 區間投信 > 0 else "法人買賣力道交錯，尚無一致共識。"
        st.write(f"● **[法人面]**: {分析法人}")
        # [籌碼面]
        分析籌碼 = "籌碼集中度偏低，且融資若續增將不利洗盤。" if 籌碼集中度 < 1 else "籌碼集中度尚可，法人控盤穩定。"
        st.write(f"● **[籌碼面]**: {分析籌碼}")
        # [指標面]
        st.write(f"● **[指標面]**: {'RSI 超賣 ({:.1f})，注意反彈訊號。'.format(最新RSI) if 最新RSI < 30 else '指標中性偏弱。'}")

        # [量價面深度分析]
        if 今日張數 > 今日5MA量 and 最新股價 > 昨日['close']:
            if 籌碼集中度 > 0 or 區間外資 > 0:
                st.write(f"● **[量價面]**: 『帶量攻擊』且法人同步買進！這是【真買盤在追】，上攻動能紮實。")
            else:
                st.write(f"● **[量價面]**: 帶量上漲但法人未買，可能是散戶進場或單純空頭回補（虛漲）。")
        elif 今日張數 < 今日5MA量:
            st.write(f"● **[量價面]**: 價漲量縮或量能低於均值，代表追價意願不足，需防高位震盪。")
        else:
            st.write(f"● **[量價面]**: 量能表現平平，目前動能尚未爆發。")

        # [買盤力道總結]
        if not 借券資料.empty and 還券比 > 15 and 最新股價 > 昨日['close']:
            st.warning(f"⚠️ **[預警]**: 今日漲幅中，有 `{還券比:.1f}%` 的交易來自還券，『軋空回補』力道大於『實體買盤』。")
        elif 籌碼集中度 > 5 and 今日張數 > 今日5MA量:
            st.success(f"✅ **[確認]**: 籌碼集中度極高且量能噴發，確認為『實體大戶買盤』強力進駐。")

        # --- 9. 圖表 ---
        st.subheader("📊 籌碼與技術戰情圖")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(股價資料['date'], 股價資料['close'], label='收盤價', color='#1f77b4', linewidth=2)
        ax.plot(股價資料['date'], 股價資料['5MA'], label='5MA', color='#ff7f0e', linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.legend()
        st.pyplot(fig)

        # --- 10. AI 投資顧問 ---
        st.markdown("---")
        st.subheader("🤖 AI 投資顧問分析")
        if "GEMINI_API_KEY" in st.secrets:
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model_names = ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-pro']
                ai_content = ""
                
                for m_name in model_names:
                    try:
                        model = genai.GenerativeModel(m_name)
                        ai_prompt = f"""
                        你是一位精通台股與籌碼分析的專家，請針對以下數據提供 300 字內的「繁體中文」大白話投資建議：
                        股票：{股票代號} {股名}
                        技術面：價格 {最新股價}，5MA {最新5MA:.2f}，RSI {最新RSI:.1f}。
                        籌碼面：外資{區間外資:+,d}，投信{區間投信:+,d}，借券餘額{最新借券餘額}張，連續回補{連續回補}天。
                        請分析亮點與風險，並做進出場建議(買進、加碼、續抱、獲利了結)。
                        """
                        response = model.generate_content(ai_prompt)
                        ai_content = response.text
                        if ai_content: break
                    except Exception: continue
                
                if ai_content:
                    st.info(f"💡 **AI 診斷結果**：\n\n{ai_content}")
                else:
                    st.warning("🕒 AI 引擎目前忙碌中，請稍後再試。")
            except Exception as e:
                st.error(f"AI 啟動失敗：{e}")
        else:
            st.warning("⚠️ 找不到 GEMINI_API_KEY。")

    except Exception as e:
        st.error(f"❌ 診斷失敗：{e}")
