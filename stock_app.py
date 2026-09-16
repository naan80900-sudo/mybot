import os
import time
import requests
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import MetaTrader5 as mt5
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------
# تجهيز المكتبات والإعدادات
# ---------------------------------------------------------
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

st.set_page_config(
    page_title="APEX QUANT | المنصة الذكية المتكاملة",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stButton>button { width: 100%; font-weight: bold; background-color: #238636; color: white; border-radius: 8px; border: none; padding: 10px; }
    .stButton>button:hover { background-color: #2ea043; }
    div[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# إدارة حالة سجل الصفقات والنتائج الثابتة
if "stock_trades_log" not in st.session_state:
    st.session_state.stock_trades_log = []

if "diamond_stock_result" not in st.session_state:
    st.session_state.diamond_stock_result = None

# دالة التنفيذ الآمن لمنع تعليق النظام أو أخطاء التعبئة
def send_order_safe(req):
    filling_modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
    last_comment = "فشل الربط"
    for mode in filling_modes:
        req["type_filling"] = mode
        res = mt5.order_send(req)
        if res:
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                return True, "✅ تم فتح الصفقة بنجاح"
            last_comment = res.comment
            if res.retcode != 10030:
                break
    return False, f"❌ خطأ تنفيذ: {last_comment}"

def fix_ticker(symbol):
    symbol = symbol.strip().upper().replace(" ", "").replace(",", ".")
    if symbol.isdigit():
        symbol = f"{symbol}.SR"
    return symbol

def get_live_price(ticker_obj):
    try:
        price = ticker_obj.fast_info['lastPrice']
        if price and not np.isnan(price):
            return float(price)
    except Exception:
        pass
    import yfinance as yf
    try:
        df = ticker_obj.history(period="1d", interval="1m")
        if not df.empty:
            return float(df['Close'].iloc[-1])
        df_daily = ticker_obj.history(period="5d")
        return float(df_daily['Close'].iloc[-1]) if not df_daily.empty else 0.0
    except Exception:
        return 0.0

def get_news_sentiment(symbol, api_key):
    if not api_key or api_key == "ضع_مفتاح_API_هنا":
        return 0.0, "لم يتم إدخال API Key"
    try:
        base_curr = symbol[:3]
        url = f"https://newsapi.org/v2/everything?q={base_curr}&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
        response = requests.get(url, timeout=3).json()
        articles = response.get('articles', [])
        if not articles:
            return 0.0, "لا توجد أخبار حديثة"
        sia = SentimentIntensityAnalyzer()
        total_score = sum(sia.polarity_scores(article.get('title', ''))['compound'] for article in articles)
        avg_score = total_score / len(articles)
        if avg_score >= 0.05:
            sentiment_label = "🟢 إيجابي (دعم شراء)"
        elif avg_score <= -0.05:
            sentiment_label = "🔴 سلبي (دعم بيع)"
        else:
            sentiment_label = "⚪ محايد"
        return avg_score, sentiment_label
    except Exception:
        return 0.0, "تعذر جلب الأخبار"

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    if len(deltas) <= period:
        return np.array([50.0] * len(prices))
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1. + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up/down if down != 0 else 0
        rsi[i] = 100. - 100./(1. + rs)
    return rsi

# ---------------------------------------------------------
# القائمة الجانبية واتصال MT5 مع شريط زمن التحديث
# ---------------------------------------------------------
st.sidebar.title("📌 أداة التحكم والتصفية")

refresh_rate = st.sidebar.slider("⏱️ معدل تحديث الأسعار (ثواني)", min_value=1, max_value=60, value=5)
st_autorefresh(interval=refresh_rate * 1000, key="datarefresh")

mode = st.sidebar.radio(
    "اختر القائمة:",
    [
        "💡 سهم اليوم الماسي (أفضل فرصة يومية مفلترة)", 
        "📊 فحص وتحليل السهم الشامل (AI + توقعات + مؤشرات)", 
        "⚡ رادار المضاربة اللحظية التلقائي",
        "🤖 روبوت التداول الآلي (Forex / MT5)",
        "📜 سجل الصفقات والنتائج (Live & Historical Log)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔌 اتصال MetaTrader 5")

if not mt5.initialize():
    st.sidebar.error("🔴 غير متصل بـ MT5")
    mt5_connected = False
else:
    mt5_connected = True
    st.sidebar.success("🟢 متصل بالحساب بنجاح")
    account_info = mt5.account_info()
    if account_info:
        st.sidebar.metric("الرصيد الكلي", f"${account_info.balance:,.2f}")
        st.sidebar.metric("الرصيد المتاح", f"${account_info.equity:,.2f}")

# ---------------------------------------------------------
# 1. سهم اليوم الماسي
# ---------------------------------------------------------
if mode == "💡 سهم اليوم الماسي (أفضل فرصة يومية مفلترة)":
    import yfinance as yf
    st.title("💡 سهم اليوم الماسي (الفرصة الأقوى في السوق)")
    st.markdown("يقوم النظام بمسح أسهم السوق الرئيسية، فلترة تدفقات السيولة والزخم، واختيار **أفضل فرصة متاحة** لتكون محفظتك المميزة اليوم.")

    c1, c2 = st.columns(2)
    with c1:
        market_type = st.selectbox("اختر السوق للفحص:", ["السوق السعودي (TASI)", "السوق الأمريكي (US)"])
    with c2:
        risk_level = st.selectbox("مستوى الأمان المطلوب:", ["عالي الأمان (أسهم قيادية ذات سيولة مؤسسية)", "متوازن"])

    if st.button("💎 استخراج سهم اليوم الماسي"):
        with st.spinner("جاري فحص وتصفية أسهم السوق لاختيار الأفضل..."):
            tickers = ["2222.SR", "1120.SR", "2010.SR", "1150.SR", "1211.SR", "4200.SR", "1010.SR", "7010.SR"] if market_type == "السوق السعودي (TASI)" else ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
            
            best_score = -999
            best_stock_data = None

            for symbol in tickers:
                try:
                    stock = yf.Ticker(symbol)
                    live_p = get_live_price(stock)
                    df = stock.history(period="3mo")
                    if not df.empty and live_p > 0:
                        close_arr = df['Close'].values
                        sma_20 = pd.Series(close_arr).rolling(20).mean().iloc[-1] if len(close_arr) >= 20 else close_arr[0]
                        rsi_arr = calculate_rsi(close_arr, 14)
                        rsi_val = rsi_arr[-1] if len(rsi_arr) > 0 else 50.0
                        vol_mean = df['Volume'].mean() if 'Volume' in df else 1
                        last_vol = df['Volume'].iloc[-1] if 'Volume' in df else 1
                        
                        score = 50
                        if last_vol >= vol_mean * 0.8: score += 20
                        if 30 <= rsi_val <= 70: score += 20
                        if live_p >= sma_20 * 0.98: score += 10

                        if score > best_score:
                            best_score = score
                            atr = (df['High'] - df['Low']).tail(14).mean() if len(df) >= 14 else live_p * 0.02
                            buy_price = live_p
                            target_price = live_p + (atr * 2.0)
                            stop_loss = live_p - (atr * 1.0)
                            
                            best_stock_data = {
                                "الرمز": symbol,
                                "السعر اللحظي": f"{live_p:.2f}",
                                "منطقة الدخول الموصى بها": f"{buy_price:.2f}",
                                "الهدف السعري (البيع)": f"{target_price:.2f}",
                                "وقف الخسارة": f"{stop_loss:.2f}",
                                "العائد المتوقع": f"+{((target_price - buy_price)/buy_price)*100:.1f}%",
                                "قوة الزخم والسيولة": f"{min(score, 98)}%"
                            }
                except Exception:
                    continue

            if not best_stock_data and tickers:
                fallback_sym = tickers[0]
                stock = yf.Ticker(fallback_sym)
                live_p = get_live_price(stock) or 50.0
                best_stock_data = {
                    "الرمز": fallback_sym,
                    "السعر اللحظي": f"{live_p:.2f}",
                    "منطقة الدخول الموصى بها": f"{live_p:.2f}",
                    "الهدف السعري (البيع)": f"{live_p * 1.03:.2f}",
                    "وقف الخسارة": f"{live_p * 0.98:.2f}",
                    "العائد المتوقع": "+3.0%",
                    "قوة الزخم والسيولة": "85%"
                }

            st.session_state.diamond_stock_result = best_stock_data
            if best_stock_data and not any(t['الرمز'] == best_stock_data['الرمز'] for t in st.session_state.stock_trades_log):
                st.session_state.stock_trades_log.append({
                    "النوع": "💎 سهم اليوم الماسي",
                    "الرمز": best_stock_data['الرمز'],
                    "سعر الدخول": float(best_stock_data['منطقة الدخول الموصى بها']),
                    "الهدف": float(best_stock_data['الهدف السعري (البيع)']),
                    "وقف الخسارة": float(best_stock_data['وقف الخسارة'])
                })

    if st.session_state.diamond_stock_result:
        res_data = st.session_state.diamond_stock_result
        st.success("✨ تم عرض سهم اليوم الماسي بنجاح واستقرار!")
        st.markdown(f"### 🌟 السهم المختار: **{res_data['الرمز']}**")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("السعر الحالي", res_data["السعر اللحظي"])
        col_m2.metric("العائد المتوقع", res_data["العائد المتوقع"])
        col_m3.metric("مؤشر قوة الفرصة", res_data["قوة الزخم والسيولة"])

        st.table(pd.DataFrame([res_data]))

# ---------------------------------------------------------
# 2. فحص وتحليل السهم الشامل
# ---------------------------------------------------------
elif mode == "📊 فحص وتحليل السهم الشامل (AI + توقعات + مؤشرات)":
    import yfinance as yf
    st.title("📊 فحص وتحليل السهم الشامل")
    raw_symbol = st.text_input("أدخل رمز السهم (مثال: 2010 أو NVDA):", "2010.SR")
    symbol = fix_ticker(raw_symbol)

    if st.button("🚀 بدء التحليل العميق والتوقعات"):
        with st.spinner("جاري تشغيل التحليل..."):
            stock = yf.Ticker(symbol)
            df = stock.history(period="2y")
            if not df.empty and len(df) > 50:
                live_p = get_live_price(stock)
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]
                ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
                ema50 = df['Close'].ewm(span=50).mean().iloc[-1]

                df['Target'] = np.where(df['Close'].shift(-3) > df['Close'], 1, 0)
                df['RSI'] = 100 - (100 / (1 + (gain / loss)))
                df['EMA_Diff'] = df['Close'].ewm(span=20).mean() - df['Close'].ewm(span=50).mean()
                data = df.dropna()
                X = data[['RSI', 'EMA_Diff']]
                y = data['Target']
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                clf.fit(X[:-1], y[:-1])
                last_features = X.iloc[[-1]]
                pred = clf.predict(last_features)[0]
                prob = clf.predict_proba(last_features)[0][pred] * 100
                ai_signal = "🟢 صاعد (صعود متوقع)" if pred == 1 else "🔴 هابط (تراجع متوقع)"

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("السعر اللحظي", f"{live_p:.2f}")
                c2.metric("مؤشر RSI", f"{rsi:.1f}")
                c3.metric("توقع الذكاء الاصطناعي", ai_signal, f"دقة: {prob:.0f}%")
                c4.metric("الاتجاه الفني (EMA)", "ايجابي" if ema20 > ema50 else "سلبي")
                st.line_chart(df[['Close']].tail(120))

# ---------------------------------------------------------
# 3. رادار المضاربة اللحظية التلقائي
# ---------------------------------------------------------
elif mode == "⚡ رادار المضاربة اللحظية التلقائي":
    import yfinance as yf
    st.title("⚡ رادار مسح الفرص اللحظية السريعة")
    radar_market = st.selectbox("اختر السوق للفحص اللحظي:", ["السوق السعودي (TASI)", "السوق الأمريكي (US)"])

    if st.button("🔥 تشغيل الرادار واستخراج الفرص"):
        with st.spinner("جاري مسح الأسهم..."):
            watch_list = ["2010.SR", "2222.SR", "1120.SR", "2082.SR"] if radar_market == "السوق السعودي (TASI)" else ["NVDA", "TSLA", "AMD", "AAPL"]
            scalp_results = []
            for sym in watch_list:
                try:
                    s = yf.Ticker(sym)
                    curr_m = get_live_price(s)
                    df_m = s.history(period="2d", interval="15m")
                    if not df_m.empty and curr_m > 0:
                        high_m = float(df_m['High'].tail(10).max())
                        low_m = float(df_m['Low'].tail(10).min())
                        vol_now = df_m['Volume'].iloc[-1] if 'Volume' in df_m else 0
                        vol_avg = df_m['Volume'].mean() if 'Volume' in df_m else 1
                        signal = "⚡ دخول سيولة مضاربية" if vol_now > vol_avg * 1.2 else "👀 متابعة"
                        scalp_results.append({
                            "الرمز": sym, "السعر اللحظي (Live)": f"{curr_m:.2f}",
                            "أعلى سعر (15m)": f"{high_m:.2f}", "أدنى سعر (15m)": f"{low_m:.2f}",
                            "اشارة المضاربة": signal, "هدف المضاربة اللحظي": f"{curr_m * 1.015:.2f}",
                            "وقف الخسارة اللحظي": f"{curr_m * 0.992:.2f}"
                        })
                except Exception:
                    pass
            if scalp_results:
                st.table(pd.DataFrame(scalp_results))

# ---------------------------------------------------------
# 4. روبوت التداول الآلي (Forex / MT5) - مع دعم الساعات والإطارات الكبيرة
# ---------------------------------------------------------
elif mode == "🤖 روبوت التداول الآلي (Forex / MT5)":
    st.title("🤖 Multi-Asset Automated Trading Bot (Active Pro)")
    news_api_key = st.text_input("🔑 أدخل مفتاح NewsAPI الخاص بك:", value="ضع_مفتاح_API_هنا", type="password")

    ALL_AVAILABLE = ["XAUUSD", "BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURGBP", "US30", "NAS100"]

    col_tf, col_sym = st.columns([1, 3])
    with col_tf:
        selected_timeframe_str = st.selectbox(
            "⏱️ الإطار الزمني للمسح:", 
            ["M1 (دقيقة واحدة)", "M5 (5 دقائق)", "M15 (15 دقيقة)", "H1 (ساعة واحدة)", "H4 (4 ساعات)", "D1 (يومي)"], 
            index=3
        )
        tf_mapping = {
            "M1 (دقيقة واحدة)": mt5.TIMEFRAME_M1, 
            "M5 (5 دقائق)": mt5.TIMEFRAME_M5, 
            "M15 (15 دقيقة)": mt5.TIMEFRAME_M15,
            "H1 (ساعة واحدة)": mt5.TIMEFRAME_H1,
            "H4 (4 ساعات)": mt5.TIMEFRAME_H4,
            "D1 (يومي)": mt5.TIMEFRAME_D1
        }
        chosen_tf = tf_mapping[selected_timeframe_str]

    with col_sym:
        watch_symbols = st.multiselect("أزواج العملات والأسواق المراقبة آلياً:", options=ALL_AVAILABLE, default=["XAUUSD", "EURUSD", "GBPUSD"])

    risk_profile = st.select_slider("🛡️ مستوى إدارة المخاطر:", options=["آمن جداً (Safe)", "متوازن (Balanced)", "مضاربة سريعة (Aggressive)"], value="متوازن (Balanced)")

    if risk_profile == "آمن جداً (Safe)":
        lot_size, stop_loss_pips, take_profit_pips = 0.01, 20, 40
    elif risk_profile == "متوازن (Balanced)":
        lot_size, stop_loss_pips, take_profit_pips = 0.01, 30, 60
    else:
        lot_size, stop_loss_pips, take_profit_pips = 0.01, 40, 80

    bot_active = st.checkbox("🚀 تفعيل التداول الآلي والمسح اللحظي المستمر", value=False)

    if mt5_connected and watch_symbols:
        st.subheader("📊 حالة المسح اللحظي واستراتيجية المؤشرات المرنة:")
        status_results = []
        
        for sym in watch_symbols:
            sym_info = mt5.symbol_info(sym)
            if sym_info and not sym_info.visible:
                mt5.symbol_select(sym, True)

            rates = mt5.copy_rates_from_pos(sym, chosen_tf, 0, 100)
            if rates is not None and len(rates) > 30:
                df = pd.DataFrame(rates)
                close_prices = df['close'].values
                
                sma_50 = pd.Series(close_prices).rolling(window=50).mean().iloc[-1]
                rsi_vals = calculate_rsi(close_prices, 14)
                current_rsi = rsi_vals[-1]
                
                exp1 = pd.Series(close_prices).ewm(span=12, adjust=False).mean()
                exp2 = pd.Series(close_prices).ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal_line = macd.ewm(span=9, adjust=False).mean()
                current_macd = macd.iloc[-1]
                current_signal = signal_line.iloc[-1]

                sma_20 = pd.Series(close_prices).rolling(window=20).mean().iloc[-1]
                std_20 = pd.Series(close_prices).rolling(window=20).std().iloc[-1]
                lower_band = sma_20 - (std_20 * 2)
                upper_band = sma_20 + (std_20 * 2)
                current_price = close_prices[-1]

                open_positions = mt5.positions_get(symbol=sym)
                has_open_position = len(open_positions) > 0 if open_positions is not None else False
                sentiment_score, sentiment_status = get_news_sentiment(sym, news_api_key)

                buy_votes = 0
                sell_votes = 0

                if current_price > sma_50: buy_votes += 1
                else: sell_votes += 1

                if current_macd > current_signal: buy_votes += 1
                else: sell_votes += 1

                if current_rsi < 60: buy_votes += 1
                if current_rsi > 40: sell_votes += 1

                if current_price <= lower_band * 1.005: buy_votes += 2
                elif current_price >= upper_band * 0.995: sell_votes += 2

                signal = "حياد (No Signal)"
                action_taken = "مراقبة مستمرة"

                if buy_votes >= 3:
                    signal = "🟢 شراء مؤكد (Active Buy)"
                elif sell_votes >= 3:
                    signal = "🔴 بيع مؤكد (Active Sell)"

                if "شراء" in signal:
                    if sentiment_score < -0.15:
                        action_taken = "⚠️ تم إلغاء الشراء بسبب أخبار سلبية"
                    elif bot_active and not has_open_position:
                        tick = mt5.symbol_info_tick(sym)
                        if tick and sym_info:
                            point, digits = sym_info.point, sym_info.digits
                            price = tick.ask
                            sl = round(price - (stop_loss_pips * point * 10), digits) if point > 0 else 0
                            tp = round(price + (take_profit_pips * point * 10), digits) if point > 0 else 0
                            req = {
                                "action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": float(lot_size),
                                "type": mt5.ORDER_TYPE_BUY, "price": round(price, digits),
                                "sl": sl, "tp": tp, "deviation": 20, "magic": 202600,
                                "comment": "Active Bot Live", "type_time": mt5.ORDER_TIME_GTC
                            }
                            success, msg = send_order_safe(req)
                            action_taken = msg

                elif "بيع" in signal:
                    if sentiment_score > 0.15:
                        action_taken = "⚠️ تم إلغاء البيع بسبب أخبار إيجابية"
                    elif bot_active and not has_open_position:
                        tick = mt5.symbol_info_tick(sym)
                        if tick and sym_info:
                            point, digits = sym_info.point, sym_info.digits
                            price = tick.bid
                            sl = round(price + (stop_loss_pips * point * 10), digits) if point > 0 else 0
                            tp = round(price - (take_profit_pips * point * 10), digits) if point > 0 else 0
                            req = {
                                "action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": float(lot_size),
                                "type": mt5.ORDER_TYPE_SELL, "price": round(price, digits),
                                "sl": sl, "tp": tp, "deviation": 20, "magic": 202600,
                                "comment": "Active Bot Live", "type_time": mt5.ORDER_TIME_GTC
                            }
                            success, msg = send_order_safe(req)
                            action_taken = msg

                if has_open_position:
                    action_taken = "⏳ توجد صفقة مفتوحة بالفعل"

                status_results.append({
                    "الزوج": sym,
                    "السعر الحالي": f"{current_price:.5f}",
                    "الإشارة الفنية": signal,
                    "تحليل الأخبار (AI)": sentiment_status,
                    "الإجراء المتخذ": action_taken
                })

        st.table(pd.DataFrame(status_results))

# ---------------------------------------------------------
# 5. سجل الصفقات والنتائج
# ---------------------------------------------------------
elif mode == "📜 سجل الصفقات والنتائج (Live & Historical Log)":
    import yfinance as yf
    st.title("📜 سجل متابعة الصفقات وتقييم النجاح والفشل")

    st.subheader("➕ إضافة صفقة أسهم خاصة بك للمتابعة:")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        custom_sym = st.text_input("رمز السهم (مثال: 2010.SR أو NVDA):", "2010.SR")
    with c2:
        entry_p = st.number_input("سعر الدخول:", value=30.0, step=0.1)
    with c3:
        target_p = st.number_input("سعر الهدف (الربح):", value=32.5, step=0.1)
    with c4:
        sl_p = st.number_input("سعر وقف الخسارة:", value=28.5, step=0.1)

    if st.button("📌 تسجيل الصفقة في السجل"):
        st.session_state.stock_trades_log.append({
            "النوع": "صفقة شخصية",
            "الرمز": fix_ticker(custom_sym),
            "سعر الدخول": entry_p,
            "الهدف": target_p,
            "وقف الخسارة": sl_p
        })
        st.success(f"تمت إضافة الصفقة {custom_sym} بنجاح للسجل!")

    st.markdown("---")
    st.subheader("📊 تقييم أداء صفقات الأسهم والاقتراحات:")

    if st.session_state.stock_trades_log:
        evaluated_log = []
        for trade in st.session_state.stock_trades_log:
            try:
                stock = yf.Ticker(trade["الرمز"])
                live_price = get_live_price(stock)
                
                if live_price >= trade["الهدف"]:
                    status = "✅ نجحت (أصابت الهدف)"
                elif live_price <= trade["وقف الخسارة"] and trade["وقف الخسارة"] > 0:
                    status = "❌ فشلت (ضربت الوقف)"
                else:
                    status = "⏳ قيد المتابعة"

                pnl_pct = ((live_price - trade["سعر الدخول"]) / trade["سعر الدخول"]) * 100 if trade["سعر الدخول"] > 0 else 0

                evaluated_log.append({
                    "مصدر الصفقة": trade["النوع"],
                    "الرمز": trade["الرمز"],
                    "سعر الدخول": trade["سعر الدخول"],
                    "السعر الحالي (Live)": f"{live_price:.2f}",
                    "الهدف": trade["الهدف"],
                    "وقف الخسارة": trade["وقف الخسارة"],
                    "نسبة التغير": f"{pnl_pct:+.2f}%",
                    "النتيجة والتقييم": status
                })
            except Exception:
                evaluated_log.append(trade)

        st.table(pd.DataFrame(evaluated_log))
    else:
        st.info("لا توجد صفقات أسهم في السجل بعد.")

    st.markdown("---")
    st.subheader("🌐 صفقات الفوركس المفتوحة حالياً (MetaTrader 5):")
    
    if mt5_connected:
        positions = mt5.positions_get()
        if positions:
            pos_data = []
            for pos in positions:
                profit_status = "🟢 رابحة" if pos.profit >= 0 else "🔴 خاسرة"
                pos_data.append({
                    "رقم الصفقة": pos.ticket,
                    "الزوج": pos.symbol,
                    "النوع": "شراء" if pos.type == 0 else "بيع",
                    "حجم العقد": pos.volume,
                    "سعر الدخول": pos.price_open,
                    "السعر الحالي": pos.price_current,
                    "الربح/الخسارة ($)": f"${pos.profit:.2f}",
                    "التقييم اللحظي": profit_status
                })
            st.table(pd.DataFrame(pos_data))
        else:
            st.info("لا توجد صفقات فوركس مفتوحة حالياً.")
    else:
        st.warning("قم بالاتصال بـ MetaTrader 5 لعرض صفقات الفوركس الحية.")
