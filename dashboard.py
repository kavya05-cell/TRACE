import time, os
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Adaptive NIDS", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1rem}
[data-testid="stMetric"]{background:#101827;padding:15px;border:1px solid #263449;border-radius:12px}
.alert-box{padding:15px;border-radius:10px;background:#4b171c;color:#ffb5b5;border:1px solid #a33;font-weight:bold}
.normal-box{padding:15px;border-radius:10px;background:#123528;color:#9af0bf;border:1px solid #26875c;font-weight:bold}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Adaptive Network Intrusion Detection System")
st.caption("Network Monitoring • Intrusion Detection • Adaptive Ensemble • Explainability")

@st.cache_data
def load_dataset():
    from app import load
    _, test = load(shuffle_train=False)
    return test

@st.cache_resource
def create_models():
    from app import make_models
    models = make_models()
    return {k: models[k] for k in ["Adaptive Random Forest", "Hoeffding Adaptive Tree", "Hoeffding Tree"]}

st.sidebar.header("Monitoring")
mode = st.sidebar.radio("Source", ["Dataset Replay", "Live Packet Capture"])

# ================= LIVE PACKET CAPTURE =================
if mode == "Live Packet Capture":
    from live_traffic import LiveTrafficMonitor

    iface = st.sidebar.text_input("Network interface (blank = default)")
    if "monitor" not in st.session_state:
        st.session_state.monitor = LiveTrafficMonitor(iface or None)

    monitor = st.session_state.monitor

    start, stop = st.sidebar.columns(2)
    if start.button("▶ Start", use_container_width=True):
        try:
            monitor.start()
        except Exception as e:
            st.sidebar.error(str(e))

    if stop.button("■ Stop", use_container_width=True):
        monitor.stop()

    snap = monitor.snapshot()
    active = monitor.running

    st.markdown(
        '<div class="normal-box">🟢 LIVE CAPTURE ACTIVE</div>' if active
        else '<div class="alert-box">🔴 CAPTURE STOPPED</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Packets Captured", f"{snap['packet_count']:,}")
    c2.metric("Current Flows", snap["flow_count"])
    c3.metric("Intrusion Alerts", len(snap["alerts"]))
    c4.metric("Capture Status", "ACTIVE" if active else "STOPPED")

    left, right = st.columns(2)

    with left:
        st.subheader("Protocol Distribution")
        protocol = pd.DataFrame(snap["protocols"].items(), columns=["Protocol", "Packets"])
        if not protocol.empty:
            st.plotly_chart(px.pie(protocol, names="Protocol", values="Packets", hole=.5), use_container_width=True)
        else:
            st.info("Waiting for packets...")

    with right:
        st.subheader("Recent Network Traffic")
        events = pd.DataFrame(snap["events"])
        if not events.empty:
            events["time"] = pd.to_datetime(events["time"], unit="s").dt.strftime("%H:%M:%S")
            st.dataframe(events[["time", "src", "dst", "sport", "dport", "proto", "bytes", "flags"]].tail(20), use_container_width=True, hide_index=True)
        else:
            st.info("No packets captured yet.")

    st.subheader("🚨 Intrusion Alerts")
    alerts = pd.DataFrame(snap["alerts"])
    if not alerts.empty:
        alerts["time"] = pd.to_datetime(alerts["time"], unit="s").dt.strftime("%H:%M:%S")
        st.dataframe(alerts[["time", "severity", "score", "src", "description", "flows", "packets"]], use_container_width=True, hide_index=True)
    else:
        st.success("No suspicious traffic patterns detected.")

    st.subheader("🔍 Explainability — Why was traffic flagged?")
    if not alerts.empty:
        idx = st.selectbox(
            "Select alert",
            range(len(alerts)),
            format_func=lambda i: f"{alerts.iloc[i]['severity']} | {alerts.iloc[i]['description']}"
        )
        selected = alerts.iloc[idx]
        st.write(selected["description"])
        st.progress(int(selected["score"]), text=f"Risk score: {selected['score']}/100")
        st.write(f"Source IP: `{selected['src']}`")
        st.write(f"Observed flows: {selected['flows']}")
        st.write(f"Packets in window: {selected['packets']}")

    st.caption("Live capture uses transparent flow-based heuristics. It does not currently apply the NSL-KDD-trained model directly to raw packets.")

    if active:
        time.sleep(1)
        st.rerun()

# ================= DATASET REPLAY =================
else:
    st.markdown('<div class="normal-box">🟢 DATASET STREAM READY</div>', unsafe_allow_html=True)

    if "demo_idx" not in st.session_state:
        st.session_state.demo_idx = 0
    if "demo_history" not in st.session_state:
        st.session_state.demo_history = []
    if "demo_alerts" not in st.session_state:
        st.session_state.demo_alerts = []
    if "demo_models" not in st.session_state:
        st.session_state.demo_models = None
    if "demo_data" not in st.session_state:
        st.session_state.demo_data = None

    if st.session_state.demo_data is None:
        with st.spinner("Loading dataset..."):
            st.session_state.demo_data = load_dataset()
            st.session_state.demo_models = create_models()

    data = st.session_state.demo_data
    models = st.session_state.demo_models

    st.sidebar.metric("Stream Position", f"{st.session_state.demo_idx:,}/{len(data):,}")
    batch = st.sidebar.slider("Records per batch", 10, 500, 50, 10)

    if st.sidebar.button("▶ Process Traffic", use_container_width=True):
        features = [c for c in data.columns if c not in ("y", "attack_type")]
        stop = min(st.session_state.demo_idx + batch, len(data))

        for i in range(st.session_state.demo_idx, stop):
            row = data.iloc[i]
            x = row[features].to_dict()
            y = int(row.y)
            label = str(row.attack_type)

            votes = []
            for name, model in models.items():
                try:
                    prediction = int(model.predict_one(x) or 0)
                except Exception:
                    prediction = 0
                votes.append(prediction)
                model.learn_one(x, y)

            pred = int(sum(votes) >= 2)
            confidence = max(sum(votes), 3-sum(votes))/3

            event = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "index": i,
                "prediction": "ATTACK" if pred else "NORMAL",
                "actual": "ATTACK" if y else "NORMAL",
                "attack_type": label,
                "correct": int(pred == y),
                "confidence": confidence
            }

            st.session_state.demo_history.append(event)

            if pred:
                st.session_state.demo_alerts.insert(0, event)

        st.session_state.demo_idx = stop

    if st.sidebar.button("↻ Reset Stream", use_container_width=True):
        st.session_state.demo_idx = 0
        st.session_state.demo_history = []
        st.session_state.demo_alerts = []
        st.session_state.demo_models = None
        st.session_state.demo_data = None
        st.rerun()

    hist = pd.DataFrame(st.session_state.demo_history)
    alerts = pd.DataFrame(st.session_state.demo_alerts)

    total = len(hist)
    detected = int((hist.prediction == "ATTACK").sum()) if total else 0
    accuracy = float(hist.correct.mean()) if total else 0
    attack_ratio = float((hist.actual == "ATTACK").mean()) if total else 0
    latest_attack = bool(total and hist.iloc[-1].prediction == "ATTACK")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records Monitored", f"{total:,}")
    c2.metric("Intrusions Detected", f"{detected:,}")
    c3.metric("Detection Accuracy", f"{accuracy:.1%}")
    c4.metric("Attack Traffic Ratio", f"{attack_ratio:.1%}")

    if latest_attack:
        st.markdown('<div class="alert-box">🚨 INTRUSION DETECTED — ATTACK TRAFFIC</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="normal-box">✅ NETWORK STATUS: NORMAL</div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("📡 Traffic Monitoring")
        if total:
            timeline = hist.tail(150)
            st.plotly_chart(
                px.scatter(
                    timeline,
                    x="index",
                    y="confidence",
                    color="prediction",
                    hover_data=["attack_type", "actual"],
                    labels={"index": "Traffic Record", "confidence": "Ensemble Confidence"}
                ),
                use_container_width=True
            )
        else:
            st.info("Click Process Traffic to start monitoring.")

    with right:
        st.subheader("🚨 Live Alert Feed")
        if not alerts.empty:
            st.dataframe(
                alerts[["time", "attack_type", "confidence", "correct"]].head(15),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No intrusion alerts yet.")

    # EXPLAINABILITY
    st.subheader("🔍 Detection Explainability")

    if total:
        last = hist.iloc[-1]
        st.write(f"**Prediction:** {last.prediction}")
        st.write(f"**Attack Type:** {last.attack_type}")
        st.write(f"**Confidence:** {last.confidence:.0%}")

        votes = hist.tail(150).prediction.value_counts().reset_index()
        votes.columns = ["Decision", "Count"]
        st.plotly_chart(px.bar(votes, x="Decision", y="Count", color="Decision"), use_container_width=True)

    shap_file = "results/explainability_drift/shap_importance_delta.csv"

if os.path.exists(shap_file):
    shap = pd.read_csv(shap_file)

    st.markdown("### 🧠 SHAP Feature Importance — Adaptation Impact")

    shap = shap.sort_values("absolute_change", ascending=False).head(12)

    st.plotly_chart(
        px.bar(
            shap,
            x="importance_change",
            y="feature",
            orientation="h",
            color="importance_change",
            color_continuous_scale="RdBu",
            title="Feature Attribution Changes After Adaptation"
        ),
        use_container_width=True
    )

    st.dataframe(
        shap[["feature", "importance_before", "importance_after", "importance_change"]],
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("SHAP results not found. Run 05_explainability_drift.py first.")

    # MODEL STATUS
    st.subheader("⚙️ Model Status")
    st.write("**Adaptive Ensemble:** ARF + HAT + Hoeffding Tree")
    st.write("Models are updated incrementally as records are processed.")
    st.caption("Dataset replay simulates streaming traffic. It is not actual network capture.")

st.divider()
st.caption("Research prototype — live capture alerts are heuristic-based; dataset replay uses the online ensemble.")