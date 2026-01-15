# rbs_ac_app.py
import json
import operator
import streamlit as st
from typing import List, Dict, Any, Tuple

# ----------------------------
# 1) Operator mapping for conditions
# ----------------------------
OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
}

# ----------------------------
# 2) Load AC JSON rules (paste your JSON here or load from file)
# ----------------------------
DEFAULT_RULES: List[Dict[str, Any]] = [
    {
        "name": "Windows open → turn AC off",
        "priority": 100,
        "conditions": [["windows_open", "==", True]],
        "action": {"ac_mode": "OFF", "fan_speed": "LOW", "setpoint": None, "reason": "Windows are open"},
    },
    {
        "name": "No one home → eco mode",
        "priority": 90,
        "conditions": [["occupancy", "==", "EMPTY"], ["temperature", ">=", 24]],
        "action": {"ac_mode": "ECO", "fan_speed": "LOW", "setpoint": 27, "reason": "Home empty; save energy"},
    },
    {
        "name": "Hot & humid (occupied) → cool strong",
        "priority": 80,
        "conditions": [["occupancy", "==", "OCCUPIED"], ["temperature", ">=", 30], ["humidity", ">=", 70]],
        "action": {"ac_mode": "COOL", "fan_speed": "HIGH", "setpoint": 23, "reason": "Hot and humid"},
    },
    {
        "name": "Hot (occupied) → cool",
        "priority": 70,
        "conditions": [["occupancy", "==", "OCCUPIED"], ["temperature", ">=", 28]],
        "action": {"ac_mode": "COOL", "fan_speed": "MEDIUM", "setpoint": 24, "reason": "Temperature high"},
    },
    {
        "name": "Slightly warm (occupied) → gentle cool",
        "priority": 60,
        "conditions": [["occupancy", "==", "OCCUPIED"], ["temperature", ">=", 26], ["temperature", "<", 28]],
        "action": {"ac_mode": "COOL", "fan_speed": "LOW", "setpoint": 25, "reason": "Slightly warm"},
    },
    {
        "name": "Night (occupied) → sleep mode",
        "priority": 75,
        "conditions": [["occupancy", "==", "OCCUPIED"], ["time_of_day", "==", "NIGHT"], ["temperature", ">=", 26]],
        "action": {"ac_mode": "SLEEP", "fan_speed": "LOW", "setpoint": 26, "reason": "Night comfort"},
    },
    {
        "name": "Too cold → turn off",
        "priority": 85,
        "conditions": [["temperature", "<=", 22]],
        "action": {"ac_mode": "OFF", "fan_speed": "LOW", "setpoint": None, "reason": "Already cold"},
    },
]

# ----------------------------
# 3) Condition evaluation
# ----------------------------
def evaluate_condition(facts: Dict[str, Any], cond: List[Any]) -> bool:
    """Evaluate a single condition: [field, op, value]."""
    if len(cond) != 3:
        return False
    field, op, value = cond
    if field not in facts or op not in OPS:
        return False
    try:
        return OPS[op](facts[field], value)
    except Exception:
        return False

def rule_matches(facts: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    """All conditions must be true (AND)."""
    return all(evaluate_condition(facts, c) for c in rule.get("conditions", []))

def run_rules(facts: Dict[str, Any], rules: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return best action (highest priority) and all matched rules."""
    fired = [r for r in rules if rule_matches(facts, r)]
    if not fired:
        return ({"ac_mode": "REVIEW", "reason": "No rule matched"}, [])
    fired_sorted = sorted(fired, key=lambda r: r.get("priority", 0), reverse=True)
    best = fired_sorted[0].get("action", {"ac_mode": "REVIEW", "reason": "No action"})
    return best, fired_sorted

# ----------------------------
# 4) Streamlit UI
# ----------------------------
st.set_page_config(page_title="Rule-Based Smart AC Controller", page_icon="❄️", layout="wide")
st.title("❄️ Rule-Based Smart Home Air Conditioner")
st.caption("Decision logic based on JSON rules with priority evaluation.")

with st.sidebar:
    st.header("Home Environment Facts")
    temperature = st.number_input("Temperature (°C)", value=22)
    humidity = st.number_input("Humidity (%)", value=46)
    occupancy = st.selectbox("Occupancy", ["OCCUPIED", "EMPTY"])
    time_of_day = st.selectbox("Time of Day", ["MORNING", "AFTERNOON", "EVENING", "NIGHT"])
    windows_open = st.checkbox("Windows Open", value=False)

    st.divider()
    st.header("Rules (JSON)")
    st.caption("Paste your JSON rules here or use defaults.")
    default_json = json.dumps(DEFAULT_RULES, indent=2)
    rules_text = st.text_area("Edit rules here", value=default_json, height=350)

    run_button = st.button("Evaluate AC Settings")

facts = {
    "temperature": float(temperature),
    "humidity": float(humidity),
    "occupancy": occupancy,
    "time_of_day": time_of_day,
    "windows_open": windows_open
}

st.subheader("Current Home Facts")
st.json(facts)

# Parse rules
try:
    rules = json.loads(rules_text)
    assert isinstance(rules, list)
except Exception as e:
    st.error(f"Invalid JSON rules, using defaults. Details: {e}")
    rules = DEFAULT_RULES

st.subheader("Active Rules")
with st.expander("Show Rules JSON", expanded=False):
    st.code(json.dumps(rules, indent=2), language="json")

st.divider()

if run_button:
    action, fired_rules = run_rules(facts, rules)

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("✅ AC Decision")
        ac_mode = action.get("ac_mode", "REVIEW")
        fan_speed = action.get("fan_speed", "-")
        setpoint = action.get("setpoint", "-")
        reason = action.get("reason", "-")

        if ac_mode == "OFF":
            st.error(f"{ac_mode} — {reason}")
        elif ac_mode == "COOL" or ac_mode == "ECO" or ac_mode == "SLEEP":
            st.success(f"{ac_mode} — {reason}")
        else:
            st.warning(f"{ac_mode} — {reason}")

        st.write(f"**Fan Speed:** {fan_speed}")
        st.write(f"**Setpoint:** {setpoint}")

    with col2:
        st.subheader("Matched Rules (by priority)")
        if not fired_rules:
            st.info("No rules matched.")
        else:
            for i, r in enumerate(fired_rules, start=1):
                st.write(f"**{i}. {r.get('name','(unnamed)')}** | priority={r.get('priority',0)}")
                st.caption(f"Action: {r.get('action',{})}")
                with st.expander("Conditions"):
                    for cond in r.get("conditions", []):
                        st.code(str(cond))
else:
    st.info("Set home facts and click **Evaluate AC Settings**.")
