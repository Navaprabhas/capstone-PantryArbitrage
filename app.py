import streamlit as st
import os
import json
import re
import google.generativeai as genai
from PIL import Image
import pdfplumber
from duckduckgo_search import DDGS
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
st.set_page_config(page_title="PantryArbitrage", page_icon="🥗", layout="wide")

# Minimalist CSS: Cleaner fonts, rounded buttons, hidden clutter
st.markdown("""
<style>
    .main { background-color: #f8f9fa; color: #212529; }
    .stButton>button { border-radius: 20px; font-weight: bold; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; color: #2E7D32; }
    .css-1544g2n { padding-top: 2rem; } /* Reduce top padding */
</style>
""", unsafe_allow_html=True)

# Retrieve API Key safely
api_key = os.environ.get("GOOGLE_API_KEY")

# ==========================================
# 2. SIDEBAR (Profile & Settings)
# ==========================================
with st.sidebar:
    # Logo / Icon
    st.image("https://cdn-icons-png.flaticon.com/512/3014/3014520.png", width=60)
    st.markdown("### **User Settings**")

    # Security: API Key Input
    if not api_key:
        api_key = st.text_input("🔑 Enter Google API Key:", type="password", help="Your key is not stored. It's used only for this session.")

    if api_key:
        genai.configure(api_key=api_key)
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            st.success("🟢 System Online")
        except Exception as e:
            st.error(f"❌ Connection Error: {e}")
            model = None
    else:
        st.warning("⚠️ Waiting for Key...")
        model = None

    st.markdown("---")
    
    # Mode Selection with clear icons
    mode = st.radio("🤖 Select Agent Mode:", ["🍳 Meal Planner", "📉 Waste Scanner"])
    
    with st.expander("👤 Edit Profile (Preferences)"):
        user_name = st.text_input("Name", "Guest")
        budget_input = st.number_input("Weekly Budget ($)", value=100.0, step=10.0)
        allergies_input = st.text_input("Allergies", "None", help="Agent will strictly avoid these.")
        dislikes_input = st.text_input("Dislikes", "Cilantro", help="Agent will try to avoid these.")

# ==========================================
# 3. HELPER FUNCTIONS (Logic Core)
# ==========================================

@dataclass
class UserProfile:
    name: str = "Guest"
    allergies: List[str] = None
    dislikes: List[str] = None
    weekly_budget_usd: float = 100.0

    def to_dict(self):
        return {
            "name": self.name,
            "allergies": self.allergies or [],
            "dislikes": self.dislikes or [],
            "weekly_budget_usd": self.weekly_budget_usd
        }

def extract_text_from_pdf(pdf_path: str) -> str:
    texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
    except Exception: return ""
    return "\n".join(texts)

def web_search(query: str, max_results: int = 5):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(r)
    except Exception: pass
    return results

def generate_text(prompt: str, image: Optional[Image.Image] = None, json_mode: bool = False) -> str:
    if not model: return "{}"
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json" if json_mode else "text/plain"
    )
    parts = [prompt]
    if image: parts.append(image)
    try:
        response = model.generate_content(parts, generation_config=generation_config)
        return response.text
    except Exception: return "{}"

def extract_json_from_text(text: str):
    try: return json.loads(text)
    except:
        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
    return {}

def transcribe_audio(audio_file):
    if not model: return ""
    try:
        prompt = "Listen to this audio and transcribe it exactly into English text. Return ONLY the text."
        audio_bytes = audio_file.read()
        response = model.generate_content([prompt, {"mime_type": "audio/mp3", "data": audio_bytes}])
        return response.text.strip()
    except Exception as e: return f"Error: {e}"

# --- AGENTS ---

def agent_a_auditor(receipt_path, fridge_img, manual_text):
    source_desc = ""
    if receipt_path: source_desc += f"RECEIPT_TEXT:\n{extract_text_from_pdf(receipt_path)}\n\n"
    prompt = """
    You are Agent A: Inventory Auditor. Analyze the input.
    Return STRICT JSON: {"ingredients": [{"name": "item", "quantity": 1, "unit": "unit", "days_until_expiry": 3, "category": "produce"}]}
    """
    if fridge_img:
        prompt += "\nAnalyze this fridge image."
        raw = generate_text(prompt, image=fridge_img, json_mode=True)
    else:
        prompt += f"\nData:\n{source_desc}"
        raw = generate_text(prompt, json_mode=True)
    data = extract_json_from_text(raw)
    return data if "ingredients" in data else {"ingredients": []}

def agent_b_chef(inventory, profile: UserProfile, request):
    ingredients = inventory.get("ingredients", [])
    ing_list = ", ".join([f"{i.get('name')} (exp {i.get('days_until_expiry')}d)" for i in ingredients])
    top_ing = ingredients[0]['name'] if ingredients else "vegetables"
    search_results = web_search(f"recipe using {top_ing} {request}", max_results=3)

    prompt = f"""
    You are Agent B: Chef.
    User: {json.dumps(profile.to_dict())}
    Inventory: {ing_list}
    Request: {request}
    Web Search Ideas: {search_results}
    Task: Create a meal plan using expiring items first.
    Return STRICT JSON:
    {{
      "recipes": [{{"name": "Recipe Name", "description": "desc", "ingredients_used": [{{"name": "x", "quantity": 1}}]}}],
      "narrative_plan": "A friendly summary of the plan."
    }}
    """
    raw = generate_text(prompt, json_mode=True)
    data = extract_json_from_text(raw)
    return data if "recipes" in data else {"recipes": [], "narrative_plan": "Could not generate plan."}

def agent_c_analyst(recipes_plan, baseline_cost=20.0, baseline_co2=3.0):
    recipes = recipes_plan.get("recipes", [])
    num_meals = max(len(recipes), 1)
    PRICE_DB = {"egg": 0.25, "milk": 1.2, "chicken": 3.0, "rice": 0.5, "default": 1.0}
    CO2_DB = {"egg": 0.2, "milk": 1.2, "chicken": 3.0, "rice": 0.1, "default": 0.5}

    total_cost = 0.0
    total_co2 = 0.0

    for r in recipes:
        for ing in r.get("ingredients_used", []):
            try: qty = float(ing.get("quantity", 1))
            except ValueError: qty = 1.0
            
            name = ing.get("name", "").lower()
            cost_unit = next((v for k,v in PRICE_DB.items() if k in name), PRICE_DB["default"])
            co2_unit = next((v for k,v in CO2_DB.items() if k in name), CO2_DB["default"])
            total_cost += cost_unit * qty
            total_co2 += co2_unit * qty

    baseline_c = baseline_cost * num_meals
    baseline_e = baseline_co2 * num_meals

    return {
        "num_meals": num_meals,
        "cooking_cost": round(total_cost, 2),
        "takeout_cost": round(baseline_c, 2),
        "money_saved": round(baseline_c - total_cost, 2),
        "co2_saved": round(baseline_e - total_co2, 2)
    }

def agent_d_scanner(img_before, img_after):
    prompt = """
    Compare these two fridge images (Before and After).
    Identify what was eaten and what is new.
    Return STRICT JSON:
    {
      "items_consumed": ["item1", "item2"],
      "waste_detected": ["item3"],
      "advice": "Short advice."
    }
    """
    raw = generate_text(prompt, image=img_after, json_mode=True)
    return extract_json_from_text(raw)

# ==========================================
# 4. MAIN UI LAYOUT
# ==========================================

# --- HERO SECTION ---
st.title("🥗 PantryArbitrage")
st.markdown("**The Zero-Waste Kitchen Concierge.** Stop wasting food, save money, and cook better.")

# --- ONBOARDING (Expandable Guide) ---
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    **1. Choose a Mode (Sidebar):**
    * **🍳 Meal Planner:** Generates recipes based on what you have.
    * **📉 Waste Scanner:** Compares fridge photos to track what you ate vs. wasted.
    
    **2. Input Data:**
    * Upload a **Fridge Photo** (Vision Agent).
    * Upload a **Receipt** (Auditor Agent).
    * Or just **Speak** your request!
    
    **3. Run:**
    * Click the **Action Button** to let the Multi-Agent System work.
    """)

# --- MAIN COLUMNS ---
col1, col2 = st.columns([1, 1.2], gap="large")

if mode == "📉 Waste Scanner":
    with col1:
        st.subheader("📸 Audit Your Fridge")
        st.info("Upload photos to track consumption habits.")
        
        img_after = st.file_uploader("Current Fridge (Today)", type=["jpg","png","jpeg"], key="after", help="Photo of your fridge right now.")
        img_before = st.file_uploader("Previous Fridge (Optional)", type=["jpg","png","jpeg"], key="before", help="Photo from last week.")
        
        scan_btn = st.button("📉 Analyze Waste Report", type="primary", use_container_width=True)

    with col2:
        if scan_btn:
            if not img_after:
                st.warning("⚠️ Please upload the 'Current Fridge' photo.")
            elif not model:
                st.error("⚠️ Please connect the API Key in the sidebar.")
            else:
                 with st.spinner("🕵️ Agent D is comparing fridge states..."):
                     try:
                         pil_after = Image.open(img_after)
                         res = agent_d_scanner(None, pil_after)
                         
                         st.subheader("📉 Analysis Results")
                         
                         # Use clean containers for results
                         c1, c2 = st.columns(2)
                         with c1:
                            st.error(f"**🗑️ Waste Detected:**\n\n" + ", ".join(res.get('waste_detected', ['None'])))
                         with c2:
                            st.success(f"**😋 Eaten:**\n\n" + ", ".join(res.get('items_consumed', ['Unknown'])))

                         if res.get('advice'):
                             st.info(f"**💡 Chef's Advice:** {res.get('advice')}")

                     except Exception as e:
                         st.error(f"Error: {e}")

else:
    # --- MEAL PLANNER MODE ---
    with col1:
        st.subheader("1. What do you have?")
        
        # Tabs for cleaner input organization
        tab1, tab2 = st.tabs(["📸 Vision Input", "🎙️ Voice/Text"])
        
        with tab1:
            uploaded_img = st.file_uploader("Upload Fridge Photo", type=["jpg", "png", "jpeg"], help="Agent A will identify ingredients.")
            uploaded_pdf = st.file_uploader("Upload Receipt (PDF)", type=["pdf"], help="Agent A will add these to your inventory.")

        with tab2:
            audio_val = st.audio_input("Record a Request")
            voice_text = ""
            if audio_val:
                with st.spinner("🎧 Transcribing..."):
                    voice_text = transcribe_audio(audio_val)
                    st.success(f"🗣️ \"{voice_text}\"")
            
            default_txt = voice_text if voice_text else "Plan dinners for 3 days. I want to save money."
            user_request = st.text_area("Your Goal:", value=default_txt, help="Tell the Chef what you crave.")

        st.markdown("###") # Spacer
        run_btn = st.button("✨ Generate Zero-Waste Plan", type="primary", use_container_width=True)

    # --- ORCHESTRATOR LOGIC ---
    if run_btn:
        if not model:
            st.error("Please enter your API Key in the sidebar.")
        else:
            # Build Profile
            profile = UserProfile(
                name=user_name,
                allergies=[x.strip() for x in allergies_input.split(",") if x.strip()],
                dislikes=[x.strip() for x in dislikes_input.split(",") if x.strip()],
                weekly_budget_usd=budget_input
            )

            # Progress Bar / Status
            status_box = st.status("🤖 AI Agents at work...", expanded=True)
            
            try:
                # 1. Inputs
                pil_image = Image.open(uploaded_img) if uploaded_img else None
                pdf_path = None
                if uploaded_pdf:
                    with open("temp_receipt.pdf", "wb") as f: f.write(uploaded_pdf.getbuffer())
                    pdf_path = "temp_receipt.pdf"

                # 2. Execution Chain
                status_box.write("👁️ **Agent A (Auditor):** Scanning inventory...")
                inventory = agent_a_auditor(pdf_path, pil_image, manual_text=None)
                st.session_state['inventory'] = inventory

                status_box.write("👨‍🍳 **Agent B (Chef):** Searching recipes & planning...")
                chef_plan = agent_b_chef(inventory, profile, user_request)
                st.session_state['plan'] = chef_plan

                status_box.write("📊 **Agent C (Analyst):** Calculating arbitrage...")
                report = agent_c_analyst(chef_plan)
                st.session_state['report'] = report

                status_box.update(label="✅ Plan Ready!", state="complete", expanded=False)
            
            except Exception as e:
                status_box.update(label="❌ Error", state="error")
                st.error(f"System Error: {str(e)}")

    # --- OUTPUT DISPLAY ---
    if 'plan' in st.session_state and 'report' in st.session_state and mode == "🍳 Meal Planner":
        plan = st.session_state['plan']
        report = st.session_state['report']
        inv = st.session_state.get('inventory', {})

        with col2:
            st.subheader("📊 The Value Proposition")
            
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("💰 Money Saved", f"${report['money_saved']}", delta="Arbitrage")
            m2.metric("🌍 CO₂ Saved", f"{report['co2_saved']} kg", delta="Impact")
            m3.metric("🥘 Meals Planned", report['num_meals'])

            st.markdown("---")
            
            st.subheader("🍽️ Your Zero-Waste Plan")
            st.info(f"**Chef's Note:** {plan.get('narrative_plan', 'Ready to cook!')}")

            # Recipe Cards
            for r in plan.get("recipes", []):
                with st.expander(f"🍳 {r.get('name')}", expanded=True):
                    st.markdown(f"*{r.get('description')}*")
                    st.markdown("**Key Ingredients Used:**")
                    for i in r.get("ingredients_used", []):
                        st.text(f"• {i.get('name')}: {i.get('quantity')}")
            
            with st.expander("🔍 View Raw Inventory Data"):
                st.json(inv)
