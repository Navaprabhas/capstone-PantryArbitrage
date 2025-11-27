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
# 1. STYLE & CONFIGURATION (THE STUNNING PART)
# ==========================================
st.set_page_config(page_title="PantryArbitrage", page_icon="🥗", layout="wide")

# Custom CSS for "Glassmorphism" look and animations
st.markdown("""
<style>
    /* Gradient Background for Header */
    .hero-header {
        background: linear-gradient(135deg, #2E7D32 0%, #00C853 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .hero-header h1 { color: white !important; margin: 0; font-size: 3rem; }
    .hero-header p { font-size: 1.2rem; opacity: 0.9; }

    /* Card Styling */
    .st-emotion-cache-1r6slb0 {
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        background-color: #1E1E1E;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #FF512F 0%, #DD2476 100%); /* Cool gradient for CTA */
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1.2rem;
        border-radius: 50px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }

    /* Highlight the Sidebar Arrow */
    .arrow-highlight {
        font-size: 2rem;
        font-weight: bold;
        color: #00E676;
        animation: bounce 1s infinite;
    }
    @keyframes bounce {
        0%, 100% { transform: translateX(0); }
        50% { transform: translateX(-10px); }
    }
</style>
""", unsafe_allow_html=True)

# Retrieve API Key safely
api_key = os.environ.get("GOOGLE_API_KEY")

# ==========================================
# 2. SIDEBAR (LOGIC)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3014/3014520.png", width=60)
    
    st.markdown("### ⚙️ Control Panel")
    
    # --- SECURITY INPUT ---
    if not api_key:
        # Add a visual cue in the sidebar itself
        st.markdown("👇 **STEP 1: ENTER KEY**")
        api_key = st.text_input("🔑 Google API Key", type="password", help="Get this from Google AI Studio. It's free!")
    
    if api_key:
        genai.configure(api_key=api_key)
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            st.success("🟢 System Online")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            model = None
    else:
        st.warning("⚠️ Waiting for Key...")
        model = None

    st.markdown("---")
    
    # Mode Selector
    st.markdown("👇 **STEP 2: CHOOSE MODE**")
    mode = st.radio("Select Agent Mode:", ["🍳 Meal Planner", "📉 Waste Scanner"], 
                    captions=["Plan new meals", "Audit fridge waste"])
    
    # Profile Settings (Hidden in Expander to keep it clean)
    with st.expander("👤 User Preferences"):
        user_name = st.text_input("Name", "Guest")
        budget_input = st.number_input("Weekly Budget ($)", value=100.0, step=10.0)
        allergies_input = st.text_input("Allergies", "None")
        dislikes_input = st.text_input("Dislikes", "Cilantro")

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

def agent_d_scanner(img_before, img_after, roast_mode=False):
    if roast_mode:
        prompt = """
        Compare these fridge images. You are NOT a helpful assistant. 
        You are a sarcastic, brutal celebrity chef (like Gordon Ramsay).
        Roast the user for what they have in their fridge. Be funny but harsh.
        Return STRICT JSON: {"items_consumed": [], "waste_detected": [], "advice": "Roast here."}
        """
    else:
        prompt = """
        Compare these images. Return STRICT JSON:
        {"items_consumed": ["x"], "waste_detected": ["y"], "advice": "Helpful advice."}
        """
    raw = generate_text(prompt, image=img_after, json_mode=True)
    return extract_json_from_text(raw)

# ==========================================
# 4. MAIN UI LAYOUT
# ==========================================

# --- HERO HEADER ---
st.markdown("""
<div class="hero-header">
    <h1>🥗 PantryArbitrage</h1>
    <p>The Zero-Waste Kitchen Concierge | Powered by Gemini 2.0 flash </p>
</div>
""", unsafe_allow_html=True)

# --- WELCOME MODE (If no API Key) ---
if not api_key:
    st.info("👋 Welcome! It looks like you haven't connected your Brain yet.")
    
    # Create columns to simulate "Pointing"
    w1, w2 = st.columns([1, 2])
    with w1:
        st.markdown("""
        <div class="arrow-highlight">
            👈 START HERE
        </div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown("### 1. Go to the Sidebar")
        st.markdown("### 2. Enter your Google API Key")
        st.markdown("### 3. Unlock the Agents")

else:
    # --- APP IS UNLOCKED ---
    
    # 1. WASTE SCANNER MODE
    if mode == "📉 Waste Scanner":
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            with st.container(border=True):
                st.subheader("📸 Step 1: Audit")
                st.info("Upload your fridge photos to track habits.")
                img_after = st.file_uploader("Current Fridge (Today)", type=["jpg","png","jpeg"], key="after")
                img_before = st.file_uploader("Previous Fridge (Last Week)", type=["jpg","png","jpeg"], key="before")
                roast = st.toggle("🔥 Enable 'Gordon Ramsay' Mode")
                
                st.markdown("###")
                scan_btn = st.button("📉 Analyze Waste Report", type="primary", use_container_width=True)

        with col2:
            if scan_btn:
                if not img_after:
                    st.warning("⚠️ Please upload the 'Current Fridge' photo.")
                else:
                     with st.spinner("🕵️ Agent D is judging your fridge..." if roast else "🕵️ Agent D is comparing states..."):
                         try:
                             pil_after = Image.open(img_after)
                             res = agent_d_scanner(None, pil_after, roast_mode=roast)
                             
                             st.balloons() # Fun!
                             
                             # Results Card
                             with st.container(border=True):
                                 st.subheader("📉 Analysis Results")
                                 c1, c2 = st.columns(2)
                                 c1.error(f"**🗑️ Waste:**\n\n" + ", ".join(res.get('waste_detected', ['None'])))
                                 c2.success(f"**😋 Eaten:**\n\n" + ", ".join(res.get('items_consumed', ['Unknown'])))
                                 
                                 advice = res.get('advice')
                                 if advice:
                                     st.markdown("---")
                                     if roast: st.warning(f"**🔥 ROAST:** {advice}")
                                     else: st.info(f"**💡 ADVICE:** {advice}")

                         except Exception as e:
                             st.error(f"Error: {e}")

    # 2. MEAL PLANNER MODE
    else:
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        # --- LEFT COLUMN (INPUTS) ---
        with col1:
            with st.container(border=True):
                st.subheader("🥣 Step 1: Input Ingredients")
                
                tab_cam, tab_mic = st.tabs(["📸 Vision Input", "🎙️ Voice Input"])
                
                with tab_cam:
                    st.caption("Upload a photo of your fridge or receipt.")
                    uploaded_img = st.file_uploader("Fridge Photo", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
                    uploaded_pdf = st.file_uploader("Receipt PDF", type=["pdf"], label_visibility="collapsed")
                
                with tab_mic:
                    st.caption("Tell the Chef what you want.")
                    audio_val = st.audio_input("Record Request")
                    voice_text = ""
                    if audio_val:
                        with st.spinner("Transcribing..."):
                            voice_text = transcribe_audio(audio_val)
                            st.success(f"🗣️ \"{voice_text}\"")
                    
                    default_txt = voice_text if voice_text else "Plan dinners for 3 days. I want to save money."
                    user_request = st.text_area("Your Goal:", value=default_txt)

            st.markdown("###")
            run_btn = st.button("✨ Generate Zero-Waste Plan", type="primary", use_container_width=True)

        # --- RIGHT COLUMN (RESULTS) ---
        with col2:
            if run_btn:
                # Build Profile
                profile = UserProfile(
                    name=user_name,
                    allergies=[x.strip() for x in allergies_input.split(",") if x.strip()],
                    dislikes=[x.strip() for x in dislikes_input.split(",") if x.strip()],
                    weekly_budget_usd=budget_input
                )

                status_box = st.status("🤖 Orchestrating Agents...", expanded=True)
                
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

                    status_box.write("👨‍🍳 **Agent B (Chef):** Searching recipes...")
                    chef_plan = agent_b_chef(inventory, profile, user_request)
                    st.session_state['plan'] = chef_plan

                    status_box.write("📊 **Agent C (Analyst):** Calculating arbitrage...")
                    report = agent_c_analyst(chef_plan)
                    st.session_state['report'] = report

                    status_box.update(label="✅ Plan Ready!", state="complete", expanded=False)
                    st.balloons()
                
                except Exception as e:
                    status_box.update(label="❌ Error", state="error")
                    st.error(f"System Error: {str(e)}")

            # Display Results if they exist
            if 'plan' in st.session_state and 'report' in st.session_state:
                plan = st.session_state['plan']
                report = st.session_state['report']
                inv = st.session_state.get('inventory', {})

                # Value Metrics
                with st.container(border=True):
                    st.subheader("📊 The Value")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("💰 Money Saved", f"${report['money_saved']}", delta="Arbitrage")
                    m2.metric("🌍 CO₂ Saved", f"{report['co2_saved']} kg", delta="Impact")
                    m3.metric("🥘 Meals", report['num_meals'])

                st.markdown("###")
                
                # Recipe Plan
                with st.container(border=True):
                    st.subheader("🍽️ The Plan")
                    st.info(f"**Chef's Note:** {plan.get('narrative_plan', 'Ready to cook!')}")
                    
                    for r in plan.get("recipes", []):
                        with st.expander(f"🍳 {r.get('name')}", expanded=True):
                            st.markdown(f"*{r.get('description')}*")
                            st.markdown("**Key Ingredients:**")
                            for i in r.get("ingredients_used", []):
                                st.text(f"• {i.get('name')}: {i.get('quantity')}")
                
                with st.expander("🔍 Debug: View Raw Inventory"):
                    st.json(inv)
