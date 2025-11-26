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
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="PantryArbitrage", page_icon="🥗", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stMetric { background-color: #262730; padding: 15px; border-radius: 5px; }
    .scanner-box { border: 2px solid #00e676; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Retrieve API Key from Environment or Sidebar
# This is safe for GitHub: It checks the server's secrets first, then asks the user.
api_key = os.environ.get("GOOGLE_API_KEY")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3014/3014520.png", width=100)
    st.title("👤 User Profile")

    if not api_key:
        api_key = st.text_input("Enter Google API Key:", type="password")

    if api_key:
        genai.configure(api_key=api_key)
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            st.success("✅ Gemini 2.0 Connected")
        except Exception as e:
            st.error(f"❌ Connection Error: {e}")
            model = None
    else:
        st.warning("⚠️ Enter API Key to proceed")
        model = None

    mode = st.radio("Select Mode:", ["Meal Planner", "Leftover Scanner 📸"])
    
    st.markdown("---")
    user_name = st.text_input("Name", "Guest")
    budget_input = st.number_input("Weekly Budget ($)", value=100.0, step=10.0)
    allergies_input = st.text_input("Allergies", "None")
    dislikes_input = st.text_input("Dislikes", "Cilantro")

# ==========================================
# 2. HELPER FUNCTIONS
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
    except Exception as e:
        return ""
    return "\n".join(texts)

def web_search(query: str, max_results: int = 5):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(r)
    except Exception:
        pass
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
    except Exception:
        return "{}"

def extract_json_from_text(text: str):
    try:
        return json.loads(text)
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
        response = model.generate_content([
            prompt,
            {"mime_type": "audio/mp3", "data": audio_bytes}
        ])
        return response.text.strip()
    except Exception as e:
        return f"Error transcribing audio: {e}"

# ==========================================
# 3. AGENTS
# ==========================================

def agent_a_auditor(receipt_path, fridge_img, manual_text):
    source_desc = ""
    if receipt_path:
        source_desc += f"RECEIPT_TEXT:\n{extract_text_from_pdf(receipt_path)}\n\n"
    if manual_text:
        source_desc += f"MANUAL_INPUT:\n{manual_text}\n\n"

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
            qty = float(ing.get("quantity", 1))
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
    # Note: In a full version, you would pass both images to Gemini.
    # Currently analyzing 'after' image for waste status.
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
# 4. ORCHESTRATOR & UI LOGIC
# ==========================================

st.title("🥗 PantryArbitrage")
st.caption("Zero-Waste Kitchen Concierge powered by Gemini 2.0 Agents")

col1, col2 = st.columns([1, 1])

if mode == "Leftover Scanner 📸":
    with col1:
        st.subheader("📸 Compare & Track")
        st.info("Upload your 'Current Fridge' (Week 2) to see what is expiring.")
        
        img_after = st.file_uploader("Current Fridge Photo (Required)", type=["jpg","png","jpeg"], key="after")
        img_before = st.file_uploader("Previous Fridge Photo (Optional)", type=["jpg","png","jpeg"], key="before")
        
        scan_btn = st.button("🔍 Analyze Waste", type="primary")

    with col2:
        if scan_btn:
            if not img_after:
                st.error("⚠️ Please upload the 'Current Fridge Photo' first.")
            elif not model:
                st.error("⚠️ Please enter your API Key in the sidebar.")
            else:
                 with st.spinner("🕵️ Agent D is comparing fridge states..."):
                     try:
                         pil_after = Image.open(img_after)
                         res = agent_d_scanner(None, pil_after)
                         
                         st.subheader("📉 Waste Analysis")
                         waste = res.get('waste_detected', [])
                         if waste and waste != ["None"]:
                             st.error(f"⚠️ Waste Detected: {', '.join(waste)}")
                         else:
                             st.success("✅ No obvious waste detected!")
                         
                         consumed = res.get('items_consumed', [])
                         if consumed and consumed != ["Unknown"]:
                             st.info(f"😋 Likely Consumed: {', '.join(consumed)}")
                             
                         advice = res.get('advice')
                         if advice:
                             st.markdown(f"**💡 Chef's Advice:**\n{advice}")
                     except Exception as e:
                         st.error(f"Error processing image: {e}")

else:
    # Meal Planner Mode
    with col1:
        st.subheader("1. Upload Inputs")
        uploaded_img = st.file_uploader("📸 Fridge Photo", type=["jpg", "png", "jpeg"])
        uploaded_pdf = st.file_uploader("📄 Grocery Receipt", type=["pdf"])

        st.subheader("2. Your Goal")
        
        # Audio input (Voice)
        voice_text = ""
        audio_val = st.audio_input("🎤 Record Request (Optional)")

        if audio_val:
            with st.spinner("🎧 Transcribing your voice..."):
                voice_text = transcribe_audio(audio_val)
                st.success(f"🗣️ Heard: \"{voice_text}\"")

        default_text = voice_text if voice_text else "Plan dinners for 3 days using these ingredients. I want to save money."
        user_request = st.text_area("Or type here:", value=default_text)

        run_btn = st.button("🚀 Run Agents", type="primary", use_container_width=True)

    if run_btn:
        if not model:
            st.error("Please configure the API Key in the sidebar first.")
        else:
            profile = UserProfile(
                name=user_name,
                allergies=[x.strip() for x in allergies_input.split(",") if x.strip()],
                dislikes=[x.strip() for x in dislikes_input.split(",") if x.strip()],
                weekly_budget_usd=budget_input
            )

            status = st.status("🤖 Orchestrator Starting...", expanded=True)
            try:
                # 1. PREPARE
                pil_image = Image.open(uploaded_img) if uploaded_img else None
                pdf_path = None
                if uploaded_pdf:
                    with open("temp_receipt.pdf", "wb") as f:
                        f.write(uploaded_pdf.getbuffer())
                    pdf_path = "temp_receipt.pdf"

                # 2. RUN AGENTS
                status.write("👁️ **Agent A (Auditor):** Scanning inventory...")
                inventory = agent_a_auditor(pdf_path, pil_image, manual_text=None)
                st.session_state['inventory'] = inventory

                status.write("👨‍🍳 **Agent B (Chef):** Searching recipes & planning...")
                chef_plan = agent_b_chef(inventory, profile, user_request)
                st.session_state['plan'] = chef_plan

                status.write("📊 **Agent C (Analyst):** Calculating metrics...")
                report = agent_c_analyst(chef_plan)
                st.session_state['report'] = report

                status.update(label="✅ Workflow Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ Error Occurred", state="error")
                st.error(f"System Error: {str(e)}")

    if 'plan' in st.session_state and mode == "Meal Planner":
        plan = st.session_state['plan']
        report = st.session_state['report']
        inv = st.session_state.get('inventory', {})

        with col2:
            st.subheader("📊 Sustainability Report")
            m1, m2, m3 = st.columns(3)
            m1.metric("💰 Money Saved", f"${report['money_saved']}")
            m2.metric("🌍 CO₂ Saved", f"{report['co2_saved']} kg")
            m3.metric("🥘 Meals", report['num_meals'])

            st.subheader("🍽️ The Plan")
            st.info(plan.get("narrative_plan", "No narrative available."))

            for r in plan.get("recipes", []):
                with st.expander(f"🍳 {r.get('name')}"):
                    st.write(f"**Description:** {r.get('description')}")
                    st.write("**Ingredients:**")
                    for i in r.get("ingredients_used", []):
                        st.text(f"- {i.get('name')}: {i.get('quantity')}")
            
            with st.expander("🔍 View Detected Inventory"):
                st.json(inv)
