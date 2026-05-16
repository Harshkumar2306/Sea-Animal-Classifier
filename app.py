import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import wikipedia
import uuid
wikipedia.set_user_agent("SeaAnimalClassifier/1.0 (contact@example.com)")

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Sea Animals Classification",
    page_icon="🌊",
    layout="wide"
)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
MODEL_PATH = "BioHMSC_best_model.pth"

# =========================================================
# CLASSES
# =========================================================
class_names = [
'Clams','Corals','Crabs','Dolphin','Eel','Fish','Jelly Fish','Lobster',
'Nudibranchs','Octopus','Otter','Penguin','Puffers','Sea Rays','Sea Urchins',
'Seahorse','Seal','Sharks','Shrimp','Squid','Starfish','Turtle_Tortoise','Whale'
]

# =========================================================
# LOAD MODEL
# =========================================================
import torch.nn as nn

class BioHMSC(nn.Module):
    def __init__(self, num_fine_classes=23, num_coarse_classes=5):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnetv2_m",
            pretrained=False,
            num_classes=0
        )
        f = self.backbone.num_features
        self.shared_mlp = nn.Sequential(
            nn.Linear(f, 512),
            nn.BatchNorm1d(512),
            nn.SiLU()
        )
        self.species_head = nn.Linear(512, num_fine_classes)
        self.coarse_head = nn.Linear(512, num_coarse_classes)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.shared_mlp(feats)
        return self.species_head(feats), self.coarse_head(feats)

@st.cache_resource
def load_model():
    model = BioHMSC(num_fine_classes=len(class_names), num_coarse_classes=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# =========================================================
# TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =========================================================
# PREDICT
# =========================================================
def predict(img):
    x = transform(img).unsqueeze(0).to(device)
    
    # Test-Time Augmentation (TTA) - Horizontal Flip
    x_flipped = torch.flip(x, dims=[3])
    
    with torch.no_grad():
        # Predict on original
        s_out_orig, _ = model(x)
        
        # Predict on flipped
        s_out_flipped, _ = model(x_flipped)
        
        # Average before softmax
        outputs = (s_out_orig + s_out_flipped) / 2.0
        
        # Temperature Scaling
        temperature = 0.7
        probs = torch.softmax(outputs / temperature, dim=1)
        
    top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
    
    results = []
    for i in range(3):
        results.append({
            "label": class_names[top3_indices[0][i].item()],
            "conf": top3_probs[0][i].item() * 100
        })
        
    return results

# =========================
# AGENT LOGIC (WIKIPEDIA)
# =========================
def agent_reason(label, conf):
    try:
        # Manual overrides for specific classes
        overrides = {
            "Sea Rays": "Batoidea",
            "Turtle_Tortoise": "Turtle",
            "Puffers": "Pufferfish",
            "Eel": "Eel",
            "Seal": "Pinniped"
        }
        
        search_label = overrides.get(label, label)
        
        with st.spinner("📚 Agent is reading knowledge base..."):
            # 1. Try exact label page first
            page = None
            try:
                page = wikipedia.page(search_label, auto_suggest=False)
            except wikipedia.DisambiguationError as e:
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                except (wikipedia.DisambiguationError, wikipedia.PageError, IndexError):
                    pass
            except wikipedia.PageError:
                pass
            
            # 2. If exact page fails, try singular label
            if not page and search_label.endswith('s'):
                try:
                     page = wikipedia.page(search_label[:-1], auto_suggest=False)
                except wikipedia.DisambiguationError as e:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                    except (wikipedia.DisambiguationError, wikipedia.PageError, IndexError):
                        pass
                except wikipedia.PageError:
                    pass

            # 3. If still no page, perform a search
            if not page:
                search_term = search_label
                if search_label.endswith('s'):
                    search_term = search_label[:-1] 
                
                query = f"{search_term} marine animal"
                results = wikipedia.search(query)
                
                if results:
                    try:
                        page = wikipedia.page(results[0], auto_suggest=False)
                    except wikipedia.DisambiguationError as e:
                        try:
                            page = wikipedia.page(e.options[0], auto_suggest=False)
                        except (wikipedia.DisambiguationError, wikipedia.PageError, IndexError):
                            pass
                    except wikipedia.PageError:
                        pass
            
            if not page:
                 return f"⚠️ No information found for **{label}**."

            # Get summary
            summary = page.summary[:500] + "..." if len(page.summary) > 500 else page.summary
            
            return f"""
**Encyclopedia Summary:**  
{summary}

**Source:** [Wikipedia]({page.url})
"""

    except Exception as e:
        return f"❌ **Knowledge Base Error:** {str(e)}"

# =========================================================
# UI
# =========================================================
st.title("🌊 Sea Animals Classification")

if "history" not in st.session_state:
    st.session_state.history = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0



files = st.file_uploader(
    "Upload images",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set()

if files:
    # 1. Identify current files using a unique key
    # We try file_id (Streamlit feature), falling back to name+size
    def get_file_id(f):
        return getattr(f, "file_id", f"{f.name}_{f.size}")

    current_file_ids = {get_file_id(f) for f in files}
    
    # 2. Cleanup: Remove processed IDs that are no longer in the uploader
    # This ensures if a user removes a file and adds it back, we re-process it
    st.session_state.processed_ids = {pid for pid in st.session_state.processed_ids if pid in current_file_ids}

    for file in files:
        fid = get_file_id(file)
        
        # 3. Process only if NOT already processed for this session
        if fid not in st.session_state.processed_ids:
            try:
                img = Image.open(file).convert("RGB")
                
                # Predict (returns top 3)
                top3_results = predict(img)
                top_label = top3_results[0]["label"]
                top_conf = top3_results[0]["conf"]
                
                # Agent Explain for top prediction
                explanation = agent_reason(top_label, top_conf)
                
                # Store
                st.session_state.history.append({
                    "id": str(uuid.uuid4()),
                    "name": file.name,
                    "image": img,
                    "top3": top3_results,
                    "explanation": explanation
                })
                
                # Mark as processed
                st.session_state.processed_ids.add(fid)
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

# History / Comparison View
if st.session_state.history:
    st.markdown("---")
    c1, c2 = st.columns([0.8, 0.2])
    c1.subheader("📜 Prediction")
    if c2.button("Clear History"):
        st.session_state.history = []
        st.session_state.uploader_key += 1
        st.rerun()

    # Iterate backwards by index to show newest first and handle deletion correctly
    for i in range(len(st.session_state.history) - 1, -1, -1):
        item = st.session_state.history[i]
        
        with st.container(border=True):
            col_img, col_info, col_del = st.columns([0.25, 0.70, 0.05])
            
            with col_img:
                st.image(item['image'], use_container_width=True)
                st.caption(f"**{item['name']}**")
            
            with col_info:
                top_label = item['top3'][0]['label']
                st.success(f"**Top Match: {top_label}**")
                
                # Display Top 3 with progress bars
                for res in item['top3']:
                    conf_val = min(max(res['conf'], 0.0), 100.0)
                    st.write(f"**{res['label']}** — {conf_val:.1f}%")
                    st.progress(int(conf_val))
                
                st.markdown("---")
                st.markdown(item['explanation'])
                
            with col_del:
                if st.button("❌", key=f"del_{item['id']}", help="Remove this result"):
                    # Find and remove by id instead of index to prevent dynamic widget bugs
                    st.session_state.history = [h for h in st.session_state.history if h['id'] != item['id']]
                    # We do NOT remove from processed_ids here.
                    # This prevents the file from reappearing immediately on rerun 
                    # (since it's still in the uploader and in processed_ids).
                    # User must remove from uploader to clear it from processed_ids.
                    st.rerun()
