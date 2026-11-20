import json

import joblib
import streamlit as st

from utils import MODELS_DIR, build_laptop_row

MODEL_PATH = MODELS_DIR / "laptop_price_model.joblib"
OPTIONS_PATH = MODELS_DIR / "input_options.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
OTHER_LINE = "Other (not listed)"
HIGH_END_EUR = 2500

st.set_page_config(page_title="Laptop Price Prediction", page_icon="💻")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    options = json.loads(OPTIONS_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, options, metadata


def typical_error(price, bands):
    for band in bands:
        upper = float("inf") if band["to"] is None else band["to"]
        if band["from"] <= price < upper:
            return band
    return None


missing = [path.name for path in (MODEL_PATH, OPTIONS_PATH, METADATA_PATH) if not path.exists()]
if missing:
    st.error(f"Missing model files in models/: {', '.join(missing)}. "
             "Run the serialization section of notebook.ipynb first.")
    st.stop()

model, options, metadata = load_artifacts()
defaults = options["defaults"]

st.title("Laptop Price Prediction")
st.caption(f"Estimates a laptop's price in euros from its specifications "
           f"({metadata['model_type']}, trained on {metadata['trained_on_rows']} laptops).")


def select(container, label, key, extra=()):
    choices = list(options[key]) + list(extra)
    index = choices.index(defaults[key]) if defaults.get(key) in choices else 0
    return container.selectbox(label, choices, index=index)


def number(container, label, key, step):
    spec = options[key]
    return container.number_input(label, min_value=spec["min"], max_value=spec["max"],
                                  value=spec["default"], step=step)


with st.form("laptop_form"):
    left, middle, right = st.columns(3)
    company = select(left, "Company", "company")
    type_name = select(left, "Type", "type_name")
    product_line = select(left, "Product line", "product_line", extra=[OTHER_LINE])
    opsys = select(left, "Operating system", "opsys")
    inches = number(middle, "Screen size (inches)", "inches", 0.1)
    resolution = select(middle, "Screen resolution", "resolution")
    touchscreen = middle.checkbox("Touchscreen")
    ips = middle.checkbox("IPS panel")
    weight = number(middle, "Weight (kg)", "weight_kg", 0.05)
    cpu = select(right, "CPU", "cpu")
    gpu = select(right, "GPU", "gpu")
    ram = select(right, "RAM (GB)", "ram_gb")
    ssd = select(right, "SSD (GB)", "ssd_gb")
    hdd = select(right, "HDD (GB)", "hdd_gb")
    flash = select(right, "Flash storage (GB)", "flash_gb")
    submitted = st.form_submit_button("Predict price")

if submitted:
    try:
        row = build_laptop_row(
            company=company, type_name=type_name,
            product_line="Other" if product_line == OTHER_LINE else product_line,
            opsys=opsys, inches=inches, resolution=resolution, touchscreen=touchscreen, ips=ips,
            weight_kg=weight, cpu=cpu, gpu=gpu, ram_gb=ram, ssd_gb=ssd, hdd_gb=hdd, flash_gb=flash,
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()

    price = float(model.predict(row)[0])
    st.metric("Estimated price", f"€{price:,.0f}")
    band = typical_error(price, metadata["error_by_predicted_price"])
    if band:
        st.caption(f"Typical absolute error for estimates in this price range: about €{band['mae']:.0f} "
                   f"(out-of-fold, {band['n']} laptops).")
    if price >= HIGH_END_EUR:
        st.warning("High-end estimates are less reliable: very few laptops above "
                   f"€{HIGH_END_EUR:,} were available for training.")

with st.expander("About this model"):
    st.write(
        f"On a held-out test set the mean absolute error was about €{metadata['test_mae']:.0f} "
        "(a random split, so many test laptops share a product name with training laptops); "
        f"for products not seen in training, cross-validation gives about €{metadata['cv_mae_grouped']:.0f}. "
        "The saved model is trained on all laptops, so these figures come from a model trained without the test set. "
        "It learns from a fixed historical dataset of laptop listings; it is a demonstration, not a pricing service."
    )