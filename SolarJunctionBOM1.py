import os
import streamlit as st
import pandas as pd
import speech_recognition as sr
from dotenv import load_dotenv
from backend import SolarBOQGenerator
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PIL import Image  # New import for handling the logo

# Set the page configuration with custom logo
st.set_page_config(
    page_title="SolarJunctionBOQ",
    page_icon=Image.open("solar_junction_logo.png"),
    layout="centered",
    initial_sidebar_state="collapsed"
)

executor = ThreadPoolExecutor(max_workers=3)

@st.cache_data
def load_initial_data(_solar_boq):
    return {
        'manufacturers': _solar_boq.get_available_manufacturers(),
        'inverters': _solar_boq.get_available_inverters(),
        'pv_modules_by_manufacturer': {
            manufacturer: _solar_boq.get_available_pv_modules(manufacturer)
            for manufacturer in _solar_boq.get_available_manufacturers()
        },
        'earthing_material_manufacturers': _solar_boq.get_available_earthing_material_manufacturers(),
        'earthing_rod_manufacturers': _solar_boq.get_earthing_rod_manufacturers()
    }

def get_voice_input():
    r = sr.Recognizer()
    text = ""
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            r.adjust_for_ambient_noise(source, duration=1)
            st.info("Please speak clearly...")
            audio = r.listen(source, timeout=5, phrase_time_limit=15)
            st.info("Processing speech...")
            text = r.recognize_google(audio)
            st.success(f"Recognized: {text}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    return text

def reset_session_state():
    # Keep track of which keys to preserve
    keys_to_keep = {'history', 'initialized', 'cached_data'}
    
    # Create a new history list if it doesn't exist
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Store the values we want to keep
    preserved_values = {key: st.session_state[key] for key in keys_to_keep if key in st.session_state}
    
    # Clear all session state
    st.session_state.clear()
    
    # Restore preserved values
    for key, value in preserved_values.items():
        st.session_state[key] = value
    
    # Set initial values
    st.session_state.step = 'initial'
    st.session_state.manufacturer = None
    st.session_state.capacity = None  
    st.session_state.selected_module = None
    st.session_state.selected_inverter = None
    st.session_state.earthing_material_manufacturer = None
    st.session_state.earthing_rod_manufacturer = None
    st.session_state.text_input = ""

def get_best_pv_module(pv_modules):
    best_module = None
    for module in pv_modules:
        if best_module is None or \
           module['Pmax'] > best_module['Pmax'] or \
           (module['Pmax'] == best_module['Pmax'] and module['Vmp'] > best_module['Vmp']):
            best_module = module
    return best_module

async def process_user_input(user_input, solar_boq):
    if not user_input.strip():
        return False
        
    loop = asyncio.get_event_loop()
    extracted_info = await loop.run_in_executor(
        executor, 
        solar_boq.extract_info_from_input,
        user_input
    )
    
    if extracted_info['manufacturer'] and extracted_info['capacity']:
        st.session_state.manufacturer = extracted_info['manufacturer']
        st.session_state.capacity = extracted_info['capacity']
        st.session_state.selected_inverter = extracted_info.get('inverter')
        st.session_state.step = 'boq_generation'
        st.rerun()
        return True
    else:
        if not extracted_info['manufacturer']:
            st.error("Could not extract manufacturer information. Please provide a valid manufacturer.")
        if not extracted_info['capacity']:
            st.error("Could not extract capacity information. Please provide a valid capacity in kW.")
        return False

def display_boq(solar_boq, selected_module):
    boq_data = solar_boq.generate_boq(
        st.session_state.capacity,
        selected_module,
        st.session_state.manufacturer,
        st.session_state.selected_inverter or "Not specified",
        st.session_state.earthing_material_manufacturer or "Not specified",
        st.session_state.earthing_rod_manufacturer or "Not specified"
    )

    st.subheader("Initial Bill of Quantities (BOQ)")
    st.table(pd.DataFrame(boq_data))
    csv = solar_boq.generate_csv(boq_data)
    st.download_button(
        label="Download Initial BOQ as CSV",
        data=csv,
        file_name="initial_boq.csv",
        mime="text/csv",
    )

def handle_boq_modifications(solar_boq, cached_data):
    change_boq = st.radio("Do you want to change the BOQ?", ("No", "Yes"))

    # Create three columns for selectors
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_inverter = st.selectbox(
            "Inverter manufacturer:", 
            cached_data['inverters'],
            index=cached_data['inverters'].index(st.session_state.selected_inverter)
            if st.session_state.selected_inverter in cached_data['inverters'] else 0
        )

    with col2:
        new_earthing_material_manufacturer = st.selectbox(
            "Earthing Material manufacturer:", 
            cached_data['earthing_material_manufacturers'],
            index=cached_data['earthing_material_manufacturers'].index(st.session_state.earthing_material_manufacturer)
            if st.session_state.earthing_material_manufacturer in cached_data['earthing_material_manufacturers'] else 0
        )

    with col3:
        new_earthing_rod_manufacturer = st.selectbox(
            "Earthing Rod manufacturer:", 
            cached_data['earthing_rod_manufacturers'],
            index=cached_data['earthing_rod_manufacturers'].index(st.session_state.earthing_rod_manufacturer)
            if st.session_state.earthing_rod_manufacturer in cached_data['earthing_rod_manufacturers'] else 0
        )

    if change_boq == "Yes":
        with col1:
            new_manufacturer = st.selectbox(
                "Select PV manufacturer:", 
                cached_data['manufacturers'],
                index=cached_data['manufacturers'].index(st.session_state.manufacturer)
                if st.session_state.manufacturer in cached_data['manufacturers'] else 0
            )

        available_pv_modules = cached_data['pv_modules_by_manufacturer'][new_manufacturer]
        pv_module_options = [f"{module['Models']} - {module.get('Pmax', 'N/A')}W" for module in available_pv_modules]
        selected_pv_module_index = st.selectbox(
            "Select PV module:", 
            range(len(pv_module_options)),
            format_func=lambda x: pv_module_options[x]
        )
        new_selected_module = available_pv_modules[selected_pv_module_index]
    else:
        new_manufacturer = st.session_state.manufacturer
        new_selected_module = st.session_state.selected_module

    # Save selections to session state
    st.session_state.selected_inverter = new_inverter
    st.session_state.earthing_material_manufacturer = new_earthing_material_manufacturer
    st.session_state.earthing_rod_manufacturer = new_earthing_rod_manufacturer

    # Create a single row with two columns for the buttons
    button_col1, button_col2 = st.columns([3, 1])

    with button_col1:
        generate_button = st.button("Update BOQ")

    with button_col2:
        back_button = st.button("← Back to Initial BOQ")

    if generate_button:
        updated_boq_data = solar_boq.generate_boq(
            st.session_state.capacity,
            new_selected_module,
            new_manufacturer,
            new_inverter,
            new_earthing_material_manufacturer,
            new_earthing_rod_manufacturer
        )

        st.subheader("Updated Bill of Quantities (BOQ)")
        st.table(pd.DataFrame(updated_boq_data))
        csv = solar_boq.generate_csv(updated_boq_data)
        st.download_button(
            label="Download Updated BOQ as CSV",
            data=csv,
            file_name="updated_boq.csv",
            mime="text/csv",
        )
    
    if back_button:
        st.session_state.step = 'boq_generation'
        st.rerun()

async def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Please set it in the .env file.")
        st.stop()

    try:
        solar_boq = SolarBOQGenerator()
    except Exception as e:
        st.error(f"Error initializing Solar BOQ Generator: {str(e)}")
        st.stop()

    if 'initialized' not in st.session_state:
        reset_session_state()
        st.session_state.initialized = True
    
    if 'cached_data' not in st.session_state:
        with st.spinner("Loading initial data..."):
            st.session_state.cached_data = load_initial_data(solar_boq)

    st.title("Solar BOQ Generator")

    st.sidebar.title("History")
    for item in st.session_state.history:
        st.sidebar.write(item)

    if st.sidebar.button("Start Over"):
        reset_session_state()
        st.rerun()

    input_method = st.radio("Choose input method:", ["Text", "Voice"])

    try:
        if st.session_state.step == 'initial':
            if input_method == "Voice":
                if st.button("🎤 Start Recording"):
                    voice_input = get_voice_input()
                    if voice_input:
                        if voice_input not in st.session_state.history:
                            st.session_state.history.append(voice_input)
                        success = await process_user_input(voice_input, solar_boq)
                        if success:
                            st.rerun()
            else:
                # Use form to handle submission
                with st.form(key='input_form'):
                    user_input = st.text_input(
                        "Enter project details (Manufacturer, Capacity, Inverter, etc.):",
                        value=st.session_state.get('text_input', '')
                    )
                    submit_button = st.form_submit_button("Submit")
                
                # Process form submission
                if submit_button:
                    # Store the input for persistence
                    st.session_state.text_input = user_input
                    
                    if user_input.strip():
                        if user_input not in st.session_state.history:
                            st.session_state.history.append(user_input)
                        
                        # Show processing indicator
                        with st.spinner("Processing input..."):
                            # Process input and store results
                            extracted_info = solar_boq.extract_info_from_input(user_input)
                            
                            # Store the extracted info in session state
                            st.session_state.extracted_info = extracted_info
                            
                            if extracted_info['manufacturer'] and extracted_info['capacity']:
                                st.session_state.manufacturer = extracted_info['manufacturer']
                                st.session_state.capacity = extracted_info['capacity']
                                st.session_state.selected_inverter = extracted_info.get('inverter')
                                st.session_state.step = 'boq_generation'
                                st.rerun()
                            else:
                                # Show specific error messages
                                if not extracted_info['manufacturer']:
                                    st.error("Could not extract manufacturer information. Please provide a valid manufacturer from the following options:")
                                    st.write(", ".join(solar_boq.get_available_manufacturers()))
                                if not extracted_info['capacity']:
                                    st.error("Could not extract capacity information. Please provide a valid capacity in kW (e.g., 30kW, 40KW, 20kw).")

        elif st.session_state.step == 'boq_generation':
            pv_modules = st.session_state.cached_data['pv_modules_by_manufacturer'][st.session_state.manufacturer]
            best_module = get_best_pv_module(pv_modules)
            st.session_state.selected_module = best_module
            display_boq(solar_boq, best_module)

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Next"):
                    st.session_state.step = 'boq_modification'
                    st.rerun()

        elif st.session_state.step == 'boq_modification':
            handle_boq_modifications(solar_boq, st.session_state.cached_data)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug information:")
        st.write(f"Current step: {st.session_state.get('step', 'unknown')}")
        st.write(f"Last input: {st.session_state.get('text_input', 'none')}")
        if 'extracted_info' in st.session_state:
            st.write("Last extracted info:", st.session_state.extracted_info)

if __name__ == "__main__":
    asyncio.run(main())