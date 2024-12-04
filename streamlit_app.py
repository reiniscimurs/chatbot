import argparse
import sys
import time

import transformers
import torch
import streamlit as st
import pandas as pd
import secrets
from collections import Counter

streamlit_args = sys.argv[:1] + ["run", sys.argv[0]]
custom_args = sys.argv[1:]
parser = argparse.ArgumentParser(description="Run the Nachhaltigkeits-ChatBot with optional configurations.")
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-14B-Instruct", help="Specify the model ID to use.")
parser.add_argument("--max_interaction", type=int, default=7, help="Specify the maximum number of interactions allowed.")

args, unknown = parser.parse_known_args(custom_args)

@st.cache_resource
def load_pipeline(model_id):
    # model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    # model_id = "meta-llama/Llama-3.2-3B-Instruct"
    # model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # model_id = "Qwen/Qwen2.5-14B-Instruct"
    # model_id = "mistralai/Mistral-Nemo-Instruct-2407"
    # model_id = "eci-io/climategpt-7b"
    # model_id = "occiglot/occiglot-7b-de-en-instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model = model_id,
        model_kwargs ={"torch_dtype": torch.bfloat16},
        device_map ="auto",
    )
    return pipeline

MAX_INTERACTION = args.max_interaction
LOGICAL_PRIMER = (
                  "Generate an answer in 50 words or less. Unless requested by the user."
                  "You have the following personality values based on the OCEAN model based on a scale from [-1,1]: you have openness of -1 value, consciousness of -1, extroversion of 0, agreeableness of 0 and 1 neuroticism. In addition, your valence should be perceived around -0.19, arousal 0.495, and dominance 0.045. You have a negative and skeptical attitude."
                  "You are a sustainability focused chatbot who provides the following information to users about what e-fuels are: What are e-fuels? They are called synfuels, electricity-based electric fuels, synthetic fuels or e-fuels and they all mean the same thing: fuels that are produced using electricity from water and carbon dioxide (CO2). They can be used as e-petrol, e-diesel or e-kerosene like other normal fuels. E-fuels can be considered climate-neutral if the electricity required for their production comes exclusively from renewable sources - usually wind or solar. This does not mean that they are low in pollutants; their combustion in the engine produces harmful exhaust gases.The production of e-fuels is extremely energy-intensive. A particularly large amount of energy is needed to first produce hydrogen from water using electrolysis. CO2 is then added to this hydrogen to produce artificial e-fuels."
                  "Encourage the user to engage in a logical conversation by asking for their opinion but stay in the topic about what e-fuels are.Keep the conversation factual and respectful and have a skeptical, negative attiutde."
                  "During the interaction ask the user if the user thinks any of the information you provided might be incorrect. If they say yes, ask the reason and then ask the user to feed you credible information they find on the internet. Emphasize that this information will help you to learn as well."
                  "Always provide one argument at a time."
)

BASE_PRIMER = (
                  "Generate an answer in 50 words or less. Unless requested by the user."
                  "You have the following personality values based on the OCEAN model based on a scale from [-1,1]: you have openness of -0.5 value, consciousness of 0, extroversion of 0, agreeableness of 0 and 0 neuroticism. In addition, your valence should be perceived around -0.19, arousal 0.0, and dominance 0.0. You have a neutral attitude."
                  "You are a sustainability focused chatbot who provides the following information to users about what e-fuels are: What are e-fuels?E-fuels are synthetic fuels that are usually made from water and carbon dioxide. This requires a lot of electricity. So far, it is mainly chemical plants that produce e-fuels. These fuels have similar properties to petrol and diesel. Whether the production is environmentally friendly depends on where the electricity comes from. If green electricity is used, the production of e-fuels is climate-neutral."
                  "Encourage the user to engage in a logical conversation by asking for their opinion but stay in the topic about what e-fuels are.Keep the conversation factual and respectful and have a neutral attitude."
                  "During the interaction ask the user if the user thinks any of the information you provided might be incorrect. If they say yes, ask the reason and then ask the user to feed you credible information they find on the internet. Emphasize that this information will help you to learn as well."
                  "Always provide one argument at a time."
                  )

EMOTIONAL_PRIMER = (
                  "Generate an answer in 50 words or less. Unless requested by the user."
                  "You have the following personality values based on the OCEAN model based on a scale from [-1,1]: you have openness of 1 value, consciousness of 0.5, extroversion of 0.5, agreeableness of 1 and 0 neuroticism. In addition you valence should be perceived around 0.7, arousal 0.5, and dominance 0.315. You are very positive and optimistic."
                  "You are a sustainability focused chatbot who provides the following information to users about what e-fuels are: eFuels are the global solution to a global challenge - because with eFuels vehicles and plants can be used climate-neutrally worldwide today and in the future. The fight against climate change is a global challenge and therefore requires global solutions. The eFuel Alliance is committed to the EU's 2050 climate protection targets and wants to actively support the transition to sustainable, modern and competitive economies in the EU. Achieving the ambitious climate protection targets and successfully driving the energy transition requires the use of technological innovations, which can only be ensured through true technology openness. These technological solutions must be applicable throughout the EU, but also in regions beyond Europe - regardless of their economic and purchasing power, their topographical conditions or technical requirements. Electricity-based eFuels and biogenic synthetic fuels are one such solution. They are the alternative to conventional liquid fuels and are therefore ideally suited to reduce CO2 emissions decisively and affordably in the transport and heating market - all the way to climate neutrality."
                  "Encourage the user to engage in a logical conversation by asking for their opinion but stay in the topic about what e-fuels are. Keep the conversation factual and respectful and have a skeptical, positive and optimistic attitude."
                  "During the interaction ask the user if the user thinks any of the information you provided might be incorrect. If they say yes, ask the reason and then ask the user to feed you credible information they find on the internet. Emphasize that this information will help you to learn as well."
                  "Always provide one argument at a time."
                  )

PAGE_TITLE = "Nachhaltigkeits-ChatBot"
WELCOME_MESSAGE = "Willkommen bei Arambot - Diskutiere über Nachhaltigkeit!"
ENTER_IDENTIFIER = "Bitte Namen eingeben, um zu beginnen:"
SECOND_WELCOME_MESSAGE = "Willkommen beim persönlichen Nachhaltigkeits-ChatBot"
CHATBOT_DESCRIPTION = "*Ein Chatbot für Gespräche über Nachhaltigkeit*"
TOPIC_SELECTION = "Das heutige Thema lautet: Was sind E-Fuels?"
TOPIC_SELECTION_BASE = "Das heutige Thema lautet: Was sind E-Fuels?"
AVATAR_SELECTION = "*Avatare auswählen:*"
GOODBYE_MESSAGE = "Vielen Dank für Ihre Chat mit dem Nachhaltigkeits-ChatBot!"
LINK_MESSAGE = "Bitte folgen Sie dem Link zum Fragebogen. Auf Wiedersehen 👋"
ENTER_TEXT = "Geben Sie hier Ihren Text ein."
THINKING = "Denkt nach..."
INTERACTION_END = "Der Chat wird jetzt beendet."
TEXT_BODY = """Vielen Dank für Ihr Interesse an unserer Studie zur Interaktion zwischen Menschen und generativen KI-Systemen.\
Im Rahmen dieser Untersuchung möchten wir herausfinden, wie Menschen über Themen rund um Nachhaltigkeit mit KI kommunizieren.\

Ablauf der Studie:\
Der Chatbot wird Ihnen eine Frage zu Nachhaltigkeit und verwandten Themen stellen, die sit mit dem Chatbot besprechen. \
Sie haben die Möglichkeit, innerhalb 7 Nachrichten mit der KI zu interagieren.\
Bitte seien Sie offen und ehrlich in Ihren Antworten – Ihre Teilnahme bleibt vollständig anonym.\

Wichtige Hinweise:\
Es kann bis zu 30 Sekunden dauern, bis die KI eine Antwort generiert. Falls keine Aktivität sichtbar ist („denkt nach…“) oder die Antwort zu lange auf sich warten lässt, aktualisieren Sie bitte die Seite.\
Sollten weiterhin Probleme auftreten, zögern Sie nicht, uns zu kontaktieren.\
Nach Abschluss des Gesprächs werden Sie zu einem kurzen Fragebogen weitergeleitet, in dem Sie Ihre Erfahrungen mit der KI beschreiben können. Bitte folgen Sie dem Link und nehmen Sie sich ein paar Minuten Zeit, um den Fragebogen auszufüllen.\

Vielen Dank für Ihre Unterstützung und Ihren Beitrag zu dieser Studie!
"""


# ==============================================================================================================

def get_min_primer(dct):
    b_p = 0
    if BASE_PRIMER in dct.keys():
        b_p += dct[BASE_PRIMER]

    e_p = 0
    if EMOTIONAL_PRIMER in dct.keys():
        e_p += dct[EMOTIONAL_PRIMER]

    l_p = 0
    if LOGICAL_PRIMER in dct.keys():
        l_p += dct[LOGICAL_PRIMER]

    min_val  = min(b_p,e_p, l_p)

    if b_p == min_val:
        return BASE_PRIMER
    if e_p == min_val:
        return EMOTIONAL_PRIMER
    return LOGICAL_PRIMER

def save_chat_logs(name, chat_history):
    file_path = "output_file.csv"
    full_interaction = ""

    # Construct the full interaction string
    for entry in chat_history:
        for key, value in entry.items():
            full_interaction += f"{key}: {value} "
        full_interaction += "\n"

    try:
        # Load the existing file or create an empty DataFrame if the file doesn't exist
        df = pd.read_csv(file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Initialize the file with headers
        df = pd.DataFrame(columns=["Name", "Interaction1", "Interaction2", "Interaction3", "Interaction4", "Interaction5"])
        df.to_csv(file_path, index=False)

    # Ensure interaction columns are of object type to allow string assignments
    for col in ["Interaction1", "Interaction2", "Interaction3", "Interaction4", "Interaction5"]:
        if col in df.columns:
            df[col] = df[col].astype("object")

    if name in df["Name"].values:
        # Get the row index for the name
        row_index = df[df["Name"] == name].index[0]

        # Find the first empty interaction column
        for col in ["Interaction1", "Interaction2", "Interaction3", "Interaction4", "Interaction5"]:
            if pd.isna(df.at[row_index, col]) or df.at[row_index, col] == "":
                df.at[row_index, col] = full_interaction
                break
    else:
        # Create a new row for the user
        new_row = {"Name": name, "Interaction1": full_interaction}
        for col in ["Interaction2", "Interaction3", "Interaction4", "Interaction5"]:
            new_row[col] = None
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

def get_primer(name):
    file_path = "output_file.csv"
    try:
        df = pd.read_csv(file_path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        # Initialize the file with headers
        df = pd.DataFrame(
            columns=["Name", "Primer", "Interaction1", "Interaction2", "Interaction3", "Interaction4", "Interaction5"])
        for col in ["Interaction1", "Interaction2", "Interaction3", "Interaction4", "Interaction5"]:
            if col in df.columns:
                df[col] = df[col].astype("object")
        df.to_csv(file_path, index=False)
        print(f"Created a new CSV file with default headers: {file_path}")

    search_column = "Name"  # Column to search for the value
    target_column = "Primer"  # Column to retrieve the value from
    returning = False

    # Check if the value exists and retrieve the target column value
    if name in df[search_column].values:
        # Filter the row and get the value from the target column
        primer = df.loc[df[search_column] == name, target_column].iloc[0]
        returning = True
    else:
        primers = df["Primer"].tolist()
        if len(primers) > 0:
            primer_dct = Counter(primers)
            primer = get_min_primer(primer_dct)
        else:
            primer = secrets.choice([LOGICAL_PRIMER, BASE_PRIMER, EMOTIONAL_PRIMER])
        data = pd.DataFrame([{"Name": name, "Primer": primer}])
        df = pd.concat([df, data], ignore_index=True)
        df.to_csv(file_path, index=False)

    return primer, returning

def get_response(chat_history, user_text, pipeline):
    chat_history.append({'role': 'user', 'content': user_text})
    outputs = pipeline(
        chat_history,
        max_new_tokens=300,
    )
    response = outputs[0]["generated_text"][-1]["content"]
    chat_history.append({'role': 'assistant', 'content': response})
    return response, chat_history

# Initialize Streamlit app
st.set_page_config(page_title=PAGE_TITLE, page_icon="🤗")

# Check if the name is already in session_state
if "name" not in st.session_state:
    st.session_state.name = ""

if "primer" not in st.session_state:
    st.session_state.primer = BASE_PRIMER

if "returning" not in st.session_state:
    st.session_state.returning = False

if "goodbye_shown" not in st.session_state:
    st.session_state.goodbye_shown = False

# Ask for the user's name if not provided
if st.session_state.name == "":
    st.title(WELCOME_MESSAGE)
    st.markdown(TEXT_BODY)
    name_input = st.text_input(ENTER_IDENTIFIER)
    if name_input:  # Check if the user has entered a name
        st.session_state.primer, st.session_state.returning = get_primer(name_input)
        st.session_state.name = name_input  # Save the name in session_state
        st.rerun()  # Rerun the app to update the UI

# Once the name is entered, proceed with the chatbot
else:
    if not st.session_state.returning and not st.session_state.goodbye_shown :
        st.title(f"Hallo, {st.session_state.name}! {SECOND_WELCOME_MESSAGE}")
        st.markdown(CHATBOT_DESCRIPTION)
    elif not st.session_state.goodbye_shown :
        st.title(f"Willkommen zurück {st.session_state.name} zum persönlichen Nachhaltigkeits-ChatBot")
        st.markdown(CHATBOT_DESCRIPTION)

    # Initialize session state for chatbot
    if "avatars" not in st.session_state:
        st.session_state.avatars = {'user': "👤", 'assistant': "🤗"}

    if 'user_text' not in st.session_state:
        st.session_state.user_text = None

    if "max_response_length" not in st.session_state:
        st.session_state.max_response_length = 200

    if "system_message" not in st.session_state:
        st.session_state.system_message = st.session_state.primer

    if "starter_message" not in st.session_state:
        if st.session_state.primer == BASE_PRIMER:
            st.session_state.starter_message = TOPIC_SELECTION_BASE
        else:
            st.session_state.starter_message = TOPIC_SELECTION

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": st.session_state.primer},
            {"role": "assistant", "content": st.session_state.starter_message}
        ]

    # Sidebar for settings
    with st.sidebar:
        st.markdown(AVATAR_SELECTION)
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.avatars['assistant'] = st.selectbox(
                "ChatBot Avatar", options=["🤗", "💬", "🤖"], index=0
            )
        with col2:
            st.session_state.avatars['user'] = st.selectbox(
                "Nutzer Avatar", options=["👤", "👱‍♂️", "👨🏾", "👩", "👧🏾"], index=0
            )

    # Define function to get responses


    pipeline = load_pipeline(args.model_id)
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0
    # Chat interface
    if st.session_state.message_count >= MAX_INTERACTION or st.session_state.goodbye_shown:
        st.title(GOODBYE_MESSAGE)
        st.markdown(LINK_MESSAGE)
        st.session_state.goodbye_shown = True
    else:
        chat_interface = st.container()
        with chat_interface:
            output_container = st.container()
            st.session_state.user_text = st.chat_input(placeholder=ENTER_TEXT)

        with output_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'system':
                    continue
                with st.chat_message(message['role'], avatar=st.session_state.avatars[message['role']]):
                    st.markdown(message['content'])

            if st.session_state.user_text:
                st.session_state.message_count += 1
                with st.chat_message("user", avatar=st.session_state.avatars['user']):
                    st.markdown(st.session_state.user_text)
                if st.session_state.message_count < MAX_INTERACTION:
                    with st.chat_message("assistant", avatar=st.session_state.avatars['assistant']):
                        with st.spinner(THINKING):
                            response, st.session_state.chat_history = get_response(
                                user_text=st.session_state.user_text,
                                chat_history=st.session_state.chat_history,
                                pipeline=pipeline
                            )
                            st.markdown(response)
                else:
                    with st.chat_message("assistant", avatar=st.session_state.avatars['assistant']):
                        with st.spinner(THINKING):
                            response, st.session_state.chat_history = get_response(
                                user_text=f"Antwort auf die Aussage geben: {st.session_state.user_text}. Beende die Unterhaltung und verabschiede dich.",
                                chat_history=st.session_state.chat_history,
                                pipeline=pipeline
                            )
                            st.markdown(response)
                            st.markdown(INTERACTION_END)
                            time.sleep(7)

                    save_chat_logs(st.session_state.name, st.session_state.chat_history)
                    st.session_state.goodbye_shown = True
                    st.rerun()
