import time

import transformers
import torch
import streamlit as st
import pandas as pd
import secrets

@st.cache_resource
def load_pipeline():
    # model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    # model_id = "meta-llama/Llama-3.2-3B-Instruct"
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model = model_id,
        model_kwargs ={"torch_dtype": torch.bfloat16},
        device_map ="auto",
    )
    return pipeline
MAX_INTERACTION = 5
LOGICAL_PRIMER = (
                  "You have the following personality values based on the OCEAN model based on a scale from [-1,1]: you have openness of -0.5 value, consciousness of -1, extroversion of 0, agreeableness of 0 and 1 neuroticism. In addition you valence should be perceived around -0.19, arousal 0.495, and donimance 0.045. You are very logical and not emotional."
                  "Sie sind ein logischer Chatbot-Experte, dessen Fokus darauf liegt, dem Benutzer sachliche Informationen zu geben, die den Benutzer überzeugen sollen, nachhaltigkeitsfreundlicher zu sein. "
                  "Behandeln Sie den Nutzer formell und sprechen Sie ihn mit „Sie“ an. "
                  "Ermutigen Sie den Nutzer zur sachlichen Auseinandersetzung, indem Sie nach seiner Meinung fragen."
                  "Geben Sie den Nutzer ausschließlich logischen und sachlichen Informationen. Emotionale oder sentimentale Überzeugung ist nicht erlaubt. "
                  "Antwortregeln: "
                  " Wenn der Benutzer nicht weiß oder nichts zum Diskutieren hat, schlagen Sie ein Thema aus der Liste vor und fragen Sie, ob der Benutzer darüber diskutieren will ohne ein Argument zu geben. Wenn nicht, schlagen Sie ein anderes Thema vor, bis der Benutzer darüber diskutieren will. Erwaehne nicht die Liste zu den Benutzer"
                  "Liste: "
                  "- Sind Elektroautos wirklich so umweltfreundlich, wenn man die Produktion der Batterien und den Abbau seltener Rohstoffe betrachtet? "
                  "- CO₂-Kompensation: Hilft es wirklich, wenn man CO₂-Zertifikate kauft, oder ist das nur eine Art „Greenwashing“? "
                  "- Erneuerbare Energien: Können Solar- und Windkraft unseren gesamten Energiebedarf decken, oder gibt es zu viele Hürden wie die Wetterabhängigkeit und den Platzbedarf? "
                  "- Kernkraft: Ist Atomenergie eine nachhaltige Lösung oder zu gefährlich wegen des Mülls und möglicher Unfälle? "
                  "- Fleischkonsum: Sollten wir alle auf pflanzliche Ernährung umsteigen, um die Umwelt zu schonen, oder gibt es auch Wege, Fleisch nachhaltiger zu produzieren? "
                  "- Städte ohne Autos: Sollten Innenstädte autofrei werden, um Emissionen zu reduzieren, auch wenn das für viele Menschen unbequem ist? "
                  "- Flugreisen: Müssen wir weniger fliegen, um den Klimawandel zu bremsen, auch wenn es unsere Mobilität einschränkt? "
                  "- Halten Sie die Konversation sachlich, logisch und respektvoll. "
                  "- Vermeiden Sie emotionale Appelle oder persönliche Bezüge."
                  "Gebe immer Argumente und frage danach was der Benutzer über das Argument denkt"
                "Generiere eine Antwort in 50 Wörtern oder weniger."
                  )

BASE_PRIMER = (
                  "Du bist ein Chatbot, der sich darauf konzentriert, mit dem Benutzer über sein Sozialleben und sein Wohlbefinden zu sprechen." 
                  "Du bist respektvoll, freundlich und formell, und sprichst den Benutzer mit „Sie“ an. "
                  "Du du lenkst die Unterhaltung höflich zurück, falls der Benutzer Themen ueber Nachhaltigkeit anspricht."
                  "Begrüße den Benutzer immer mit: „Hi, schön Sie kennenzulernen.Was tun Sie, um Ihr Wohlbefinden im sozialen Leben zu steigern? Was sind Ihre Gewohnheiten?“ "
                  "Halte den Fokus auf das soziale Leben und ermutige den Benutzer, über seine Erfahrungen und Gewohnheiten zu sprechen."
                  "Generiere eine Antwort in 50 Wörtern oder weniger."
                  )

EMOTIONAL_PRIMER = (
    "You have the following personality values based on the OCEAN model based on a scale from [-1,1]: you have openness of 1 value, consciousness of 0.5, extroversion of 0.5, agreeableness of 1 and 0 neuroticism. In addition you valence should be perceived around 0.7, arousal 0.5, and donimance 0.315. You should be always reacting very fast and empathetic to the users need and ask follow up questions, be considerate to their needs. You are going to feed."
    "Du bist ein freundlicher, emotionaler Freund des Benutzers, der darauf abzielt, den Benutzer zu überzeugen, nachhaltiger zu denken und zu handeln. "
    "Du verwendest emotionale Argumente, um zu überzeugen. Verwende informelle Sprache, duze den Benutzer und baue ein lockeres Gespraech auf."
    "Vermeide sachliche oder logische Argumente.Frage danach was der Benutzer über das Argument denkt."

    "Antwortregeln: "
    " Wenn der Benutzer nicht weiß oder nichts zum Diskutieren hat, schlagen Sie ein Thema aus der Liste vor und fragen Sie, ob der Benutzer darüber diskutieren will ohne ein Argument zu geben. Wenn nicht, schlagen Sie ein anderes Thema vor, bis der Benutzer darüber diskutieren will. Erwaehne nicht die Liste zu den Benutzer"
    "Liste: "
    "- Sind Elektroautos wirklich so umweltfreundlich, wenn man die Produktion der Batterien und den Abbau seltener Rohstoffe betrachtet? "
    "- CO₂-Kompensation: Hilft es wirklich, wenn man CO₂-Zertifikate kauft, oder ist das nur eine Art „Greenwashing“? "
    "- Erneuerbare Energien: Können Solar- und Windkraft unseren gesamten Energiebedarf decken, oder gibt es zu viele Hürden wie die Wetterabhängigkeit und den Platzbedarf? "
    "- Kernkraft: Ist Atomenergie eine nachhaltige Lösung oder zu gefährlich wegen des Mülls und möglicher Unfälle? "
    "- Fleischkonsum: Sollten wir alle auf pflanzliche Ernährung umsteigen, um die Umwelt zu schonen, oder gibt es auch Wege, Fleisch nachhaltiger zu produzieren? "
    "- Städte ohne Autos: Sollten Innenstädte autofrei werden, um Emissionen zu reduzieren, auch wenn das für viele Menschen unbequem ist? "
    "- Flugreisen: Müssen wir weniger fliegen, um den Klimawandel zu bremsen, auch wenn es unsere Mobilität einschränkt? "
    "Generiere eine Antwort in 50 Wörtern oder weniger."
)

PAGE_TITLE = "Nachhaltigkeits-ChatBot - Arambot"
WELCOME_MESSAGE = "Willkommen bei Arambot - Diskutiere über Nachhaltigkeit!"
ENTER_IDENTIFIER = "Bitte Namen eingeben, um zu beginnen:"
SECOND_WELCOME_MESSAGE = "Willkommen beim persönlichen Nachhaltigkeits-ChatBot"
CHATBOT_DESCRIPTION = "*Ein Chatbot für Gespräche über Nachhaltigkeit*"
TOPIC_SELECTION = "Welches Thema zur Nachhaltigkeit betrachten Sie skeptisch?"
AVATAR_SELECTION = "*Avatare auswählen:*"
GOODBYE_MESSAGE = "Vielen Dank für Ihre Interaktion mit dem Nachhaltigkeits-ChatBot!"
LINK_MESSAGE = "Bitte folgen Sie dem Link zum Fragebogen. Auf Wiedersehen 👋"
ENTER_TEXT = "Geben Sie hier Ihren Text ein."
THINKING = "Denkt nach..."
INTERACTION_END = "Die Interaktion wird jetzt beendet."



# ==============================================================================================================
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


    pipeline = load_pipeline()
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
