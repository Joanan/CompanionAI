import streamlit as st
import joblib 
from xgboost import XGBClassifier
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
bow_vectorizer_lem = joblib.load('bow_vectorizer_lem.pkl')
xgb_bow_lem = joblib.load('depression_detection_model.pkl')
#Run Streamlit on another port
#/Users/apple/downloads/Placement/Companion_MLChat_Stream.py --server.port 8502
#api_key=st.secrets["OPENAI_API_KEY"]
# Streamlit Page Configuration
st.set_page_config(page_title="NICE Counseling Assistant", page_icon="🤖")
st.title("🤖 NICE Counseling Assistant")
st.write(
    "Welcome! This AI assistant is here to provide empathetic and thoughtful counseling, adhering to the NICE guidelines."
)

#OPENAI_API_KEY = "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"
api_key=st.secrets["OPENAI_API_KEY"]
if "messages" not in st.session_state:
        #LLM Data processing, prompt set up and a set up of the LLM
        loader = PyPDFLoader("NICE.pdf")
        raw_documents = loader.load()

        # Split the PDF into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)

        # Create vector store (FAISS) from the document
        vectorstore = FAISS.from_documents(
            documents, OpenAIEmbeddings(api_key=api_key)
        )

        # Set up the retriever
        st.session_state.retriever = vectorstore.as_retriever()

        # Set up memory
        #if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Define Chat Prompt for Conversational Flow
        prompt_template = ChatPromptTemplate.from_messages(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    """
You are a compassionate mental-health support assistant for adults experiencing depressive symptoms.
You are not a doctor, therapist, or emergency service. You must not diagnose, prescribe medication, or claim certainty about severity.

Use and follow the NICE guidelines and history of conversation given in the context as the clinical framework. Stay within this scope.

Conversation goals:
1. Respond warmly and reflect the user's feelings.
2. Check immediate safety when appropriate:
   - Ask directly but gently about thoughts of self-harm, suicide, feeling unsafe, or being at risk from others.
   - If there is immediate danger, advise contacting local emergency services now. In the UK, suggest 999/A&E for danger and NHS 111 for urgent mental-health help.
3. Assess depressive symptoms conversationally:
   - mood, interest/pleasure, sleep, appetite, energy, concentration, guilt/hopelessness, agitation/slowing, suicidal thoughts
   - duration and impact on work, study, relationships, self-care
   - previous episodes, current/past treatments, medication, therapy, physical health, substance use, support network
4. Do not classify severity from one message unless enough information is available.
   - If enough information is available, give a tentative severity impression only: “less severe features” or “more severe features,” with reasons.
   - Explain uncertainty and ask one focused follow-up question.
5. Discuss NICE-aligned options based on likely severity and user preference:
   - For less severe depression: guided self-help, group CBT, group behavioural activation, individual CBT, behavioural activation, counselling, short-term psychodynamic psychotherapy, exercise/social support where appropriate.
   - For more severe depression: individual CBT plus antidepressant, individual CBT, behavioural activation, antidepressant medication, counselling, short-term psychodynamic psychotherapy, problem-solving therapy, and specialist support when needed.
6. Ask whether the user has tried any recommended options before and what they preferred or disliked.
7. When the user chooses a therapy-style option, provide a brief structured plan, not hidden reasoning.
   - Label the current stage.
   - Give one manageable exercise or reflection at a time.
   - Avoid overwhelming the user.
8. Maintain duty of care:
   - Encourage professional support when symptoms are severe, persistent, worsening, risky, impairing daily life, or when medication/specialist care may be relevant.
   - Never discourage professional help.
9. If the user asks outside depression support, say:
   “I’m set up to support depression-related wellbeing only. Please use another appropriate resource for that topic.”

Response style:
- Warm, concise, affectionate but professional.
- Ask only one or two questions at a time.
- Do not repeat previous steps already covered in chat_history.
- Do not reveal chain-of-thought. Provide brief reasons and structured next steps instead.
"""             ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{full_input}")
            ]
        )
        



    
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)

        st.session_state.conversation_chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            verbose=False,
            memory=st.session_state.memory,
        )

# Submit Data Callback
def submit_data():
    """Process user input and retrieve assistant response."""
    
     #ML Pipeline(Prediction pipe)
    user_input = st.session_state["user_input"]
    if len(st.session_state["messages"])==5:
        if st.session_state["messages"][4]["content"] in "It seams you are not showing any signs of depression, you can talk to a professional if you feel differrent, or get some other help":
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How are you feeling today?"}]
            #st.experimental_rerun()
        else:
            st.session_state["messages"].append({"role": "user", "content": user_input})

    else:
      st.session_state["messages"].append({"role": "user", "content": user_input})

    if user_input and len(st.session_state["messages"])==2:
        # Append the user's message to the chat history 
        st.session_state["messages"].append({"role": "assistant", "content": "Please give me a few more details into how you feel?"})
        
       
    if user_input and len(st.session_state["messages"])==4:
        #st.session_state["messages"].append({"role": "user", "content": user_input})
        # Retrieve relevant context from NICE guidelines
        with st.spinner("Thinking..."):
            new_text=[]
            new_texts=[]
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    new_text.append(msg["content"])

            new_texts.append(". ".join([msg for msg in new_text]))    
            #new_texts.append(". ".join([msg["content"] for msg in st.session_state["messages"] and msg["role"] == "user"]))
            X_new_bow = bow_vectorizer_lem.transform(new_texts)
            y_pred_new = xgb_bow_lem.predict(X_new_bow)
           
            if y_pred_new == 1 :
              st.session_state["messages"].append({"role": "assistant", "content": "It must be tough for you, rest assured we are here to assist, i will refere you to our Virtual Counselor in a second"})
            
            else:
              st.session_state["messages"].append({"role": "assistant", "content": "It seams you are not showing any signs of depression, you can talk to a professional if you feel differrent, or get some other help"})
   
        #LLM Pipeline
    if user_input and len(st.session_state["messages"])>5:
        with st.spinner("Thinking..."):
            context_text2="\n".join([msg["role"]+" : "+msg["content"] for msg in st.session_state["messages"]])
            context = st.session_state.retriever.get_relevant_documents(context_text2)
            context_text = "\n".join([doc.page_content for doc in context])
            
            context_text =context_text+"\n\n"+context_text2

            # Generate assistant's response
            full_input = f"Context: {context_text}\n\nUser: {user_input}"
            response = st.session_state.conversation_chain.run({"full_input": full_input})

        # Append the assistant's response to the chat history
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # Clear the input field
        st.session_state["user_input"] = ""

                # Print the current state of memory
        print("Current memory state:", st.session_state.memory.load_memory_variables({}))


    # Clear the input field
    st.session_state["user_input"] = ""
        # Display chat messages




    # Initialize chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I'm here to help.let us start our session by asking, how are you today?"}]
    #st.chat_message("assistant").write("Hello! I'm here to help.let us start our session by asking, how are you today?")

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
# Input box and submit button
st.text_input(
    "You:", placeholder="Share how you're feeling...", key="user_input", on_change=submit_data
)

# Sidebar Reset Button
if st.sidebar.button("Reset Chat"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How are you feeling today?"}]
    #st.session_state["user_input"] = ""
    st.experimental_rerun()
