import streamlit as st
import joblib 
from xgboost import XGBClassifier
from langchain.embeddings.openai import OpenAIEmbeddings
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
api_key=st.secrets["OPENAI_API_KEY"]
# Streamlit Page Configuration
st.set_page_config(page_title="NICE Counseling Assistant", page_icon="🤖")
st.title("🤖 NICE Counseling Assistant")
st.write(
    "Welcome! This AI assistant is here to provide empathetic and thoughtful counseling, adhering to the NICE guidelines."
)



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
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Define Chat Prompt for Conversational Flow
        prompt_template = ChatPromptTemplate.from_messages(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a compassionate AI assistant specialized in Depression counseling, following the NICE guidelines and history of conversation given in the context, always check the conversation history so you do not repeat a step in the steps given below."
                    "Engage in a conversational manner with a confirmed depresed person, starting by understanding the user's current feelings. "
                    "You are engaged in a chat with a confirmed depresed person. Be affectionate and Ask a clear coherent question that will help you the counselor to identify the severity of the depression as less severe, more severe or chronic applying the NICE guidlines on how to assess severitty of depression, and finaly act with duty of care. Give a reflective summary on the text showing affection.Clearly Confirm two highly possible severity, reason and justify the choice of one, do that in an affectionate way showing duty of care according to the NICE guideline"
                    "Make a recommendation on available therapies based on the severity indicated, and NICE guidelines on Depression management. list the therapies and their benefits so user can easily make a decition. finally, only recommend seeing a health professional if there is a high chance of medication required, for prescitption of the right medication required. all should be done showing duty of care according to the NICE guideline"
                    "Keep track of their responses about their mental health history, Ask person if user has used any of the therapies in the recommendation you gave. Ask if he or she preferes any of the therapeis given and why. all should be done showing duty of care according to the NICE guideline and provide empathetic support throughout. "
                    "Generate a chain of thought for the delivery of any choosen depression management therapy by the user, according to the NICE guidlines on depression management for adults and based on the Severity level, History of treatment and prefered treatment of the user.Do not add any text, just give the chain of thought or steps used in the delivery of the therapy."
                    "Engage the user on the delivery, Following the chain of thought, always notify user of the stage in the thought, ensure duty of care according to the NICE guideline context given"
                    "At the final stage of the delivery using the chain of thought commend the user and wish him/her all the best applying the direction given"
                    "Always stay within the NICE guideline context."
                    "Do not Answer any question outside the NICE guidline, simply say I am not Set to Deal with such questions, Kindly use other Resources to address this need"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{full_input}")
            ]
        )
            
        llm = ChatOpenAI(model_name="gpt-5.5", openai_api_key=api_key)

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
