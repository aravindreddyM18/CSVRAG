import pandas as pd
import streamlit as st
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama
import loadcsv
#import modin.pandas as pd
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain.agents import AgentExecutor
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import PromptTemplate


st.set_page_config(page_title="Chat bot",initial_sidebar_state='expanded')
st.title("KBE CSV chatbot")
python_repl = PythonREPL()

    
with st.sidebar:
    uploaddocs=st.file_uploader(label="Upload file",type=['csv','xlsx'])

if uploaddocs:
    filename=uploaddocs.name
    dff =loadcsv.datload(uploaddocs,filename)
    pandasoutputparser=PandasDataFrameOutputParser(dataframe=dff)
    

if  "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

for meassage in st.session_state.chat_history:
    with st.chat_message(meassage["role"]):
        st.markdown(meassage["content"])

outputparser=CommaSeparatedListOutputParser()


user_input=st.text_input(label="",placeholder="Ask here...")


prompttemplete=PromptTemplate(template=f""" if user greets you greet them back. Read provided data and while giving output verify it with provided data and answer the question {input}
                              """)
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history .append({"role":"user","content":user_input}) 
    llm=ChatOllama(model = "mistral",  temperature = 0) 
    pandas_df_agent=create_pandas_dataframe_agent(llm,dff,verbose=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,allow_dangerous_code=True,handle_parsing_errors=True,prompt=prompttemplete,return_intermediate_steps=True,include_df_in_prompt=True,number_of_head_rows=5,max_iterations=7) # can add prompt and tools
   # meassage=[{"role":"system","content":"you are helpful assistent"}, *st.session_state.chat_history]
   # You can create the tool to pass to an agent
   # repl_tool = Tool(
   # name="python_repl",
   # description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
   # func=python_repl.run,
    #)
    #agentexe=AgentExecutor(agent=pandas_df_agent,tools=pandas_df_agent.tools,handle_parsing_errors=True,verbose=True) 
    try:

        response=pandas_df_agent.invoke({"input":user_input},handle_parsing_errors=True)# commented to add agent executor and get response from it
        #response=agentexe.invoke({"input":user_input})
   # outputparser=CommaSeparatedListOutputParser()
    #outputparser.invoke()
        #chain = pandas_df_agent| pandasoutputparser
        #response=chain.invoke({"input":user_input})

        bot_output=response["output"]
        st.markdown(response)
    except Exception as e:
        bot_output="Sorry... unable to process and generate output!"
        print(f"ERROR is: {e}")
        st.markdown(response)
    #st.write(bot_output)
    

    st.session_state.chat_history.append({"role":"ai","content":bot_output})
    with st.chat_message('ai'):
        st.markdown(bot_output)
  

    
    






