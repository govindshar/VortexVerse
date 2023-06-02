from gpt_index import SimpleDirectoryReader ,GPTListIndex,GPTSimpleVectorIndex,LLMPredictor,PromptHelper
from langchain import OpenAI
import sys
import os

os.environ["OPENAI_API_KEY"]="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"

def createVectorIndex(path):
    max_input=4096
    tokens=256
    chunk_size=400
    max_chunk_overlap=20
    prompt_helper=PromptHelper(max_input,tokens,max_chunk_overlap,chunk_size_limit=chunk_size )
    #define LLM
    llMPredictor=LLMPredictor(llm=OpenAI(temperature=0.3,model_name="text-davinci-003" ,max_tokens=tokens))
    #laod data
    docs=SimpleDirectoryReader(path).load_data()
    #create vector index
    vectorIndex=GPTSimpleVectorIndex(documents=docs,llm_predictor=llMPredictor,prompt_helper=prompt_helper)
    vectorIndex.save_to_disk("vectorIndex1.json")
    return vectorIndex

vectorIndex=createVectorIndex("k")



#def anserMe(vectorIndex):
#    vIndex =GPTSimpleVectorIndex.load_from_disk(vectorIndex)
#    while True:
#        prompt=input("Please ask:")
#        response=vIndex.query(prompt,response_mode="compact")
#        print(f"Response: {response} \n")

#anserMe("vectorIndex.json")
