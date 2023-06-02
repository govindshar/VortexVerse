from gpt_index import  GPTSimpleVectorIndex
import os

os.environ["OPENAI_API_KEY"]="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"
#os.environ["OPENAI_API_KEY"]="9ca7dc1c9e694e8985872fea56c26a78"
def anserMe(vectorIndex):
    vIndex =GPTSimpleVectorIndex.load_from_disk(vectorIndex)
    while True:
        prompt=input("Please ask:")
        response=vIndex.query(prompt,response_mode="compact")
        print(f"Response: {response} \n")

anserMe("vectorIndex1.json")