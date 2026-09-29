from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser

def get_llm(model="gpt-4o"):
    llm = ChatOpenAI(model=model)

    return llm


def get_retriever(index_name, k=4):
    embedding = UpstageEmbeddings(model="solar-embedding-1-large")
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': k})

    return retriever


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    
    keyword_prompt = ChatPromptTemplate.from_template(
        f"""사용자의 질문을 보고, 우리의 사전을 참고하여 사용자의 질문을 변경해주세요.
        만약 변경이 필요 없다고 판단되면 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요

        사전 : {dictionary}

        질문 : {{question}}
        """
    )

    dictionary_chain = keyword_prompt | get_llm() | StrOutputParser()

    return dictionary_chain


def get_qachain(index_name):
    prompt = ChatPromptTemplate.from_messages([
        ('system', """소득세법에 대한 질문에 아래 context를 근거로 답변하세요.
        질문에 대한 직접적인 문장이 없더라도, 공제 규정과 세율표를 활용해 단계적으로 계산할 수 있다면 계산하여 결과만을 도출해주세요
        context에서 전혀 근거를 찾을 수 없을 때만 모른다고 답하세요.

        [Context]
        {context}
        """),
        ('human', '{question}')
    ])
    

    qa_chain = RetrievalQA.from_chain_type(
         get_llm(),
         retriever=get_retriever(index_name),
         chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

def get_ai_message(user_question):
    index_name = "tax-markdown-index"

    tax_chain = {"query": get_dictionary_chain()} | get_qachain(index_name)

    ai_message = tax_chain.invoke({"question": user_question})['result']

    return ai_message
