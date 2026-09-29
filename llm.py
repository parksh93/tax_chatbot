from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import uuid

store = {}

# -> BaseChatMessageHistory : BaseChatMessageHistory 타입을 반환한다는 명시적 의미
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]

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


def get_rag_chain(index_name):
    retriever = get_retriever(index_name)
    llm = get_llm()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick("answer")

    return conversational_rag_chain

def get_ai_response(user_question):
    index_name = "tax-markdown-index"

    tax_chain = {"input": get_dictionary_chain()} | get_rag_chain(index_name)

    ai_response = tax_chain.stream(
        {"question": user_question},
        config={'configurable': {'session_id': str(uuid.uuid4())}}
    )

    return ai_response
