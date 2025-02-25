# src/rag_strategy.py

from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from src.rag.knowledge_loader import load_knowledge_docs, create_vector_store
import pandas as pd
from dotenv import load_dotenv
import os
from transformers import pipeline

load_dotenv()

def load_llm():
    """
    Loads a publicly available Falcon instruct model without requiring tokens.
    We'll use the HuggingFace pipeline approach so that LangChain sees it as a valid LLM.
    """
    # Create a transformers pipeline for text generation with a public model
    hf_pipe = pipeline(
        task="text-generation",
        model="tiiuae/falcon-7b-instruct",  # public instruct model
        max_length=512,                    # adjust as needed
        do_sample=True,
        temperature=0.7
    )
    # Wrap this pipeline in a LangChain HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    return llm

def generate_business_strategy(csv_filepath: str) -> str:
    # 1. Load lead data and pick top leads
    df = pd.read_csv(csv_filepath)
    top_leads = df.sort_values('lead_score', ascending=False).head(5)
    summary = "Key Metrics from Lead Data:\n"
    summary += top_leads[['company_name', 'industry', 'lead_score']].to_string(index=False)
    
    # 2. Load knowledge docs and create the vector store
    docs = load_knowledge_docs()
    vector_store = create_vector_store(docs)
    retriever = vector_store.as_retriever()

    # 3. Load the LLM (HuggingFacePipeline) so it's recognized as a valid Runnable
    llm = load_llm()

    # 4. Create the QA chain with chain_type="stuff"
    #    Note: For advanced usage, you can also use chain_type="map_reduce" or "refine".
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    # 5. Define the final prompt
    prompt = (
        "Use the following pieces of context to answer the question at the end. "
        "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
        "Title: Advanced Lead Generation Strategies\n"
        "Content:\n"
        "In today's competitive market, advanced lead generation requires a multi-channel approach. Key strategies include:\n"
        "- Leveraging data analytics to segment customers by behavior and firmographics.\n"
        "- Implementing predictive models to score leads and predict conversion likelihood.\n"
        "- Using personalized, AI-driven messaging based on customer interactions.\n"
        "- Integrating social media insights, web analytics, and CRM data to form a comprehensive view.\n"
        "- Continuous testing and optimization of ad creatives across channels.\n\n"
        "Title: The Future of AI in Sales and Lead Generation\n"
        "Content:\n"
        "Artificial intelligence is revolutionizing sales by automating lead generation and optimizing marketing strategies. Emerging trends include:\n"
        "- The use of Retrieval-Augmented Generation (RAG) to combine real-time data with expert knowledge.\n"
        "- Integration of advanced vector databases to store and retrieve contextual information.\n"
        "- Development of autonomous agents that learn and adapt over time.\n"
        "- Implementation of deep learning models to predict customer behavior and conversion rates.\n"
        "- Streamlined workflows that allow sales teams to focus on high-value interactions.\n\n"
        "Title: Best Practices in Marketing & Advertising\n"
        "Content:\n"
        "Effective marketing and advertising are grounded in understanding customer journeys and optimizing touchpoints. Best practices include:\n"
        "- Utilizing multi-channel advertising campaigns to maximize reach.\n"
        "- Adopting A/B testing to refine ad creatives and messages.\n"
        "- Leveraging AI to dynamically adjust bids and targeting based on performance data.\n"
        "- Integrating customer feedback loops and sentiment analysis.\n"
        "- Employing machine learning for precise targeting and personalization.\n\n"
        f"Question: Based on the following key metrics from our lead generation data:\n{summary}\n\n"
        "And considering best practices in lead generation, marketing, and advertising, provide a detailed business strategy to improve lead quality and boost sales. "
        "Include recommendations on multi-channel marketing, personalized outreach, advertising optimization, and the use of AI-driven insights. "
        "Present your answer in a structured, step-by-step format with clear headings and bullet points where appropriate."
    )
    
    # 6. Use qa_chain.invoke(...) to get the answer
    result = qa_chain.invoke(prompt)
    
    # 7. If result is a dict, parse out the text. Otherwise, convert to str.
    if isinstance(result, dict):
        # Some chain types return dict with "result" or "output_text"
        final_answer = result.get("result") or result.get("output_text") or str(result)
    else:
        final_answer = str(result)

    return final_answer

if __name__ == "__main__":
    strategy = generate_business_strategy("data/processed/scored_data.csv")
    with open("data/final/business_strategy.txt", "w") as f:
        f.write(strategy)
    print("Business strategy generated and saved to data/final/business_strategy.txt")
    print("\nGenerated Strategy:\n", strategy)
