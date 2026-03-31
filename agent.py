from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from ml_model import get_ml_signal
from ddgs import DDGS
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class TradingState(TypedDict):
    ticker: str
    ml_signal: str                    
    ml_confidence: float              
    recent_news: List[str]           
    final_decision: str               
    reasoning: str
    bull_reason: str


def quant_node(state: TradingState):
    ml_results=get_ml_signal()
    
    return {
        "ml_signal": ml_results["ml_signal"],
        "ml_confidence": ml_results["ml_confidence"]
    }

def researcher_node(state: TradingState):    
    ticker = state['ticker']
    company_name = ticker.split('.')[0]
    search_query = f"{company_name} stock financial news India"
    
    recent_news = []
    
    try:
        with DDGS() as ddgs:
            results = ddgs.text(search_query, max_results=5, timelimit='w')
            
            for i, article in enumerate(results):
                title = article.get('title', 'No Title')
                snippet = article.get('body', 'No summary available.')
                
                formatted_news = f"{i+1}. {title}\nSummary: {snippet}\n"
                recent_news.append(formatted_news)
                
    except Exception as e:
        print(f"Search Error: {e}")
        recent_news = ["Could not fetch recent news. Rely entirely on technical data."]

    return {"recent_news": recent_news}

class BullResponse(BaseModel):
    bull_reason: str

def bull_node(state):
    ticker = state.get('ticker', 'Unknown')
    
    client = OpenAI()
    news = "\n".join(state.get("recent_news", ["No news found."]))

    user_prompt = f"""
    Ticker: {ticker}
    News: {news}
    """
    
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are a highly optimistic Bullish Analyst. Your job is to read the news and ONLY provide the strongest possible arguments for why we should BUY this stock right now. Ignore negative news or spin it positively. Keep it to 2-3 compelling sentences."},
                {"role": "user", "content": user_prompt}
            ],
            response_format=BullResponse
        )
        
        reasoning = response.choices[0].message.parsed.bull_reason
                
        return {
            "bull_reason": reasoning
        }
        
    except Exception as e:
        return {
            "bull_reason": "Could not generate bullish due to a system error."
        }


class CIODecision(BaseModel):
    final_decision: str
    reasoning: str

def cio_node(state):
    client = OpenAI()
    
    news = "\n".join(state.get("recent_news", ["No news found."]))
    user_prompt = f"""
    Ticker: {state.get('ticker')}
    ML Signal: {state.get('ml_signal')} (Confidence: {state.get('ml_confidence')}%)
    News: {news}
    """
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a quant fund CIO. Decide: 'EXECUTE_BUY', 'EXECUTE_SELL', or 'HOLD'. Severe news overrides the ML signal. Otherwise, trust the ML."},
                {"role": "user", "content": user_prompt}
            ],
            response_format=CIODecision
        )
        
        decision = response.choices[0].message.parsed
        
        return {
            "final_decision": decision.final_decision,
            "reasoning": decision.reasoning
        }
        
    except Exception as e:
        return {"final_decision": "HOLD", "reasoning": "System failure. Defaulting to HOLD."}

workflow = StateGraph(TradingState)

workflow.add_node("quant_model", quant_node)
workflow.add_node("news_researcher", researcher_node)
workflow.add_node("chief_investment_officer", cio_node)
workflow.add_node("bullish_node",bull_node)

workflow.add_edge(START, "quant_model")
workflow.add_edge("quant_model", "news_researcher")
workflow.add_edge("news_researcher","bullish_node")
workflow.add_edge("news_researcher", "chief_investment_officer")
workflow.add_edge("chief_investment_officer", END)

app = workflow.compile()

if __name__ == "__main__":
    
    initial_input = {"ticker": "RELIANCE.BO"}
    
    final_state = app.invoke(initial_input)
    
    print(f"Decision:  {final_state['final_decision']}")
    print(f"Reasoning: {final_state['reasoning']}")