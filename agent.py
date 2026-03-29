from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from ml_model import get_ml_signal
from ddgs import DDGS

class TradingState(TypedDict):
    ticker: str
    ml_signal: str                    
    ml_confidence: float              
    recent_news: List[str]           
    final_decision: str               
    reasoning: str


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

def cio_node(state: TradingState):    
    
    signal = state.get("ml_signal")
    confidence = state.get("ml_confidence")
    news = state.get("recent_news")
    
    reasoning = f"The ML model suggests {signal} ({confidence}%). News is mostly positive. Approving trade."
    
    return {
        "final_decision": "EXECUTE_BUY",
        "reasoning": reasoning
    }

workflow = StateGraph(TradingState)

workflow.add_node("quant_model", quant_node)
workflow.add_node("news_researcher", researcher_node)
workflow.add_node("chief_investment_officer", cio_node)

workflow.add_edge(START, "quant_model")
workflow.add_edge("quant_model", "news_researcher")
workflow.add_edge("news_researcher", "chief_investment_officer")
workflow.add_edge("chief_investment_officer", END)

app = workflow.compile()

if __name__ == "__main__":
    
    initial_input = {"ticker": "RELIANCE.BO"}
    
    final_state = app.invoke(initial_input)
    
    print(f"Decision:  {final_state['final_decision']}")
    print(f"Reasoning: {final_state['reasoning']}")