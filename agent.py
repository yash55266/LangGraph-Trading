from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from ml_model import get_ml_signal

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
    mock_news = [
        "Reliance announces massive new green energy plant.",
        "Oil prices drop, putting slight pressure on refinery margins."
    ]
    return {"recent_news": mock_news}

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