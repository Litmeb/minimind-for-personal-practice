from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from dataclasses import dataclass
# from langchain_community.tools.tavily_search import TavilySearchResults
def judge(title,content,assistant_response,valid_categories,correct_category):
    @dataclass
    class ResponseFormat:
        """Output the evaluation result."""
        correctness: bool
    model = ChatDeepSeek(model="deepseek-chat")
    SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of text classification. You will be given a user query that includes a title, content, and a set of allowed categories. The assistant's response is a single predicted category.

    Your task is to judge whether the assistant’s classification is correct and well-justified based on the provided title and content.

    Correctness: Does the predicted category align with the correct category?
    Output your evaluation in the following JSON format strictly without any additional text:"""
    # 3.创建Agent
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        response_format=ResponseFormat
    )

    # 4.运行Agent获得结果
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f'Title: {title}\nContent: {content}\nValid Categories: {valid_categories}\nCorrect Category: {correct_category}\nAssistant Response: {assistant_response}'}]}
    )
    return result['structured_response']
if __name__ == "__main__":
    while True:
        title = input("请输入标题: ")
        content = input("请输入内容: ")
        assistant_response = input("请输入助手回答: ")
        valid_categories = input("请输入允许的分类: ")
        correct_category = input("请输入正确分类: ")
        result = judge(title,content,assistant_response,valid_categories,correct_category)
        print(result)