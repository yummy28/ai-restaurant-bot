from config import openai_client, OPENAI_AGENT_MODEL
from schemas import KeywordsList, DeliveryInfo, OrderConfirmed, MessageHistory
from .utils import convert_message_history, convert_message_history_to_str

class OpenAIManager:

    @staticmethod
    def get_answer(user_query: str, message_history: MessageHistory = [], context_str="") -> str:

        history = convert_message_history(user_query, message_history)

        with open("prompts/answer_user.txt", encoding="utf-8") as f:
            PROMPT = f.read()

        PROMPT += f"Context: {context_str}"
        
        response = openai_client.responses.create(
            model=OPENAI_AGENT_MODEL,
            input=history,
            instructions=PROMPT,
            temperature=0
        )

        return response.output_text

    @staticmethod
    def extract_keywords(text: str) -> list[str]:

        with open("prompts/extract_keywords.txt", encoding="utf-8") as f:
            PROMPT = f.read()
        
        result = openai_client.chat.completions.parse(
            model=OPENAI_AGENT_MODEL,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": text}
            ],
            response_format=KeywordsList,
        )
        return result.choices[0].message.parsed.keywords
    
    @staticmethod
    def extract_delivery_info(message_history: MessageHistory) -> DeliveryInfo:

        history = convert_message_history_to_str(message_history)

        with open("prompts/extract_delivery_info.txt", encoding="utf-8") as f:
            PROMPT = f.read()
        
        result = openai_client.chat.completions.parse(
            model=OPENAI_AGENT_MODEL,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": history}
            ],
            response_format=DeliveryInfo,
        )
        return result.choices[0].message.parsed
    
    @staticmethod
    def is_order_confirmed(message_history: MessageHistory) -> bool:

        message_history_str = convert_message_history_to_str(message_history)

        with open("prompts/confirm_order.txt", encoding="utf-8") as f:
            PROMPT = f.read()
        
        result = openai_client.chat.completions.parse(
            model=OPENAI_AGENT_MODEL,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": message_history_str}
            ],
            response_format=OrderConfirmed,
        )

        return result.choices[0].message.parsed.confirmed

    @staticmethod
    def get_rag_fusion_query(user_query: str, message_history: MessageHistory = []) -> str:

        history = convert_message_history(user_query, message_history)

        with open("prompts/rag_fusion1.txt", encoding="utf-8") as f:
            PROMPT = f.read()

        query = f"""
        Chat History: 
        {history}
        Follow Up Input: {user_query}
        Standalone Question:
        """

        response =  openai_client.responses.create(
            model=OPENAI_AGENT_MODEL,
            input=query,
            instructions=PROMPT
        )

        return response.output_text