from vector_search.hybrid_search import HybridSearchManager
from database.sqllite_core import DatabaseManager
from .openai_manager import OpenAIManager
from .messages import DISPUTE_RESPONSE, PARTNERSHIP_RESPONSE
from classifier import classify_query
from schemas import Message, MessageHistory, User


class AgentService:

    @staticmethod
    def get_answer_user_query(user_id: int, user_query: str) -> str:
        if not DatabaseManager.if_user_exists(user_id):
            DatabaseManager.add_user(User(user_id=user_id))

        query_category = DatabaseManager.get_user_query_category(user_id)
        if query_category is None:
            query_category = classify_query(user_query)
            DatabaseManager.update_user_query_category(user_id, query_category)

        if query_category == "order":
            answer = AgentService.get_answer_order(user_id, user_query)
        elif query_category == "disputes":
            answer = AgentService.get_answer_dispute()
        elif query_category == "partnership":
            answer = AgentService.get_answer_partnership()

        DatabaseManager.add_message(Message(user_id=user_id, role="user", text=user_query))
        DatabaseManager.add_message(Message(user_id=user_id, role="assistant", text=answer))
        
        return answer
    
    @staticmethod
    def get_answer_order(user_id: int, user_query: str) -> str:
        message_history = DatabaseManager.get_message_history(user_id)
        better_query = OpenAIManager.get_rag_fusion_query(message_history=message_history, user_query=user_query)

        points = HybridSearchManager.query(text=better_query)
        context_str = [str(point.payload) for point in points]
        answer = OpenAIManager.get_answer(user_query=user_query, message_history=message_history, context_str=context_str)

        last_messages = MessageHistory(messages=message_history.messages[-4:])
        last_messages.messages.append(Message(user_id=user_id, role="user", text=user_query))

        if OpenAIManager.is_order_confirmed(last_messages) and not DatabaseManager.is_user_finished(user_id=user_id):
            delivery_info = OpenAIManager.extract_delivery_info(message_history)
            DatabaseManager.finish_user(user_id)
            print(delivery_info)
        return answer

    @staticmethod
    def get_answer_dispute() -> str:
        return DISPUTE_RESPONSE

    @staticmethod
    def get_answer_partnership() -> str:
        return PARTNERSHIP_RESPONSE
