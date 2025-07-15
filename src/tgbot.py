import telebot
from telebot import types
from config import BOT_TOKEN
from agent.service import AgentService


bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(func=lambda m: True)
def start_message(message:types.Message):
    user_id = message.from_user.id
    user_query = message.text

    res = AgentService.get_answer_user_query(user_id=user_id, user_query=user_query)
    bot.send_message(message.from_user.id, res)

bot.infinity_polling()