from pydantic import BaseModel
from typing import Literal
from enum import Enum

class KeywordsList(BaseModel):
    keywords: list[str]

class PaymentType(str, Enum):
    cash = "готівка"
    card = "картка"

class DeliveryTime(str, Enum):
    asap = "якнайшвидше"
    scheduled = "конкретна година"

class DeliveryInfo(BaseModel):
    name: str
    phone_number: str
    payment_type: PaymentType
    delivery_time: DeliveryTime
    menu_positions: list[str]

class OrderConfirmed(BaseModel):
    confirmed: bool

class Message(BaseModel):
    user_id: int
    role: Literal["user", "assistant"]
    text: str

class MessageHistory(BaseModel):
    messages: list[Message]

class User(BaseModel):
    user_id: int