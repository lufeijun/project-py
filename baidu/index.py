from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.messages import HumanMessage


import os

os.environ["QIANFAN_AK"] = ""
os.environ["QIANFAN_SK"] = ""

chat = QianfanChatEndpoint(
    streaming=True,
     model="ERNIE-4.0-8K",
)
res = chat([HumanMessage(content="你是谁")])
print(res.content)
