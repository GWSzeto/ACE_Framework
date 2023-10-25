from .AceLayer import AceLayer
from .customagents.l3agent.SelfModel import SelfModel
from .Chat import Chatbot

# This layer appears to be responsible for ingesting user messages
# and sending them through the other layers
class L3Agent(AceLayer):

    chat_bot = Chatbot()
    input_data = None
    proposed_response = None

    def initialize_agents(self):
        self.agent = SelfModel()

    def load_relevant_data(self):
        self.interface.refresh_info()

        self.input_data = (f"Operating System Name: {self.interface.os_name}\n"
                           f"Operating System Version: {self.interface.os_version}\n"
                           f"System: {self.interface.system}\n"
                           f"Architecture: {self.interface.architecture}\n"
                           f"Current Date and Time: {self.interface.date_time}")

    def run_agents(self):
        # Call individual Agents From Each Layer

        print(f"\nProposed Response:\n{self.proposed_response}\n")

        self.result = self.agent.run(top_message=self.top_layer_message,
                                     bottom_message=self.bottom_layer_message,
                                     input_data=self.input_data,
                                     proposed_response=self.proposed_response)

        self.proposed_response = None

    def get_proposed_response(self):
        last_message = self.interface.get_chat_messages(1)
        response = self.chat_bot.run(last_message)

        print(f"\nUnfiltered Response:\n{response}\n")

        self.proposed_response = response

    # Pull chat history last message from user
    # send message variable to .custom_agents.modules.chat.chatbot.run()
    # return bot response
    # Run response through self-model
    # send bot response down south bus
