import asyncio
import logging
import aio_pika
from abc import ABC
from base.settings import Settings
from base.amqp.connection import get_connection
from base.amqp.exchange import create_exchange
from base import ai
from base import prompts

import openai
import re
import tiktoken
import json
from database.connection import get_db
from database.dao import (
    get_layer_state_by_name,
    get_layer_config,
    get_active_ancestral_prompt,
    update_layer_state,
)
from database.dao_models import LayerConfigModel, AncestralPromptModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Layer Config
# contains the prompts to be used for each layer (including reasoning and action decision)
#
# contains the prompts for the ancestral prompts?? Still have no idea what that is really
# 
# Current issue right now is that the db is not pre-populated with data, so there is no layer config
# noe ancestral prompt to grab
class BaseLayer(ABC):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loop = asyncio.get_event_loop()
        self.connection = None
        self.channel = None
        self.llm_messages = []
        self.ancestral_prompt: AncestralPromptModel
        self.layer_config: LayerConfigModel
        self._fetch_layer_config()
        self._fetch_ancestral_prompt()

    async def data_bus_message_handler(self, message: aio_pika.IncomingMessage):
        logger.info("data_bus_message_handler")
        # try:
        if self.settings.debug:
            # click through for more detail
            await self.wait_for_signal()

            with get_db() as db:
                update_layer_state(
                    db=db,
                    layer_name=self.settings.role_name,
                    process_messages=False,
                )

        # where the message gets processed
        # Focus on this part
        await self._process_message(
            message=message, 
            source_bus="Data Bus",
        )
        await message.ack()
    # except:
        # await message.nack()

    async def control_bus_message_handler(self, message: aio_pika.IncomingMessage):
        logger.info("control_bus_message_handler")
    # try:
        if self.settings.debug:
            await self.wait_for_signal()

        await self._process_message(
            message=message, 
            source_bus="Control Bus",
        )
        await message.ack()
    # except:
        # await message.nack()

    # Checks if the layer in question is ready to process messages or not
    # Waits until the layer is ready to process messages
    async def wait_for_signal(self):
        while True:
            logger.info("wait_for_signal")
            with get_db() as session:
                process_messages = get_layer_state_by_name(
                    db=session,
                    layer_name=self.settings.role_name,
                ).process_messages
                logger.info(f"{process_messages=}")

            if process_messages:
                break
            await asyncio.sleep(3)

    async def _process_message(
        self, message: aio_pika.IncomingMessage, source_bus: str
    ):
        logger.info(f"Processing message from {source_bus}")
        # if debug == True
        # fetches config from the layer config table
        self._fetch_layer_config()
        # fetches the ancestral prompt from the ancestral prompt table
        self._fetch_ancestral_prompt()

        await self._handle_bus_message(
            message=message,
            source_bus=source_bus,
        )

    # uses the reason prompt from layer config along with the identity of the layer as the system message
    # using the message passed in for context, generates a response
    def _reason(
        self,
        input: str,
        source_bus: str,
    ):
        return ai.reason(
            ancestral_prompt=self.ancestral_prompt.prompt,
            input=input,
            source_bus=source_bus,
            prompts=self.layer_config.prompts,
            llm_model_parameters=self.layer_config.llm_model_parameters,
            llm_messages=self.llm_messages,
        )

    async def _handle_bus_message(self, message: aio_pika.IncomingMessage, source_bus):

        logger.info(f"handling message from {source_bus}")

        # Passes in the message into the layer as context
        # Generates an LLM output based on the message
        reasoning_completion = self._reason(
            input=input,
            source_bus=source_bus,
        )

        logger.info(f"{reasoning_completion=}")

        # using the message, reasoning and bus rules, decides what should be
        # put on the subsequent north (data) and south (control) busses
        data_bus_message, control_bus_message = self._determine_action(
            source_bus,
            reasoning_completion,
        )

        logger.info(f"{data_bus_message=}")
        logger.info(f"{control_bus_message=}")

        logger.info(f"{ai.determine_none(data_bus_message['content'])=}")
        logger.info(f"{ai.determine_none(control_bus_message['content'])=}")

        # If the decision is to put a message on to the north and south (data and control) busses
        # It creates the queues and exchanges
        # forms the body of the message
        # then sends the message off
        if ai.determine_none(data_bus_message['content']) != "none":
            await self._publish(
                queue_name=self.settings.data_bus_pub_queue,
                message=data_bus_message,
                destination_bus="Data Bus",
                source_bus=source_bus,
                input_message=message,
                reasoning_message=reasoning_completion,
            )
            self.llm_messages.append(data_bus_message)

        if ai.determine_none(control_bus_message['content']) != "none":
            await self._publish(
                queue_name=self.settings.control_bus_pub_queue,
                message=control_bus_message,
                destination_bus="Control Bus",
                source_bus=source_bus,
                input_message=message,
                reasoning_message=reasoning_completion,
            )
            # create setting to disable this.
            self.llm_messages.append(control_bus_message)

        # used when the message history is too larger than the token max limit
        # agent summarizes the message history
        self._compact_llm_messages()

    # using the message, reasoning and bus rules, decides what should be
    # put on the subsequent north and south busses
    def _determine_action(
        self,
        source_bus,
        reasoning_completion,
    ):
        return ai.determine_action(
            ancestral_prompt=self.ancestral_prompt.prompt,
            source_bus=source_bus,
            reasoning_completion=reasoning_completion,
            prompts=self.layer_config.prompts,
            llm_model_parameters=self.layer_config.llm_model_parameters,
            role_name=self.settings.role_name,
            llm_messages=self.llm_messages,
        )

    async def _publish(
        self,
        queue_name,
        message,
        destination_bus,
        source_bus,
        input_message: aio_pika.IncomingMessage,
        reasoning_message,
    ):
        # creates the queue and exchange for the queue_name passed in
        # binds the queue to the exchange
        # exchange is fanout, so all queues receive the messages
        # logging queue is also bound to the exchange
        exchange = await create_exchange(
            connection=self.connection,
            queue_name=queue_name,
        )

        headers = {
            "source_bus": source_bus,
            "parent_message_id": str(input_message.message_id),
            "destination_bus": destination_bus,
            "layer_name": self.settings.role_name or "user input",
            "llm_messages": json.dumps(self.llm_messages),
            "config_id": str(self.layer_config.config_id),
            "input": input_message.body.decode(),
            "reasoning": json.dumps(reasoning_message),
        }

        logger.info(f"message {headers=}")

        message_body = aio_pika.Message(
            body=message["content"].encode(),
            headers=headers,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type="text/plain",
        )

        logger.info(f"publishing {queue_name=}, {destination_bus=}, {source_bus=}")

        await exchange.publish(
            message_body,
            routing_key=queue_name,
        )

    async def _connect(self):
        self.connection = await get_connection(
            loop=self.loop,
            amqp_host_name=self.settings.amqp_host_name,
            username=self.settings.amqp_username,
            password=self.settings.amqp_password,
            role_name=self.settings.role_name,
        )
        self.channel = await self.connection.channel()
        logger.info(f"{self.settings.role_name} connection established...")

    # data bus is north bus, control bus is south bus
    # if it's sub (subscribe), then it's pushing data onto the bus
    # if it's pub (publish), then it's retrieving data from the bus
    # so for something like layer 2:
    # data_bus_sub_queue => north bus from layer 3
    # data_bus_pub_queue => north bus to layer 1
    # similarly
    # control_bus_sub_queue => south bus from layer 1
    # control_bus_pub_queue => south bus to layer 3
    async def _subscribe(self):
        nb_queue = await self.channel.declare_queue(
            self.settings.data_bus_sub_queue,
            durable=True,
        )
        sb_queue = await self.channel.declare_queue(
            self.settings.control_bus_sub_queue,
            durable=True,
        )

        await nb_queue.consume(self.data_bus_message_handler)
        await sb_queue.consume(self.control_bus_message_handler)

    # when the llm message history token count has become too large
    # it summarizes the history of the messages and uses the new generated message
    # as the replacement for the message history
    def _compact_llm_messages(self):
        token_count = 0
        for message in self.llm_messages:
            token_count += self._count_tokens(message)
        logger.info(f"Current {token_count=}")
        if token_count > self.settings.memory_max_tokens:
            logger.info("compacting initiated...")
            self._update_llm_messages()
            token_count = self._count_tokens(self.llm_messages[0])
            logger.info(f"After compaction memory {token_count=}")
        else:
            logger.info("No compaction required")

    def _update_llm_messages(self):
        openai.api_key = self.settings.openai_api_key
        identity = {"role": "system", "content": self.layer_config.prompts.identity}
        summarization_prompt = {
            "role": "user",
            "content": prompts.memory_compaction_prompt,
        }

        conversation = [identity] + self.llm_messages + [summarization_prompt]

        completion = openai.ChatCompletion.create(
            model=self.settings.model,
            messages=conversation,
            temperature=self.settings.temperature,
        )
        self.llm_messages = [completion.choices[0].message]

    def _count_tokens(self, message: str) -> int:
        encoding = tiktoken.encoding_for_model(self.settings.model)

        logger.info(f"{message=}")

        num_tokens = len(encoding.encode(message["content"]))
        return num_tokens

    def _fetch_layer_config(self):
        with get_db() as db:
            config = get_layer_config(
                db=db,
                layer_name=self.settings.role_name,
            )
            self.layer_config = LayerConfigModel.model_validate(config)


    def _fetch_ancestral_prompt(self):
        with get_db() as db:
            prompt = get_active_ancestral_prompt(db=db)
            self.ancestral_prompt = AncestralPromptModel.model_validate(prompt)

    async def _run_layer(self):
        logger.info(f"Running {self.settings.role_name}")
        # connects to rabbitMQ server and creates channel
        await self._connect()
        await self._subscribe()
        logger.info(
            f"{self.settings.role_name} Subscribed to {self.settings.data_bus_sub_queue} and {self.settings.control_bus_sub_queue}"
        )

    # starts here
    def run(self):
        self.loop.create_task(self._run_layer())
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()
