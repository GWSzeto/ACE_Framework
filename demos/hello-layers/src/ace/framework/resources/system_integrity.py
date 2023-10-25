import aio_pika
import asyncio
import yaml

from ace.settings import Settings
from ace.framework.resource import Resource


class SystemIntegritySettings(Settings):
    pass

# This file seems to be responsible for the flow of the whole system
# sets up the system integrity queues and exchanges for each bus between each layer
# the callbacks for each queue setup are responsible for the following:
# 1. sending the "run_layer" for each layer
# 2. sending the "begin_work" command to the top layer


class SystemIntegrity(Resource):

    def __init__(self):
        super().__init__()
        self.post_complete = False
        self.shutdown_complete = False
        # Sets up a dict of all of the busses between each layer
        # Assuming this is designed to status test the ping and pong endpoints for each layer
        self.post_verification_matrix = self.compute_ping_pong_combinations()

    @property
    def settings(self):
        return SystemIntegritySettings(
            name="system_integrity",
            label="System Integrity",
        )

    # TODO: Add valid status checks.
    def status(self):
        self.log.debug(f"Checking {self.labeled_name} status")
        return self.return_status(True)

    async def post_connect(self):
        # The callback for subbing to this queue
        # is responsible for running the layers and beginning work
        await self.subscribe_system_integrity()
        # The callback for subbing to this queue
        # is responsible for stopping all the layers when a done message is received
        await self.subscribe_system_integrity_data()

    # Called after the post_connect
    def post_start(self):
        asyncio.set_event_loop(self.bus_loop)
        self.bus_loop.create_task(self.post_layers())

    async def system_integrity_pre_disconnect(self):
        await self.unsubscribe_system_integrity()
        await self.unsubscribe_system_integrity_data()

    async def publish_message(self, queue_name, message, delivery_mode=2):
        message = aio_pika.Message(
            body=message,
            delivery_mode=delivery_mode
        )
        # publishing it through a default exchange sends the message directly to the queue
        await self.publisher_channel.default_exchange.publish(message, routing_key=queue_name)

    # This is where it's responsible for calling the ping and pong status check for each layer
    async def post_layers(self):
        for layer in self.settings.layers:
            await self.post_layer(layer)

    # utility function responsible for sending commands to the respective system integrity resource queue
    # resource could be layer, logging, telemetry, etc
    async def execute_resource_command(self, resource, command, kwargs=None):
        kwargs = kwargs or {}
        self.log.debug(
            f"[{self.labeled_name}] sending command '{command}' to resource: {resource}")
        queue_name = self.build_system_integrity_queue_name(resource)
        message = self.build_message(
            resource, message={'method': command, 'kwargs': kwargs}, message_type='command')
        await self.publish_message(queue_name, message)

    async def post_layer(self, layer):
        self.log.info(
            f"[{self.labeled_name}] sending POST command to layer: {layer}")
        await self.execute_resource_command(layer, 'schedule_post')

    # Sends the 'run_layer' command to each system_integrity queue for each layer
    async def run_layers(self):
        for layer in self.settings.layers:
            self.log.info(f"[{self.labeled_name}] Running layer: {layer}")
            await self.execute_resource_command(layer, 'run_layer')

    # stops layers, then services, then the bus, then this resource
    async def stop_layers(self):
        # stops all layers in reverse order
        for layer in reversed(self.settings.layers):
            self.log.info(f"[{self.labeled_name}] Stopping layer: {layer}")
            await self.execute_resource_command(layer, 'stop_resource')
        # stops logging and telemetry
        for resource in self.settings.other_resources:
            if resource != 'busses':
                self.log.info(
                    f"[{self.labeled_name}] Stopping resource: {resource}")
                await self.execute_resource_command(resource, 'stop_resource')
        # unsubs from the system integrity queue and the system integrity data queue
        await self.system_integrity_pre_disconnect()
        self.log.info(f"[{self.labeled_name}] Stopping resource: 'busses'")
        await self.execute_resource_command('busses', 'stop_resource')
        self.stop_resource()
        self.shutdown_complete = True

    # signals the begin_work command to the top layer
    async def begin_work(self):
        top_layer = self.settings.layers[0]
        self.log.info(
            f"[{self.labeled_name}] Beginning work from top layer: {top_layer}")
        await self.execute_resource_command(top_layer, 'begin_work')

    async def message_handler(self, message: aio_pika.IncomingMessage):
        async with message.process():
            body = message.body.decode()
        self.log.debug(f"[{self.labeled_name}] received a message: {body}")
        try:
            data = yaml.safe_load(body)
        except yaml.YAMLError as e:
            self.log.error(
                f"[{self.labeled_name}] could not parse message: {e}")
            return
        if not self.post_complete:
            await self.check_post_complete(data)

    async def message_data_handler(self, message: aio_pika.IncomingMessage):
        async with message.process():
            body = message.body.decode()
        self.log.debug(
            f"[{self.labeled_name}] received a data message: {body}")
        try:
            data = yaml.safe_load(body)
        except yaml.YAMLError as e:
            self.log.error(
                f"[{self.labeled_name}] could not parse data message: {e}")
            return
        if not self.shutdown_complete:
            await self.check_done(data)

    # If all of the ping pong status checks are complete for every bus between every layer
    # then the layers are good to go
    async def check_post_complete(self, data):
        if data['type'] in ['ping', 'pong']:
            # If all of the ping and pongs for each bus between each layer has been checked
            # The layers are good to go then and ready to run_layers and begin_work
            if self.verify_ping_pong_sequence_complete(f"{data['type']}.{data['resource']['source']}.{data['resource']['destination']}"):
                self.log.info(
                    f"[{self.labeled_name}] verified POST complete for all layers")
                self.post_complete = True
                # click through for more info
                await self.run_layers()
                await self.begin_work()

    async def check_done(self, data):
        if data['type'] == 'done':
            self.log.info(
                f"[{self.labeled_name}] ACE mission done, initiating shutdown of all layers")
            await self.stop_layers()

    # any time a message goes to the system integrity queue
    # the message will be sent to the message handler where it runs the layers and begins work
    async def subscribe_system_integrity(self):
        self.log.debug(
            f"{self.labeled_name} subscribing to system integrity queue...")
        queue_name = self.settings.system_integrity_queue
        self.consumers[queue_name] = await self.try_queue_subscribe(queue_name, self.message_handler)
        self.log.info(
            f"{self.labeled_name} Subscribed to system integrity queue")

    # subs to the system integrity data queue
    # if the message sent to the queue is of type "done"
    # then the ACE mission is done and it shuts down all of the layers
    async def subscribe_system_integrity_data(self):
        self.log.debug(
            f"{self.labeled_name} subscribing to system integrity data queue...")
        queue_name = self.settings.system_integrity_data_queue
        self.consumers[queue_name] = await self.try_queue_subscribe(queue_name, self.message_data_handler)
        self.log.info(
            f"{self.labeled_name} Subscribed to system integrity data queue")

    async def unsubscribe_system_integrity(self):
        queue_name = self.settings.system_integrity_queue
        if queue_name in self.consumers:
            queue, consumer_tag = self.consumers[queue_name]
            self.log.debug(
                f"{self.labeled_name} unsubscribing from system integrity queue...")
            await queue.cancel(consumer_tag)
            self.log.info(
                f"{self.labeled_name} Unsubscribed from system integrity queue")

    async def unsubscribe_system_integrity_data(self):
        queue_name = self.settings.system_integrity_data_queue
        if queue_name in self.consumers:
            queue, consumer_tag = self.consumers[queue_name]
            self.log.debug(
                f"{self.labeled_name} unsubscribing from system integrity data queue...")
            await queue.cancel(consumer_tag)
            self.log.info(
                f"{self.labeled_name} Unsubscribed from system integrity data queue")

    def compute_ping_pong_combinations(self):
        layers = self.settings.layers
        combinations = {}
        for i in range(len(layers)):
            # First layer has no northen layer.
            if i != 0:
                combinations[f"ping.{layers[i-1]}.{layers[i]}"] = False
                combinations[f"pong.{layers[i]}.{layers[i-1]}"] = False
            # Last layer has no southern layer.
            if i != len(layers) - 1:
                combinations[f"ping.{layers[i+1]}.{layers[i]}"] = False
                combinations[f"pong.{layers[i]}.{layers[i+1]}"] = False
        return combinations

    # this checks if all of the ping and pongs were successful for each bus between each layer
    def verify_ping_pong_sequence_complete(self, step):
        if step in self.post_verification_matrix:
            self.post_verification_matrix[step] = True
        return all(value for value in self.post_verification_matrix.values())
