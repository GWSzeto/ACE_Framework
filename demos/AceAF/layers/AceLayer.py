from . import LAYER_REGISTRY
from agentforge.utils.storage_interface import StorageInterface
import threading
from agentforge.config import Config
from .Interface import Interface
from pprint import pprint 


class AceLayer:

    interface = Interface()

    def __init__(self):
        self.layer_name = self.__class__.__name__
        self.layer_number = int(self.layer_name[1])  # Strip the 'L' prefix and layer name to get the number
        self.north_layer = self.layer_number - 1
        self.south_layer = self.layer_number + 1

        self.storage = StorageInterface().storage_utils
        self.config = Config()
        self.interface = Interface()

        self.bus = {'NorthBus': None, 'SouthBus': None}
        self.my_messages = {'NorthBus': None, 'SouthBus': None}
        self.top_layer_message = None
        self.bottom_layer_message = None

        self.result = None
        self.agent = None
        self.event = None
        self.event_type = None  # variable to store the type of event

        LAYER_REGISTRY[self.layer_number] = self

    # -------------------------------- THREADS AND EVENTS --------------------------------

    def create_event_thread(self):
        def event_loop():
            while True:
                self.event.wait()  # Wait for any event to be triggered

                # Depending on the event type, call the appropriate handler
                if self.event_type == 'NorthBusUpdate':
                    self.handle_north_bus_update()
                elif self.event_type == 'SouthBusUpdate':
                    self.handle_south_bus_update()
                elif self.event_type == 'InputUpdate':
                    self.handle_input_update()
                elif self.event_type == 'UserUpdate':
                    self.handle_user_update()

                # Reset the event
                self.event.clear()

        thread = threading.Thread(target=event_loop)
        thread.daemon = True
        thread.start()

    def stand_by(self):
        self.event = threading.Event()
        self.event_type = None
        self.create_event_thread()

    def handle_north_bus_update(self):
        # Load Data From North Bus and process
        self.run()

    def handle_south_bus_update(self):
        # Load Data From South Bus and process
        self.run()

    def handle_input_update(self):
        # Load Relevant Data From Input and process
        self.run()

    # This only called when the /bot endpoint is hit
    # Interestingly enough, the L3 Agent is responsible for ingesting user messages
    # And sending it through out the other layers
    def handle_user_update(self):
        # Load Relevant Data From Input and process
        LAYER_REGISTRY[self.layer_number].get_proposed_response()
        self.run()

    def trigger_event(self, event_type):
        """Trigger the event and set the event type."""
        self.event_type = event_type
        self.event.set()

    def trigger_next_layer(self):
        if self.south_layer < 7:
            LAYER_REGISTRY[self.south_layer].trigger_event('SouthBusUpdate')
        else:
            # This will most likely use the interface method handle_south_bus
            LAYER_REGISTRY[self.layer_number].parse_agent_output()

    # -------------------------------- MAIN LOGIC --------------------------------

    def run(self):
        self.interface.output_message(self.layer_number,
                                      f"\n--------------------Running {self.layer_name}--------------------")
        self.initialize_agents() # pass
        self.load_relevant_data() # pass
        self.load_data_from_bus(bus="SouthBus") # loads SouthBus collection to self.bus.SouthBus
        self.load_data_from_bus(bus="NorthBus") # loads NorthBus collection to self.bus.NorthBus
        self.process_data_from_buses() # Loads the top and bottom layer messages. North layer => South Bus => Top Layer Message, vice versa
        self.run_agents() # calls "run" on the agents (reference AgentForge architecture)
        self.parse_results() # splits llm message into north and south bus messages
        self.update_bus(bus="SouthBus", message=self.my_messages['SouthBus']) # saves message to "SouthBus" collection for this layer number
        self.update_bus(bus="NorthBus", message=self.my_messages['NorthBus']) # saves message to "NorthBus" collection for this layer number 
        self.trigger_next_layer()

    def initialize_agents(self):
        # Meant for Individual Layers to override
        pass

    def load_relevant_data(self):
        # Meant for Individual Layers to override
        pass

    def load_data_from_bus(self, **kwargs):  # North Bus
        bus_name = kwargs['bus']
        params = {"collection_name": bus_name}
        self.bus[bus_name] = self.storage.load_collection(params)

    def process_data_from_buses(self):
        north_bus = self.bus.get("NorthBus", None)
        south_bus = self.bus.get("SouthBus", None)

        north_layer = self.north_layer.__str__()
        south_layer = self.south_layer.__str__()

        # Having a hard time understanding how a collection could be in it's own referenced "ids" field
        # printing south_bus to get a better idea of what's going on

        print("South Bus")
        pprint(south_bus)
        pprint(south_bus['ids'])
        print(south_bus in south_bus['ids'])

        # North Layer Writes to South Bus, Hence it's a Message from the Top Layer
        if south_bus and north_layer in south_bus['ids']:
            index = south_bus['ids'].index(north_layer)
            self.top_layer_message = south_bus['documents'][index]

        # South Layer Writes to North Bus, Hence it's a Message from the Bottom Layer
        if north_bus and south_layer in north_bus['ids']:
            index = north_bus['ids'].index(north_layer)
            self.bottom_layer_message = north_bus['documents'][index]

    def run_agents(self):
        # Call individual Agents From Each Layer
        self.result = self.agent.run(top_message=self.top_layer_message,
                                     bottom_message=self.bottom_layer_message)
                                     # self_message=self.my_messages['SouthBus'])

    def parse_results(self):
        result = self.result.__str__()

        # Splitting the string on "Northbound:" to separate the sections again
        if "---Northbound---" in result:
            southbound_str, northbound_str = result.split("---Northbound---")
            northbound_str = northbound_str.strip()
        else:
            northbound_str = None
            southbound_str = result

        southbound_str = southbound_str.replace("---Southbound---", "").strip()

        self.my_messages['SouthBus'] = southbound_str
        self.my_messages['NorthBus'] = northbound_str

        print(f"SOUTH BUS MESSAGE:\n\n{self.my_messages['SouthBus']}\n\n")
        print(f"NORTH BUS MESSAGE:\n{self.my_messages['NorthBus']}\n\n")

    def update_bus(self, **kwargs):

        if not kwargs['message']:
            return

        params = {
            'collection_name': kwargs['bus'],
            'ids': [self.layer_number.__str__()],
            'data': [kwargs['message']]
        }

        self.storage.save_memory(params)
        self.interface.output_message(self.layer_number,
                                      f"\n-----------------------{kwargs['bus']}-----------------------\n"
                                      f"{kwargs['message']}\n")


