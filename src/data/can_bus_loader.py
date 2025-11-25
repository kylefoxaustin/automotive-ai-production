class CANBusDataLoader:
    def __init__(self, interface, bitrate):
        self.interface = interface
        self.bitrate = bitrate
    
    def connect(self):
        print(f"Connected to {self.interface} at {self.bitrate} bps")
