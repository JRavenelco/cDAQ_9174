"""
Digilent AnalogIn acquisition thread using DWF C library directly (like projectAD).
Based on projectAD implementation: https://github.com/dimschlukas/projectAD

Uses ctypes to call DWF functions directly instead of pydwf wrapper.
"""

import threading
import queue
import time
import sys
import numpy as np
from ctypes import *
from dwfconstants import *
from typing import List, Optional


class DigilentDWFThread(threading.Thread):
    """Continuous AnalogIn acquisition thread using DWF C library directly (projectAD style)."""
    
    def __init__(
        self,
        data_queue: Optional[queue.Queue] = None,
        channels: Optional[List[int]] = None,
        sample_rate: float = 20000.0,
        channel_range: float = 5.0,
        buffer_size: Optional[int] = None,
        acquisition_time: float = 0.1,  # Time per acquisition block in seconds
        name: str = "DigilentDWFThread",
        daemon: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(name=name, daemon=daemon)
        self._stop_event = threading.Event()
        self.queue = data_queue if data_queue is not None else queue.Queue(maxsize=10)
        self.channels = channels if channels is not None else [0, 1]  # Default both channels
        self.sample_rate = float(sample_rate)
        self.channel_range = float(channel_range)
        self.buffer_size = buffer_size
        self.acquisition_time = acquisition_time
        self.verbose = verbose
        
        self.exception: Optional[BaseException] = None
        self.dwf = None
        self.hdwf = c_int()
        
    def stop(self) -> None:
        """Request the thread to stop."""
        self._stop_event.set()
        
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[DigilentDWF] {msg}")
    
    def _open_device(self) -> bool:
        """Open first available Digilent device. Returns True if successful."""
        try:
            # Load DWF library
            if sys.platform.startswith("win"):
                self.dwf = cdll.dwf
            elif sys.platform.startswith("darwin"):
                self.dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
            else:
                self.dwf = cdll.LoadLibrary("libdwf.so")
            
            # Check for available devices
            cdevices = c_int()
            self.dwf.FDwfEnum(c_int(0), byref(cdevices))
            self._log(f"Number of Devices: {cdevices.value}")
            
            if cdevices.value == 0:
                self._log("No device available")
                return False
            
            # Open first device
            self.dwf.FDwfDeviceOpen(c_int(0), byref(self.hdwf))
            
            if self.hdwf.value == hdwfNone.value:
                self._log("Failed to open device - may be in use by another process")
                return False
                
            self._log("Device opened successfully")
            return True
            
        except Exception as e:
            self._log(f"Error opening device: {e}")
            return False
    
    def _configure_acquisition(self) -> bool:
        """Configure analog input acquisition like projectAD. Returns True if successful."""
        try:
            # Calculate buffer size if not specified (like projectAD approach)
            if self.buffer_size is None:
                self.buffer_size = int(self.sample_rate * self.acquisition_time)
            
            self._log(f"Configuring acquisition: {self.sample_rate} Hz, {self.buffer_size} samples")
            
            # Set sampling frequency
            self.dwf.FDwfAnalogInFrequencySet(self.hdwf, c_double(self.sample_rate))
            
            # Set range for all channels (like projectAD uses -1 for all channels)
            self.dwf.FDwfAnalogInChannelRangeSet(self.hdwf, c_int(-1), c_double(self.channel_range))
            
            # Set buffer size
            self.dwf.FDwfAnalogInBufferSizeSet(self.hdwf, c_int(self.buffer_size))
            
            # Enable only requested channels
            for ch in range(4):  # Assume max 4 channels
                enable = ch in self.channels
                self.dwf.FDwfAnalogInChannelEnableSet(self.hdwf, c_int(ch), c_bool(enable))
                if enable:
                    self._log(f"Enabled channel {ch}")
            
            return True
            
        except Exception as e:
            self._log(f"Error configuring acquisition: {e}")
            return False
    
    def _acquire_data(self) -> Optional[np.ndarray]:
        """Acquire one block of data like projectAD. Returns data array or None on error."""
        try:
            # Start acquisition (like projectAD does it)
            self.dwf.FDwfAnalogInConfigure(self.hdwf, c_int(1), c_int(1))
            
            # Wait for acquisition to complete (like projectAD)
            sts = c_int()
            timeout_count = 0
            max_timeout = int(self.acquisition_time * 1000 + 1000)  # ms
            
            while timeout_count < max_timeout:
                self.dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(sts))
                if sts.value == DwfStateDone.value:
                    break
                time.sleep(0.001)  # 1ms sleep
                timeout_count += 1
                
                if self._stop_event.is_set():
                    return None
            
            if sts.value != DwfStateDone.value:
                self._log(f"Acquisition timeout, status: {sts.value}")
                return None
            
            # Read data from all enabled channels
            channel_data = []
            for ch in self.channels:
                # Create buffer for this channel
                rg = (c_double * self.buffer_size)()
                
                # Get data for this channel
                self.dwf.FDwfAnalogInStatusData(self.hdwf, c_int(ch), rg, len(rg))
                
                # Convert to numpy array
                data = np.array([rg[i] for i in range(self.buffer_size)], dtype=np.float32)
                channel_data.append(data)
            
            # Stack into shape (n_channels, n_samples)
            if len(channel_data) > 1:
                result = np.vstack(channel_data)
            else:
                result = channel_data[0].reshape(1, -1)
            
            return result
            
        except Exception as e:
            self._log(f"Error acquiring data: {e}")
            return None
    
    def _close_device(self) -> None:
        """Close the device."""
        try:
            if self.dwf is not None and self.hdwf.value != hdwfNone.value:
                self.dwf.FDwfDeviceCloseAll()
                self._log("Device closed")
        except Exception as e:
            self._log(f"Error closing device: {e}")
    
    def run(self) -> None:
        """Main thread loop."""
        try:
            # Open device
            if not self._open_device():
                raise RuntimeError("Failed to open Digilent device")
            
            # Configure acquisition
            if not self._configure_acquisition():
                raise RuntimeError("Failed to configure acquisition")
            
            self._log("Starting continuous acquisition")
            
            # Main acquisition loop
            while not self._stop_event.is_set():
                # Acquire one block of data
                data_block = self._acquire_data()
                
                if data_block is not None:
                    # Check for NaNs
                    if np.isnan(data_block).any():
                        self._log("NaN detected in acquired block; dropping this block.")
                        continue
                    
                    # Add to queue with drop-oldest policy
                    try:
                        self.queue.put_nowait(data_block)
                    except queue.Full:
                        try:
                            _ = self.queue.get_nowait()  # Remove oldest
                        except queue.Empty:
                            pass
                        self.queue.put_nowait(data_block)
                
                # Brief pause between acquisitions (like projectAD)
                time.sleep(0.001)
            
            self._log("Acquisition stopped")
            
        except Exception as e:
            self.exception = e
            self._log(f"Exception in acquisition thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._close_device()


# Test function
if __name__ == "__main__":
    import sys
    
    q = queue.Queue(maxsize=5)
    t = DigilentDWFThread(
        data_queue=q,
        channels=[0, 1],
        sample_rate=20000.0,
        channel_range=5.0,
        acquisition_time=0.1,
        verbose=True,
    )
    
    t.start()
    t0 = time.time()
    
    try:
        while time.time() - t0 < 5.0:
            try:
                blk = q.get(timeout=0.2)
                print(f"Got block shape={blk.shape}, CH0: min={blk[0].min():.3f}V max={blk[0].max():.3f}V, CH1: min={blk[1].min():.3f}V max={blk[1].max():.3f}V")
            except queue.Empty:
                pass
    finally:
        t.stop()
        t.join(timeout=3.0)
        if t.exception:
            print("Thread exception:", t.exception, file=sys.stderr)
