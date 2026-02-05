"""
Digilent AnalogIn acquisition thread using pydwf.

- Opens the first available Digilent WaveForms device using openDwfDevice.
- Configures AnalogIn for continuous streaming (Record mode with recordLength=0.0).
- Reads available samples in a loop and pushes numpy arrays of shape (n_channels, n_samples)
  into a queue (dropping oldest on overflow), with basic data-quality checks (NaN guard).
- Robust start/stop/error handling suitable for integration in a PyQt GUI.

Dependencies:
  - pydwf==1.1.19 (tested API)
  - numpy

Typical usage from GUI:
  q = queue.Queue(maxsize=10)
  th = DigilentAnalogInThread(data_queue=q, channels=[0,1], sample_rate=20000.0,
                              channel_range=5.0, poll_interval_s=0.002)
  th.start()
  ... periodically read q.get_nowait() in a QTimer to update plots ...
  th.stop(); th.join()
"""
from __future__ import annotations

import threading
import queue
import time
import numpy as np
from typing import List, Optional, Tuple

try:
    from pydwf import DwfLibrary, DwfAcquisitionMode, PyDwfError
    from pydwf.utilities import openDwfDevice
except Exception as e:  # pragma: no cover
    # Lazy import error; actual GUI may import this module without a device present.
    raise


class DigilentAnalogInThread(threading.Thread):
    """Continuous AnalogIn acquisition thread for Digilent WaveForms devices (pydwf).

    Produces blocks of float voltage samples to a queue as 2D numpy arrays
    with shape (n_channels, n_samples).
    """

    def __init__(
        self,
        data_queue: Optional[queue.Queue] = None,
        channels: Optional[List[int]] = None,
        sample_rate: float = 20000.0,
        channel_range: float = 5.0,
        buffer_size: Optional[int] = None,
        poll_interval_s: float = 0.01,  # Increased from 0.001 to reduce overhead
        name: str = "DigilentAnalogInThread",
        daemon: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(name=name, daemon=daemon)
        self._stop_event = threading.Event()
        self.queue = data_queue if data_queue is not None else queue.Queue(maxsize=10)
        self.channels = channels if channels is not None else [0]
        self.sample_rate = float(sample_rate)
        self.channel_range = float(channel_range)
        self.buffer_size = buffer_size  # if None, device default/max used
        self.poll_interval_s = float(poll_interval_s)
        self.verbose = verbose

        self.exception: Optional[BaseException] = None
        self._device = None
        self._analogIn = None

    def stop(self) -> None:
        """Request the thread to stop."""
        self._stop_event.set()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[DigilentAnalogIn] {msg}")

    def run(self) -> None:  # noqa: C901 - single-thread loop logic
        try:
            dwf = DwfLibrary()
            with openDwfDevice(dwf) as device:
                self._device = device
                ai = device.analogIn
                self._analogIn = ai

                # Basic reset
                ai.reset()

                # Determine channel count and validate requested channels
                total_ch = ai.channelCount()
                for ch in self.channels:
                    if ch < 0 or ch >= total_ch:
                        raise ValueError(f"Requested channel index {ch} out of range [0, {total_ch-1}]")

                # Enable requested channels and set range
                for ch in range(total_ch):
                    ai.channelEnableSet(ch, ch in self.channels)
                for ch in self.channels:
                    ai.channelRangeSet(ch, self.channel_range)

                # Configure sampling
                # Frequency
                # Clamp to allowed range if necessary
                fmin, fmax = ai.frequencyInfo()
                sr = min(max(self.sample_rate, fmin), fmax)
                if abs(sr - self.sample_rate) > 1e-9:
                    self._log(f"Adjusted sample_rate from {self.sample_rate} to device limits {sr} Hz [{fmin}, {fmax}]")
                ai.frequencySet(sr)

                # Buffer size (device-side circular buffer) - Use reasonable default based on sample rate
                if self.buffer_size is not None:
                    bmin, bmax = ai.bufferSizeInfo()
                    bsz = int(max(min(self.buffer_size, bmax), bmin))
                    if bsz != self.buffer_size:
                        self._log(f"Adjusted buffer_size from {self.buffer_size} to device limits {bsz} [{bmin}, {bmax}]")
                    ai.bufferSizeSet(bsz)
                else:
                    # Set reasonable default buffer based on sample rate (like projectAD approach)
                    # Use ~0.1 seconds worth of samples as buffer
                    default_buffer = int(sr * 0.1)
                    bmin, bmax = ai.bufferSizeInfo()
                    bsz = int(max(min(default_buffer, bmax), bmin))
                    ai.bufferSizeSet(bsz)
                    self._log(f"Set default buffer_size to {bsz} samples (~0.1s at {sr:.0f} Hz)")

                # Acquisition mode: Record with recordLength=0 for continuous stream
                ai.acquisitionModeSet(DwfAcquisitionMode.Record)
                ai.recordLengthSet(0.0)

                # Start acquisition
                ai.configure(reconfigure=True, start=True)
                self._log("Acquisition started (Record mode, recordLength=0)")

                # Producer loop
                while not self._stop_event.is_set():
                    # Request status without bulk data to query availability
                    ai.status(False)
                    available, lost, corrupt = ai.statusRecord()

                    if lost or corrupt:
                        self._log(f"Warning: record status: available={available}, lost={lost}, corrupt={corrupt}")

                    if available > 0:
                        # Only process if we have a reasonable amount of data to avoid too many small blocks
                        min_samples = max(1, int(sr * 0.001))  # At least 1ms worth of data
                        if available >= min_samples:
                            # Retrieve available samples for each channel
                            blocks: List[np.ndarray] = []
                            for ch in self.channels:
                                data = ai.statusData(ch, available)  # ndarray float Volts
                                # Ensure we got the expected count (defensive)
                                if data.size != available:
                                    self._log(f"Channel {ch}: expected {available} samples, got {data.size}")
                                blocks.append(np.asarray(data, dtype=np.float32))

                            # Stack into shape (n_channels, n_samples)
                            block = np.vstack(blocks)

                            # Data quality check: NaN guard
                            if np.isnan(block).any():
                                self._log("NaN detected in acquired block; dropping this block.")
                            else:
                                # Queue with drop-oldest on overflow
                                try:
                                    self.queue.put_nowait(block)
                                except queue.Full:
                                    try:
                                        _ = self.queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                    self.queue.put_nowait(block)
                        else:
                            # Not enough samples yet, wait a bit more
                            time.sleep(self.poll_interval_s * 0.5)
                    else:
                        # No data yet; brief sleep to avoid busy-wait
                        time.sleep(self.poll_interval_s)

                # Graceful stop: stop acquisition
                try:
                    ai.configure(reconfigure=False, start=False)
                except Exception:
                    pass
                self._log("Acquisition stopped.")

        except (PyDwfError, Exception) as e:
            # Save exception for GUI to inspect and stop thread
            self.exception = e
            self._log(f"Exception: {e}")
        finally:
            self._analogIn = None
            self._device = None


# Optional simple manual test (run standalone)
if __name__ == "__main__":  # pragma: no cover
    import sys

    q: queue.Queue = queue.Queue(maxsize=3)
    t = DigilentAnalogInThread(
        data_queue=q,
        channels=[0],
        sample_rate=20000.0,
        channel_range=5.0,
        poll_interval_s=0.002,
        verbose=True,
    )
    t.start()
    t0 = time.time()
    try:
        while time.time() - t0 < 5.0:
            try:
                blk = q.get(timeout=0.2)
                print(f"Got block shape={blk.shape}, min={blk.min():.3f}V, max={blk.max():.3f}V")
            except queue.Empty:
                pass
    finally:
        t.stop()
        t.join(timeout=2.0)
        if t.exception:
            print("Thread exception:", t.exception, file=sys.stderr)
