"""
grbl_uploader.py
----------------
A lightweight Python library to upload and stream G-code files to a GRBL controller.

Compatible with GRBL 1.1+
Tested with: Python 3.9+, Windows / macOS / Linux

Author: [Your Name or Team]
License: MIT
"""

import serial
import time
from threading import Event

class GRBLUploader:
    """A simple wrapper for streaming G-code to a GRBL controller."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        """
        Initialize GRBLUploader with serial port configuration.

        :param port: Serial port path (e.g. 'COM13' on Windows or '/dev/ttyUSB0' on Linux)
        :param baudrate: Communication speed, default 115200
        :param timeout: Serial timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    # ------------------------------------------------------------------------

    @staticmethod
    def _remove_comment(line: str) -> str:
        """Remove comments starting with ';'."""
        return line.split(';', 1)[0] if ';' in line else line

    @staticmethod
    def _remove_eol_chars(line: str) -> str:
        """Strip trailing whitespace and newlines."""
        return line.strip()

    # ------------------------------------------------------------------------

    def _send_wake_up(self):
        """Send wake-up sequence to initialize GRBL."""
        self.ser.write(b"\r\n\r\n")
        time.sleep(2)
        self.ser.flushInput()

    def _wait_for_idle(self, line: str):
        """Poll GRBL status until 'Idle' is detected."""
        Event().wait(1)

        # ignore special setup commands
        if line in ("$X", "$$"):
            return

        idle_counter = 0
        while True:
            self.ser.reset_input_buffer()
            self.ser.write(b"?\n")
            response = self.ser.readline().decode("utf-8", errors="ignore").strip()

            if response and "Idle" in response:
                idle_counter += 1

            if idle_counter > 10:
                break

    # ------------------------------------------------------------------------

    def connect(self):
        """Open serial connection to GRBL."""
        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        self._send_wake_up()
        print(f"[CONNECTED] to GRBL at {self.port}")

    def disconnect(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[DISCONNECTED]")

    # ------------------------------------------------------------------------

    def send_line(self, line: str):
        """Send one line of G-code and wait for response."""
        clean_line = self._remove_eol_chars(self._remove_comment(line))
        if not clean_line:
            return None

        cmd = (clean_line + "\n").encode("utf-8")
        self.ser.write(cmd)
        self._wait_for_idle(clean_line)
        response = self.ser.readline().decode("utf-8", errors="ignore").strip()
        print(f"→ {clean_line} | ← {response}")
        return response

    # ------------------------------------------------------------------------

    def stream_file(self, filepath: str):
        """Stream G-code file to GRBL using buffer-aware streaming (much faster)."""
        GRBL_BUFFER_SIZE = 128  # GRBL serial buffer size in bytes
        RX_BUFFER_SIZE = GRBL_BUFFER_SIZE
        RX_BUFFER_FREE = RX_BUFFER_SIZE
        RX_BUFFER_USED = 0

        with open(filepath, "r") as f:
            lines = [self._remove_eol_chars(self._remove_comment(l)) for l in f if l.strip()]

        i = 0
        responses = []
        while i < len(lines):
            line = lines[i]
            if not line:
                i += 1
                continue
            cmd = (line + "\n").encode("utf-8")
            cmd_len = len(cmd)

            # Send next line only if enough buffer space
            if RX_BUFFER_USED + cmd_len <= GRBL_BUFFER_SIZE:
                self.ser.write(cmd)
                RX_BUFFER_USED += cmd_len
                i += 1
            else:
                # Read any available responses to free buffer
                response = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if response:
                    print(f"← {response}")
                    if response == "ok":
                        RX_BUFFER_USED -= cmd_len  # Free one command’s worth of space
                    responses.append(response)

        # Drain remaining responses
        while True:
            response = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if not response:
                break
            print(f"← {response}")
            responses.append(response)

        print("[UPLOAD COMPLETE]")
        return responses


# ------------------------------------------------------------------------
# Example CLI usage (only runs if executed directly)
# ------------------------------------------------------------------------
if __name__ == "__main__":
    uploader = GRBLUploader(port="COM13")
    uploader.connect()
    uploader.stream_file("grbl_test.gcode")
    uploader.disconnect()
