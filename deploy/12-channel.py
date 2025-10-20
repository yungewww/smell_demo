#!/usr/bin/env python
"""
Serial Data Collection CSV

Collects raw data in CSV form over a serial connection and saves them to files.

Install dependencies:

    python -m pip install pyserial

The first line should be header information. Each sample should be on a newline.
Here is a raw accelerometer data sample (in m/s^2):

    accX,accY,accZ
    -0.22,0.82,10.19
    -0.05,0.77,9.63
    -0.01,1.10,8.50
    ...

The end of the sample should contain an empty line (e.g. \r\n\r\n).

Call this script as follows:

    python serial-data-collect-csv.py
    
Author: Shawn Hymel (EdgeImpulse, Inc.)
Date: June 17, 2022
License: Apache-2.0 (apache.org/licenses/LICENSE-2.0)


python3 iclr-serial-data-collect-csv.py -p /dev/cu.SLAB_USBtoUART -b 115200 -d ./iclr -l "ambient"


"""

import argparse
import os
import uuid

# Third-party libraries
import serial
import serial.tools.list_ports
import time

# Settings
DEFAULT_BAUD = 115200       # Must match transmitting program baud rate
DEFAULT_LABEL = "_unknown"  # Label prepended to all CSV files

TARGET_COLUMNS = ["NO2", "C2H5OH", "VOC", "CO"]

# Generate unique ID for file (last 12 characters from uuid4 method)
uid = str(uuid.uuid4())[-12:]

# Create a file with unique filename and write CSV data to it
def write_csv(data, dir, label):

    # Keep trying if the file exists
    # exists = True
    # while exists:
    filename = label + "." + uid + ".csv"
    
    # Create and write to file if it does not exist
    out_path = os.path.join(dir, filename)

    with open(out_path, 'a+') as file:
        file.write(data)
    print("Data written to:", out_path)
        # if not os.path.exists(out_path):
        #     exists = False
        #     try:
        #     except IOError as e:
        #         print("ERROR", e)
        #         return
    

# Command line arguments
parser = argparse.ArgumentParser(description="Serial Data Collection CSV")
parser.add_argument('-p',
                    '--port',
                    dest='port',
                    type=str,
                    required=True,
                    help="Serial port to connect to")
parser.add_argument('-b',
                    '--baud',
                    dest='baud',
                    type=int,
                    default=DEFAULT_BAUD,
                    help="Baud rate (default = " + str(DEFAULT_BAUD) + ")")
parser.add_argument('-d',
                    '--directory',
                    dest='directory',
                    type=str,
                    default="./data",
                    help="Output directory for files (default = .)")
parser.add_argument('-l',
                    '--label',
                    dest='label',
                    type=str,
                    default=DEFAULT_LABEL,
                    help="Label for files (default = " + DEFAULT_LABEL + ")")
parser.add_argument('-t', 
                    '--time',
                    dest="time",
                    type=int, 
                    required=True, 
                    help="Measurement Time")

                    
# Print out available serial ports
print()
print("Available serial ports:")
available_ports = serial.tools.list_ports.comports()
for port, desc, hwid in sorted(available_ports):
    print("  {} : {} [{}]".format(port, desc, hwid))
    
# Parse arguments
args = parser.parse_args()
port = args.port
baud = args.baud
out_dir = args.directory
label = args.label
duration = args.time

# Configure serial port
ser = serial.Serial()
ser.port = port
ser.baudrate = baud

# Attempt to connect to the serial port
try:
    ser.open()
except Exception as e:
    print("ERROR:", e)
    exit()
print()
print("Connected to {} at a baud rate of {}".format(port, baud))
print("Press 'ctrl+c' to exit")

# Serial receive buffer
rx_buf = b''

# Make output directory
try:
    os.makedirs(out_dir)
except FileExistsError:
    pass

# Loop forever (unless ctrl+c is captured)

header = "NO2,C2H5OH,VOC,CO,Alcohol,LPG,Benzene,Temperature,Pressure,Humidity,Gas_Resistance,Altitude"
write_csv(header + "\n", out_dir, label)

try:
    start_time = None
    first_measurement = False
    while True:
        if first_measurement:
            if time.time()-start_time > duration:
                end_time = time.time()
                break

        if ser.in_waiting > 0:
            while ser.in_waiting:

                rx_buf += ser.read()

                # Check for end of line
                if rx_buf[-2:] == b'\r\n':
                    buf_str = rx_buf.decode('utf-8').replace('\r', '').strip()
                    rx_buf = b''  # clear buffer regardless

                    if not buf_str:
                        continue  # skip blank lines

                    parts = buf_str.split(',')
                    values = []

                    for part in parts:
                        if ':' in part:
                            _, val = part.split(':', 1)
                            try:
                                float(val.strip())  # validate numeric
                                values.append(val.strip())
                            except ValueError:
                                break
                        else:
                            break

                    if len(values) == 12:
                        csv_line = ','.join(values) + '\n'
                        write_csv(csv_line, out_dir, label)
                        if not first_measurement:
                            start_time = time.time()
                            first_measurement = True
                    else:
                        print(len(values))
                        print("Skipping malformed row:", buf_str)


# Look for keyboard interrupt (ctrl+c)
except KeyboardInterrupt:
    pass

# Close serial port
print("Closing serial port\n")

try:
    elapsed = round(start_time - end_time)   # total elapsed seconds, rounded
    mins, secs = divmod(elapsed, 60)
    mins = max(0, mins)
    print(f"Ran for {mins} mins {secs} secs\n")
except:
    pass

print(f"Data written to {out_dir}/{label}.{uid}.csv\n")
ser.close()