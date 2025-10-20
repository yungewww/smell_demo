def twelve_channel(port, baud, outcsv, duration):
    import serial, serial.tools.list_ports, time

    def write_csv(data, file_path):
        with open(file_path, 'a+', encoding='utf-8') as f:
            f.write(data)

    # List ports (avoid shadowing the function arg!)
    print("\nAvailable serial ports:")
    available_ports = serial.tools.list_ports.comports()
    for dev, desc, hwid in sorted((p.device, p.description, p.hwid) for p in available_ports):
        print(f"  {dev} : {desc} [{hwid}]")

    # Configure serial port
    ser = serial.Serial()
    ser.port = port                 # <- uses the function argument you intended
    ser.baudrate = baud
    ser.timeout = 1                 # non-blocking-ish reads if logic ever changes

    try:
        ser.open()
    except Exception as e:
        print("ERROR:", e)
        return
    print(f"\nConnected to {port} at a baud rate of {baud}")
    print("Press 'ctrl+c' to exit")

    rx_buf = b''
    header = "NO2,C2H5OH,VOC,CO,Alcohol,LPG,Benzene,Temperature,Pressure,Humidity,Gas_Resistance,Altitude"
    write_csv(header + "\n", outcsv)

    try:
        start_time = None
        first_measurement = False

        while True:
            # If we’ve started measuring, stop after duration seconds
            if first_measurement and (time.time() - start_time > duration):
                end_time = time.time()
                break

            if ser.in_waiting > 0:
                while ser.in_waiting:
                    b = ser.read()
                    if not b:
                        break
                    rx_buf += b

                    # Only check for EOL when we have at least 2 bytes
                    if len(rx_buf) >= 2 and rx_buf[-2:] == b'\r\n':
                        buf_str = rx_buf.decode('utf-8', errors='ignore').replace('\r', '').strip()
                        rx_buf = b''

                        if not buf_str:
                            continue

                        parts = buf_str.split(',')
                        values = []
                        for part in parts:
                            if ':' in part:
                                _, val = part.split(':', 1)
                                val = val.strip()
                                try:
                                    float(val)  # validate numeric
                                    values.append(val)
                                except ValueError:
                                    values = []
                                    break
                            else:
                                values = []
                                break

                        if len(values) == 12:
                            write_csv(','.join(values) + '\n', outcsv)
                            if not first_measurement:
                                start_time = time.time()
                                first_measurement = True
                                print(f"Writing to {outcsv}")
                            print(".", end="")
                            
                        else:
                            # Optional: debug malformed lines
                            # print(f"Skipping malformed row ({len(values)} vals): {buf_str}")
                            pass

    except KeyboardInterrupt:
        end_time = time.time() if first_measurement else None
    finally:
        print("Closing serial port\n")
        try:
            ser.close()
        except Exception:
            pass

    # Elapsed time printout
    try:
        if first_measurement and end_time is not None:
            elapsed = round(end_time - start_time)
            mins, secs = divmod(max(0, elapsed), 60)
            print(f"Ran for {mins} mins {secs} secs\n")
    except Exception:
        pass

    print(f"Data written to {outcsv}\n")
