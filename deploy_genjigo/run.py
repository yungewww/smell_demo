import os

os.chdir(os.path.dirname(__file__))
import json
import re
import uuid
import datetime

from serialcollect import twelve_channel

CONFIG_PATH = "config.json"

# Retrieve stimuli push
_STIM_RE = re.compile(r"^\s*([A-Za-z][A-Za-z _-]*?)(\d+(?:\.\d+)?)?\s*$")
SEP = "_"  # change to "" if you want no separator between multiple items


def format_stimuli(d):
    vals = [d.get(k, "").strip() for k in ("stimuli1", "stimuli2", "stimuli3")]
    vals = [v for v in vals if v]

    if not vals:
        return ""

    parsed = []
    for v in vals:
        m = _STIM_RE.match(v)
        if not m:
            raise ValueError(
                f"Bad stimulus: {v!r} (expected like 'apple' or 'apple50')"
            )
        name, num = m.group(1).strip(), (m.group(2) or "")
        parsed.append((name, num))

    # Single: return just the name (no percentage)
    if len(parsed) == 1:
        return parsed[0][0]

    if len(parsed) == 0:
        return "ambient"

    # Multiple: sort by name, then concatenate each as name+number
    parsed.sort(key=lambda t: t[0].lower())
    return SEP.join(name + num for name, num in parsed)


def standardize_time_seconds(d):
    """
    Expects:
      d = {"time": "<number>", "time_units": "seconds|minutes|milliseconds", ...}

    Returns the time in seconds as a float and (optionally) updates d in-place
    to {"time": "<seconds-as-string>", "time_units": "seconds"}.
    """
    val = float(d.get("time", 0))
    if val == 0:
        raise ValueError("Must input a valid integer in config file.")

    unit = (d.get("time_units") or "seconds").strip().lower()

    multipliers = {
        "seconds": 1,
        "second": 1,
        "s": 1,
        "minutes": 60,
        "minute": 60,
        "m": 60,
        "milliseconds": 1e-3,
        "millisecond": 1e-3,
        "ms": 1e-3,
    }

    if unit not in multipliers:
        raise ValueError(
            f"Unsupported time unit: {unit!r}. Use seconds, minutes, or milliseconds."
        )

    seconds = val * multipliers[unit]

    # If you want to standardize the dict too:
    d["time"] = str(seconds)
    d["time_units"] = "seconds"

    return seconds


if __name__ == "__main__":
    # Read config file
    config_path = CONFIG_PATH
    with open(config_path, "r") as c:
        config = json.load(c)

    # Retrieve stimuli push
    _STIM_RE = re.compile(r"^\s*([A-Za-z][A-Za-z _-]*?)(\d+(?:\.\d+)?)?\s*$")
    SEP = "_"  # change to "" if you want no separator between multiple items

    measurement_name = format_stimuli(config)
    uid = uid = str(uuid.uuid4())[-12:]
    unique_tag = measurement_name + "." + uid
    if config["data_dir"]:
        data_dir = os.path.join(config["data_dir"], measurement_name)
    else:
        data_dir = os.path.join("./data", measurement_name)

    try:
        os.makedirs(data_dir)
    except FileExistsError:
        pass

    output_csv_path = os.path.join(data_dir, unique_tag) + ".csv"
    output_json_path = os.path.join(data_dir, unique_tag) + ".json"

    try:
        twelve_channel(
            config["port"],
            config["baudrate"],
            output_csv_path,
            standardize_time_seconds(config),
        )
    except Exception as e:
        print("Error in twelve_channel:", e)

    # check if output csv has at least one data point, and if so write to output json

    def has_one_data_point(file_path) -> bool:
        with open(file_path, "r") as f:
            for i, _ in enumerate(f, 1):
                if i > 1:
                    return True
            return False

    if has_one_data_point(output_csv_path):
        # add current data and time to config and save to output json
        config["date"] = str(datetime.datetime.now())
        with open(output_json_path, "w") as f:
            json.dump(config, f, sort_keys=True, indent=1)
    else:
        print("No data collected")
